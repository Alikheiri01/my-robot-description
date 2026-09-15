#!/usr/bin/env python3
"""
Phase C variant of build_occupancy_grid.py -- replaces the manual, one-time
CLI --exclude snapshot with a LIVE subscription to tracked pedestrian
positions, so the exclusion zone tracks a moving agent automatically
instead of needing a hand-typed circle re-specified every test.

Deliberately a SEPARATE file, not an edit to build_occupancy_grid.py --
that file stays untouched as the validated, known-good static-scene
baseline. All of its core algorithm (raycasting, direct splat, angular
gap-fill, exclusion filtering) is reused HERE UNCHANGED; only the source of
exclusion-zone positions changes, from a CLI snapshot to a live topic.

WHAT'S NEW vs build_occupancy_grid.py
------------------------------------------------------------------
- Subscribes to /odom to track the robot's own LIVE world pose (reusing
  the exact same odom-to-world composition already validated in
  hunav_model_bridge.py -- DiffDrive's odometry starts at (0,0,0)
  regardless of the robot's true spawn pose, so it must be composed with
  the known SPAWN_X/SPAWN_Y/SPAWN_YAW constants to recover real world pose).
- Subscribes to /people (people_msgs/People, published by
  hunav_model_bridge.py) to track each tracked pedestrian's LIVE
  world-frame position.
- Every incoming point-cloud frame, recomputes the exclusion zone list
  fresh from whatever the latest /odom + /people data says -- no CLI
  snapshot, no staleness if the robot or pedestrian moves.
- PEDESTRIAN_EXCLUSION_RADIUS is a single hardcoded constant (0.5m) rather
  than a per-agent CLI radius -- matches (with a small safety margin) the
  known agent radius (0.4m) in thesis_static_agent.yaml. MUST be updated by
  hand if that scenario's agent radius ever changes; there is no shared
  single source of truth between the two files for this value yet.

MAINTENANCE WARNING -- SPAWN_X/SPAWN_Y/SPAWN_YAW are duplicated here AND in
hunav_model_bridge.py, both hardcoded to match the launch file's actual
spawn_entity arguments. If that spawn pose ever changes, BOTH copies need
updating or robot-relative exclusion positions will silently go wrong in
one of the two nodes. Logged as a known fragility for the final version.

GRACEFUL START-UP: until at least one /odom message has been received,
exclusion zones are treated as empty (nothing excluded yet) rather than
blocking grid publication entirely -- a brief, low-risk window right at
node startup, not a steady-state concern.
"""
import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry
from people_msgs.msg import People

# =============================================================================
# UNCHANGED from build_occupancy_grid.py -- same validated core algorithm,
# copied verbatim. See that file for the full derivation/history of each
# constant and function.
# =============================================================================

FAR_CLIP = 8.0
CAMERA_HFOV_DEG = 73.0
CAMERA_XY_OFFSET_BASE = (0.366, 0.0)

FLOOR_Z_BASE = -0.0925

RESOLUTION = 0.10

FORWARD_MARGIN = 6.0
BEHIND_MARGIN = 2.0
SIDE_MARGIN = 4.0
SAFETY_MARGIN = 1.0

ANGULAR_RESOLUTION_DEG = 0.3

RAY_TOLERANCE = RESOLUTION * 0.5

MAX_GAP_BINS = 10
RANGE_CONTINUITY_TOL = 0.15  # meters


def _compute_base_extent() -> float:
    radius = FAR_CLIP + max(FORWARD_MARGIN, BEHIND_MARGIN, SIDE_MARGIN) + SAFETY_MARGIN
    return 2.0 * radius


BASE_EXTENT = _compute_base_extent()
GRID_N = int(np.ceil(BASE_EXTENT / RESOLUTION))
if GRID_N % 2 == 0:
    GRID_N += 1
GRID_HALF_EXTENT = (GRID_N * RESOLUTION) / 2.0

_HALF_FOV_RAD = np.deg2rad(CAMERA_HFOV_DEG) / 2.0
_N_ANGULAR_BINS = int(np.ceil(np.deg2rad(CAMERA_HFOV_DEG) / np.deg2rad(ANGULAR_RESOLUTION_DEG)))
_BIN_EDGES = np.linspace(-_HALF_FOV_RAD, _HALF_FOV_RAD, _N_ANGULAR_BINS + 1)

UNKNOWN, FREE, OCCUPIED = -1, 0, 100


def world_to_cell(x: float, y: float):
    col = int(np.floor((x + GRID_HALF_EXTENT) / RESOLUTION))
    row = int(np.floor((y + GRID_HALF_EXTENT) / RESOLUTION))
    return row, col


def cell_to_world(row: int, col: int):
    x = (col + 0.5) * RESOLUTION - GRID_HALF_EXTENT
    y = (row + 0.5) * RESOLUTION - GRID_HALF_EXTENT
    return x, y


def world_xy_to_base(x, y, robot_pos, robot_yaw):
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    dx, dy = x - robot_pos[0], y - robot_pos[1]
    return c * dx + s * dy, -s * dx + c * dy


def filter_excluded_points(points_xy: np.ndarray, exclusion_zones):
    if not exclusion_zones or points_xy.shape[0] == 0:
        return points_xy
    mask = np.ones(points_xy.shape[0], dtype=bool)
    for (zx, zy, zr) in exclusion_zones:
        d = np.hypot(points_xy[:, 0] - zx, points_xy[:, 1] - zy)
        mask &= (d > zr)
    return points_xy[mask]


def _fill_angular_gaps(obstacle_min_range: np.ndarray) -> np.ndarray:
    filled = obstacle_min_range.copy()
    n = len(filled)
    is_gap = ~np.isfinite(filled)

    i = 0
    while i < n:
        if not is_gap[i]:
            i += 1
            continue
        start = i
        while i < n and is_gap[i]:
            i += 1
        end = i
        run_len = end - start
        left_val = filled[start - 1] if start > 0 else None
        right_val = filled[end] if end < n else None
        if (left_val is not None and right_val is not None
                and run_len <= MAX_GAP_BINS
                and abs(left_val - right_val) <= RANGE_CONTINUITY_TOL):
            filled[start:end] = np.linspace(left_val, right_val, run_len + 2)[1:-1]
    return filled


def build_occupancy_grid(points_xy: np.ndarray):
    grid = np.full((GRID_N, GRID_N), UNKNOWN, dtype=np.int8)

    sx, sy = CAMERA_XY_OFFSET_BASE

    cell_centers = (np.arange(GRID_N) + 0.5) * RESOLUTION - GRID_HALF_EXTENT
    X, Y = np.meshgrid(cell_centers, cell_centers)
    cell_dx, cell_dy = X - sx, Y - sy
    cell_range = np.hypot(cell_dx, cell_dy)
    cell_angle = np.arctan2(cell_dy, cell_dx)

    in_fov = np.abs(cell_angle) <= _HALF_FOV_RAD
    in_range = cell_range <= FAR_CLIP
    visible = in_fov & in_range

    n_dropped = 0
    obstacle_min_range = np.full(_N_ANGULAR_BINS, np.inf)

    if points_xy.shape[0] > 0:
        pdx = points_xy[:, 0] - sx
        pdy = points_xy[:, 1] - sy
        prange = np.hypot(pdx, pdy)
        pangle = np.arctan2(pdy, pdx)

        p_in_envelope = (np.abs(pangle) <= _HALF_FOV_RAD) & (prange <= FAR_CLIP)
        n_dropped = int(np.count_nonzero(~p_in_envelope))

        pts_in = points_xy[p_in_envelope]
        prange_in = prange[p_in_envelope]
        pangle_in = pangle[p_in_envelope]

        if pts_in.shape[0] > 0:
            bin_idx = np.clip(np.digitize(pangle_in, _BIN_EDGES) - 1, 0, _N_ANGULAR_BINS - 1)
            np.minimum.at(obstacle_min_range, bin_idx, prange_in)
            obstacle_min_range = _fill_angular_gaps(obstacle_min_range)

            cell_bin_idx = np.clip(np.digitize(cell_angle, _BIN_EDGES) - 1, 0, _N_ANGULAR_BINS - 1)
            cell_obstacle_range = obstacle_min_range[cell_bin_idx]
            free_mask = visible & (cell_range < (cell_obstacle_range - RAY_TOLERANCE))
            grid[free_mask] = FREE

            col_idx = np.clip(
                np.floor((pts_in[:, 0] + GRID_HALF_EXTENT) / RESOLUTION).astype(int), 0, GRID_N - 1)
            row_idx = np.clip(
                np.floor((pts_in[:, 1] + GRID_HALF_EXTENT) / RESOLUTION).astype(int), 0, GRID_N - 1)
            grid[row_idx, col_idx] = OCCUPIED
        else:
            grid[visible] = FREE
    else:
        grid[visible] = FREE

    return grid, n_dropped


def decode_channels(grid: np.ndarray):
    known = grid != UNKNOWN
    occupied = grid == OCCUPIED
    return occupied, known


# =============================================================================
# NEW for Phase C -- live pose tracking + live exclusion zones
# =============================================================================

# Must match spawn_entity's -x/-y/-Y arguments in the launch file exactly --
# same constants, same caveat, as hunav_model_bridge.py.
SPAWN_X = 0.0
SPAWN_Y = 0.0
SPAWN_YAW = 0.7854

# Matches (with a small safety margin) thesis_static_agent.yaml's agent
# radius (0.4m) -- update by hand if that scenario's radius ever changes.
PEDESTRIAN_EXCLUSION_RADIUS = 0.5


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_world(odom_x: float, odom_y: float, odom_yaw: float):
    """Identical composition to hunav_model_bridge.py's own version -- see
    that file for the confirmed-bug backstory (odometry starts at (0,0,0)
    regardless of true spawn pose)."""
    c, s = math.cos(SPAWN_YAW), math.sin(SPAWN_YAW)
    world_x = SPAWN_X + odom_x * c - odom_y * s
    world_y = SPAWN_Y + odom_x * s + odom_y * c
    world_yaw = SPAWN_YAW + odom_yaw
    return world_x, world_y, world_yaw


class OccupancyGridBuilderDynamic(Node):
    def __init__(self):
        super().__init__('occupancy_grid_builder_dynamic')

        self.robot_world_pose = None  # (x, y, yaw), from /odom, live
        self.tracked_people = {}  # name -> (world_x, world_y), from /people, live

        self.pub = self.create_publisher(OccupancyGrid, '/occupancy_grid/base', 10)
        self.create_subscription(
            PointCloud2, '/depth_cam/fused/points', self.callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(People, '/people', self._people_cb, 10)

        self.get_logger().info(
            f'Dynamic occupancy grid builder started. Grid: {GRID_N}x{GRID_N} cells '
            f'@ {RESOLUTION:.2f} m/cell. Exclusion zones now come from live /odom + '
            f'/people subscriptions, not a CLI snapshot. Pedestrian exclusion radius: '
            f'{PEDESTRIAN_EXCLUSION_RADIUS} m.'
        )

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.robot_world_pose = odom_to_world(p.x, p.y, odom_yaw)

    def _people_cb(self, msg: People):
        self.tracked_people = {
            person.name: (person.position.x, person.position.y)
            for person in msg.people
        }

    def _current_exclusion_zones(self):
        """
        Recomputed fresh every call from whatever the latest /odom + /people
        data says -- no persistence, no staleness. Returns an empty list
        (nothing excluded) if the robot's own pose isn't known yet, which
        only happens briefly at startup before the first /odom message.
        """
        if self.robot_world_pose is None or not self.tracked_people:
            return []
        zones = []
        for (world_x, world_y) in self.tracked_people.values():
            bx, by = world_xy_to_base(world_x, world_y, self.robot_world_pose, self.robot_world_pose[2])
            zones.append((bx, by, PEDESTRIAN_EXCLUSION_RADIUS))
        return zones

    def callback(self, msg: PointCloud2):
        structured = pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True)
        if structured.shape[0] == 0:
            points_xy = np.zeros((0, 2))
        else:
            points_xy = np.column_stack([structured['x'], structured['y']]).astype(np.float64)

        exclusion_zones = self._current_exclusion_zones()
        points_xy = filter_excluded_points(points_xy, exclusion_zones)

        grid, n_dropped = build_occupancy_grid(points_xy)

        if n_dropped > 0:
            self.get_logger().warn(
                f'{n_dropped} obstacle points fell outside the assumed sensor '
                f'envelope (HFOV={CAMERA_HFOV_DEG} deg, far_clip={FAR_CLIP} m) and were '
                f'ignored for occupancy purposes.',
                throttle_duration_sec=5.0
            )

        self.publish_grid(grid, msg.header)

    def publish_grid(self, grid: np.ndarray, header):
        out = OccupancyGrid()
        out.header = header
        out.header.frame_id = 'base_link'
        out.info.resolution = float(RESOLUTION)
        out.info.width = GRID_N
        out.info.height = GRID_N
        out.info.origin.position.x = -GRID_HALF_EXTENT
        out.info.origin.position.y = -GRID_HALF_EXTENT
        out.info.origin.position.z = FLOOR_Z_BASE
        out.info.origin.orientation.w = 1.0
        out.data = grid.flatten(order='C').tolist()
        self.pub.publish(out)


def main():
    # No CLI args needed anymore -- exclusion zones are entirely live now.
    # Kept a bare parser (no arguments) so `--help` still works cleanly and
    # this stays consistent with every other node in this project.
    argparse.ArgumentParser(description=__doc__).parse_args()

    rclpy.init()
    node = OccupancyGridBuilderDynamic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()