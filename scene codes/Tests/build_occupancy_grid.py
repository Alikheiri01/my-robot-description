#!/usr/bin/env python3
"""
Builds a single robot-frame (base_link) occupancy grid, recomputed fresh from
scratch every incoming fused point cloud frame -- no persistence, no SLAM, per
the project's explicit "not a persistent map" decision.

Subscribes to:  /depth_cam/fused/points   (already ground-removed, obstacle-only,
                                            base_link frame -- see fuse_depth_clouds.py)
Publishes to:   /occupancy_grid/base      (nav_msgs/OccupancyGrid, for RViz + for
                                            the ground-truth checker script)

WHAT THIS IS (and isn't) -- scope check against the project plan
------------------------------------------------------------------
This is step 1 of the occupancy grid module: the whole-scene, robot-frame base
grid. It does NOT do pedestrian-centric cropping/heading-alignment yet (step 2)
-- that's a separate pure function to be built and tested next, once we have a
cube ground-truth pose to construct a closed-loop test pose against.

TWO CHANNELS, ONE ARRAY
------------------------
Rather than publishing two separate channel arrays, this reuses the standard
nav_msgs/OccupancyGrid int8 convention, which already natively encodes both of
the spec's channels in one array:
    -1  = unknown        (channel 2: known/unknown mask -> unknown)
     0  = free            (channel 2: known;  channel 1: free)
     100 = occupied        (channel 2: known;  channel 1: occupied)
This is equivalent information to two separate boolean channels and trivial to
split back out later (see `decode_channels()` at the bottom) when it's time to
feed the encoder -- so nothing is lost, we're just not maintaining two parallel
arrays for no reason during the visualization/MVP phase.

ALGORITHM -- hybrid raycast + direct splat
------------------------------------------------------------------
Pure angular raycasting alone can miss thin or steeply-angled obstacle
surfaces that fall between two angular bins without ever being the "nearest
point" in either. So this uses two passes:

  A) Direct splat: every raw obstacle point is placed directly into its own
     grid cell and marked OCCUPIED. This guarantees no obstacle point is ever
     lost to angular quantization, regardless of bin width.
  B) Raycast free-space pass: points are binned by angle from the sensor
     origin; for each angular bin we find the nearest obstacle range. Every
     cell within the sensor's FOV/range envelope that is closer than that
     bin's nearest-obstacle range is marked FREE. Cells at/behind the nearest
     obstacle are left UNKNOWN (occluded) unless a direct splat already
     marked them OCCUPIED.
  C) Everything else -- outside the FOV wedge, beyond far_clip, or an
     occluded cell behind an obstacle -- stays UNKNOWN, since the grid is
     initialized to UNKNOWN before A/B ever run.

Because the input cloud has ALREADY had the ground plane removed upstream (in
fuse_depth_clouds.py), the absence of a point along a ray up to far_clip
correctly means "clear floor, free space" -- not "no data". This is exactly
the assumption occupancy raycasting needs, and it falls out naturally from the
existing verified pipeline rather than requiring anything new.

FLAGGED ASSUMPTIONS -- status
------------------------------------------------------------------
1. CAMERA_XY_OFFSET_BASE: CONFIRMED via the xacro joint chain
   (handle_middle_joint -> depth_camera_back_joint -> left/right_camera_joint,
   cross-checked with tf2_echo) at X = +0.366m forward of base_link, Y = 0.0
   (the ±0.03m left/right offsets are symmetric and cancel for this
   single-combined-origin approximation). That's ~3-4 grid cells at 0.10m
   resolution -- not negligible, now corrected below.
2. CAMERA_HFOV_DEG = 73.0, applied as a single combined forward-facing wedge:
   confirmed as the intended, accepted simplification (same Option-A
   co-located-cameras choice already documented in the project handoff doc),
   not an oversight -- no change needed here.
3. FAR_CLIP = 8.0: duplicated from depth_camera.xacro's <clip><far>. If that
   value changes again (the handoff doc notes it already moved once, 50m ->
   8m), update it here too -- BASE_EXTENT is derived from it automatically,
   so the grid will resize correctly, but this constant itself doesn't watch
   the xacro file, it has to be updated by hand.

KNOWN FIX -- angular sampling gaps causing free-space leakage behind objects
------------------------------------------------------------------
Observed in Gazebo testing: cells directly behind a thin panel target were
showing a scattered mix of FREE and UNKNOWN instead of solidly UNKNOWN
(occluded). Root cause: the fused cloud is voxel-downsampled (VOXEL_SIZE =
0.03m in fuse_depth_clouds.py). At close range, that ~3cm point spacing
subtends an angle LARGER than this file's 0.3 deg angular bins (e.g. at 1m,
adjacent points are ~1.7 deg apart) -- leaving genuinely-empty bins between
two real points on the same continuous surface. Each empty bin reported
obstacle_min_range = inf ("no return"), which the free-space pass read as
"clear line of sight to far_clip", so free-space rays leaked straight through
those sampling gaps in the panel. Fixed below via `_fill_angular_gaps()`,
which closes small bin-runs bounded by two similar-range neighbors (same
surface, sampling artifact) while leaving large/mismatched gaps alone (real
edges, doorways, separate objects at different depths) -- see that function's
docstring for the full reasoning and tuning knobs.

EXCLUSION ZONES -- manual stand-in for the spec's actor-ID exclusion
------------------------------------------------------------------
Per the project spec's "Build now" list: "Exclusion of pedestrians/robot from
the static channel (free via sim ground truth)." This wasn't implemented
before because no dynamic actors existed yet to exclude -- but a physical
stand-in object (e.g. a cylinder placed to mark a synthetic pedestrian pose
for crop testing) exposes the same problem today: the depth camera has no way
to know an object is "just a marker" and correctly reports it as a real
obstacle, contaminating both the base grid and any crop resampled from it.

Real per-actor-ID exclusion needs Hunav's ground-truth poses, which don't
exist yet. Until then, this file accepts exclusion zones manually via CLI --
same "hand-supplied ground truth standing in for a future automated source"
pattern already used for the synthetic pedestrian pose in
pedestrian_crop_view.py. Any obstacle point whose (x, y) falls inside a given
world-frame circle is dropped before it can contribute to either the direct
splat or the raycasting pass, so line-of-sight correctly passes through
whatever used to be there.

Usage (no change to the default no-argument invocation the launch file
already uses -- exclusion zones are entirely opt-in):

    python3 build_occupancy_grid.py \
        --robot-pos X Y Z --robot-yaw YAW_RAD \
        --exclude X Y Z RADIUS [--exclude X2 Y2 Z2 RADIUS2 ...]

--robot-pos/--robot-yaw are WORLD frame, same Component Inspector values
used everywhere else, and are only required if --exclude is given (needed to
transform the exclusion center into base_link). Z components are accepted
for CLI consistency but unused (2D top-down exclusion). Same snapshot
caveat as pedestrian_crop_view.py: this transform happens once at startup,
not via a live TF lookup -- if the robot moves after this script starts,
restart it.
"""
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid

# ---------------------------------------------------------------------------
# Constants -- module-level, matching the style of fuse_depth_clouds.py
# (FLOOR_Z_BASE, VOXEL_SIZE, etc.) rather than ROS2 declared parameters, for
# MVP simplicity. Easy to promote to declared params later if needed.
# ---------------------------------------------------------------------------

FAR_CLIP = 8.0                      # meters; must match depth_camera.xacro <clip><far>
CAMERA_HFOV_DEG = 73.0              # degrees; single-camera HFOV (see assumption 2 above)
CAMERA_XY_OFFSET_BASE = (0.366, 0.0)  # meters, (x, y) in base_link -- confirmed via
                                       # xacro joint chain + tf2_echo (see docstring)

FLOOR_Z_BASE = -0.0925
# True floor height in base_link, same constant/value as fuse_depth_clouds.py's
# FLOOR_Z_BASE -- base_link sits at wheel-center height (wheel radius 0.0925m),
# not at the floor, so anything rendered/reasoned about in base_link needs this
# offset applied explicitly. Used below purely for the *published* grid's
# z-height (so it visually sits on the floor in RViz, not at wheel-center
# height) -- the grid's actual occupancy math is 2D/top-down and doesn't
# depend on z at all, only this one publish-time value does.

RESOLUTION = 0.10  # m/cell -- matches the crop resolution spec, so step 2's
                    # crop resampling only has to handle rotation, not also a
                    # resolution change

# Crop margins, duplicated here from the occupancy grid spec (6m fwd / 2m
# behind / 4m each side) purely to size the base grid -- NOT used for any
# cropping logic in this file, that's step 2.
FORWARD_MARGIN = 6.0
BEHIND_MARGIN = 2.0
SIDE_MARGIN = 4.0
SAFETY_MARGIN = 1.0  # extra buffer on top of the geometric minimum

ANGULAR_RESOLUTION_DEG = 0.3
# Chosen so that at FAR_CLIP, one angular bin's arc length (~0.3 deg * 8m in
# radians ~= 0.042m) stays comfortably under one cell (0.10m) -- fine enough
# resolution that no cell row along a ray is skipped between bins.

RAY_TOLERANCE = RESOLUTION * 0.5
# Cells within half a cell-width of a bin's nearest-obstacle range are
# treated as "at the obstacle boundary", not spuriously marked free due to
# floating point / bin-quantization noise.

MAX_GAP_BINS = 10
# Max run length of consecutive "no return" angular bins that will be
# bridged as a sampling artifact rather than left as a real gap. Sized off
# the worst-case realistic testing scenario: ~3cm voxel-downsampled point
# spacing (VOXEL_SIZE in fuse_depth_clouds.py) at ~1m range subtends about
# 1.7 deg =~ 5-6 bins at ANGULAR_RESOLUTION_DEG=0.3 -- 10 bins gives
# headroom above that. Gaps shrink (in bin-count) as range increases, so
# this single constant, sized for the closest expected range, stays
# conservative-safe further out. If testing is ever done closer than ~1m to
# an object and leakage reappears, raise this.

RANGE_CONTINUITY_TOL = 0.15  # meters
# How much a bridged gap's two bounding ranges are allowed to differ and
# still be treated as "the same surface, sampled unevenly" rather than two
# different objects at genuinely different depths (e.g. a doorway with a
# wall behind it must NOT get bridged -- that gap's bounding ranges would
# differ by far more than this).


def _compute_base_extent() -> float:
    """
    Full base grid width/height (meters), derived from far_clip + crop
    margins rather than hardcoded -- see project decision: if far_clip moves
    again (already happened once, 50m -> 8m), this recomputes automatically
    instead of silently becoming stale.

    Reasoning: a pedestrian could be sensed anywhere up to far_clip from the
    robot, and their heading-aligned crop can then extend up to
    max(margins) further in an arbitrary direction (the crop is
    pedestrian-heading-aligned, not robot-aligned, so that extension isn't
    confined to "further forward"). So the base grid must cover a radius of
    (far_clip + max_margin) from the robot in every direction, plus a safety
    margin -- hence full width/height = 2x that radius.
    """
    radius = FAR_CLIP + max(FORWARD_MARGIN, BEHIND_MARGIN, SIDE_MARGIN) + SAFETY_MARGIN
    return 2.0 * radius


BASE_EXTENT = _compute_base_extent()
GRID_N = int(np.ceil(BASE_EXTENT / RESOLUTION))
if GRID_N % 2 == 0:
    GRID_N += 1  # snap to odd so there's a true center cell at the robot origin,
                 # which makes both manual reasoning and test-pose math simpler
GRID_HALF_EXTENT = (GRID_N * RESOLUTION) / 2.0  # meters, half-width of the square grid

_HALF_FOV_RAD = np.deg2rad(CAMERA_HFOV_DEG) / 2.0
_N_ANGULAR_BINS = int(np.ceil(np.deg2rad(CAMERA_HFOV_DEG) / np.deg2rad(ANGULAR_RESOLUTION_DEG)))
_BIN_EDGES = np.linspace(-_HALF_FOV_RAD, _HALF_FOV_RAD, _N_ANGULAR_BINS + 1)

UNKNOWN, FREE, OCCUPIED = -1, 0, 100  # nav_msgs/OccupancyGrid convention


def world_to_cell(x: float, y: float):
    """
    Converts a base_link (x, y) coordinate to (row, col) grid indices, using
    the same origin convention as the published OccupancyGrid (origin at the
    grid's -x,-y corner). Exposed for reuse by the ground-truth checker
    script, so both sides agree on grid geometry by construction -- the
    thing under test there is the occupancy LOGIC, not this coordinate
    convention (same pattern as fuse_depth_clouds.py's FLOOR_Z_BASE being
    reused rather than re-derived by its own checker script).
    """
    col = int(np.floor((x + GRID_HALF_EXTENT) / RESOLUTION))
    row = int(np.floor((y + GRID_HALF_EXTENT) / RESOLUTION))
    return row, col


def cell_to_world(row: int, col: int):
    """Inverse of world_to_cell, returning the cell's center coordinate."""
    x = (col + 0.5) * RESOLUTION - GRID_HALF_EXTENT
    y = (row + 0.5) * RESOLUTION - GRID_HALF_EXTENT
    return x, y


def _fill_angular_gaps(obstacle_min_range: np.ndarray) -> np.ndarray:
    """
    Closes small runs of "no return" angular bins that are sandwiched between
    two bins reporting similar obstacle range -- almost always a point-cloud
    sampling gap on a continuous surface, not a real opening (see the "KNOWN
    FIX" section in this module's docstring for the full diagnosis). Genuine
    gaps (object edges, doorways, open space) are left untouched: either the
    empty run is too long (> MAX_GAP_BINS), or the two bounding ranges
    disagree too much (> RANGE_CONTINUITY_TOL) to plausibly be one surface.

    A run touching either end of the bin array (i.e. only one side is
    bounded) is also left unfilled -- not enough information to interpolate
    confidently, so it stays "no return" (correctly free/unknown-by-default)
    rather than risk fabricating an obstacle at the FOV edge.
    """
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
        end = i  # empty run is bins [start, end)
        run_len = end - start
        left_val = filled[start - 1] if start > 0 else None
        right_val = filled[end] if end < n else None
        if (left_val is not None and right_val is not None
                and run_len <= MAX_GAP_BINS
                and abs(left_val - right_val) <= RANGE_CONTINUITY_TOL):
            # Linearly interpolate across the gap, treating it as the same
            # continuous surface sampled a little unevenly.
            filled[start:end] = np.linspace(left_val, right_val, run_len + 2)[1:-1]
    return filled


def world_xy_to_base(x, y, robot_pos, robot_yaw):
    """
    Transforms a world-frame (x, y) into base_link -- same rotation
    convention already used in check_occupancy_grid_accuracy.py and
    pedestrian_crop_view.py. Only position matters for a circular exclusion
    zone, so no yaw/heading component here.
    """
    c, s = np.cos(robot_yaw), np.sin(robot_yaw)
    dx, dy = x - robot_pos[0], y - robot_pos[1]
    return c * dx + s * dy, -s * dx + c * dy


def filter_excluded_points(points_xy: np.ndarray, exclusion_zones):
    """
    Drops any point falling inside a given (base_x, base_y, radius) circle --
    see the module docstring's "EXCLUSION ZONES" section. Called before
    build_occupancy_grid() so excluded points never contribute to either the
    direct splat or the raycasting pass; line-of-sight correctly passes
    through whatever used to be there.
    """
    if not exclusion_zones or points_xy.shape[0] == 0:
        return points_xy
    mask = np.ones(points_xy.shape[0], dtype=bool)
    for (zx, zy, zr) in exclusion_zones:
        d = np.hypot(points_xy[:, 0] - zx, points_xy[:, 1] - zy)
        mask &= (d > zr)
    return points_xy[mask]


def build_occupancy_grid(points_xy: np.ndarray):
    """
    points_xy: (M, 2) array of obstacle points' (x, y) in base_link -- already
    ground-removed, obstacle-only points from the fused cloud. May be empty
    (M=0), which correctly yields an all-free-within-FOV, all-unknown-outside
    grid (the "empty scene" case, same one fuse_depth_clouds.py's own tests
    already check for at the point-cloud level).

    Returns:
      grid:      (GRID_N, GRID_N) np.int8 array, row=y-index, col=x-index,
                 values in {UNKNOWN, FREE, OCCUPIED}.
      n_dropped: number of input points that fell outside the assumed sensor
                 envelope (HFOV / far_clip) and were therefore ignored -- see
                 the docstring's flagged assumptions if this is ever nonzero
                 and large; it usually means CAMERA_HFOV_DEG or
                 CAMERA_XY_OFFSET_BASE don't match the real mount.
    """
    grid = np.full((GRID_N, GRID_N), UNKNOWN, dtype=np.int8)

    sx, sy = CAMERA_XY_OFFSET_BASE

    # ---- Every cell's (range, angle) relative to the sensor origin ----
    cell_centers = (np.arange(GRID_N) + 0.5) * RESOLUTION - GRID_HALF_EXTENT
    X, Y = np.meshgrid(cell_centers, cell_centers)  # X[row,col]=x, Y[row,col]=y (numpy 'xy' indexing)
    cell_dx, cell_dy = X - sx, Y - sy
    cell_range = np.hypot(cell_dx, cell_dy)
    cell_angle = np.arctan2(cell_dy, cell_dx)

    in_fov = np.abs(cell_angle) <= _HALF_FOV_RAD
    in_range = cell_range <= FAR_CLIP
    visible = in_fov & in_range  # cells the sensor could possibly have seen this frame

    # ---- Filter obstacle points to the same envelope ----
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
            # ---- Pass B setup: nearest-obstacle range per angular bin ----
            bin_idx = np.clip(np.digitize(pangle_in, _BIN_EDGES) - 1, 0, _N_ANGULAR_BINS - 1)
            np.minimum.at(obstacle_min_range, bin_idx, prange_in)
            obstacle_min_range = _fill_angular_gaps(obstacle_min_range)

            # ---- Pass B: mark free cells (closer than their bin's obstacle) ----
            cell_bin_idx = np.clip(np.digitize(cell_angle, _BIN_EDGES) - 1, 0, _N_ANGULAR_BINS - 1)
            cell_obstacle_range = obstacle_min_range[cell_bin_idx]
            free_mask = visible & (cell_range < (cell_obstacle_range - RAY_TOLERANCE))
            grid[free_mask] = FREE

            # ---- Pass A: direct splat, overwrites free with occupied where
            # an actual point landed -- always wins, run after the free pass ----
            col_idx = np.clip(
                np.floor((pts_in[:, 0] + GRID_HALF_EXTENT) / RESOLUTION).astype(int), 0, GRID_N - 1)
            row_idx = np.clip(
                np.floor((pts_in[:, 1] + GRID_HALF_EXTENT) / RESOLUTION).astype(int), 0, GRID_N - 1)
            grid[row_idx, col_idx] = OCCUPIED
        else:
            # No obstacles in range at all this frame -- everything visible is free.
            grid[visible] = FREE
    else:
        # Empty input cloud entirely -- same result, everything visible is free.
        grid[visible] = FREE

    return grid, n_dropped


def decode_channels(grid: np.ndarray):
    """
    Splits the packed OccupancyGrid-convention array back into the spec's two
    explicit boolean channels, for whenever the encoder/dataset pipeline
    needs them separately rather than packed. Not used by this node itself.
    """
    known = grid != UNKNOWN
    occupied = grid == OCCUPIED
    return occupied, known


class OccupancyGridBuilder(Node):
    def __init__(self, exclusion_zones=None):
        super().__init__('occupancy_grid_builder')
        self.exclusion_zones = exclusion_zones or []
        self.pub = self.create_publisher(OccupancyGrid, '/occupancy_grid/base', 10)
        self.create_subscription(
            PointCloud2, '/depth_cam/fused/points', self.callback, qos_profile_sensor_data)
        excl_msg = (f'{len(self.exclusion_zones)} exclusion zone(s) active (base_link): '
                    f'{self.exclusion_zones}' if self.exclusion_zones else 'no exclusion zones active')
        self.get_logger().info(
            f'Occupancy grid builder started. Grid: {GRID_N}x{GRID_N} cells '
            f'@ {RESOLUTION:.2f} m/cell ({GRID_N * RESOLUTION:.1f} m x '
            f'{GRID_N * RESOLUTION:.1f} m), centered on base_link. '
            f'FAR_CLIP={FAR_CLIP} m, HFOV={CAMERA_HFOV_DEG} deg. {excl_msg}.'
        )

    def callback(self, msg: PointCloud2):
        structured = pc2.read_points(msg, field_names=('x', 'y'), skip_nans=True)
        if structured.shape[0] == 0:
            points_xy = np.zeros((0, 2))
        else:
            # Vectorized field extraction -- same idiom as fuse_depth_clouds.py's
            # transform_to_base(), not a per-point python loop.
            points_xy = np.column_stack([structured['x'], structured['y']]).astype(np.float64)

        points_xy = filter_excluded_points(points_xy, self.exclusion_zones)

        grid, n_dropped = build_occupancy_grid(points_xy)

        if n_dropped > 0:
            self.get_logger().warn(
                f'{n_dropped} obstacle points fell outside the assumed sensor '
                f'envelope (HFOV={CAMERA_HFOV_DEG} deg, far_clip={FAR_CLIP} m) and were '
                f'ignored for occupancy purposes. If this is persistently large, '
                f'check the CAMERA_HFOV_DEG / CAMERA_XY_OFFSET_BASE assumptions '
                f'against depth_camera.xacro.',
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-pos', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                         help='Robot position in WORLD frame. Required only if --exclude is given.')
    parser.add_argument('--robot-yaw', type=float, default=None,
                         help='Robot heading in WORLD frame, radians. Required only if --exclude is given.')
    parser.add_argument('--exclude', type=float, nargs=4, action='append', default=None,
                         metavar=('X', 'Y', 'Z', 'RADIUS'),
                         help='WORLD-frame exclusion zone center (Z unused) + radius, in meters. '
                              'Repeatable for multiple zones.')
    args = parser.parse_args()
    if args.exclude and (args.robot_pos is None or args.robot_yaw is None):
        parser.error('--exclude requires --robot-pos and --robot-yaw to transform it into base_link')
    return args


def main():
    args = parse_args()
    exclusion_zones = []
    if args.exclude:
        for (x, y, z, radius) in args.exclude:
            bx, by = world_xy_to_base(x, y, args.robot_pos, args.robot_yaw)
            exclusion_zones.append((bx, by, radius))

    rclpy.init()
    node = OccupancyGridBuilder(exclusion_zones=exclusion_zones)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()