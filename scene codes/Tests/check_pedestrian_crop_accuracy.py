#!/usr/bin/env python3
"""
Ground-truth check for /occupancy_grid/pedestrian_crop (see
extract_pedestrian_crop.py + pedestrian_crop_view.py).

Same "predict independently via fresh geometry, then compare" pattern as
check_occupancy_grid_accuracy.py, but routed through a DIFFERENT
intermediate frame on purpose.

WHY NOT JUST REUSE THE SAME TRANSFORM CHAIN
------------------------------------------------------------------
The real pipeline computes the crop as: pedestrian world pose --
[pedestrian_crop_view.py's world_pose_to_base_link] --> pedestrian pose in
base_link -- [extract_pedestrian_crop.py's local_to_base, used internally
during resampling] --> base_link coordinates -> looked up in the base grid.
If either of those two transforms has a sign error, an axis swap, or a bug
in how array indices map to physical positions, and this checker predicted
the same way (even by "independently" re-deriving the identical formula), a
consistent bug in the real chain would just agree with an equally-shaped
prediction -- the check would pass while being wrong.

So this checker's prediction is computed a different way: cube geometry
stays in WORLD frame throughout (never transformed into base_link at all),
the sensor's position is placed in world frame from the robot's world pose,
and each output cell's world position comes DIRECTLY from the pedestrian's
OWN world pose -- never via a computed "pedestrian pose in base_link"
intermediate. If the real system's world->base_link or local->base_link
transform is wrong, this independently-routed prediction will disagree
with it, because it never passes through that same intermediate value.

The small rotation / ray-intersection helpers below are therefore
deliberately DUPLICATED from check_occupancy_grid_accuracy.py rather than
imported -- reusing that code here would defeat the purpose. Only shared
SIZE/geometry CONSTANTS (crop dimensions, sensor mount offset, FOV, far
clip, tolerances already empirically tuned in check_occupancy_grid_accuracy.py)
are reused, same "reuse the constant, not the logic" pattern as before.

REQUIRES pedestrian_crop_view.py to already be running, publishing
/occupancy_grid/pedestrian_crop, started with the EXACT SAME
--ped-pos/--ped-yaw/--robot-pos/--robot-yaw as given to this script --
otherwise this is comparing against a different assumed scenario than the
one actually running.

Usage (all world-frame, straight off Gazebo's Component Inspector):

    python3 check_pedestrian_crop_accuracy.py \
        --cube-pos X Y Z --cube-size SX SY SZ --cube-yaw YAW_RAD \
        --robot-pos X Y Z --robot-yaw YAW_RAD \
        --ped-pos X Y Z --ped-yaw YAW_RAD
"""
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

import build_occupancy_grid as bog
import extract_pedestrian_crop as cropmod

BOUNDARY_TOL = bog.RESOLUTION
OCCUPIED_SHELL_TOL = 0.5 * bog.RESOLUTION
# Both values carried over as-is from check_occupancy_grid_accuracy.py, where
# OCCUPIED_SHELL_TOL was narrowed from an initial 1.5 cells down to 0.5 cells
# based on live diagnostic evidence (real occupied cells landed within ~1cm
# of the analytic surface -- near-noiseless simulated depth, see that
# script's history). Same physical sensor, same expected precision, so the
# same tolerance applies here.


def yaw_rot_2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def cube_footprint_world(cube_pos, cube_size, cube_yaw):
    """Cube's 4 top-down footprint corners, in WORLD frame, CCW order. No
    robot transform at all -- deliberately staying in world frame."""
    hx, hy = cube_size[0] / 2.0, cube_size[1] / 2.0
    local_corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    R = yaw_rot_2d(cube_yaw)
    return (R @ local_corners.T).T + np.array(cube_pos[:2])


def ray_quad_intersection_range(sx, sy, dx, dy, corners, eps=1e-12):
    """Same ray/segment intersection as check_occupancy_grid_accuracy.py,
    duplicated (see module docstring for why)."""
    n = corners.shape[0]
    best_t = np.full(dx.shape, np.inf)
    for i in range(n):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        rx, ry = x1 - sx, y1 - sy
        det = ex * dy - ey * dx
        valid = np.abs(det) > eps

        t = np.full(dx.shape, np.inf)
        u = np.full(dx.shape, np.inf)
        t[valid] = (ex * ry - ey * rx) / det[valid]
        u[valid] = (dx[valid] * ry - dy[valid] * rx) / det[valid]

        hit = valid & (t >= 0) & (u >= 0) & (u <= 1)
        best_t = np.where(hit & (t < best_t), t, best_t)
    return best_t


def build_prediction(cube_pos, cube_size, cube_yaw, robot_pos, robot_yaw, ped_pos, ped_yaw):
    """
    Returns (predicted, categories, debug) in the (CROP_ROWS, CROP_COLS)
    pedestrian-local frame -- same shapes/semantics as
    check_occupancy_grid_accuracy.py's build_prediction, just computed via
    the independently-routed world-frame path described in this module's
    docstring.
    """
    # Sensor position in WORLD frame, from the robot's world pose + the
    # fixed base_link-frame mount offset, rotated into world orientation.
    R_robot = yaw_rot_2d(robot_yaw)
    sensor_world = np.array(robot_pos[:2]) + R_robot @ np.array(bog.CAMERA_XY_OFFSET_BASE)
    sensor_heading_world = robot_yaw  # camera assumed fixed-forward on the robot, no separate pan

    footprint_world = cube_footprint_world(cube_pos, cube_size, cube_yaw)

    # Each output cell's WORLD position, computed DIRECTLY from the
    # pedestrian's own world pose -- never via a "pedestrian pose in
    # base_link" intermediate.
    col_idx = np.arange(cropmod.CROP_COLS)
    row_idx = np.arange(cropmod.CROP_ROWS)
    local_x = (col_idx + 0.5) * cropmod.CROP_RESOLUTION - cropmod.CROP_BEHIND
    local_y = (row_idx + 0.5) * cropmod.CROP_RESOLUTION - cropmod.CROP_SIDE
    LX, LY = np.meshgrid(local_x, local_y)  # LX[row,col], LY[row,col]

    cp, sp = np.cos(ped_yaw), np.sin(ped_yaw)
    world_x = ped_pos[0] + LX * cp - LY * sp
    world_y = ped_pos[1] + LX * sp + LY * cp

    dx, dy = world_x - sensor_world[0], world_y - sensor_world[1]
    cell_range = np.hypot(dx, dy)
    abs_bearing = np.arctan2(dy, dx)
    rel_bearing = np.arctan2(np.sin(abs_bearing - sensor_heading_world),
                              np.cos(abs_bearing - sensor_heading_world))  # wrap to [-pi, pi]

    half_fov = np.deg2rad(bog.CAMERA_HFOV_DEG) / 2.0
    visible = (np.abs(rel_bearing) <= half_fov) & (cell_range <= bog.FAR_CLIP)

    udx, udy = np.cos(abs_bearing), np.sin(abs_bearing)
    cube_hit_range = ray_quad_intersection_range(
        sensor_world[0], sensor_world[1], udx.ravel(), udy.ravel(), footprint_world
    ).reshape(LX.shape)
    has_hit = np.isfinite(cube_hit_range)

    occupied = visible & has_hit & (np.abs(cell_range - cube_hit_range) <= OCCUPIED_SHELL_TOL)
    free = visible & (cell_range < (cube_hit_range - BOUNDARY_TOL))
    occluded = visible & has_hit & (cell_range > (cube_hit_range + OCCUPIED_SHELL_TOL))
    out_of_fov = ~visible
    boundary = visible & ~occupied & ~free & ~occluded

    predicted = np.full(LX.shape, bog.UNKNOWN, dtype=np.int8)
    predicted[occupied] = bog.OCCUPIED
    predicted[free] = bog.FREE

    categories = {
        'occupied': occupied, 'free': free, 'occluded': occluded,
        'out_of_fov': out_of_fov, 'boundary': boundary,
    }
    debug = {'cube_hit_range': cube_hit_range, 'cell_range': cell_range}
    return predicted, categories, debug


POSE_MISMATCH_TOL = 1e-3  # meters/radians -- allows for float formatting round-trip, not real slack


def parse_pose_stamp(frame_id: str):
    """
    Parses the pose values pedestrian_crop_view.py stamps into frame_id (see
    that script's publish_crop docstring). Returns None if the field isn't
    present -- e.g. an older viewer version, or a genuinely different topic.
    """
    if ';' not in frame_id:
        return None
    fields = dict(kv.split('=') for kv in frame_id.split(';')[1:] if '=' in kv)
    try:
        return {k: float(v) for k, v in fields.items()}
    except (ValueError, KeyError):
        return None


class CropChecker(Node):
    def __init__(self, predicted, categories, debug, expected_pose):
        super().__init__('pedestrian_crop_accuracy_checker')
        self.predicted = predicted
        self.categories = categories
        self.debug = debug
        self.expected_pose = expected_pose
        self.done = False
        self.create_subscription(OccupancyGrid, '/occupancy_grid/pedestrian_crop', self.callback, 10)

    def callback(self, msg: OccupancyGrid):
        if self.done:
            return
        if msg.info.width != cropmod.CROP_COLS or msg.info.height != cropmod.CROP_ROWS:
            print(f'WARNING: received crop is {msg.info.width}x{msg.info.height}, expected '
                  f'{cropmod.CROP_COLS}x{cropmod.CROP_ROWS} -- did the crop size change since '
                  f'this checker was written? Aborting comparison.')
            self.done = True
            return

        stamped = parse_pose_stamp(msg.header.frame_id)
        if stamped is None:
            print('WARNING: the live crop message does not carry a pose stamp (older '
                  'pedestrian_crop_view.py version?) -- cannot verify it matches the poses '
                  'given to this script. Proceeding anyway, but treat the result with caution.')
        else:
            mismatches = []
            for key, expected_val in self.expected_pose.items():
                actual_val = stamped.get(key)
                if actual_val is None or abs(actual_val - expected_val) > POSE_MISMATCH_TOL:
                    mismatches.append((key, expected_val, actual_val))
            if mismatches:
                print('\n*** ABORTING: pedestrian_crop_view.py is running with DIFFERENT pose '
                      'values than were given to this checker -- likely still running from an '
                      'earlier launch. Restart pedestrian_crop_view.py with the exact same '
                      '--ped-pos/--ped-yaw/--robot-pos/--robot-yaw as this command, then rerun. ***')
                print(f'{"field":<18}{"expected":>14}{"actual (live)":>16}')
                for key, exp, act in mismatches:
                    act_str = f'{act:.6f}' if act is not None else 'MISSING'
                    print(f'{key:<18}{exp:>14.6f}{act_str:>16}')
                print()
                self.done = True
                return

        actual = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        print('\n--- Pedestrian crop accuracy report ---')
        print(f'Crop: {cropmod.CROP_ROWS}x{cropmod.CROP_COLS} @ {cropmod.CROP_RESOLUTION} m/cell '
              f'({cropmod.CROP_FORWARD}m ahead / {cropmod.CROP_BEHIND}m behind / '
              f'{cropmod.CROP_SIDE}m each side)\n')

        total_scored = 0
        total_correct = 0
        header = f'{"category":<12}{"cells":>10}{"agree":>10}{"agree %":>10}'
        print(header)
        print('-' * len(header))
        for name in ('occupied', 'free', 'occluded', 'out_of_fov'):
            mask = self.categories[name]
            n_cells = int(np.count_nonzero(mask))
            if n_cells == 0:
                print(f'{name:<12}{0:>10}{"--":>10}{"--":>10}')
                continue
            agree = int(np.count_nonzero(actual[mask] == self.predicted[mask]))
            pct = 100.0 * agree / n_cells
            print(f'{name:<12}{n_cells:>10}{agree:>10}{pct:>9.1f}%')
            total_scored += n_cells
            total_correct += agree

        n_boundary = int(np.count_nonzero(self.categories['boundary']))
        print(f'{"boundary*":<12}{n_boundary:>10}{"n/a":>10}{"excluded":>10}')
        print('-' * len(header))
        if total_scored > 0:
            overall_pct = 100.0 * total_correct / total_scored
            print(f'{"OVERALL":<12}{total_scored:>10}{total_correct:>10}{overall_pct:>9.1f}%')
        print('\n* boundary: within tolerance of a predicted classification transition --')
        print('  discretization noise, excluded from scoring, not an error.\n')

        has_hit = np.isfinite(self.debug['cube_hit_range'])
        actual_occ_mask = (actual == bog.OCCUPIED) & has_hit
        n_actual_occ = int(np.count_nonzero(actual_occ_mask))
        if n_actual_occ > 0:
            offsets = (self.debug['cell_range'][actual_occ_mask]
                       - self.debug['cube_hit_range'][actual_occ_mask])
            print(f'Diagnostic: {n_actual_occ} cells the real crop marked OCCUPIED along '
                  f'directions where the cube was analytically hit.')
            print(f'  Range offset from predicted surface (actual - predicted), meters:')
            print(f'    mean={offsets.mean():+.3f}  median={np.median(offsets):+.3f}  '
                  f'std={offsets.std():.3f}  min={offsets.min():+.3f}  max={offsets.max():+.3f}')
            if abs(offsets.mean()) > OCCUPIED_SHELL_TOL:
                print(f'  NOTE: mean offset exceeds the shell tolerance -- looks like a '
                      f'systematic bias, not noise. Worth tracking down.\n')
            else:
                print(f'  Mean offset is within tolerance -- consistent with the same '
                      f'near-noiseless sensor precision already established.\n')

        print('------------------------------------\n')
        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube-pos', type=float, nargs=3, required=True)
    parser.add_argument('--cube-size', type=float, nargs=3, required=True)
    parser.add_argument('--cube-yaw', type=float, default=0.0)
    parser.add_argument('--robot-pos', type=float, nargs=3, required=True)
    parser.add_argument('--robot-yaw', type=float, required=True)
    parser.add_argument('--ped-pos', type=float, nargs=3, required=True)
    parser.add_argument('--ped-yaw', type=float, required=True)
    args = parser.parse_args()

    predicted, categories, debug = build_prediction(
        args.cube_pos, args.cube_size, args.cube_yaw,
        args.robot_pos, args.robot_yaw,
        args.ped_pos, args.ped_yaw)

    expected_pose = {
        'ped_world_x': args.ped_pos[0], 'ped_world_y': args.ped_pos[1], 'ped_world_yaw': args.ped_yaw,
        'robot_world_x': args.robot_pos[0], 'robot_world_y': args.robot_pos[1], 'robot_world_yaw': args.robot_yaw,
    }

    rclpy.init()
    node = CropChecker(predicted, categories, debug, expected_pose)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()