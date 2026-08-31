#!/usr/bin/env python3
"""
Ground-truth check for /occupancy_grid/base (see build_occupancy_grid.py).

Given the cube's and robot's exact world poses (read from Gazebo's Component
Inspector -- same values you'd already read for check_fused_cloud_accuracy.py),
this INDEPENDENTLY predicts, via analytic geometry, what every cell in the
grid should be, then compares against what the running node actually
publishes.

Deliberately does NOT import or reuse build_occupancy_grid.py's raycasting
logic for the prediction -- reusing the code under test would just test it
against itself. The only things imported from that module are shared
coordinate/geometry CONSTANTS (grid resolution, sensor mount offset, FOV,
far clip) -- same "reuse the constant, not the logic" pattern
check_fused_cloud_accuracy.py already uses with FLOOR_Z_BASE.

Usage example (same CLI shape as check_fused_cloud_accuracy.py -- z values in
--cube-pos / --robot-pos are accepted for consistency with that script but
unused here, since this is a 2D top-down grid):

    python3 check_occupancy_grid_accuracy.py \
        --cube-pos 2.0 0.3 0.25 --cube-size 0.5 0.5 0.5 --cube-yaw 0 \
        --robot-pos 0.0 0.0 0.0925 --robot-yaw 0.0

Run this from the same directory as build_occupancy_grid.py (or add that
directory to PYTHONPATH) so the `import build_occupancy_grid` below resolves.

FIX -- occupied prediction was over-filling the cube's interior
------------------------------------------------------------------
An earlier version of this checker predicted "occupied" as the entire
interior of the cube's footprint polygon. That's wrong: a single-viewpoint
depth sensor can never see through a solid object to its far side or
interior, so the real pipeline only ever splats points on the near-facing
surface it actually gets a return from -- a thin shell, not a filled solid.
Comparing against the full polygon interior showed up exactly as observed
in real testing: "occupied" agreement near 0% while "free"/"occluded"/
"out_of_fov" stayed at 98-100%, since those large regions barely touch the
mistake while "occupied" (small, localized right at it) took the full hit.
This is the same category of error the project's own
check_fused_cloud_accuracy.py test already ran into once and documented
(comparing a single-viewpoint cloud against a solid object's full bounding
volume instead of what that viewpoint can actually see). Fixed below:
"occupied" is now predicted as a thin shell right at each ray's first
intersection with the cube (see OCCUPIED_SHELL_TOL), and the polygon
interior beyond that shell is correctly predicted UNKNOWN/occluded, same
as the space behind the cube.
"""
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

import build_occupancy_grid as bog

BOUNDARY_TOL = bog.RESOLUTION
# Cells within one cell-width of the free/occupied transition are expected
# to disagree sometimes purely due to 0.10m discretization -- same
# "noise-level, not a bug" tolerance philosophy as check_fused_cloud_accuracy.py's
# 1-6cm bbox-edge agreement band. These are reported separately, not counted
# as errors.

OCCUPIED_SHELL_TOL = 0.5 * bog.RESOLUTION
# Diagnostic-driven, not assumed: an earlier version of this constant (1.5
# cells) assumed real splatted points would scatter meaningfully around the
# true surface and needed generous slack to catch them. Live diagnostic data
# (see the offset report this checker prints) showed the opposite -- real
# occupied cells landed within ~1cm of the analytic surface (std=0.001m),
# essentially a single cell-deep row, because the sim currently gives each
# camera perfect ground-truth depth directly (the documented Option-A
# simplification -- no simulated stereo-matching noise yet). A 1.5-cell-wide
# tolerance was therefore mostly capturing neighboring FREE/UNKNOWN cells as
# false "should be occupied" -- hence the low agreement despite the real
# splat being essentially exact. Half a cell comfortably contains the
# observed ~1cm offset without over-claiming neighboring rows.
# NOTE: this will likely need widening again once real stereo-disparity
# noise is introduced (later-phase fidelity work, see the project handoff
# doc's Section 5) -- it's tuned to today's near-noiseless ground-truth
# depth, not a universal constant.


def yaw_rot_2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def cube_footprint_in_base(cube_pos, cube_size, cube_yaw, robot_pos, robot_yaw):
    """Returns the cube's 4 top-down footprint corners in base_link, in CCW order."""
    hx, hy = cube_size[0] / 2.0, cube_size[1] / 2.0
    # CCW order: bottom-left, bottom-right, top-right, top-left
    local_corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])

    R_cube = yaw_rot_2d(cube_yaw)
    world_corners = (R_cube @ local_corners.T).T + np.array(cube_pos[:2])

    R_robot = yaw_rot_2d(robot_yaw)
    base_corners = (R_robot.T @ (world_corners - np.array(robot_pos[:2])).T).T
    return base_corners  # (4, 2), CCW (rotation preserves winding, no reflection)


def ray_quad_intersection_range(sx, sy, dx, dy, corners, eps=1e-12):
    """
    For each ray (origin (sx,sy), unit direction (dx[i],dy[i])), returns the
    nearest positive distance at which it crosses the quad's boundary, or
    np.inf if that ray never hits it. Standard ray/segment intersection via
    Cramer's rule, solved once per edge, vectorized over all rays.
    """
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


def build_prediction(footprint_corners):
    """
    Returns:
      predicted: (GRID_N, GRID_N) int8 array in the same -1/0/100 convention
                 as the real grid.
      categories: dict of {name: boolean mask} for reporting, partitioning
                 every cell into exactly one of: occupied, free, occluded,
                 out_of_fov, boundary (ambiguous, excluded from scoring).

    "occupied" is a thin shell right at each visible ray's first
    intersection with the cube -- NOT the cube's full footprint interior,
    since a single-viewpoint sensor never sees past that surface (see the
    FIX note in this module's docstring).
    """
    cell_centers = (np.arange(bog.GRID_N) + 0.5) * bog.RESOLUTION - bog.GRID_HALF_EXTENT
    X, Y = np.meshgrid(cell_centers, cell_centers)  # X[row,col], Y[row,col] -- matches bog's own convention

    sx, sy = bog.CAMERA_XY_OFFSET_BASE
    cell_dx, cell_dy = X - sx, Y - sy
    cell_range = np.hypot(cell_dx, cell_dy)
    cell_angle = np.arctan2(cell_dy, cell_dx)

    half_fov = np.deg2rad(bog.CAMERA_HFOV_DEG) / 2.0
    visible = (np.abs(cell_angle) <= half_fov) & (cell_range <= bog.FAR_CLIP)

    udx, udy = np.cos(cell_angle), np.sin(cell_angle)
    cube_hit_range = ray_quad_intersection_range(
        sx, sy, udx.ravel(), udy.ravel(), footprint_corners).reshape(X.shape)
    has_hit = np.isfinite(cube_hit_range)

    occupied = visible & has_hit & (np.abs(cell_range - cube_hit_range) <= OCCUPIED_SHELL_TOL)
    free = visible & (cell_range < (cube_hit_range - BOUNDARY_TOL))
    occluded = visible & has_hit & (cell_range > (cube_hit_range + OCCUPIED_SHELL_TOL))
    out_of_fov = ~visible
    # anything visible and not cleanly in one of the above is within the
    # tolerance band of a transition -- ambiguous, reported not scored
    boundary = visible & ~occupied & ~free & ~occluded

    predicted = np.full(X.shape, bog.UNKNOWN, dtype=np.int8)
    predicted[occupied] = bog.OCCUPIED
    predicted[free] = bog.FREE
    # occluded + out_of_fov + boundary all stay UNKNOWN (bog.UNKNOWN already
    # the init value) -- boundary cells are excluded from scoring separately,
    # not from the published value itself, since we don't know which side of
    # the transition they'd fall on.

    categories = {
        'occupied': occupied,
        'free': free,
        'occluded': occluded,
        'out_of_fov': out_of_fov,
        'boundary': boundary,
    }
    debug = {
        'cube_hit_range': cube_hit_range,  # analytic range to the cube surface along each cell's ray
        'cell_range': cell_range,          # each cell's own range from the sensor
    }
    return predicted, categories, debug


class GridChecker(Node):
    def __init__(self, predicted, categories, debug):
        super().__init__('occupancy_grid_accuracy_checker')
        self.predicted = predicted
        self.categories = categories
        self.debug = debug
        self.done = False
        self.create_subscription(OccupancyGrid, '/occupancy_grid/base', self.callback, 10)

    def callback(self, msg: OccupancyGrid):
        if self.done:
            return
        if msg.info.width != bog.GRID_N or msg.info.height != bog.GRID_N:
            print(f'WARNING: received grid is {msg.info.width}x{msg.info.height}, '
                  f'expected {bog.GRID_N}x{bog.GRID_N} -- did GRID_N change since '
                  f'this checker was written? Aborting comparison.')
            self.done = True
            return

        actual = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))

        print('\n--- Occupancy grid accuracy report ---')
        print(f'Grid: {bog.GRID_N}x{bog.GRID_N} @ {bog.RESOLUTION} m/cell '
              f'({bog.GRID_N * bog.RESOLUTION:.1f}m x {bog.GRID_N * bog.RESOLUTION:.1f}m)\n')

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
        print('\n* boundary: within one cell-width of a predicted classification')
        print('  transition -- discretization noise, excluded from scoring, not an error.\n')

        # Flag the worst offender specifically -- if "occluded" agreement is
        # low while the others are high, that's the exact failure mode the
        # angular-gap leakage bug produced, worth calling out by name.
        occ_mask = self.categories['occluded']
        if np.count_nonzero(occ_mask) > 0:
            occ_agree_pct = 100.0 * np.count_nonzero(
                actual[occ_mask] == self.predicted[occ_mask]) / np.count_nonzero(occ_mask)
            if occ_agree_pct < 95.0:
                mism = actual[occ_mask] != self.predicted[occ_mask]
                free_leak = int(np.count_nonzero(actual[occ_mask][mism] == bog.FREE))
                print(f'NOTE: occluded-region agreement is {occ_agree_pct:.1f}% -- '
                      f'{free_leak} of the disagreeing cells were reported FREE where '
                      f'occlusion was expected. This is the same failure signature as '
                      f'the angular-gap leakage bug already fixed once; if it reappears, '
                      f'check MAX_GAP_BINS / RANGE_CONTINUITY_TOL in build_occupancy_grid.py.\n')

        # ---- Diagnostic: is "occupied" disagreement noise, or a systematic bias? ----
        # Every cell the real grid marked OCCUPIED, restricted to angles where the
        # cube was actually hit analytically -- their signed range offset from the
        # predicted surface tells us whether mismatches are symmetric scatter
        # (real depth noise/quantization, tolerance is just strict) or a
        # consistent directional bias (an actual remaining offset bug).
        has_hit = np.isfinite(self.debug['cube_hit_range'])
        actual_occ_mask = (actual == bog.OCCUPIED) & has_hit
        n_actual_occ = int(np.count_nonzero(actual_occ_mask))
        if n_actual_occ > 0:
            offsets = (self.debug['cell_range'][actual_occ_mask]
                       - self.debug['cube_hit_range'][actual_occ_mask])
            print(f'Diagnostic: {n_actual_occ} cells the real grid marked OCCUPIED '
                  f'along angles where the cube was analytically hit.')
            print(f'  Range offset from predicted surface (actual - predicted), meters:')
            print(f'    mean={offsets.mean():+.3f}  median={np.median(offsets):+.3f}  '
                  f'std={offsets.std():.3f}  min={offsets.min():+.3f}  max={offsets.max():+.3f}')
            if abs(offsets.mean()) > OCCUPIED_SHELL_TOL:
                print(f'  NOTE: mean offset ({offsets.mean():+.3f}m) exceeds the shell tolerance '
                      f'({OCCUPIED_SHELL_TOL:.2f}m) -- looks like a SYSTEMATIC bias (cube/robot pose '
                      f'reading, or a residual coordinate offset), not symmetric noise. Worth tracking '
                      f'down before touching OCCUPIED_SHELL_TOL.\n')
            else:
                print(f'  Mean offset is within the shell tolerance and roughly centered on zero -- this '
                      f'looks like symmetric real-sensor noise/quantization scatter around the true '
                      f'surface, not a bug. "occupied" reads harsh here because it IS entirely a boundary '
                      f'region by definition, so noise that barely dents the large free/occluded regions '
                      f'shows up amplified in this one category\'s percentage.\n')

        print('------------------------------------\n')
        self.done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube-pos', type=float, nargs=3, required=True)
    parser.add_argument('--cube-size', type=float, nargs=3, required=True)
    parser.add_argument('--cube-yaw', type=float, default=0.0)
    parser.add_argument('--robot-pos', type=float, nargs=3, required=True)
    parser.add_argument('--robot-yaw', type=float, default=0.0)
    args = parser.parse_args()

    footprint = cube_footprint_in_base(
        args.cube_pos, args.cube_size, args.cube_yaw, args.robot_pos, args.robot_yaw)

    predicted, categories, debug = build_prediction(footprint)

    rclpy.init()
    node = GridChecker(predicted, categories, debug)
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