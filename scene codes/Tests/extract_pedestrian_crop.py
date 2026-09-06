#!/usr/bin/env python3
"""
Pure resampling function: extracts a pedestrian-centric, heading-aligned
crop from the whole-scene base occupancy grid (build_occupancy_grid.py's
/occupancy_grid/base). No sensing computation happens here -- this is
purely a geometric lookup into the base grid, per the "one projection, many
crops" efficiency decision in the project's occupancy grid spec.

CROP GEOMETRY -- fixed by spec, not tunable per-call
------------------------------------------------------------------
8m x 8m total, asymmetric: 6m ahead of the pedestrian, 2m behind, 4m to
each side. At the base grid's own 0.10 m/cell resolution, that's exactly
80x80 cells in both axes (6+2=8m -> 80 cells; 4+4=8m -> 80 cells) -- same
resolution as the base grid, on purpose, so this step is a pure rotation +
translation lookup, never also a resolution change.

INDEXING CONVENTION -- locks in the format the dataset/trajectory pipeline
must also use, per the spec's "coordinate consistency" requirement (get
this right once, here, since it's expensive to change later once data has
been generated against it)
------------------------------------------------------------------
Local frame: origin at the pedestrian's current position, local +x along
the pedestrian's smoothed heading (forward), local +y to the pedestrian's
left -- same right-handed x-forward/y-left convention base_link itself
uses, just recentered/rotated onto the pedestrian instead of the robot.
  local_x in [-CROP_BEHIND, +CROP_FORWARD]   (-2m .. +6m)
  local_y in [-CROP_SIDE,   +CROP_SIDE]       (-4m .. +4m)
Array shape (80, 80): row = local_y index, col = local_x index -- the same
"row follows the Y-like axis, col follows the X-like axis" rule the base
grid already uses (matching nav_msgs/OccupancyGrid's own convention),
just applied in the pedestrian's local frame instead of base_link's.

OUT-OF-BOUNDS HANDLING
------------------------------------------------------------------
Any crop cell whose corresponding base_link location falls outside the
published base grid's extent is UNKNOWN, never wrapped or index-erroring --
per the project's explicit graceful-out-of-bounds requirement.
"""
import numpy as np
import build_occupancy_grid as bog

CROP_FORWARD = 6.0
CROP_BEHIND = 2.0
CROP_SIDE = 4.0
CROP_RESOLUTION = bog.RESOLUTION  # deliberately identical to the base grid's resolution

CROP_COLS = int(round((CROP_FORWARD + CROP_BEHIND) / CROP_RESOLUTION))  # 80
CROP_ROWS = int(round((2 * CROP_SIDE) / CROP_RESOLUTION))               # 80
assert CROP_COLS == 80 and CROP_ROWS == 80, \
    "crop margins no longer match the spec's 80x80 assumption -- check CROP_* constants"


def local_to_base(local_x, local_y, ped_x, ped_y, ped_heading):
    """Rotates + translates pedestrian-local (x, y) into base_link (x, y)."""
    c, s = np.cos(ped_heading), np.sin(ped_heading)
    base_x = ped_x + local_x * c - local_y * s
    base_y = ped_y + local_x * s + local_y * c
    return base_x, base_y


def base_to_local(base_x, base_y, ped_x, ped_y, ped_heading):
    """Inverse of local_to_base -- rotates + translates base_link (x, y) into pedestrian-local (x, y)."""
    dx, dy = base_x - ped_x, base_y - ped_y
    c, s = np.cos(ped_heading), np.sin(ped_heading)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return local_x, local_y


def extract_pedestrian_crop(base_grid: np.ndarray, ped_x: float, ped_y: float, ped_heading: float) -> np.ndarray:
    """
    base_grid: (bog.GRID_N, bog.GRID_N) int8 array, the published base grid
    (same -1/0/100 convention). ped_x, ped_y, ped_heading: synthetic
    pedestrian pose in base_link (x, y in meters, heading in radians).

    Returns an (80, 80) int8 array in the pedestrian-local frame described
    in this module's docstring.
    """
    col_idx = np.arange(CROP_COLS)
    row_idx = np.arange(CROP_ROWS)
    local_x = (col_idx + 0.5) * CROP_RESOLUTION - CROP_BEHIND
    local_y = (row_idx + 0.5) * CROP_RESOLUTION - CROP_SIDE
    LX, LY = np.meshgrid(local_x, local_y)  # LX[row,col], LY[row,col] -- matches this module's row=y,col=x rule

    base_x, base_y = local_to_base(LX, LY, ped_x, ped_y, ped_heading)

    base_col = np.floor((base_x + bog.GRID_HALF_EXTENT) / bog.RESOLUTION).astype(np.int64)
    base_row = np.floor((base_y + bog.GRID_HALF_EXTENT) / bog.RESOLUTION).astype(np.int64)

    in_bounds = (base_row >= 0) & (base_row < bog.GRID_N) & (base_col >= 0) & (base_col < bog.GRID_N)

    crop = np.full((CROP_ROWS, CROP_COLS), bog.UNKNOWN, dtype=np.int8)
    # Clamp indices purely so the gather itself never sees an invalid index --
    # the gathered value at any originally-out-of-bounds cell is discarded
    # below via the in_bounds mask, so the clamped value is never actually used.
    safe_row = np.clip(base_row, 0, bog.GRID_N - 1)
    safe_col = np.clip(base_col, 0, bog.GRID_N - 1)
    gathered = base_grid[safe_row, safe_col]
    crop[in_bounds] = gathered[in_bounds]
    return crop


def self_test():
    """
    Pure algebraic round-trip check, no ROS/Gazebo needed: pick an arbitrary
    local (x, y) and pedestrian pose, transform out to base_link and back,
    confirm the original point is recovered. Catches sign errors and axis
    swaps immediately, without needing the sim running.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        lx = rng.uniform(-CROP_BEHIND, CROP_FORWARD)
        ly = rng.uniform(-CROP_SIDE, CROP_SIDE)
        ped_x, ped_y = rng.uniform(-5, 5, size=2)
        ped_heading = rng.uniform(-np.pi, np.pi)

        bx, by = local_to_base(lx, ly, ped_x, ped_y, ped_heading)
        lx2, ly2 = base_to_local(bx, by, ped_x, ped_y, ped_heading)

        assert abs(lx - lx2) < 1e-9 and abs(ly - ly2) < 1e-9, \
            f'round-trip failed: ({lx},{ly}) -> ({bx},{by}) -> ({lx2},{ly2})'
    print('self_test: 200/200 round-trip checks passed.')


if __name__ == '__main__':
    self_test()