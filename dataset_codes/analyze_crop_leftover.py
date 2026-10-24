#!/usr/bin/env python3
"""
analyze_crop_leftover.py -- what is the occupied stuff (black cells) that
survives near the pedestrian in the recorded crops, and where does it sit?

WHY THIS EXISTS
The pedestrian is removed from the occupancy grid by an exclusion circle
(PEDESTRIAN_EXCLUSION_RADIUS) centred on the pedestrian's pose at the depth
data's time. Any occupied cell left close to the anchor of a crop is a piece
of the pedestrian that the circle missed. Different problems leave such
pieces, and they look different in the data:

  * TIMING (offset too large in size): the circle is centred where the
    pedestrian WAS, so the body sits a little AHEAD of it while walking. The
    missed pieces are almost all on the heading side, and only when the
    pedestrian is moving.
  * TIMING (offset too small in size): the opposite, the missed pieces are
    almost all BEHIND the anchor, and only when moving.
  * EXTREMITIES (arms / feet swinging beyond the radius): the missed pieces
    are spread on BOTH sides of the anchor (leading and trailing limb), and
    at about the same distance whatever the speed.

This script reads the .npz samples of a recording (time-aligned recorder,
2026-09-30) and says which of these the data looks like, with numbers.
It only READS the samples; it changes nothing.

Crop layout (same as inspect_dataset_samples.py --view): array row = y
(left +), array column = x (ahead of the pedestrian +); x spans
-CROP_BEHIND..CROP_FORWARD, y spans -CROP_SIDE..CROP_SIDE, anchor at (0, 0).

OBSTACLES (2026-10-06): with real obstacles in the scene, their cells near
the pedestrian are NOT pedestrian leftovers. If the run folder has an
obstacles.yaml (record_session.py saves one) -- or --obstacles is given --
cells within OBSTACLE_MARGIN_M of a known obstacle are left out. Without it
the result is as before (and wrong in scenes with obstacles).

USAGE   python3 analyze_crop_leftover.py [samples_dir] [--obstacles PATH]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'scene codes'))

try:
    from extract_pedestrian_crop import CROP_FORWARD, CROP_BEHIND, CROP_SIDE
except Exception:                       # only when run away from the project; same values the project uses
    CROP_FORWARD, CROP_BEHIND, CROP_SIDE = 6.0, 2.0, 4.0

NEAR_R = 1.5        # m: occupied cells within this of the anchor count as "leftover pedestrian"
MOVING = 0.5        # m/s
SLOW = 0.3          # m/s
MIN_LEFTOVER_SAMPLES = 10
OBSTACLE_MARGIN_M = 0.25   # grid cells of an obstacle face sit up to ~1-2 cells off its true outline
SPEED_BINS = [(0.0, 0.3), (0.3, 0.7), (0.7, 5.0)]
OCCUPIED = 100


def cell_centres(shape):
    """x (ahead) of every column and y (left) of every row, in metres, like the viewer's extent."""
    rows, cols = shape
    xs = -CROP_BEHIND + (np.arange(cols) + 0.5) * (CROP_FORWARD + CROP_BEHIND) / cols
    ys = -CROP_SIDE + (np.arange(rows) + 0.5) * (2.0 * CROP_SIDE) / rows
    return xs, ys


def find_obstacles_file(samples_dir: Path, given=None):
    if given:
        return Path(given)
    p = samples_dir.parent / 'obstacles.yaml'
    return p if p.exists() else None


def on_obstacle_mask(x, y, anchor, obstacles):
    """True for crop cells (anchor-local x, y) lying on a known obstacle."""
    from hunav_config import odom_to_world
    from world_obstacles import min_distance
    ax, ay, ayaw = anchor
    c, s = np.cos(ayaw), np.sin(ayaw)
    out = np.zeros(len(x), dtype=bool)
    for i, (lx, ly) in enumerate(zip(x, y)):
        wx, wy, _ = odom_to_world(ax + c * lx - s * ly, ay + s * lx + c * ly, 0.0)
        out[i] = min_distance(wx, wy, obstacles)[0] <= OBSTACLE_MARGIN_M
    return out


def load(samples_dir: Path, obstacles=None):
    rows = []
    load.ignored = 0
    for f in sorted(samples_dir.glob('*.npz')):
        with np.load(f, allow_pickle=True) as d:
            crop, hist, fut = d['crop'], d['trajectory_history'], d['future_target']
            period = float(d['sample_period_sec'])
            # last history row is the anchor itself (0,0,0): the row before it is one step back
            speed = 0.5 * (np.hypot(*fut[0, :2]) + np.hypot(*hist[-2, :2])) / period
            xs, ys = cell_centres(crop.shape)
            r_idx, c_idx = np.nonzero(crop == OCCUPIED)
            x, y = xs[c_idx], ys[r_idx]
            near = np.hypot(x, y) <= NEAR_R
            if obstacles and 'ped_anchor_odom' in d and near.any():
                idx = np.nonzero(near)[0]
                on = on_obstacle_mask(x[idx], y[idx], d['ped_anchor_odom'], obstacles)
                near[idx[on]] = False
                load.ignored += int(on.sum())
            to_robot = None
            if 'robot_pose_odom' in d and 'ped_anchor_odom' in d:
                r, a = d['robot_pose_odom'], d['ped_anchor_odom']
                dx, dy = r[0] - a[0], r[1] - a[1]
                c, s = np.cos(a[2]), np.sin(a[2])
                v = np.array([c * dx + s * dy, -s * dx + c * dy])   # robot position in the pedestrian frame
                to_robot = v / max(np.hypot(*v), 1e-9)
            rows.append({'speed': float(speed), 'x': x[near], 'y': y[near], 'to_robot': to_robot})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples_dir', nargs='?', default=str(_THIS_DIR / 'samples'))
    ap.add_argument('--obstacles', default=None, help="obstacle yaml (default: the run folder's obstacles.yaml)")
    args = ap.parse_args()
    sdir = Path(args.samples_dir).expanduser().resolve()
    obs_file = find_obstacles_file(sdir, args.obstacles)
    obstacles = []
    if obs_file:
        sys.path.insert(0, str(_THIS_DIR.parent / 'agent_control'))
        from world_obstacles import load_obstacles
        obstacles = load_obstacles(obs_file)
    rows = load(sdir, obstacles)
    if not rows:
        print('no samples found')
        return
    speed = np.array([r['speed'] for r in rows])
    has = np.array([len(r['x']) > 0 for r in rows])
    moving, slow = speed >= MOVING, speed < SLOW
    print(f'{len(rows)} samples: {int(moving.sum())} moving (>= {MOVING} m/s), {int(slow.sum())} slow (< {SLOW} m/s)')
    print(f'Occupied cells within {NEAR_R} m of the anchor are counted as leftover pedestrian.')
    if obs_file:
        print(f'Known obstacles: {len(obstacles)} from {obs_file}; {load.ignored} cells on them left out.\n')
    else:
        print('Known obstacles: none given (no obstacles.yaml in the run folder): obstacle cells near the '
              'pedestrian would be counted as leftover.\n')

    print(f'{"speed (m/s)":14s} {"samples":>8s} {"with leftover":>14s} {"mean x of leftover cells (+ = ahead)":>38s}')
    for lo, hi in SPEED_BINS:
        m = (speed >= lo) & (speed < hi)
        if not m.any():
            continue
        idx = np.nonzero(m & has)[0]
        xs = np.concatenate([rows[i]['x'] for i in idx]) if len(idx) else np.array([])
        label = f'{lo:.1f} .. {hi:.1f}'
        print(f'{label:14s} {int(m.sum()):8d} {100 * np.mean(has[m]):13.0f}% '
              f'{(f"{xs.mean():+.2f} m" if len(xs) else "-"):>38s}')

    mv = np.nonzero(moving & has)[0]
    share_mv = float(np.mean(has[moving])) if moving.any() else 0.0
    share_sl = float(np.mean(has[slow])) if slow.any() else 0.0
    print(f'\nleftover present: {100 * share_mv:.0f}% of moving samples, {100 * share_sl:.0f}% of slow samples')
    if len(mv) == 0:
        print('No leftover pedestrian cells near the anchor in any moving sample: the exclusion is clean.')
        return
    if len(mv) < MIN_LEFTOVER_SAMPLES:
        print(f'Only {len(mv)} moving samples have leftover (fewer than {MIN_LEFTOVER_SAMPLES}): '
              f'too few to say where it sits. Record more if you want a verdict.')
        return

    mean_x = np.array([rows[i]['x'].mean() for i in mv])
    dist = np.concatenate([np.hypot(rows[i]['x'], rows[i]['y']) for i in mv])
    xs_all = np.concatenate([rows[i]['x'] for i in mv])
    p_ahead = float(np.mean(mean_x > 0.0))
    print(f'among the {len(mv)} moving samples that have leftover:')
    print(f'  the leftover is AHEAD of the anchor (heading side) in {100 * p_ahead:.0f}% of them; '
          f'cells: median x {np.median(xs_all):+.2f} m, median distance {np.median(dist):.2f} m, '
          f'90th percentile {np.percentile(dist, 90):.2f} m')
    facing = [float(np.mean(r['x'] * r['to_robot'][0] + r['y'] * r['to_robot'][1] > 0))
              for r in (rows[i] for i in mv) if r['to_robot'] is not None]
    if facing:
        print(f'  the leftover is mostly on the robot-facing side of the anchor in '
              f'{100 * np.mean(np.array(facing) > 0.5):.0f}% of them')

    print('\nREADING (a rule of thumb, not a proof)')
    if p_ahead >= 0.85 and share_mv >= 2 * max(share_sl, 0.02):
        need = max(0.5, float(np.percentile(dist, 90)) + 0.05)
        print('  Almost always AHEAD of the anchor and mostly while walking. Two causes look the same here:\n'
              '   (a) TIMING: the exclusion circle is centred behind the body. Decide with validate_time_alignment.py:\n'
              '       if its timing offset is beyond +-0.1 s, fix the timing first and do NOT enlarge the radius.\n'
              '   (b) REACH: the leading foot/arm of a walker sticks out ahead of the circle. If the validator passes\n'
              '       (offset within +-0.1 s), this is it: raise PEDESTRIAN_EXCLUSION_RADIUS in hunav_config.py\n'
              f'       (about {need:.2f} m would have covered 90% of these cells), then record again and re-run this.')
    elif p_ahead <= 0.15 and share_mv >= 2 * max(share_sl, 0.02):
        print('  Almost always BEHIND the anchor and mostly while walking: the exclusion circle is ahead of the body.\n'
              '  That points to TIMING with POSE_LOOKUP_OFFSET_SEC too small in size for this session.\n'
              '  Run validate_time_alignment.py; it prints the offset to use.')
    elif 0.25 <= p_ahead <= 0.75:
        need = max(0.5, float(np.percentile(dist, 90)) + 0.05)
        print('  On BOTH sides of the anchor: swinging limbs beyond the radius, not a timing problem.\n'
              f'  Raise PEDESTRIAN_EXCLUSION_RADIUS in hunav_config.py (about {need:.2f} m would have covered 90% of it).')
    else:
        print('  Mixed picture (mostly one side, but not overwhelmingly). Run validate_time_alignment.py too and send both.')


if __name__ == '__main__':
    main()