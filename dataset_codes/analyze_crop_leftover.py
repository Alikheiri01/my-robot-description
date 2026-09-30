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

USAGE   python3 analyze_crop_leftover.py [samples_dir]
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
SPEED_BINS = [(0.0, 0.3), (0.3, 0.7), (0.7, 5.0)]
OCCUPIED = 100


def cell_centres(shape):
    """x (ahead) of every column and y (left) of every row, in metres, like the viewer's extent."""
    rows, cols = shape
    xs = -CROP_BEHIND + (np.arange(cols) + 0.5) * (CROP_FORWARD + CROP_BEHIND) / cols
    ys = -CROP_SIDE + (np.arange(rows) + 0.5) * (2.0 * CROP_SIDE) / rows
    return xs, ys


def load(samples_dir: Path):
    rows = []
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
    args = ap.parse_args()
    rows = load(Path(args.samples_dir).expanduser().resolve())
    if not rows:
        print('no samples found')
        return
    speed = np.array([r['speed'] for r in rows])
    has = np.array([len(r['x']) > 0 for r in rows])
    moving, slow = speed >= MOVING, speed < SLOW
    print(f'{len(rows)} samples: {int(moving.sum())} moving (>= {MOVING} m/s), {int(slow.sum())} slow (< {SLOW} m/s)')
    print(f'Occupied cells within {NEAR_R} m of the anchor are counted as leftover pedestrian.\n')

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
        print('  Almost always AHEAD of the anchor and mostly while walking: the body sits ahead of the pose used for the\n'
              '  exclusion circle. That points to TIMING with POSE_LOOKUP_OFFSET_SEC too large in size for this session.\n'
              '  Run validate_time_alignment.py; it prints the offset to use. Do NOT just enlarge the radius.')
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