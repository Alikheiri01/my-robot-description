#!/usr/bin/env python3
"""
analyze_walk_smoothness.py -- does the recorded pedestrian walk like a person,
or does it zig-zag?

WHY THIS EXISTS (2026-10-05)
On a straight walk the crops kept tilting +-15 deg from sample to sample. The
cause is HuNav's social-force integration at 2 Hz: with relaxation time 0.5 s
and goal force factor 2, one 0.5 s Euler step flips the sideways velocity
error exactly (factor -1), so the walk zig-zags forever and the heading swings
with it. validate_time_alignment.py cannot see this: the heading really does
follow the (zig-zagging) body. This script measures the zig-zag itself.

WHAT IT MEASURES, on the samples' own pose rows (history + future, 0.5 s apart)
  * TURN PER STEP: heading change between consecutive 0.5 s steps while
    walking (speed >= 0.5 m/s), leaving out real turnarounds (> 120 deg).
    A person walking straight: a few degrees. The 2 Hz zig-zag: ~25-30 deg,
    alternating in sign.
  * SIGN FLIPS: share of consecutive turns that alternate left/right with
    both above 5 deg -- the zig-zag signature.
  * CROP TILT: angle between the anchor's heading (the crop's x axis) and the
    direction the pedestrian travels from 0.5 s before to 0.5 s after the
    anchor. This is the tilt you see in the crop viewer.

USAGE   python3 analyze_walk_smoothness.py <samples_dir>
"""
import math
import sys
from pathlib import Path

import numpy as np

MOVING = 0.5           # m/s
REVERSAL_DEG = 120.0
FLIP_MIN_DEG = 5.0
TURN_LIMIT_DEG = 10.0  # p90 turn per 0.5 s step on walking stretches
TILT_LIMIT_DEG = 10.0  # p90 crop tilt


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def analyze(samples_dir):
    turns, flips_num, flips_den, tilts = [], 0, 0, []
    n = 0
    for f in sorted(Path(samples_dir).glob('*.npz')):
        with np.load(f, allow_pickle=True) as d:
            h, fu = d['trajectory_history'], d['future_target']
            P = float(d['sample_period_sec'])
        n += 1
        rows = np.concatenate([h[:, :2], fu[:, :2]])
        steps = np.diff(rows, axis=0)
        speed = np.hypot(steps[:, 0], steps[:, 1]) / P
        head = np.degrees(np.arctan2(steps[:, 1], steps[:, 0]))
        prev = None
        for i in range(1, len(steps)):
            if speed[i] < MOVING or speed[i - 1] < MOVING:
                prev = None
                continue
            t = wrap(head[i] - head[i - 1])
            if abs(t) > REVERSAL_DEG:
                prev = None
                continue
            turns.append(abs(t))
            if prev is not None and abs(prev) > FLIP_MIN_DEG and abs(t) > FLIP_MIN_DEG:
                flips_den += 1
                flips_num += int(np.sign(prev) != np.sign(t))
            elif prev is not None:
                flips_den += 1
            prev = t
        # crop tilt: anchor heading is local +x; chord from history[-2] to future[0]
        a, b = h[-2, :2], fu[0, :2]
        if np.hypot(*(b - a)) / (2 * P) >= MOVING:
            v_in, v_out = h[-1, :2] - h[-2, :2], fu[0, :2] - h[-1, :2]
            if abs(wrap(math.degrees(math.atan2(v_out[1], v_out[0]) - math.atan2(v_in[1], v_in[0])))) < REVERSAL_DEG:
                tilts.append(abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))))
    return n, np.array(turns), (flips_num / flips_den if flips_den else float('nan')), np.array(tilts)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    n, turns, flip, tilts = analyze(sys.argv[1])
    if n == 0 or len(turns) < 20:
        print(f'{n} samples: too few walking steps to judge the walk.')
        return
    p90t, p90c = float(np.percentile(turns, 90)), float(np.percentile(tilts, 90)) if len(tilts) else float('nan')
    print(f'{n} samples, {len(turns)} walking step pairs, {len(tilts)} walking anchors')
    print(f'  turn per 0.5 s step : median {np.median(turns):.1f} deg, p90 {p90t:.1f} deg (limit {TURN_LIMIT_DEG:.0f})')
    print(f'  left/right flips    : {100 * flip:.0f}% of consecutive turns (zig-zag signature; straight walking ~ low)')
    print(f'  crop tilt           : median {np.median(tilts):.1f} deg, p90 {p90c:.1f} deg (limit {TILT_LIMIT_DEG:.0f})')
    ok = p90t <= TURN_LIMIT_DEG and (math.isnan(p90c) or p90c <= TILT_LIMIT_DEG)
    print(f'WALK SMOOTHNESS: {"PASS" if ok else "FAIL"} -- '
          + ('the walk is smooth.' if ok else 'the pedestrian zig-zags; check BRIDGE_UPDATE_RATE_HZ (>= 10) in hunav_config.py.'))


if __name__ == '__main__':
    main()