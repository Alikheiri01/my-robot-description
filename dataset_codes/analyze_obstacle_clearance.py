#!/usr/bin/env python3
"""
analyze_obstacle_clearance.py -- did the recorded pedestrian ever walk INTO an
obstacle? (2026-10-05)

WHY
Before obstacles were fed to HuNav, the pedestrian walked straight through a
box. A model trained on such samples learns that people pass through things.
This check makes sure no recording with that problem is used.

TWO INDEPENDENT CHECKS on the recorded samples
  1. CROP CHECK (independent of what the pedestrian was told): every history
     and future position of a sample is put into that sample's own crop, and
     its distance to the nearest OCCUPIED cell is measured. Static obstacles do
     not move, so a position closer than TOUCH_M to an occupied cell means the
     body was in or against something the robot really saw. To stay clear of
     pedestrian leftovers (pieces of the pedestrian's own body near the
     anchor), cells within IGNORE_NEAR_ANCHOR_M of the anchor and positions
     within SKIP_ROWS_NEAR_ANCHOR_M of it are not used -- other samples, whose
     anchors are elsewhere, cover those places. Catches a wrong obstacles.yaml.
  2. GROUND-TRUTH CHECK (if obstacles.yaml exists): the same positions, taken
     to the world frame, against the obstacle outlines of the yaml. Exact.

PASS: ground truth -- no position closer than TOUCH_M to an obstacle outline;
      crop -- at most CROP_TOUCH_LIMIT_PCT % of the checked positions closer
      than TOUCH_M to an occupied cell (a little room for depth-camera specks).
With no obstacle anywhere near the path both pass, and say so.

USAGE   python3 analyze_obstacle_clearance.py <samples_dir> [--obstacles PATH]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'scene codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'agent_control'))

from analyze_crop_leftover import cell_centres, OCCUPIED  # noqa: E402

TOUCH_M = 0.25                 # body half-width: a centre this close = touching / inside
NEAR_M = 1.0                   # positions this close to an obstacle are "passing an obstacle"
IGNORE_NEAR_ANCHOR_M = 1.0     # crop cells this close to the anchor may be the pedestrian itself
SKIP_ROWS_NEAR_ANCHOR_M = 1.2
CROP_TOUCH_LIMIT_PCT = 1.0
DEFAULT_OBSTACLES = _THIS_DIR.parent / 'agent_control' / 'obstacles.yaml'


def to_odom(anchor, lx, ly):
    ax, ay, ayaw = anchor
    c, s = math.cos(ayaw), math.sin(ayaw)
    return ax + c * lx - s * ly, ay + s * lx + c * ly


def analyze(samples_dir, obstacles):
    try:
        from hunav_config import odom_to_world
    except Exception:
        odom_to_world = None
    crop_d, truth_d = [], []
    worst = (math.inf, None)
    n = 0
    for f in sorted(Path(samples_dir).glob('*.npz')):
        with np.load(f, allow_pickle=True) as d:
            crop, h, fu = d['crop'], d['trajectory_history'], d['future_target']
            anchor = d['ped_anchor_odom'] if 'ped_anchor_odom' in d else None
        n += 1
        rows = np.concatenate([h[:, :2], fu[:, :2]])
        # --- crop check
        xs, ys = cell_centres(crop.shape)
        ri, ci = np.nonzero(crop == OCCUPIED)
        cx, cy = xs[ci], ys[ri]
        keep = np.hypot(cx, cy) > IGNORE_NEAR_ANCHOR_M
        cx, cy = cx[keep], cy[keep]
        for lx, ly in rows:
            if math.hypot(lx, ly) < SKIP_ROWS_NEAR_ANCHOR_M:
                continue
            if not (xs[0] <= lx <= xs[-1] and ys[0] <= ly <= ys[-1]):
                continue
            crop_d.append(float(np.min(np.hypot(cx - lx, cy - ly))) if len(cx) else math.inf)
        # --- ground truth check
        if obstacles and anchor is not None and odom_to_world is not None:
            from world_obstacles import min_distance
            for lx, ly in rows:
                ox, oy = to_odom(anchor, float(lx), float(ly))
                wx, wy, _ = odom_to_world(ox, oy, 0.0)
                dist, o = min_distance(wx, wy, obstacles)
                truth_d.append(dist)
                if dist < worst[0]:
                    worst = (dist, f'{f.name}: ({wx:.2f}, {wy:.2f}) world, {o.name}')
    return n, np.array(crop_d), np.array(truth_d), worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples_dir')
    ap.add_argument('--obstacles', default=str(DEFAULT_OBSTACLES))
    args = ap.parse_args()
    obstacles = []
    if Path(args.obstacles).exists():
        from world_obstacles import load_obstacles
        obstacles = load_obstacles(args.obstacles)
    n, crop_d, truth_d, worst = analyze(args.samples_dir, obstacles)
    if n == 0:
        print('0 samples: nothing to check.')
        return
    print(f'{n} samples')
    ok = True

    fin = crop_d[np.isfinite(crop_d)]
    near = int(np.sum(fin < NEAR_M))
    touch = int(np.sum(fin < TOUCH_M))
    pct = 100.0 * touch / max(1, len(crop_d))
    if near == 0:
        print(f'  crop check   : {len(crop_d)} positions checked, none within {NEAR_M:.1f} m of an occupied cell '
              f'(open floor, nothing to avoid)')
    else:
        print(f'  crop check   : {len(crop_d)} positions checked, {near} pass within {NEAR_M:.1f} m of an occupied '
              f'cell, closest {fin.min():.2f} m; touching (< {TOUCH_M:.2f} m): {touch} = {pct:.1f}% '
              f'(limit {CROP_TOUCH_LIMIT_PCT:.0f}%)')
        ok &= pct <= CROP_TOUCH_LIMIT_PCT

    if not obstacles:
        print(f'  ground truth : no obstacle file / empty list ({args.obstacles}), skipped')
    elif len(truth_d) == 0:
        print('  ground truth : samples have no anchor pose (old schema), skipped')
    else:
        t_touch = int(np.sum(truth_d < TOUCH_M))
        t_near = int(np.sum(truth_d < NEAR_M))
        print(f'  ground truth : {len(truth_d)} positions vs {len(obstacles)} obstacle(s): {t_near} within '
              f'{NEAR_M:.1f} m, closest {truth_d.min():.2f} m ({worst[1]}); touching: {t_touch}')
        ok &= t_touch == 0
    print(f'OBSTACLE CLEARANCE: {"PASS" if ok else "FAIL"} -- '
          + ('the pedestrian never walks into an obstacle.' if ok else
             'the pedestrian walks into an obstacle. Is the bridge hunav_model_bridge_nav.py, and does '
             'obstacles.yaml match Gazebo (spawn_obstacles.py)?'))


if __name__ == '__main__':
    main()