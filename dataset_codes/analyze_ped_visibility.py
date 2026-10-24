#!/usr/bin/env python3
"""
analyze_ped_visibility.py -- in how many recorded samples could the ROBOT
actually see the pedestrian? (2026-10-20, recorded flags 2026-10-24)

WHY
The trajectories come from the simulator's ground truth, so a sample is saved
even when the pedestrian is outside the camera's 73 deg view, beyond its 8 m
range, or hidden behind a box. A real robot could not make such a sample: it
has no current position of a person it cannot see. In the crop this shows as
the anchor sitting in GREY (unknown) instead of WHITE (seen, free). Nothing is
wrong with those crops (they show exactly what the robot knew), but training
must be able to leave them out. The samples are KEPT; this only measures.

RULE FOR "SEEN" (crop_seen, also used by dataset_recorder.py and the viewer):
the cells within ANCHOR_R of the pedestrian are mostly KNOWN (not grey).

TWO INDEPENDENT ANSWERS AT THE ANCHOR (they should agree)
  crop     : crop_seen() on the sample's own crop
  geometry : from robot_pose_odom and ped_anchor_odom: is the pedestrian
             inside the camera's wedge (+-36.5 deg, <= 8 m, camera 0.37 m ahead
             of the robot centre), allowing for its body width?
  crop=seen but geometry=out of view  -> must never happen (a bug signal: FAIL)
  crop=unseen but geometry=in view    -> hidden behind an obstacle, or at the edge

HISTORY
  Samples recorded from 2026-10-24 carry ped_seen_history (one flag per history
  step, judged by the recorder on the grid of that moment: 1 seen, 0 not seen,
  -1 no grid near that time). Older samples do not; for them each history step
  is looked up in the sample anchored near that time, which leaves many steps
  "unknown" (anchors are not evenly spaced) -- those numbers are APPROXIMATE
  and too low.

USABLE (the proposed training filter, decided at training time, not here):
  seen at the anchor AND seen in at least USABLE_MIN_HISTORY_SEEN of the
  history steps (unknown counts as not seen).

It only READS the samples; it changes nothing.

USAGE
  python3 analyze_ped_visibility.py runs/<run>/samples      (one run, details)
  python3 analyze_ped_visibility.py runs/*/samples          (one line per run + total)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))

try:
    from extract_pedestrian_crop import CROP_FORWARD, CROP_BEHIND, CROP_SIDE
except Exception:                       # only when run away from the project; same values the project uses
    CROP_FORWARD, CROP_BEHIND, CROP_SIDE = 6.0, 2.0, 4.0

UNKNOWN = -1
ANCHOR_R = 0.25          # m: cells around the anchor that tell whether the pedestrian's spot was seen
KNOWN_SHARE = 0.5        # this share of those cells known -> "seen"
CAMERA_AHEAD = 0.37      # m: camera ahead of the robot centre
HALF_FOV_DEG = 36.5
MAX_RANGE = 8.0
BODY_R = 0.25            # m: half the body width (a body at the edge of the wedge is partly seen)
HIST_MATCH_SEC = 0.3     # old samples only
USABLE_MIN_HISTORY_SEEN = 6   # of TRAJECTORY_HISTORY_LENGTH (8); the anchor step is one of them


def crop_seen(crop):
    """(seen, known share) for a pedestrian-centred crop (row = y, column = x, viewer convention)."""
    crop = np.asarray(crop)
    rows, cols = crop.shape
    xs = -CROP_BEHIND + (np.arange(cols) + 0.5) * (CROP_FORWARD + CROP_BEHIND) / cols
    ys = -CROP_SIDE + (np.arange(rows) + 0.5) * (2.0 * CROP_SIDE) / rows
    X, Y = np.meshgrid(xs, ys)
    near = np.hypot(X, Y) <= ANCHOR_R
    share = float((crop[near] != UNKNOWN).mean())
    return share >= KNOWN_SHARE, share


def geometry_in_view(robot, ped):
    rx, ry, ryaw = robot
    cx, cy = rx + CAMERA_AHEAD * math.cos(ryaw), ry + CAMERA_AHEAD * math.sin(ryaw)
    dx, dy = ped[0] - cx, ped[1] - cy
    rng = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx) - ryaw)
    bearing = (bearing + 180.0) % 360.0 - 180.0
    margin = math.degrees(math.atan2(BODY_R, max(rng, 1e-6)))
    inside = abs(bearing) <= HALF_FOV_DEG + margin and rng - BODY_R <= MAX_RANGE
    return inside, bearing, rng


def analyze(samples_dir: Path):
    rows = []
    for f in sorted(samples_dir.glob('*.npz')):
        with np.load(f, allow_pickle=True) as d:
            seen, share = crop_seen(d['crop'])
            geo = None
            if 'robot_pose_odom' in d and 'ped_anchor_odom' in d:
                geo = geometry_in_view(d['robot_pose_odom'], d['ped_anchor_odom'])
            rows.append({'file': f.name, 'seen': seen, 'share': share, 'geo': geo,
                         't': float(d['anchor_time_sim']) if 'anchor_time_sim' in d else None,
                         'hist_t': np.array(d['history_times']) if 'history_times' in d else None,
                         'flags': np.array(d['ped_seen_history']) if 'ped_seen_history' in d else None})
    recorded = bool(rows) and all(r['flags'] is not None for r in rows)
    times = np.array([r['t'] if r['t'] is not None else np.nan for r in rows])
    for r in rows:
        r['hist'] = None
        if r['flags'] is not None:
            fl = r['flags']
            r['hist'] = (int((fl == 1).sum()), int((fl == 0).sum()), int((fl == -1).sum()))
            continue
        if r['hist_t'] is None or np.isnan(times).all():
            continue
        seen_n = unseen_n = unknown_n = 0
        for t in r['hist_t']:
            j = int(np.nanargmin(np.abs(times - t)))
            if abs(times[j] - t) > HIST_MATCH_SEC:
                unknown_n += 1
            elif rows[j]['seen']:
                seen_n += 1
            else:
                unseen_n += 1
        r['hist'] = (seen_n, unseen_n, unknown_n)
    for r in rows:
        r['usable'] = bool(r['seen'] and r['hist'] is not None and r['hist'][0] >= USABLE_MIN_HISTORY_SEEN)
    return rows, recorded


def pct(a, n):
    return f'{100.0 * a / n:.0f}%' if n else '-'


def report(name, rows, recorded, detail=True):
    n = len(rows)
    if not n:
        print(f'{name}: no samples')
        return None
    seen = sum(r['seen'] for r in rows)
    usable = sum(r['usable'] for r in rows)
    with_geo = [r for r in rows if r['geo'] is not None]
    both = sum(1 for r in with_geo if r['seen'] and r['geo'][0])
    crop_only = [r for r in with_geo if r['seen'] and not r['geo'][0]]
    geo_only = [r for r in with_geo if not r['seen'] and r['geo'][0]]
    neither = sum(1 for r in with_geo if not r['seen'] and not r['geo'][0])
    src = 'recorded flags' if recorded else 'APPROXIMATE, old samples'
    if not detail:
        print(f'{name:40s} {n:5d} samples  seen at anchor {pct(seen, n):>4s}  usable {pct(usable, n):>4s}'
              f'{"" if recorded else "*"}  contradictions {len(crop_only)}')
        return n, seen, usable, len(crop_only), recorded

    print(f'PEDESTRIAN VISIBILITY   {name}   ({n} samples)')
    print(f'  seen at the anchor (crop: spot around the anchor known)   {seen:4d}  = {pct(seen, n)}')
    print(f'  NOT seen at the anchor (anchor in grey)                   {n - seen:4d}  = {pct(n - seen, n)}')
    if with_geo:
        print('  cross-check with the camera geometry (robot pose + 73 deg / 8 m wedge):')
        print(f'    seen in crop and inside the wedge        {both:4d}   (agree)')
        print(f'    not seen and outside the wedge           {neither:4d}   (agree: robot looked away / too far)')
        print(f'    not seen but inside the wedge            {len(geo_only):4d}   (hidden behind an obstacle, or at the very edge)')
        print(f'    SEEN in crop but OUTSIDE the wedge       {len(crop_only):4d}   (must be 0)')
    hs = [r['hist'] for r in rows if r['seen'] and r['hist'] is not None]
    if hs:
        H = sum(hs[0])
        print(f'  history steps seen, for the samples seen at the anchor ({src}):')
        for k in range(H, -1, -1):
            c = sum(1 for h in hs if h[0] == k)
            if c:
                print(f'    {k} of {H} steps seen   {c:4d}  = {pct(c, len(hs))}')
        unk = sum(h[2] for h in hs)
        if unk:
            print(f'    ({unk} history steps had no grid/sample near their time and count as not seen)')
    unseen = [i for i, r in enumerate(rows) if not r['seen']]
    if unseen:
        shown = ', '.join(str(i) for i in unseen[:40]) + (' ...' if len(unseen) > 40 else '')
        print(f'  not-seen sample indices (open with --view --start <i>): {shown}')
    if crop_only:
        print('  CONTRADICTIONS (crop says seen, geometry says out of view):')
        for r in crop_only[:10]:
            b, rng = r['geo'][1], r['geo'][2]
            print(f'    {r["file"]}: bearing {b:+.0f} deg, range {rng:.1f} m, known share {r["share"]:.2f}')
    agree = ('crops and camera geometry agree' if not crop_only else
             f'{len(crop_only)} CONTRADICTIONS between crops and camera geometry')
    print(f'VISIBILITY: seen at anchor {pct(seen, n)}, usable {pct(usable, n)} '
          f'(anchor + >= {USABLE_MIN_HISTORY_SEEN}/{TRAJ_H} history steps seen{"" if recorded else ", approximate"}); {agree}.')
    return n, seen, usable, len(crop_only), recorded


try:
    from hunav_config import TRAJECTORY_HISTORY_LENGTH as TRAJ_H
except Exception:
    TRAJ_H = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples_dirs', nargs='*', default=[str(_THIS_DIR / 'samples')])
    args = ap.parse_args()
    dirs = [Path(p).expanduser().resolve() for p in args.samples_dirs]
    if len(dirs) == 1:
        report(dirs[0].parent.name, *analyze(dirs[0]))
        return
    tot = [0, 0, 0, 0]
    old = False
    for d in dirs:
        res = report(d.parent.name, *analyze(d), detail=False)
        if res:
            tot = [a + b for a, b in zip(tot, res[:4])]
            old = old or not res[4]
    print(f'{"TOTAL":40s} {tot[0]:5d} samples  seen at anchor {pct(tot[1], tot[0]):>4s}  '
          f'usable {pct(tot[2], tot[0]):>4s}  contradictions {tot[3]}')
    if old:
        print('* = old samples without recorded flags: history looked up between samples, usable is too low.')


if __name__ == '__main__':
    main()