#!/usr/bin/env python3
"""
analyze_sample_fields.py -- checks the fields added to the samples on 2026-10-27:
the robot track (robot_history_odom, robot_future_odom) and the episode labels
(episode_id, episode_time_sec). Offline, reads saved samples only.

WHY AN INDEPENDENT CHECK
robot_pose_odom (the robot at the grid stamp t_g) is an old field, verified in the
T17 tests. Every sample also carries the new robot track at 20 times around its
anchor. For each sample j, other samples i whose window covers t_g(j) must put the
robot at the same place: interpolate i's track at t_g(j) and compare it with
robot_pose_odom(j). Two different code paths (anchor lookup vs history/future rows),
so a wrong time or a wrong row order would show up here.

CHECKS
  shapes        robot_history_odom (H, 3), robot_future_odom (F, 3), all finite
  robot track   p90 position error of the cross-check above <= TRACK_P90_LIMIT_M
  robot speed   no jump faster than ROBOT_MAX_SPEED between consecutive rows
  last row      robot_history_odom[-1] (robot at T) within LAST_ROW_LIMIT_M of
                robot_pose_odom (robot at t_g, ~0.05 s earlier)
  episodes      ids and sample counts; episode_time_sec consistent inside an episode
                (T - episode_time_sec = the same start time); no sample window crosses
                an episode change (consecutive samples of different episodes have
                anchors at least one window length apart)

USAGE
    python3 analyze_sample_fields.py runs/<run>/samples
Last line:  SAMPLE FIELDS: PASS | FAIL | NO DATA: <details>   (read by record_session.py)
"""
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

TRACK_P90_LIMIT_M = 0.10
LAST_ROW_LIMIT_M = 0.25
ROBOT_MAX_SPEED = 2.0       # m/s between 0.5 s rows (the robot drives < 1 m/s)
MAX_PARTNERS = 6            # samples i compared with each sample j (nearest anchors)


def load(samples_dir):
    files = sorted(Path(samples_dir).rglob('*.npz'))
    out = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            out.append({k: d[k] for k in d.files} | {'_file': f.name})
        except Exception as e:
            print(f'  could not read {f.name}: {e}')
    return out


def interp_track(times, rows, t):
    """robot x, y at time t from a sample's 0.5 s rows (linear), or None outside them."""
    if t < times[0] or t > times[-1]:
        return None
    return float(np.interp(t, times, rows[:, 0])), float(np.interp(t, times, rows[:, 1]))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    samples = load(sys.argv[1])
    n = len(samples)
    print(f'{n} samples in {sys.argv[1]}')
    if n == 0:
        print('SAMPLE FIELDS: NO DATA: no samples')
        return
    new = [s for s in samples if 'robot_history_odom' in s and 'robot_future_odom' in s and 'episode_id' in s]
    if not new:
        print('SAMPLE FIELDS: NO DATA: samples recorded before 2026-10-27 (no robot track / episode fields)')
        return
    problems = []
    if len(new) < n:
        problems.append(f'{n - len(new)} samples without the new fields')

    # --- shapes ---------------------------------------------------------------------
    bad_shape = 0
    for s in new:
        H, F = len(s['history_times']), len(s['future_times'])
        rh, rf = s['robot_history_odom'], s['robot_future_odom']
        if rh.shape != (H, 3) or rf.shape != (F, 3) or not (np.isfinite(rh).all() and np.isfinite(rf).all()):
            bad_shape += 1
    print(f'shapes        : {len(new) - bad_shape} ok, {bad_shape} wrong shape or not finite')
    if bad_shape:
        problems.append(f'{bad_shape} samples with wrong robot track shape / NaN')
    new = [s for s in new if s['robot_history_odom'].shape[0] == len(s['history_times'])
           and s['robot_future_odom'].shape[0] == len(s['future_times'])
           and np.isfinite(s['robot_history_odom']).all() and np.isfinite(s['robot_future_odom']).all()]
    new.sort(key=lambda s: float(s['anchor_time_sim']))

    # --- last history row vs robot_pose_odom -------------------------------------------
    last = np.array([math.hypot(*(s['robot_history_odom'][-1, :2] - s['robot_pose_odom'][:2])) for s in new])
    print(f'last row      : robot at T vs robot at the grid stamp: median {np.median(last):.3f} m, '
          f'max {last.max():.3f} m (limit {LAST_ROW_LIMIT_M} m)')
    if last.max() > LAST_ROW_LIMIT_M:
        problems.append(f'robot at T is {last.max():.2f} m from robot_pose_odom')

    # --- speed between rows ------------------------------------------------------------
    vmax = 0.0
    for s in new:
        times = np.concatenate([s['history_times'], s['future_times']])
        rows = np.concatenate([s['robot_history_odom'], s['robot_future_odom']])
        d = np.hypot(np.diff(rows[:, 0]), np.diff(rows[:, 1])) / np.maximum(np.diff(times), 1e-6)
        vmax = max(vmax, float(d.max()) if len(d) else 0.0)
    print(f'robot speed   : fastest between two rows {vmax:.2f} m/s (limit {ROBOT_MAX_SPEED})')
    if vmax > ROBOT_MAX_SPEED:
        problems.append(f'robot jumps ({vmax:.1f} m/s)')

    # --- cross-check against other samples' robot_pose_odom ---------------------------------
    anchors = np.array([float(s['anchor_time_sim']) for s in new])
    errs = []
    for j, sj in enumerate(new):
        tg = float(sj['grid_stamp_sim'])
        near = np.argsort(np.abs(anchors - anchors[j]))
        used = 0
        for i in near:
            if i == j:
                continue
            si = new[i]
            if str(si['episode_id']) != str(sj['episode_id']):
                continue
            times = np.concatenate([si['history_times'], si['future_times']])
            rows = np.concatenate([si['robot_history_odom'], si['robot_future_odom']])
            p = interp_track(times, rows, tg)
            if p is None:
                continue
            errs.append(math.hypot(p[0] - sj['robot_pose_odom'][0], p[1] - sj['robot_pose_odom'][1]))
            used += 1
            if used >= MAX_PARTNERS:
                break
    if errs:
        e = np.array(errs)
        p90 = float(np.percentile(e, 90))
        print(f'robot track   : {len(e)} cross-checks: median {np.median(e):.3f} m, p90 {p90:.3f} m, '
              f'max {e.max():.3f} m (p90 limit {TRACK_P90_LIMIT_M} m)')
        if p90 > TRACK_P90_LIMIT_M:
            problems.append(f'robot track disagrees with robot_pose_odom (p90 {p90:.2f} m)')
    else:
        p90 = None
        print('robot track   : no overlapping samples to cross-check')

    # --- episodes --------------------------------------------------------------------
    eps = OrderedDict()
    for s in new:
        eps.setdefault(str(s['episode_id']), []).append(s)
    window = None
    print('episodes      :')
    for eid, ss in eps.items():
        ta = [float(s['anchor_time_sim']) for s in ss]
        ets = np.array([float(s['episode_time_sec']) for s in ss]) if 'episode_time_sec' in ss[0] else None
        line = f'    {eid or "(none)":20s} {len(ss):5d} samples, anchors {min(ta):8.1f} .. {max(ta):8.1f} s'
        if eid and ets is not None:
            starts = np.array(ta) - ets
            spread = float(starts.max() - starts.min())
            line += f', episode start {np.median(starts):.2f} s'
            if spread > 1e-3 or (ets < 0).any():
                problems.append(f'episode {eid}: episode_time_sec inconsistent (spread {spread:.3f} s)')
                line += f'  <-- episode_time_sec inconsistent (spread {spread:.3f} s)'
        elif not eid and ets is not None and (ets != -1.0).any():
            problems.append('samples without an episode have episode_time_sec != -1')
        print(line)
    crossing = 0
    for a, b in zip(new, new[1:]):
        if str(a['episode_id']) != str(b['episode_id']):
            window = (len(a['history_times']) - 1 + len(a['future_times'])) * float(a['sample_period_sec'])
            gap = float(b['anchor_time_sim']) - float(a['anchor_time_sim'])
            ok = gap >= window - 0.2
            print(f'    change {str(a["episode_id"]) or "(none)"} -> {str(b["episode_id"]) or "(none)"}: '
                  f'anchors {gap:.1f} s apart (a window is {window:.1f} s) {"ok" if ok else "<-- A WINDOW CROSSES THE CHANGE"}')
            crossing += 0 if ok else 1
    if crossing:
        problems.append(f'{crossing} episode change(s) inside a sample window')

    detail = (f'{len(new)} samples; robot track p90 {p90:.3f} m' if p90 is not None else f'{len(new)} samples; robot track not cross-checked') + \
        f'; {len(eps)} episode(s): ' + ', '.join(f'{k or "(none)"} {len(v)}' for k, v in eps.items())
    if problems:
        print(f'SAMPLE FIELDS: FAIL: {"; ".join(problems)} ({detail})')
    else:
        print(f'SAMPLE FIELDS: PASS: {detail}')


if __name__ == '__main__':
    main()