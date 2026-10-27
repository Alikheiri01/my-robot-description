#!/usr/bin/env python3
"""
analyze_goal_approach.py -- how does the (HuNav) pedestrian slow down when it
arrives at a goal / stops? (2026-10-26)

WHY
HuNav's social force has no "slow down before the goal" in the textbook sense: the
goal force pulls at full speed and the pedestrian brakes only when it overshoots.
Whether the data contains a gradual, scene-explainable slowdown decides scenario C
(stop in front of a wall) and WHEN the start gate must close around an arrival.

WHAT IT DOES (offline, reads saved samples only, touches nothing else)
  1. Rebuilds the pedestrian's track in the ODOM frame from every sample's history and
     future rows (local frame -> odom with ped_anchor_odom), merged by simulation time
     (median of the duplicates). Resampled to 0.1 s; NEVER interpolated across a gap
     longer than 0.6 s (the track is cut into segments there).
  2. Speed = central difference over +-0.2 s.
  3. Finds ARRIVALS
        default (--stops)   a stop = speed < 0.1 m/s for >= 1 s; arrival time = first slow
                            step, arrival point = mean standing position
        --goals "x,y; x,y"  goals in ODOM coordinates (as in the `goals` shell function);
                            an arrival = the closest approach to a goal within 0.5 m; the
                            distances in the profile are to THAT position (where it turned)
                            (if it dwells there: the first step within 5 cm of the closest distance)
     Only the APPROACH side (before the arrival) is used, because at a looping goal the
     reversal brakes the pedestrian anyway.
  4. Per arrival. Everything is limited to the CURRENT LEG: after the previous arrival
     (closest approach) or the end of the previous stop. (2026-10-27: with looping goals
     5 m apart a leg lasts only 3-10 s, and without this the "approach" contained the
     pedestrian LEAVING the same point one leg earlier -- slow, accelerating, close to it.)
        cruise speed      median speed from 8 s to 4 s before arrival (skipped if < 0.3 m/s
                          or too few valid samples). If that window reaches back before
                          the leg start: the median speed 2-3 m before arrival instead
                          (column 'from' = time / dist); skipped if the leg is too short.
        slowdown start    the last time before the arrival with speed >= 0.9 x cruise:
                          its distance and time before the arrival
        braking time      (stops only) from the slowdown start to the first slow step
        closest approach  (goals only) speed there / cruise, and the closest distance
  5. Profile pooled over the run: speed / cruise against distance to the arrival point,
     bins 0-.25 .25-.5 .5-.75 .75-1 1-1.5 1.5-2 2-3 m (median, 10th-90th percentile, n),
     plus speed / cruise at 3, 2, 1.5, 1, 0.5, 0 s before arrival.

RESOLUTION: speed is a central difference over +-0.2 s, so a sharp brake looks up to
speed x 0.2 s earlier (0.1 m at 0.5 m/s, 0.3 m at 1.5 m/s). Tested on synthetic tracks
(2026-10-27): constant-deceleration stops found 0.53 / 1.05 / 1.58 m before arrival vs
0.51 / 1.01 / 1.52 m true; full-speed-then-brake loops gave NO ANTICIPATORY SLOWDOWN.
A sanity check of the track rebuild: the cruise speed must match the speed you set.

HOW TO READ IT
  * speed stays >= ~0.9 x cruise until < 0.5 m from the arrival point -> NO anticipatory
    slowdown: scenario C needs a bridge-side slowdown or stays out; the gate closes AT arrival.
  * gradual slowdown over >= 1 m -> C is feasible as HuNav does it; in A/B/D/E the gate
    closes at the slowdown start.
  * compare runs with different speeds (0.5 / 1.0 / 1.5 m/s): does the slowdown distance
    grow with speed? Then a fixed closing distance is not enough.

USAGE (from dataset_codes/, venv with numpy; matplotlib only for --plot)
    python3 analyze_goal_approach.py runs/*T10_stop_go* --plot
    python3 analyze_goal_approach.py runs/*T05_slow*       --goals "4,-2.5; 4,2.5"
    python3 analyze_goal_approach.py runs/*T01_cross_mid* --goals "4,-2.5; 4,2.5"
    python3 analyze_goal_approach.py runs/*T06_fast*      --goals "4,-2.5; 4,2.5" --plot
Each argument is a run folder (all sample_*.npz below it are used). With --plot a PNG is
saved next to the run (goal_approach.png) and its path printed.

OUTPUT: per run the table and profiles above, then a VERDICT line (the reading rules
below, applied automatically), and with several runs a SUMMARY table (cruise speed vs
slowdown distance) for the speed-scaling question.
  VERDICT rules  NO ANTICIPATORY SLOWDOWN  median speed/cruise >= 0.9 in every bin beyond 0.5 m
                 GRADUAL SLOWDOWN          median slowdown start >= 1.0 m before arrival
                 SHORT SLOWDOWN            in between (starts 0.5 .. 1.0 m before arrival)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

DT = 0.1                 # s: resampling step
MAX_GAP = 0.6            # s: never interpolate across a hole longer than this
HALF_WIN = 0.2           # s: central-difference half window
STOP_SPEED = 0.1         # m/s
STOP_MIN_SEC = 1.0
GOAL_RADIUS = 0.5        # m
CRUISE_FROM, CRUISE_TO = 8.0, 4.0     # s before arrival
CRUISE_MIN_SPEED = 0.3   # m/s
CRUISE_MIN_FRAC = 0.5    # at least this fraction of the cruise window must have valid speed
CRUISE_DIST = (2.0, 3.0) # m before arrival: fallback cruise band when the leg is too short in time
CRUISE_DIST_MIN_N = 5    # valid speed points needed in that band
SLOW_FRAC = 0.9
BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SEC_BEFORE = [3.0, 2.0, 1.5, 1.0, 0.5, 0.0]
GRADUAL_M = 1.0          # slowdown starting at least this far before arrival = gradual
NO_ANTICIP_M = 0.5       # beyond this distance the speed must stay >= SLOW_FRAC x cruise


# --- loading and track rebuilding -------------------------------------------------
def load_track(run_dir):
    """Merged odom track from all samples: (times, x, y) sorted, duplicates merged by median."""
    files = sorted(Path(run_dir).rglob('sample_*.npz'))
    if not files:
        raise SystemExit(f'no sample_*.npz under {run_dir}')
    T, X, Y = [], [], []
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            a = d['ped_anchor_odom'].astype(np.float64)
            c, s = math.cos(a[2]), math.sin(a[2])
            for rows, times in ((d['trajectory_history'], d['history_times']),
                                (d['future_target'], d['future_times'])):
                x, y = rows[:, 0].astype(np.float64), rows[:, 1].astype(np.float64)
                T.append(times.astype(np.float64))
                X.append(a[0] + c * x - s * y)
                Y.append(a[1] + s * x + c * y)
        except Exception as e:                    # a broken file must not stop the analysis
            print(f'  skipped {f.name}: {e}')
    T, X, Y = np.concatenate(T), np.concatenate(X), np.concatenate(Y)
    key = np.round(T / 0.01).astype(np.int64)     # same instant (10 ms) -> one point, median
    order = np.argsort(key, kind='stable')
    key, T, X, Y = key[order], T[order], X[order], Y[order]
    uniq, start = np.unique(key, return_index=True)
    ends = list(start[1:]) + [len(key)]
    t = np.array([np.median(T[s:e]) for s, e in zip(start, ends)])
    x = np.array([np.median(X[s:e]) for s, e in zip(start, ends)])
    y = np.array([np.median(Y[s:e]) for s, e in zip(start, ends)])
    return len(files), t, x, y


def resample(t, x, y):
    """0.1 s grid, NaN where the nearest raw points are more than MAX_GAP apart."""
    g = np.arange(math.ceil(t[0] / DT) * DT, t[-1], DT)
    gx, gy = np.interp(g, t, x), np.interp(g, t, y)
    idx = np.searchsorted(t, g)
    lo, hi = np.clip(idx - 1, 0, len(t) - 1), np.clip(idx, 0, len(t) - 1)
    gap = t[hi] - t[lo]
    bad = (gap > MAX_GAP) & (g > t[lo]) & (g < t[hi])
    gx[bad], gy[bad] = np.nan, np.nan
    return g, gx, gy


def speeds(g, gx, gy):
    k = int(round(HALF_WIN / DT))
    v = np.full(len(g), np.nan)
    for i in range(k, len(g) - k):
        dx, dy = gx[i + k] - gx[i - k], gy[i + k] - gy[i - k]
        v[i] = math.hypot(dx, dy) / (2 * k * DT)
    return v


# --- arrivals ---------------------------------------------------------------------
def find_stops(g, gx, gy, v):
    """[(i_first_slow, (px, py), i_moving_again)]: speed < STOP_SPEED for >= STOP_MIN_SEC."""
    need = int(round(STOP_MIN_SEC / DT))
    slow = np.where(np.isnan(v), False, v < STOP_SPEED)
    out, i, n = [], 0, len(g)
    while i < n:
        if slow[i]:
            j = i
            while j < n and slow[j]:
                j += 1
            if j - i >= need:
                out.append((i, (float(np.nanmean(gx[i:j])), float(np.nanmean(gy[i:j]))), j))
            i = j
        else:
            i += 1
    return out


def find_goal_arrivals(g, gx, gy, goals):
    """[(i_closest, (goal_x, goal_y), i_closest)]: one per pass within GOAL_RADIUS of a goal."""
    out = []
    for gxx, gyy in goals:
        d = np.hypot(gx - gxx, gy - gyy)
        inside = np.where(np.isnan(d), False, d < GOAL_RADIUS)
        i, n = 0, len(g)
        while i < n:
            if inside[i]:
                j = i
                while j < n and inside[j]:
                    j += 1
                dmin = float(np.nanmin(d[i:j]))
                # closest approach; if the pedestrian dwells there (stays within 5 cm of the
                # minimum), the FIRST such step, so the approach side contains no standing frames
                k = i + int(np.nanargmax(np.where(np.isnan(d[i:j]), False, d[i:j] <= dmin + 0.05)))
                out.append((k, (gxx, gyy), k))
                i = j
            else:
                i += 1
    return sorted(out)


# --- one arrival ------------------------------------------------------------------
def analyse_arrival(g, gx, gy, v, ia, point, mode, leg_start):
    """leg_start: first index of the current leg (after the previous arrival / stop), or None.
    point: the stop position, or (goals mode) the goal itself. In goals mode the distances are
    measured to where the pedestrian REALLY turned around (its closest approach, typically
    0.3-0.45 m short of the goal point = HuNav's goal radius), not to the goal point: otherwise
    that fixed 0.35 m is counted as 'slowdown' (2026-10-27)."""
    goal = point
    if mode == 'goals':
        point = (float(gx[ia]), float(gy[ia]))
    n_c0, n_c1 = int(round(CRUISE_FROM / DT)), int(round(CRUISE_TO / DT))
    if leg_start is None and ia - n_c0 < 0:
        return None, 'not enough track before it (start of the recording)'
    lo = max(ia - n_c0, leg_start if leg_start is not None else 0, 0)
    dist = np.hypot(gx - point[0], gy - point[1])
    if lo == ia - n_c0:                                   # the whole 8..4 s window is in this leg
        win = v[ia - n_c0: ia - n_c1 + 1]
        ok = win[~np.isnan(win)]
        if len(ok) < CRUISE_MIN_FRAC * len(win):
            return None, 'too few valid speed samples in the cruise window'
        how = 'time'
    else:                                                 # short leg: 2..3 m before arrival
        seg = np.arange(leg_start, ia + 1)
        m = (dist[seg] >= CRUISE_DIST[0]) & (dist[seg] < CRUISE_DIST[1]) & ~np.isnan(v[seg])
        ok = v[seg][m]
        if len(ok) < CRUISE_DIST_MIN_N:
            return None, 'leg too short for a cruise speed (< 3 m or < 8 s of track since the previous arrival)'
        how = 'dist'
    cruise = float(np.median(ok))
    if cruise < CRUISE_MIN_SPEED:
        return None, f'cruise speed {cruise:.2f} m/s < {CRUISE_MIN_SPEED} (no approach / start-up)'
    # slowdown start: last time before arrival with speed >= SLOW_FRAC * cruise
    i_s = None
    for i in range(ia, lo - 1, -1):
        if not np.isnan(v[i]) and v[i] >= SLOW_FRAC * cruise:
            i_s = i
            break
    res = {'t_arrival': float(g[ia]), 'cruise': cruise, 'point': point, 'cruise_from': how}
    if i_s is not None:
        res['slow_start_dist'] = float(math.hypot(gx[i_s] - point[0], gy[i_s] - point[1]))
        res['slow_start_sec'] = float((ia - i_s) * DT)
        if mode == 'stops':
            res['brake_sec'] = float((ia - i_s) * DT)      # slow step = arrival
    else:
        res['slow_start_dist'] = res['slow_start_sec'] = None
    if mode == 'goals':
        res['closest_dist'] = float(math.hypot(point[0] - goal[0], point[1] - goal[1]))
        res['v_at_closest_ratio'] = float(v[ia] / cruise) if not np.isnan(v[ia]) else None
    # profile points (approach side only): (distance to point, speed / cruise)
    prof = []
    for i in range(lo, ia + 1):
        if not np.isnan(v[i]):
            prof.append((math.hypot(gx[i] - point[0], gy[i] - point[1]), float(v[i] / cruise), (ia - i) * DT))
    res['profile'] = prof
    return res, ''


# --- reporting --------------------------------------------------------------------
def fmt(x, nd=2):
    return '  n/a' if x is None else f'{x:.{nd}f}'


def verdict(arrivals, pts):
    sd = [r['slow_start_dist'] for r in arrivals if r['slow_start_dist'] is not None]
    far_ok = True
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        if lo < NO_ANTICIP_M:
            continue
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if m.sum() and np.median(pts[m, 1]) < SLOW_FRAC:
            far_ok = False
    med = float(np.median(sd)) if sd else None
    if far_ok and (med is None or med < NO_ANTICIP_M):
        return 'NO ANTICIPATORY SLOWDOWN', med
    if med is not None and med >= GRADUAL_M:
        return 'GRADUAL SLOWDOWN', med
    return 'SHORT SLOWDOWN', med


def report(name, n_files, g, v, arrivals, skipped, mode, plot_path=None):
    print(f'\n=== {name}: {n_files} samples, track {g[0]:.1f}..{g[-1]:.1f} s '
          f'({np.count_nonzero(~np.isnan(v))} valid speed points) ===')
    print(f'  {len(arrivals)} usable arrival(s), {len(skipped)} skipped by the rules')
    for t, why in skipped:
        print(f'    skipped arrival at t={t:.1f} s: {why}')
    if not arrivals:
        print('  VERDICT: NO DATA (no usable arrival)')
        return None
    head = '  arrival t   cruise from   slowdown starts (dist, sec before)'
    head += '   braking s' if mode == 'stops' else '   turns m short of goal   v/cruise there'
    print(head)
    for r in arrivals:
        line = (f'  {r["t_arrival"]:8.1f}   {r["cruise"]:5.2f} {r["cruise_from"]:4s}   '
                f'{fmt(r["slow_start_dist"]):>6} m, {fmt(r["slow_start_sec"], 1):>5} s')
        line += f'        {fmt(r.get("brake_sec"), 1):>5}' if mode == 'stops' else \
            f'        {fmt(r.get("closest_dist")):>5}       {fmt(r.get("v_at_closest_ratio")):>5}'
        print(line)
    cr = [r['cruise'] for r in arrivals]
    sd = [r['slow_start_dist'] for r in arrivals if r['slow_start_dist'] is not None]
    ss = [r['slow_start_sec'] for r in arrivals if r['slow_start_sec'] is not None]
    print(f'  cruise median {np.median(cr):.2f} m/s' +
          (f'; slowdown starts median {np.median(sd):.2f} m ({np.median(ss):.1f} s) before arrival' if sd else ''))

    pts = np.array([p for r in arrivals for p in r['profile']])
    print('\n  speed / cruise vs distance to the arrival point (approach side, pooled):')
    print('    distance m      n   median   p10..p90')
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if m.sum():
            q = np.percentile(pts[m, 1], [10, 50, 90])
            print(f'    {lo:4.2f}-{hi:4.2f}  {int(m.sum()):6d}   {q[1]:5.2f}    {q[0]:.2f}..{q[2]:.2f}')
        else:
            print(f'    {lo:4.2f}-{hi:4.2f}       0')
    print('\n  speed / cruise at fixed times before arrival (median over arrivals):')
    for s in SEC_BEFORE:
        m = np.abs(pts[:, 2] - s) < DT / 2 + 1e-9
        print(f'    {s:3.1f} s before: ' + (f'{np.median(pts[m, 1]):.2f}  (n={int(m.sum())})' if m.sum() else 'n/a'))

    vd, med = verdict(arrivals, pts)
    print(f'\n  VERDICT: {vd}' + (f' (slowdown starts {med:.2f} m before arrival, median)' if med is not None else ''))
    if plot_path is not None:
        plot(name, g, v, arrivals, pts, plot_path, mode)
    return {'name': name, 'n': len(arrivals), 'cruise': float(np.median(cr)), 'slow_m': med,
            'slow_s': float(np.median(ss)) if ss else None, 'verdict': vd}


def plot(name, g, v, arrivals, pts, path, mode):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (matplotlib not installed: no plot)')
        return
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    ax[0].plot(g, v, lw=1)
    for r in arrivals:
        ax[0].axvline(r['t_arrival'], color='r', ls='--', lw=0.8)
    ax[0].axhline(STOP_SPEED, color='gray', ls=':', lw=0.8)
    ax[0].set(xlabel='simulation time (s)', ylabel='speed (m/s)',
              title=f'{name}: speed (red = arrival, {"stop" if mode == "stops" else "closest goal approach"})')
    ax[1].scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.4)
    cs, med = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if m.sum():
            cs.append((lo + hi) / 2)
            med.append(np.median(pts[m, 1]))
    ax[1].plot(cs, med, 'r-o', label='binned median')
    ax[1].axhline(SLOW_FRAC, color='gray', ls=':', lw=0.8)
    ax[1].set(xlabel='distance to the arrival point (m)', ylabel='speed / cruise', xlim=(3.2, -0.1),
              title='approach profile (x axis reversed: arrival on the right)')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'\n  plot saved: {path}')


def parse_goals(text):
    out = []
    for part in text.split(';'):
        if part.strip():
            x, y = part.split(',')
            out.append((float(x), float(y)))
    return out


def main():
    ap = argparse.ArgumentParser(description='HuNav slowdown near goals / stops, from saved samples.')
    ap.add_argument('runs', nargs='+', help='run folder(s) containing sample_*.npz')
    ap.add_argument('--goals', help='goals in ODOM coordinates, e.g. "4,-2.5; 4,2.5" (default: detect stops)')
    ap.add_argument('--plot', action='store_true', help='save goal_approach.png in each run folder')
    args = ap.parse_args()
    goals = parse_goals(args.goals) if args.goals else None
    mode = 'goals' if goals else 'stops'

    summary = []
    for run in args.runs:
        run = Path(run)
        if not run.is_dir():
            print(f'not a folder: {run}')
            continue
        n_files, t, x, y = load_track(run)
        g, gx, gy = resample(t, x, y)
        v = speeds(g, gx, gy)
        cands = find_goal_arrivals(g, gx, gy, goals) if goals else find_stops(g, gx, gy, v)
        arrivals, skipped = [], []
        prev_end = None
        for ia, point, i_end in cands:
            r, why = analyse_arrival(g, gx, gy, v, ia, point, mode, prev_end)
            prev_end = i_end + 1
            if r:
                arrivals.append(r)
            else:
                skipped.append((float(g[ia]), why))
        r = report(run.name, n_files, g, v, arrivals, skipped, mode,
                   (run / 'goal_approach.png') if args.plot else None)
        if r:
            summary.append(r)
    if len(summary) > 1:
        print('\n=== SUMMARY (does the slowdown distance grow with speed?) ===')
        print('  run                                   arrivals  cruise m/s  slowdown starts   verdict')
        for r in sorted(summary, key=lambda r: r['cruise']):
            print(f'  {r["name"][:38]:38s} {r["n"]:6d}     {r["cruise"]:5.2f}     '
                  f'{fmt(r["slow_m"]):>5} m {fmt(r["slow_s"], 1):>4} s    {r["verdict"]}')


if __name__ == '__main__':
    main()