#!/usr/bin/env python3
"""
check_run.py -- checks a run recorded by episode_runner.py, per episode (P5, 2026-11-01).

record_session.py's report.txt judges the whole recording (timing, exclusion, ...).
This adds what only the episodes know:
  outcome       every episode ended 'ok' (not stuck / timeout / aborted)
  samples       saved samples per episode; samples that belong to no episode
                ('reset_...', 'end_...', 'idle') -- should be 0: such windows mix two episodes
  usable        measured usable share per episode (anchor seen and >= 6 of 8 history steps seen,
                the recorder's own flags) vs the planned share; run total >= 50 %
  event         measured event time vs planned (s after the walk start), and how many samples
                have the event INSIDE their 6 s future (the samples that teach the manoeuvre)

USAGE
    python3 check_scripts/check_run.py datasets/dev/A_detour/env_A_000/runs/<run folder>
    python3 check_scripts/check_run.py datasets/dev/A_detour/env_A_000            # its newest run
    python3 check_scripts/check_run.py datasets/dev/A_detour                      # newest run of every environment
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mvp_config as C  # noqa: E402


def find_runs(p):
    p = Path(p)
    if (p / 'report.txt').exists():
        return [p]
    if (p / 'environment.json').exists():
        runs = sorted((p / 'runs').glob('*'))
        return runs[-1:] if runs else []
    out = []
    for e in sorted(p.iterdir()):
        out += find_runs(e) if (e / 'environment.json').exists() else []
    return out


def check_run(run):
    probs = []
    eps = {}
    for f in sorted((run / 'episodes').glob('*.json')):
        e = json.loads(f.read_text())
        eps[e['episode_id']] = e
    groups = defaultdict(list)
    for f in sorted((run / 'samples').glob('*.npz')):
        d = np.load(f, allow_pickle=True)
        eid = str(d['episode_id']) if 'episode_id' in d.files else ''
        hist = d['ped_seen_history']
        usable = int(d['ped_seen_anchor']) == 1 and int((hist == 1).sum()) >= C.USABLE_MIN_HISTORY_SEEN
        groups[eid].append((float(d['anchor_time_sim']), usable))
    rep = (run / 'report.txt').read_text() if (run / 'report.txt').exists() else ''
    overall = next((l.split('OVERALL:')[1].strip() for l in rep.splitlines() if 'OVERALL:' in l), 'NO REPORT')
    print(f'== {run}')
    print(f'   record_session report: {overall}')
    if not overall.startswith('PASS'):
        probs.append(f'report.txt: {overall}')
    stray = {k: len(v) for k, v in groups.items() if k not in eps}
    if stray:
        probs.append('samples outside any episode: ' + ', '.join(f'{k or "(none)"} {n}' for k, n in stray.items()))
    tot_n = tot_u = 0
    print(f'   {"episode":18s} {"outcome":8s} {"samples":>7s} {"usable":>7s} {"planned":>8s}   event (s after walk start)'
          f'  samples with the event in their future')
    for eid, e in eps.items():
        m = e['measured']
        s = groups.get(eid, [])
        n, u = len(s), sum(x[1] for x in s)
        tot_n, tot_u = tot_n + n, tot_u + u
        share = u / n if n else 0.0
        ev = ''
        if e['event'] and 't_event_sim' in m:
            te = m['t_event_sim']
            inside = [x for x in s if x[0] < te <= x[0] + C.FUTURE_STEPS * C.PERIOD_SEC]
            ev = (f'{te - m["t_walk_start_sim"]:5.1f} (planned {e["event"]["t_planned_sec"]:5.1f})   '
                  f'{len(inside)} ({sum(x[1] for x in inside)} usable)')
            if not inside:
                probs.append(f'{eid}: no sample has the event in its future')
        elif e['event']:
            ev = 'NOT MEASURED'
            probs.append(f'{eid}: event time not measured')
        print(f'   {eid:18s} {m["outcome"]:8s} {n:7d} {share:7.0%} {e["planned"]["usable_share"]:8.0%}   {ev}')
        if m['outcome'] != 'ok':
            probs.append(f'{eid}: outcome {m["outcome"]}')
    run_share = tot_u / tot_n if tot_n else 0.0
    print(f'   run: {tot_n} samples, usable {run_share:.0%}')
    if run_share < 0.5:
        probs.append(f'usable {run_share:.0%} < 50%')
    for p in probs:
        print(f'   - {p}')
    print(f'   {"PASS" if not probs else "NOT YET"}')
    return not probs


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = [r for a in sys.argv[1:] for r in find_runs(a)]
    if not runs:
        sys.exit('no recorded runs found')
    ok = all([check_run(r) for r in runs])
    print('\nOVERALL:', 'PASS' if ok else 'NOT YET')
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()