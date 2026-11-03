#!/usr/bin/env python3
"""
episode_samples.py -- lists the saved samples of ONE episode of a recorded run, one line per
sample, with the recorder's own visibility flags (2026-11-03).

Use it when check_run.py shows an episode whose measured usable share is far below the
planned one: it tells whether the anchor was not seen, or the history steps were not seen,
and when in the episode that happened.

  t        anchor time, seconds after the walk start (episodes/<id>.json)
  anchor   1 = pedestrian seen at the anchor
  history  the 8 history steps, oldest first: 1 = seen, 0 = not seen, . = no grid near that time
           (counts as not seen)
  usable   anchor seen and >= USABLE_MIN_HISTORY_SEEN history steps seen

USAGE
    python3 check_scripts/episode_samples.py <run folder or environment folder> <episode id>
    python3 check_scripts/episode_samples.py datasets/dev/B_corner/env_B_001 env_B_001_ep0
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mvp_config as C  # noqa: E402


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run, eid = Path(sys.argv[1]), sys.argv[2]
    if (run / 'environment.json').exists():                 # environment folder -> its newest run
        runs = sorted((run / 'runs').glob('*'))
        if not runs:
            sys.exit('no recorded runs found')
        run = runs[-1]
    ep_file = run / 'episodes' / f'{eid}.json'
    if not ep_file.exists():
        sys.exit(f'{ep_file} not found')
    meas = json.loads(ep_file.read_text())['measured']
    t0 = meas['t_walk_start_sim']
    rows, fields = [], None
    for f in sorted((run / 'samples').glob('*.npz')):
        d = np.load(f, allow_pickle=True)
        if 'episode_id' not in d.files or str(d['episode_id']) != eid:
            continue
        fields = fields or d.files
        hist = np.asarray(d['ped_seen_history']).astype(int).ravel()
        anchor = int(d['ped_seen_anchor'])
        usable = anchor == 1 and int((hist == 1).sum()) >= C.USABLE_MIN_HISTORY_SEEN
        rows.append((float(d['anchor_time_sim']) - t0, anchor, ''.join('.' if v < 0 else str(v) for v in hist), usable, f.name))
    print(f'== {run}')
    print(f'   {eid}: {len(rows)} samples, walk start at sim {t0:.1f} s, event at '
          f'{meas.get("t_event_after_walk_start_sec", "-")} s, end at {meas.get("t_end_after_walk_start_sec", "-")} s')
    print(f'   fields in a sample: {", ".join(fields) if fields else "-"}')
    print(f'   {"t (s)":>7s}  anchor  history   usable  file')
    for t, a, h, u, name in sorted(rows):
        print(f'   {t:7.1f}  {a:6d}  {h:8s}  {"yes" if u else "no":6s}  {name}')
    if rows:
        print(f'   usable {sum(r[3] for r in rows)}/{len(rows)}, anchor seen {sum(r[1] for r in rows)}/{len(rows)}')


if __name__ == '__main__':
    main()