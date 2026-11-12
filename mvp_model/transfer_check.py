#!/usr/bin/env python3
"""
transfer_check.py -- H10 (exploratory): does the model work on a scenario TYPE it never saw in training? numpy only, laptop.
    python3 transfer_check.py                  # uses runs/ (oof_<variant>_hold<S>_s<seed>.npz from the final Kaggle run)

For each held-out scenario S (B, C, D) the models were trained on all the other scenarios and tested on S. Printed, on near
events of S, ADE@3s of: constant velocity, hist_hn (no map) and full_hn (map), both held out, and for reference full_hn from
the normal cross-validation (where S was in the training data). Report only (no pass/fail): "transfers" means the held-out
map model beats the held-out no-map model by at least 5 % with the upper end of the interval below 0.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reports import find_files  # noqa: E402
from motion.baselines import cvm  # noqa: E402
from motion.data import load_dataset  # noqa: E402
from motion.stats import Boot, fmt  # noqa: E402
from wedge_check import ade3_mean  # noqa: E402

PAT = re.compile(r'^oof_(?P<v>.+)_hold(?P<S>[A-E])_s(?P<s>\d+)\.npz$')
NEAR_S, H3 = 4.0, 6


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--data', default=None)
    ap.add_argument('--map', default='full_hn')
    ap.add_argument('--nomap', default='hist_hn')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    data = args.data or ('packed/mvp_v1_packed.npz' if (here / 'packed' / 'mvp_v1_packed.npz').exists()
                         else '../mvp_collection/datasets/mvp_v1')
    ds = load_dataset(data, verbose=False)
    hold = {}
    for p in sorted(Path(args.runs).rglob('oof_*_hold*_s*.npz')):
        m = PAT.match(p.name)
        if m and 'smoke' not in str(p):
            hold.setdefault((m['v'], m['S']), {})[int(m['s'])] = p
    if not hold:
        sys.exit(f'no oof_*_hold*_s*.npz found under {args.runs}')
    found = find_files(args.runs)
    boot = Boot(ds)
    sc = np.array([s[0] for s in ds['scen']])
    near = ds['usable'] & ds['ev_future'] & (np.nan_to_num(ds['t_to_event'], nan=99) <= NEAR_S)
    d = np.linalg.norm(cvm(ds['hist'], 4) - ds['fut'][:, :, :2], axis=2)[:, :H3].mean(1)
    print('H10 transfer (exploratory): ADE@3s [m] on near events of the held-out scenario\n')
    for S in ('B', 'C', 'D'):
        if (args.map, S) not in hold or (args.nomap, S) not in hold:
            print(f'scenario {S}: files missing\n')
            continue
        m = near & (sc == S)
        vm = ade3_mean(ds, [hold[(args.map, S)][k] for k in sorted(hold[(args.map, S)])])
        vn = ade3_mean(ds, [hold[(args.nomap, S)][k] for k in sorted(hold[(args.nomap, S)])])
        print(f'scenario {S} held out (n={int(m.sum())})')
        print(f'   constant velocity          {fmt(boot.mean(d, m))}')
        print(f'   {args.nomap} (held out)     {fmt(boot.mean(vn, m))}')
        print(f'   {args.map} (held out)     {fmt(boot.mean(vm, m))}')
        fs = found.get((args.map, ''), {})
        if fs:
            print(f'   {args.map} (S in training)  {fmt(boot.mean(ade3_mean(ds, [fs[k] for k in sorted(fs)]), m))}   <- normal cross-validation')
        r = boot.mean(vm - vn, m)
        gain = -r[0] / boot.mean(vn, m)[0]
        print(f'   held-out map model minus held-out no-map model: {fmt(r, 3)} m, gain {gain * 100:.1f} %  ->  '
              f'{"TRANSFERS" if (r[2] < 0 and gain >= 0.05) else "does NOT transfer"}\n')


if __name__ == '__main__':
    main()