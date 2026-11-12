#!/usr/bin/env python3
"""
wedge_check.py -- H11, the wedge-only leak check. numpy only, laptop. Run from mvp_model/ :
    python3 wedge_check.py                 # uses runs/ (finds oof_wedge_s*, oof_hist_s*, oof_noocc_s*, oof_full_s* by name)

Rule (fixed in docs/evaluation-plan.md before the run), on B near events, ADE@3s of `wedge` minus `hist`:
    LEAK    if the upper end of the 95 % interval is below 0 AND the gain is at least 5 % of hist's ADE
    NO LEAK otherwise.
Also printed: how much of the noocc gain the wedge explains,  share = (hist - wedge) / (hist - noocc)  (B near), and the same
numbers for C near (information). `wedge` = the full network whose crop is only the camera cone (no walls, no shadows).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reports import find_files  # noqa: E402
from motion.data import N_FUT, load_dataset  # noqa: E402
from motion.stats import Boot, fmt  # noqa: E402

NEAR_S, H3 = 4.0, 6


def ade3_mean(ds, paths):
    index = {i: j for j, i in enumerate(ds['ids'])}
    per = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        a = np.full((len(ds['ids']), N_FUT, 2), np.nan)
        for i, q in zip(z['ids'], z['pred']):
            j = index.get(str(i))
            if j is not None:
                a[j] = q
        per.append(np.linalg.norm(a - ds['fut'][:, :, :2], axis=2)[:, :H3].mean(1))
    return np.mean(per, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--data', default=None)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    data = args.data or ('packed/mvp_v1_packed.npz' if (here / 'packed' / 'mvp_v1_packed.npz').exists()
                         else '../mvp_collection/datasets/mvp_v1')
    ds = load_dataset(data, verbose=False)
    found = find_files(args.runs)
    v = {}
    for name in ('wedge', 'hist', 'noocc', 'full'):
        fs = found.get((name, ''), {})
        if fs:
            v[name] = ade3_mean(ds, [fs[s] for s in sorted(fs)])
    for need in ('wedge', 'hist'):
        if need not in v:
            sys.exit(f'no oof_{need}_s*.npz found under {args.runs}')
    boot = Boot(ds)
    sc = np.array([s[0] for s in ds['scen']])
    near = ds['usable'] & ds['ev_future'] & (np.nan_to_num(ds['t_to_event'], nan=99) <= NEAR_S)
    print('H11 wedge-only leak check, ADE@3s [m], near events\n')
    for g in ('B', 'C'):
        m = near & (sc == g)
        print(f'scenario {g} (n={int(m.sum())})')
        for name in ('hist', 'noocc', 'wedge', 'full'):
            if name in v:
                print(f'   {name:6s} {fmt(boot.mean(v[name], m))}')
        d = boot.mean(v['wedge'] - v['hist'], m)
        h = boot.mean(v['hist'], m)[0]
        gain = -d[0] / h
        leak = d[2] < 0 and gain >= 0.05
        print(f'   wedge - hist: {fmt(d, 3)} m   gain {gain * 100:.1f} % of hist   ->  '
              f'{"LEAK (robot placement alone predicts it)" if leak else "NO LEAK"}' + ('   [PRIMARY]' if g == 'B' else '   [information]'))
        if 'noocc' in v:
            sH, c = boot.agg(v['hist'], m)
            sW, _ = boot.agg(v['wedge'], m)
            sN, _ = boot.agg(v['noocc'], m)
            r = boot.ci(lambda s: ((sH[s].sum(1) - sW[s].sum(1)) / c[s].sum(1)) / ((sH[s].sum(1) - sN[s].sum(1)) / c[s].sum(1)),
                        [(sH, c)])
            print(f'   share of the noocc gain explained by the wedge: {fmt(r)}   (0 = none, 1 = all)')
        print()


if __name__ == '__main__':
    main()