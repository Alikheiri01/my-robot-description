#!/usr/bin/env python3
"""
seed_check.py -- is the edge of model A over model B (e.g. full vs concat) stable, or the size of the training noise?
numpy only, laptop. Run from mvp_model/ :
    python3 seed_check.py                      # full vs concat, files found under runs/
    python3 seed_check.py --a full --b noocc
For every seed (alone) and every fold it prints mean ADE@3s of A minus B on the near events of B and of C (negative = A
better). The 95 % bootstrap interval in evaluate.py averages the 3 seeds first, so it does not show seed noise; this does.
A claim "A is better" is only kept if the sign is the same for every seed (and for most folds).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reports import find_files  # noqa: E402
from motion.data import N_FUT, load_dataset  # noqa: E402

NEAR_S, H3 = 4.0, 6


def ade3_of(ds, path):
    z = np.load(path, allow_pickle=False)
    index = {i: j for j, i in enumerate(ds['ids'])}
    a = np.full((len(ds['ids']), N_FUT, 2), np.nan)
    for i, p in zip(z['ids'], z['pred']):
        j = index.get(str(i))
        if j is not None:
            a[j] = p
    return np.linalg.norm(a - ds['fut'][:, :, :2], axis=2)[:, :H3].mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--data', default=None)
    ap.add_argument('--a', default='full')
    ap.add_argument('--b', default='concat')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    data = args.data or ('packed/mvp_v1_packed.npz' if (here / 'packed' / 'mvp_v1_packed.npz').exists()
                         else '../mvp_collection/datasets/mvp_v1')
    ds = load_dataset(data, verbose=False)
    found = find_files(args.runs)
    fa, fb = found.get((args.a, ''), {}), found.get((args.b, ''), {})
    seeds = sorted(set(fa) & set(fb))
    if not seeds:
        sys.exit(f'no common seeds for {args.a} and {args.b} under {args.runs}')
    sc = np.array([s[0] for s in ds['scen']])
    near = ds['usable'] & ds['ev_future'] & (np.nan_to_num(ds['t_to_event'], nan=99) <= NEAR_S)
    print(f'{args.a} minus {args.b}, ADE@3s [m] on near events (negative = {args.a} better)\n')
    diffs = {s: ade3_of(ds, fa[s]) - ade3_of(ds, fb[s]) for s in seeds}
    for g in ('B', 'C'):
        m = near & (sc == g)
        print(f'scenario {g} (n={int(m.sum())})')
        per_seed = []
        for s in seeds:
            v = np.nanmean(diffs[s][m])
            per_seed.append(v)
            folds = ' '.join(f'{np.nanmean(diffs[s][m & (ds["fold"] == k)]):+.3f}' for k in range(int(ds['fold'].max()) + 1))
            print(f'   seed {s}: {v:+.4f}    per fold: {folds}')
        allf = np.array([[np.nanmean(diffs[s][m & (ds["fold"] == k)]) for k in range(int(ds['fold'].max()) + 1)] for s in seeds])
        print(f'   seed means: {np.mean(per_seed):+.4f}  (spread between seeds {np.std(per_seed):.4f})   '
              f'seeds negative: {sum(v < 0 for v in per_seed)}/{len(seeds)}   seed-fold cells negative: '
              f'{int((allf < 0).sum())}/{allf.size}\n')


if __name__ == '__main__':
    main()