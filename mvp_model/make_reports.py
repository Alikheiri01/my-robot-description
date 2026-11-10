#!/usr/bin/env python3
"""
make_reports.py -- builds ALL evaluation reports from the prediction files of every Kaggle run. numpy only, runs on
your laptop (no GPU, no Kaggle). Run from mvp_model/ :

    python3 make_reports.py                       # searches runs/ (and everything below it) for oof_*.npz
    python3 make_reports.py --runs runs --data packed/mvp_v1_packed.npz

It finds the files by NAME, wherever you unzipped them (runs/run_1, runs/run_2/runs, ...):
    oof_<variant>_s<seed>.npz   oof_<variant>_s<seed>_shufscn.npz   oof_<variant>_s<seed>_shufall.npz
and writes
    reports/eval_full.txt         residual family:  full vs hist, + shuffled crops (H4), noocc (H5), concat (H7)
    reports/eval_full_direct.txt  direct family:    full_direct vs hist_direct, + shuffled crops (H4)
It prints which files it found, so you can see at a glance if something is missing.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAT = re.compile(r'^oof_(?P<v>.+?)_s(?P<s>\d+)(?P<suf>_shufscn|_shufall)?\.npz$')


def find_files(root):
    found = {}                                        # (variant, suffix) -> {seed: path}
    for p in sorted(Path(root).rglob('oof_*.npz')):
        m = PAT.match(p.name)
        if not m or 'smoke' in p.parts:
            continue
        key = (m['v'], m['suf'] or '')
        seed = int(m['s'])
        if seed in found.setdefault(key, {}):
            print(f'[warn] {p.name} found twice, using {found[key][seed]} and ignoring {p}')
            continue
        found[key][seed] = p
    return found


def spec(found, v, suf=''):
    seeds = found.get((v, suf), {})
    return ','.join(str(seeds[s]) for s in sorted(seeds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--data', default=None, help='packed .npz or dataset folder (default: packed/mvp_v1_packed.npz, '
                                                  'else ../mvp_collection/datasets/mvp_v1)')
    ap.add_argument('--out', default='reports')
    args = ap.parse_args()
    data = args.data
    if data is None:
        data = 'packed/mvp_v1_packed.npz' if (HERE / 'packed' / 'mvp_v1_packed.npz').exists() else \
            '../mvp_collection/datasets/mvp_v1'
    found = find_files(args.runs)
    print(f'[files] under {args.runs}:')
    for (v, suf), seeds in sorted(found.items()):
        print(f'   {v + suf:28s} seeds {sorted(seeds)}   from {sorted({str(p.parent) for p in seeds.values()})}')
    if not found:
        sys.exit(f'no oof_*.npz found under {args.runs}')

    for test, ref, extras in (('full', 'hist', ('noocc', 'concat')), ('full_direct', 'hist_direct', ())):
        if not (spec(found, test) and spec(found, ref)):
            print(f'[skip] {test}: files for {test} and/or {ref} not found')
            continue
        cmd = [sys.executable, str(HERE / 'evaluate.py'), data, '--test', test, '--ref', ref,
               '--preds', f'{test}={spec(found, test)}', '--preds', f'{ref}={spec(found, ref)}',
               '--out', f'{args.out}/eval_{test}.txt']
        if spec(found, test, '_shufscn'):
            cmd += ['--preds', f'{test}_shufscn={spec(found, test, "_shufscn")}', '--shuf-scn', f'{test}_shufscn']
        if spec(found, test, '_shufall'):
            cmd += ['--preds', f'{test}_shufall={spec(found, test, "_shufall")}']
        for extra, flag in zip(extras, ('--noocc', '--concat')):
            if spec(found, extra):
                cmd += ['--preds', f'{extra}={spec(found, extra)}', flag, extra]
            else:
                print(f'[note] {extra}: no files, {flag} test not run')
        print(f'\n[run] {test} vs {ref}')
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            sys.exit(f'evaluate.py failed for {test}')
        # provenance: which folder every predictor in this report came from
        names = [test, ref, f'{test}_shufscn', f'{test}_shufall'] + [e for e in extras if spec(found, e)]
        with open(f'{args.out}/eval_{test}.txt', 'a') as f:
            f.write('\nSOURCES (folder of the prediction files of each predictor):\n')
            for n in names:
                v, suf = (n[:-8], '_shufscn') if n.endswith('_shufscn') else (n[:-8], '_shufall') if n.endswith('_shufall') else (n, '')
                if (v, suf) in found:
                    f.write(f'  {n:24s} {sorted({str(p.parent) for p in found[(v, suf)].values()})}\n')
    print(f'\n[done] reports are in {args.out}/ : eval_full.txt (read the HYPOTHESES block at the bottom) and eval_full_direct.txt')


if __name__ == '__main__':
    main()