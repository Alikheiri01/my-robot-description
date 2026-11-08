#!/usr/bin/env python3
"""
evaluate.py -- ONE evaluation for every predictor (baselines now, the transformer later). numpy only.

Run from mvp_model/ :
    python3 evaluate.py                                   # step M1: baselines only
    python3 evaluate.py --preds full=runs/oof_full_s0.npz,runs/oof_full_s1.npz,runs/oof_full_s2.npz \\
        --preds hist=runs/oof_hist_s0.npz,... --test full --ref hist \\
        --shuf-scn full_shufscn=runs/oof_full_s0_shufscn.npz,...   (see docs/evaluation-plan.md for the full command)

Built-in predictors (all out-of-fold, environment-grouped folds, usable samples only):
    stand, cvm2, cvm4, ridge_hist (history-only ridge regression = the floor every model has to beat)
Model predictions are added with --preds name=file.npz[,file2.npz,...]  (files written by train.py; several files =
several seeds, the per-sample errors are averaged).

Metrics (metres, pedestrian frame): ADE@3s / ADE@6s = mean distance to the label over the first 3 s / all 6 s,
FDE = distance at the end of that horizon, occ = share of predicted points inside occupied crop cells.
Every mean has a 95 % interval from a bootstrap over ENVIRONMENTS.

Hypotheses H1-H7 are printed when --test is given; the rules are fixed in docs/evaluation-plan.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion.baselines import baseline_predictions  # noqa: E402
from motion.data import N_FUT, load_dataset  # noqa: E402
from motion.metrics import all_metrics, label_occ  # noqa: E402
from motion.stats import Boot, fmt  # noqa: E402

DEFAULT_DATASET = '../mvp_collection/datasets/mvp_v1'
NEAR_S = 4.0                     # "near event": the event happens within 4 s of the anchor


def load_preds(ds, spec):
    """'name=f1.npz,f2.npz' -> (name, [pred arrays aligned to the dataset order])"""
    name, files = spec.split('=', 1)
    index = {i: j for j, i in enumerate(ds['ids'])}
    arrs = []
    for f in files.split(','):
        z = np.load(f, allow_pickle=False)
        a = np.full((len(ds['ids']), N_FUT, 2), np.nan)
        for i, p in zip(z['ids'], z['pred']):
            j = index.get(str(i))
            if j is not None:
                a[j] = p
        arrs.append(a)
    return name, arrs


def compute_all(ds, extra_specs):
    """-> dict name -> dict metric -> (N,) per-sample values (seeds averaged)"""
    out = {}
    for k, p in baseline_predictions(ds).items():
        out[k] = all_metrics(p, ds)
    for spec in extra_specs:
        name, arrs = load_preds(ds, spec)
        ms = [all_metrics(a, ds) for a in arrs]
        out[name] = {m: np.mean([x[m] for x in ms], axis=0) for m in ms[0]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', nargs='?', default=DEFAULT_DATASET)
    ap.add_argument('--preds', action='append', default=[], help='name=file.npz[,file2.npz]')
    ap.add_argument('--ref', default='ridge_hist', help='reference predictor (the same model WITHOUT the crop: hist)')
    ap.add_argument('--test', default=None, help='predictor under test, normally full (enables H1-H7)')
    ap.add_argument('--shuf-scn', default=None, help='name of the within-scenario shuffled-crop predictions (H4)')
    ap.add_argument('--noocc', default=None, help='name of the no-occupied-channel predictions (H5)')
    ap.add_argument('--concat', default=None, help='name of the simple-fusion predictions (H7)')
    ap.add_argument('--out', default='reports/eval_report.txt')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    lines = []

    def out(s=''):
        print(s)
        lines.append(s)

    ds = load_dataset(args.dataset, folds=args.folds)
    u = ds['usable']
    E = compute_all(ds, args.preds)
    names = list(E)
    boot = Boot(ds)
    scens = sorted(set(ds['scen']))
    ev = ds['ev_future']
    tte = ds['t_to_event']
    near = u & ev & (tte <= NEAR_S)
    far = u & ev & (tte > NEAR_S)

    def sc_mask(*letters):
        return np.isin(ds['scen'], [s for s in scens if s[0] in letters])

    out('=' * 120)
    out(f'EVALUATION  {args.dataset}   usable samples {int(u.sum())} of {len(u)}   environments {len(boot.envs)}   '
        f'folds {args.folds} (grouped by environment)')
    out('Metres, mean [95 % interval from a bootstrap over environments]. Lower is better. near event = event within '
        f'{NEAR_S:g} s of the anchor.')
    out('=' * 120)

    def table(title, metric, subsets):
        out()
        out(title)
        out(f'  {"scenario":9s} {"subset":10s} {"n":>6s} {"envs":>5s} | ' + ' | '.join(f'{k:^19s}' for k in names))
        for sc in scens + ['ALL']:
            base = u & ((ds['scen'] == sc) if sc != 'ALL' else True)
            for sub, m in subsets(base):
                if not m.any():
                    continue
                cells = [f'{fmt(boot.mean(E[k][metric], m)):^19s}' for k in names]
                out(f'  {sc:9s} {sub:10s} {int(m.sum()):6d} {len(set(ds["env_key"][m])):5d} | ' + ' | '.join(cells))

    subs = lambda b: [('all', b), ('near event', b & near), ('far event', b & far), ('no event', b & ~ev)]  # noqa: E731
    table('E1  ADE@3s  (PRIMARY: the first 3 s, where the crop can still see the cause)', 'ade3', subs)
    table('E2  ADE@6s  (secondary: the full 6 s)', 'ade6', subs)

    out()
    out('E3  FDE (point values): distance at 3 s / at 6 s')
    out(f'  {"scenario":9s} {"subset":10s} | ' + ' | '.join(f'{k:>12s}' for k in names))
    for sc in scens:
        base = u & (ds['scen'] == sc)
        for sub, m in (('all', base), ('near event', base & near), ('no event', base & ~ev)):
            if m.any():
                out(f'  {sc:9s} {sub:10s} | ' + ' | '.join(
                    f'{np.mean(E[k]["fde3"][m]):5.2f} /{np.mean(E[k]["fde6"][m]):5.2f}' for k in names))

    out()
    out('E4  OCCUPIED SHARE: fraction of the 12 predicted points that lie inside an occupied crop cell (label = reference, ~0)')
    out(f'  {"scenario":9s} {"subset":10s} {"label":>7s} | ' + ' | '.join(f'{k:>10s}' for k in names))
    lab = label_occ(ds)
    for sc in scens:
        base = u & (ds['scen'] == sc)
        for sub, m in (('all', base), ('near event', base & near)):
            if m.any():
                out(f'  {sc:9s} {sub:10s} {np.mean(lab[m]):7.3f} | ' + ' | '.join(
                    f'{np.mean(E[k]["occ"][m]):10.3f}' for k in names))

    out()
    out('E5  ADE@6s on event samples by time to the event (B, C, D pooled per bin)')
    out(f'  {"bin":8s} {"n":>6s} | ' + ' | '.join(f'{k:^19s}' for k in names))
    evm = u & ev & sc_mask('B', 'C', 'D')
    for lo, hi in ((0, 2), (2, 4), (4, 6.01)):
        m = evm & (tte > lo) & (tte <= hi)
        if m.any():
            out(f'  {lo:g}-{min(hi, 6):g} s   {int(m.sum()):6d} | ' + ' | '.join(
                f'{fmt(boot.mean(E[k]["ade6"], m)):^19s}' for k in names))

    out()
    out('E6  ADE@3s on near events (B, C, D) by visibility of the event point in the crop')
    out(f'  {"event point":14s} {"n":>6s} | ' + ' | '.join(f'{k:^19s}' for k in names))
    nm = near & sc_mask('B', 'C', 'D')
    for labl, m in (('visible', nm & ds['ev_visible']), ('not visible', nm & ~ds['ev_visible'])):
        if m.any():
            out(f'  {labl:14s} {int(m.sum()):6d} | ' + ' | '.join(
                f'{fmt(boot.mean(E[k]["ade3"], m)):^19s}' for k in names))

    # ----------------------------------------------------------------------------------------------- hypotheses
    if args.test:
        t, r = args.test, args.ref
        for k in (t, r):
            if k not in E:
                sys.exit(f'unknown predictor {k}; have {names}')
        out()
        out('=' * 120)
        out(f'HYPOTHESES   test = {t}   reference = {r}   (difference = test - reference; negative = test better)')
        out('=' * 120)

        def gain_line(tag, label, mask, rule_txt):
            a = boot.mean(E[t]['ade3'] - E[r]['ade3'], mask)
            ref_m = np.nanmean(E[r]['ade3'][mask]) if mask.any() else np.nan
            gain = -a[0] / ref_m * 100 if ref_m else np.nan
            ok = np.isfinite(a[2]) and a[2] < 0 and gain >= 5
            out(f'  {tag:20s} {"PASS" if ok else "FAIL"}   {label}: ADE@3s diff {fmt(a, 3)} m = gain {gain:.1f} % of the '
                f'reference ({int(a[3])} samples, {a[4]} envs)   [{rule_txt}]')
            return ok

        rule = 'PASS: upper bound < 0 and gain >= 5 %'
        gain_line('H1 B (PRIMARY)', 'B near events', near & sc_mask('B'), rule)
        gain_line('H1 C', 'C near events', near & sc_mask('C'), rule)
        for sc in scens:
            if sc[0] == 'D':
                m = near & (ds['scen'] == sc)
                if m.any():
                    a = boot.mean(E[t]['ade3'] - E[r]['ade3'], m)
                    out(f'       {sc:9s} near events: diff {fmt(a, 3)} m ({int(a[3])} samples; descriptive only)')
        a = boot.mean(E[t]['ade6'] - E[r]['ade6'], u & ev & sc_mask('B', 'C'))
        out(f'       B+C all events, ADE@6s (secondary): diff {fmt(a, 3)} m ({int(a[3])} samples; descriptive only)')

        ctrl = u & (sc_mask('A', 'E') | (sc_mask('B', 'C', 'D') & ~ev))
        a = boot.rel_diff(E[t]['ade3'], E[r]['ade3'], ctrl)
        ok = np.isfinite(a[2]) and a[2] < 0.03
        out(f'  {"H2 no harm":20s} {"PASS" if ok else "FAIL"}   controls (A, E all; B, C, D without event): ADE@3s relative '
            f'change {fmt(a, 3)} ({int(a[3])} samples, {a[4]} envs)   [PASS: upper bound < +0.03 = at most 3 % worse]')

        a = boot.diff_of_means(E[t]['ade3'] - E[r]['ade3'], nm & ds['ev_visible'],
                               E[t]['ade3'] - E[r]['ade3'], nm & ~ds['ev_visible'])
        ok = np.isfinite(a[2]) and a[2] < 0
        out(f'  {"H3 visibility":20s} {"PASS" if ok else "FAIL"}   (diff on visible) - (diff on not visible), B,C,D near events: '
            f'{fmt(a, 3)} m   [exploratory; PASS: upper bound < 0]')

        if args.shuf_scn:
            if args.shuf_scn not in E:
                sys.exit(f'unknown predictor {args.shuf_scn}; have {names}')
            for tag, sc in (('H4 B (PRIMARY)', 'B'), ('H4 C', 'C')):
                m = near & sc_mask(sc)
                a = boot.mean(E[args.shuf_scn]['ade3'] - E[t]['ade3'], m)
                ok = np.isfinite(a[1]) and a[1] > 0
                out(f'  {tag:20s} {"PASS" if ok else "FAIL"}   {sc} near events, crops shuffled WITHIN the scenario minus normal: '
                    f'{fmt(a, 3)} m   [PASS: lower bound > 0]')
        else:
            out('  H4                   not run (needs --shuf-scn)')

        if args.noocc:
            if args.noocc not in E:
                sys.exit(f'unknown predictor {args.noocc}; have {names}')
            for tag, sc in (('H5 B (PRIMARY)', 'B'), ('H5 C', 'C')):
                m = near & sc_mask(sc)
                a = boot.mean(E[args.noocc]['ade3'] - E[t]['ade3'], m)
                ok = np.isfinite(a[1]) and a[1] > 0
                out(f'  {tag:20s} {"PASS" if ok else "FAIL"}   {sc} near events, no-occupied-channel minus full: {fmt(a, 3)} m   '
                    f'[PASS: lower bound > 0]')
        else:
            out('  H5                   not run (needs --noocc)')

        wall = u & sc_mask('A', 'B', 'D')
        a = boot.mean(E[t]['occ'] - E[r]['occ'], wall)
        ok = np.isfinite(a[2]) and a[2] < 0.005
        out(f'  {"H6 not through walls":20s} {"PASS" if ok else "FAIL"}   occupied share, A, B, D all samples: diff {fmt(a, 4)} '
            f'(label {np.mean(lab[wall]):.4f})   [PASS: upper bound < +0.005 = not worse than the reference]')

        if args.concat:
            if args.concat not in E:
                sys.exit(f'unknown predictor {args.concat}; have {names}')
            for tag, sc in (('H7 B', 'B'), ('H7 C', 'C')):
                m = near & sc_mask(sc)
                a = boot.mean(E[t]['ade3'] - E[args.concat]['ade3'], m)
                verdict = ('transformer better' if a[2] < 0 else ('simple fusion better' if a[1] > 0 else 'TIE'))
                out(f'  {tag:20s} {verdict:20s} {sc} near events, {t} minus {args.concat}: {fmt(a, 3)} m   '
                    f'[information only: TIE means the architecture is not the contribution]')
        else:
            out('  H7                   not run (needs --concat)')
        out()
        out('  H8 counterfactual edits, H9 speed, H10 transfer: later steps (see docs/evaluation-plan.md)')

    p = Path(args.out)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('\n'.join(lines) + '\n')
        print(f'\n[report written to {p}]')
    except OSError as e:
        print(f'[could not write report: {e}]')


if __name__ == '__main__':
    main()
