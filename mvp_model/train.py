#!/usr/bin/env python3
"""
train.py -- trains one model variant with environment-grouped cross-validation and writes OUT-OF-FOLD predictions
that evaluate.py compares with the baselines.

!! NOT RUN YET (written without torch). Order: python3 tests/test_torch.py  ->  smoke run  ->  full run:
!!     python3 train.py --variant full --seeds 0 --epochs 2 --fold 0        (smoke: it runs, the loss is finite)
!!     python3 train.py --variant full                                       (3 seeds x 5 folds)
Look at a smoke-run ADE only as "it runs and the loss goes down", not as a result.

Per seed and per fold k (5 folds, grouped by environment, stratified by scenario):
    test  = usable samples of fold k (never used for any decision)
    pool  = usable samples of the other folds; ~1/8 of their ENVIRONMENTS (per scenario) become the validation set
    train on the rest: AdamW, one-cycle schedule, loss = ADE (mean L2 over the 12 steps), random left/right mirror
    (training only), early stopping on the validation ADE; the best weights predict the test fold.
    Variants that use the crop also write two "shuffled crop" predictions (the same trained model, crops swapped):
      _shufscn  crops swapped with a sample of ANOTHER environment of the SAME scenario  -> used by H4
      _shufall  crops swapped with any sample of the test fold (information only: does it only know the scenario?)

Variants: full | hist (no crop, same depth) | noocc (obstacles shown as free) | concat (simple fusion) | robot
          each also as  <name>_direct  = the network outputs the path itself instead of a correction to CVM
Outputs in --out (default runs/):  oof_<variant>_s<seed>.npz  (+ _shufscn.npz, _shufall.npz)   keys: ids, pred (M,12,2)
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion.data import load_dataset  # noqa: E402
from motion.models import VARIANTS, count_params, flip_batch  # noqa: E402

DEFAULT_DATASET = '../mvp_collection/datasets/mvp_v1'


def split_val(ds, pool_mask, seed, frac=1 / 8):
    """Choose validation ENVIRONMENTS inside the pool, stratified by scenario (at least one per scenario)."""
    rng = np.random.RandomState(seed)
    val_envs = set()
    for sc in sorted(set(ds['scen'][pool_mask])):
        envs = sorted(set(ds['env_key'][pool_mask & (ds['scen'] == sc)]))
        rng.shuffle(envs)
        val_envs.update(envs[:max(1, int(round(len(envs) * frac)))])
    isval = np.array([e in val_envs for e in ds['env_key']])
    return pool_mask & ~isval, pool_mask & isval


def shuffle_partner(ds, te, within_scenario, seed):
    """For every test sample the index of the sample whose crop it receives (never from its own environment if
    another environment exists)."""
    rng = np.random.RandomState(seed)
    partner = np.empty(len(te), dtype=np.int64)
    for a, i in enumerate(te):
        pool = te[(ds['scen'][te] == ds['scen'][i])] if within_scenario else te
        other = pool[ds['env_key'][pool] != ds['env_key'][i]]
        pool = other if len(other) else pool
        partner[a] = pool[rng.randint(len(pool))]
    return partner


def make_tensors(ds, dev):
    return dict(hist=torch.tensor(ds['hist'], device=dev), crop=torch.tensor(ds['crop'], device=dev),
                fut=torch.tensor(ds['fut'], device=dev), robot=torch.tensor(ds['robot'], device=dev))


@torch.no_grad()
def predict(model, T, idx, crop_idx=None, bs=512):
    """crop_idx: use the crops of these samples instead (for the shuffle tests)."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        cj = j if crop_idx is None else crop_idx[i:i + bs]
        out.append(model(T['hist'][j], T['crop'][cj], T['robot'][j]).cpu())
    return torch.cat(out).numpy()


def ade_of(model, T, idx):
    p = torch.tensor(predict(model, T, idx), device=T['fut'].device)
    return (p - T['fut'][idx][:, :, :2]).norm(dim=2).mean().item()


def train_one(args, T, tr, va, seed, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VARIANTS[args.variant]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(tr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    best, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.bs):
            j = perm[i:i + args.bs]
            h, c, f, r = T['hist'][j], T['crop'][j], T['fut'][j], T['robot'][j]
            if not args.no_flip:
                m = torch.rand(len(j), device=dev) < 0.5
                if m.any():
                    h2, c2, f2 = flip_batch(h[m], c[m], f[m])
                    r2 = r[m].clone()
                    r2[:, 1] = -r2[:, 1]
                    h, c, f, r = h.clone(), c.clone(), f.clone(), r.clone()
                    h[m], c[m], f[m], r[m] = h2, c2, f2, r2
            pred = model(h, c, r)
            loss = (pred - f[:, :, :2]).norm(dim=2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        v = ade_of(model, T, va)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if args.verbose:
            print(f'      epoch {ep + 1:3d}  train ADE {tot / nb:.3f}  val ADE {v:.3f}  best {best:.3f}')
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', nargs='?', default=DEFAULT_DATASET)
    ap.add_argument('--variant', default='full', choices=sorted(VARIANTS))
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fold', type=int, default=None, help='run only this fold (smoke test)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--out', default='runs')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    dev = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                       ('cpu' if args.device == 'auto' else args.device))
    ds = load_dataset(args.dataset, folds=args.folds, cache_dir=args.out)
    T = make_tensors(ds, dev)
    u = ds['usable']
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'[train] variant {args.variant}  device {dev}  parameters {count_params(VARIANTS[args.variant]())}')
    uses_crop = not args.variant.startswith('hist')
    for seed in [int(s) for s in args.seeds.split(',')]:
        pred = np.full((len(u), 12, 2), np.nan, np.float32)
        p_scn, p_all = pred.copy(), pred.copy()
        t0 = time.time()
        for k in ([args.fold] if args.fold is not None else range(args.folds)):
            te = np.nonzero(u & (ds['fold'] == k))[0]
            tr_m, va_m = split_val(ds, u & (ds['fold'] != k), seed * 100 + k)
            tr, va = np.nonzero(tr_m)[0], np.nonzero(va_m)[0]
            model, best = train_one(args, T, tr, va, seed, dev)
            pred[te] = predict(model, T, te)
            if uses_crop:
                p_scn[te] = predict(model, T, te, crop_idx=torch.tensor(shuffle_partner(ds, te, True, seed), device=dev))
                p_all[te] = predict(model, T, te, crop_idx=torch.tensor(shuffle_partner(ds, te, False, seed), device=dev))
            err = np.linalg.norm(pred[te] - ds['fut'][te, :, :2], axis=2).mean()
            print(f'[train] seed {seed} fold {k}: train {len(tr)}  val {len(va)}  test {len(te)}  '
                  f'val ADE {best:.3f}  test ADE@6s {err:.3f}   ({time.time() - t0:.0f} s)')
        m = u & np.isfinite(pred[:, 0, 0])
        base = out / f'oof_{args.variant}_s{seed}'
        np.savez(f'{base}.npz', ids=ds['ids'][m], pred=pred[m])
        if uses_crop:
            np.savez(f'{base}_shufscn.npz', ids=ds['ids'][m], pred=p_scn[m])
            np.savez(f'{base}_shufall.npz', ids=ds['ids'][m], pred=p_all[m])
        print(f'[train] wrote {base}.npz ({int(m.sum())} samples)')


if __name__ == '__main__':
    main()
