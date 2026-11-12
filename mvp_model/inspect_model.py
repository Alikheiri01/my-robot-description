#!/usr/bin/env python3
"""
inspect_model.py -- step M5: look at the trained model and try to break it. Laptop, CPU is enough (no training).

Run from mvp_model/ :
    python3 inspect_model.py                       # everything (pictures, damage table, H8)
    python3 inspect_model.py --seeds 0 --what pics # quick look: one seed, only the pictures
    python3 inspect_model.py --models runs/run_2   # models of another run (e.g. the damage-trained one)

It needs the trained weights  <models>/**/full_s<seed>_f<fold>.pt  (found by name, anywhere below --models) and the
packed dataset. EVERY sample is predicted by the fold model that never saw its environment (same rule as train.py),
and seeds are averaged like evaluate.py does.

Outputs (reports/inspect/):  inspect_report.txt   pics_B.png  pics_C.png   damage.png
What it does:
  pics    predicted vs true path for the best / typical / worst B and C near-event samples
  damage  the SAME test samples with a damaged input; ADE@3s per condition (the "before" table of M5)
            map:     no map, shifted 0.2 / 0.5 m, wrong cells 1 % / 5 %, rotated 5 / 10 degrees
            history: tracker noise 0.05 / 0.1 / 0.2 m (velocities and heading are recomputed from the noisy points)
  h8      counterfactual edits: change ONLY the map, keep the history, see if the prediction reacts like physics says

THE H8 RULES ARE FIXED HERE, BEFORE THE FIRST RUN (changing them after seeing numbers is not allowed; a change needs a dated note):
  H8a remove: C near-event samples whose stop point is visible; the occupied cells within 1.5 m of the stop point are
      made free (unknown cells stay unknown). The prediction must move TOWARD constant velocity (CVM4) over the whole 6 s by
      at least 0.05 m (the mean distance to CVM4 gets smaller by 0.05 m). CONTROL: the same removal 2.5 m to the side of the
      stop point (the side with more obstacle cells). PASS: share of samples that react >= 60 % AND (share_edit - share_control) >= 30 points.
  H8b insert: A and E samples that walk (speed >= 0.8 m/s). A wall (1.6 m long, 0.3 m thick) is put across the walking
      direction 2.5 m ahead, only on known cells, and the cells behind it (seen from the robot) become unknown.
      Look only at samples whose normal prediction crosses that wall line. The edited prediction must NOT cross it.
      CONTROL: the same wall 2.5 m to the side, parallel to the walking direction (it does not block the way); the
      mean change of the prediction it causes is the reference. PASS: share that no longer cross >= 50 % AND the mean change with
      the real wall >= 3 x the control change. H8b is exploratory (a wall in the way is outside the training scenes).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion import damage as D  # noqa: E402
from motion.baselines import cvm  # noqa: E402
from motion.data import DT, N_FUT, event_point, load_dataset  # noqa: E402
from motion.stats import Boot, fmt  # noqa: E402

NEAR_S = 4.0
H3 = 6
H8A_MOVE, H8A_SHARE, H8A_GAP, H8A_R, H8A_SIDE = 0.05, 0.60, 0.30, 1.5, 2.5
H8B_SHARE, H8B_RATIO, H8B_DIST, H8B_MIN_SPEED = 0.50, 3.0, 2.5, 0.8
WALL_HALF_LEN, WALL_HALF_THICK = 0.8, 0.15
GROUPS = ('B near', 'C near', 'A all', 'E all')


# ------------------------------------------------------------------------------------------------ model ensemble
class Ensemble:
    """All trained fold models of one variant; predict(...) -> (seeds, n, 12, 2)."""

    def __init__(self, root, variant='full', seeds=None, device='cpu'):
        import torch
        from motion.models import VARIANTS
        self.torch, self.device = torch, torch.device(device)
        files = {}
        for p in sorted(Path(root).rglob(f'{variant}_s*_f*.pt')):
            try:
                s, f = p.stem[len(variant) + 2:].split('_f')
                files[(int(s), int(f))] = p
            except ValueError:
                continue
        if not files:
            raise SystemExit(f'no {variant}_s<seed>_f<fold>.pt found under {root}')
        self.seeds = sorted({s for s, _ in files if seeds is None or s in seeds})
        self.models = {}
        for (s, f), p in files.items():
            if s in self.seeds:
                m = VARIANTS[variant]()
                m.load_state_dict(torch.load(p, map_location='cpu'))
                self.models[(s, f)] = m.to(self.device).eval()
        self.variant = variant
        print(f'[models] {variant}: {len(self.models)} files, seeds {self.seeds}, from {root}')

    def predict(self, hist, crop, folds, robot=None, bs=512):
        torch = self.torch
        out = np.zeros((len(self.seeds), len(hist), N_FUT, 2), np.float32)
        with torch.no_grad():
            for si, s in enumerate(self.seeds):
                for f in sorted(set(folds)):
                    idx = np.nonzero(folds == f)[0]
                    m = self.models[(s, f)]
                    for i in range(0, len(idx), bs):
                        j = idx[i:i + bs]
                        h = torch.as_tensor(hist[j], device=self.device)
                        c = torch.as_tensor(crop[j], device=self.device)
                        r = None if robot is None else torch.as_tensor(robot[j], device=self.device)
                        out[si, j] = m(h, c, r).cpu().numpy()
        return out


# ------------------------------------------------------------------------------------------------ helpers
def group_masks(ds):
    u = ds['usable']
    sc = np.array([s[0] for s in ds['scen']])
    near = ds['ev_future'] & (np.nan_to_num(ds['t_to_event'], nan=99) <= NEAR_S)
    return {'B near': u & (sc == 'B') & near, 'C near': u & (sc == 'C') & near,
            'A all': u & (sc == 'A'), 'E all': u & (sc == 'E')}


def ade3(pred, ds, idx):
    """pred (S,n,12,2) for the samples idx -> per-sample ADE@3s averaged over seeds, as a full-length array (nan elsewhere)."""
    d = np.linalg.norm(pred - ds['fut'][idx][None, :, :, :2], axis=3)[:, :, :H3].mean(2).mean(0)
    out = np.full(len(ds['ids']), np.nan)
    out[idx] = d
    return out


def table_row(boot, vals, masks):
    return [fmt(boot.mean(vals, m)) for m in masks.values()]


def print_table(lines, title, rows, masks, boot):
    lines.append('')
    lines.append(title)
    w = max(len(n) for n, _ in rows) + 2
    lines.append(f'{"":{w}s}' + ''.join(f'{g:>22s}' for g in masks))
    for name, vals in rows:
        lines.append(f'{name:{w}s}' + ''.join(f'{c:>22s}' for c in table_row(boot, vals, masks)))


def load_oof(root, variant):
    """Stored out-of-fold predictions of a variant (for the self-check and the 'hist' reference row), or None."""
    files = sorted(p for p in Path(root).rglob(f'oof_{variant}_s*.npz')
                   if 'smoke' not in p.parts and '_shuf' not in p.name)
    return files


def oof_ade3(ds, files):
    index = {i: j for j, i in enumerate(ds['ids'])}
    per = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        a = np.full((len(ds['ids']), N_FUT, 2), np.nan)
        for i, p in zip(z['ids'], z['pred']):
            j = index.get(str(i))
            if j is not None:
                a[j] = p
        per.append(a)
    d = np.stack([np.linalg.norm(a - ds['fut'][:, :, :2], axis=2)[:, :H3].mean(1) for a in per]).mean(0)
    return d, per


# ------------------------------------------------------------------------------------------------ pictures
def pictures(ds, ens, hist_ens, vals_full, vals_hist, outdir, per_row=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#c8c8c8', '#ffffff', '#202020'])
    masks = group_masks(ds)
    rng = np.random.RandomState(0)
    for g in ('B near', 'C near'):
        idx = np.nonzero(masks[g] & np.isfinite(vals_full))[0]
        if len(idx) < 3:
            continue
        order = idx[np.argsort(vals_full[idx])]
        picks = {'best': [], 'typical': [], 'worst': []}
        seen = set()
        def take(cands, key):
            for i in cands:
                if ds['env_key'][i] not in seen and len(picks[key]) < per_row:
                    picks[key].append(i)
                    seen.add(ds['env_key'][i])
        take(order, 'best')
        take(order[::-1], 'worst')
        mid = len(order) // 2
        take(sorted(order[max(0, mid - 40):mid + 40], key=lambda i: rng.rand()), 'typical')
        fig, axs = plt.subplots(3, per_row, figsize=(3.4 * per_row, 3.2 * 3), squeeze=False)
        for r, key in enumerate(('best', 'typical', 'worst')):
            for c in range(per_row):
                ax = axs[r][c]
                if c >= len(picks[key]):
                    ax.axis('off')
                    continue
                i = picks[key][c]
                arr = np.where(ds['crop'][i] == -1, 0, np.where(ds['crop'][i] == 0, 1, 2))
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4), interpolation='nearest')
                h = ds['hist'][i]
                f = ds['fut'][i, :, :2]
                cv = cvm(ds['hist'][i:i + 1], 4)[0]
                pf = ens.predict(ds['hist'][i:i + 1], ds['crop'][i:i + 1], ds['fold'][i:i + 1]).mean(0)[0]
                ax.plot(h[:, 0], h[:, 1], 'o-', color='#555555', ms=3, lw=1, label='history')
                ax.plot(np.r_[0, f[:, 0]], np.r_[0, f[:, 1]], '-', color='#1b9e77', lw=2.2, label='truth')
                ax.plot(np.r_[0, cv[:, 0]], np.r_[0, cv[:, 1]], '--', color='#7570b3', lw=1.5, label='CVM4')
                if hist_ens is not None:
                    ph = hist_ens.predict(ds['hist'][i:i + 1], ds['crop'][i:i + 1], ds['fold'][i:i + 1]).mean(0)[0]
                    ax.plot(np.r_[0, ph[:, 0]], np.r_[0, ph[:, 1]], '-', color='#d95f02', lw=1.5, label='hist (no map)')
                ax.plot(np.r_[0, pf[:, 0]], np.r_[0, pf[:, 1]], '-', color='#e7298a', lw=2, label='full (map)')
                ax.plot(*ds['robot'][i], '^', color='#1f78b4', ms=7, label='robot')
                ax.set_xlim(-2, 6)
                ax.set_ylim(-4, 4)
                ax.set_aspect('equal')
                ax.tick_params(labelsize=7)
                t = f'{key}: {ds["env"][i]}   t_event {ds["t_to_event"][i]:.1f}s\nADE@3s full {vals_full[i]:.2f}'
                if vals_hist is not None and np.isfinite(vals_hist[i]):
                    t += f'  hist {vals_hist[i]:.2f}'
                ax.set_title(t, fontsize=8)
                if r == 0 and c == 0:
                    handles, labels = ax.get_legend_handles_labels()
        fig.suptitle(f'{g}: grey = unknown, white = free, black = occupied (pedestrian at 0,0 looking right)', fontsize=10)
        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = outdir / f'pics_{g[0]}.png'
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f'[pics] {path}')


def damage_chart(rows_by_cond, refs, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    conds = [c for c, _ in rows_by_cond]
    for ax, g in zip(axs, ('B near', 'C near')):
        v = [r[g] for _, r in rows_by_cond]
        ax.barh(range(len(conds)), v, color='#e7298a')
        for k, name, col in (('hist', 'hist (no map)', '#d95f02'), ('cvm4', 'CVM4', '#7570b3')):
            if refs.get(k, {}).get(g) is not None:
                ax.axvline(refs[k][g], color=col, ls='--', lw=1.5, label=name)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('ADE@3s on near events [m]  (lower is better)')
        ax.set_title(g)
        ax.legend(fontsize=8)
    fig.suptitle('full model with damaged input (bars) against the no-map model and constant velocity (dashed lines)')
    fig.tight_layout()
    fig.savefig(outdir / 'damage.png', dpi=110)
    plt.close(fig)
    print(f'[damage] {outdir / "damage.png"}')


# ------------------------------------------------------------------------------------------------ damage table
def damage_table(ds, ens, boot, masks, idx, lines, ref_vals, outdir, want_chart=True, hist_ens=None):
    folds, hist, crop = ds['fold'][idx], ds['hist'][idx], ds['crop'][idx]
    conds = [('clean (reference)', lambda: (hist, crop, 0))]
    conds += [('map: none', lambda: (hist, D.no_map(crop), 0))]
    conds += [(f'map: shifted {m} m', (lambda m=m: (hist, D.shift_map(crop, m, 1), 0))) for m in (0.2, 0.5)]
    conds += [(f'map: {int(p * 100)} % wrong cells', (lambda p=p: (hist, D.noise_cells(crop, p, 2), 0))) for p in (0.01, 0.05)]
    conds += [(f'map: rotated {a} deg', (lambda a=a: (hist, D.rotate_map(crop, a, 3), 0))) for a in (5, 10)]
    for s in (0.05, 0.1, 0.2):
        def mk(s=s):
            hn, sh = D.noisy_history(hist, s, 4)
            return hn, crop, sh
        conds.append((f'history noise {s} m', mk))
    rows, per_cond, noise_refs = [], [], []
    for name, make in conds:
        h, c, shift = make()
        pred = ens.predict(h, c, folds)
        pred = pred + (shift[None, :, None, :] if np.ndim(shift) else 0)     # back to the true frame
        v = ade3(pred, ds, idx)
        rows.append((name, v))
        if name.startswith('history noise'):                                 # fair opponents under the SAME noisy history
            cvn = (cvm(h, 4) + shift[:, None, :])[None]
            noise_refs.append((name.replace('history noise', 'cvm4, history noise'), ade3(cvn, ds, idx)))
            if hist_ens is not None:
                ph = hist_ens.predict(h, c, folds) + shift[None, :, None, :]
                noise_refs.append((name.replace('history noise', 'hist, history noise'), ade3(ph, ds, idx)))
        per_cond.append((name, {g: boot.mean(v, m)[0] for g, m in masks.items()}))
        print(f'[damage] {name}')
    for name, v in ref_vals + noise_refs:
        rows.append((name, v))
    print_table(lines, 'DAMAGED INPUT: ADE@3s [m], mean and 95 % interval (bootstrap over environments), same samples in every row',
                rows, masks, boot)
    refs = {}
    for k, v in ref_vals:
        key = 'hist' if k.startswith('hist') else 'cvm4' if k.startswith('cvm4') else None
        if key:
            refs[key] = {g: boot.mean(v, m)[0] for g, m in masks.items()}
    if want_chart:
        damage_chart(per_cond, refs, outdir)
    return rows


# ------------------------------------------------------------------------------------------------ H8
def to_cvm_dist(pred, cv):
    return np.linalg.norm(pred - cv, axis=1).mean()                 # whole 6 s: the stop may come after 3 s


def crosses(path, d, u, half_len):
    """Does the polyline (0,0),p1..p12 cross the wall line (at distance d along u, |across| <= half_len)?"""
    p = np.vstack([[0.0, 0.0], path])
    n = np.array([-u[1], u[0]])
    a = p @ u - d
    for k in range(len(p) - 1):
        if a[k] < 0 <= a[k + 1]:
            t = -a[k] / (a[k + 1] - a[k]) if a[k + 1] != a[k] else 0.0
            q = p[k] + t * (p[k + 1] - p[k])
            if abs(q @ n) <= half_len:
                return True
    return False


def h8(ds, ens, masks, lines, max_n=600):
    rng = np.random.RandomState(0)
    lines.append('')
    lines.append('H8 COUNTERFACTUAL MAP EDITS (rules fixed in the docstring of inspect_model.py before the first run)')
    # ---- H8a: remove the obstacle at the stop point (C near, event point visible)
    idx = np.nonzero(masks['C near'] & ds['ev_visible'])[0]
    idx = rng.permutation(idx)[:max_n]
    crops_e, crops_c, keep = [], [], []
    for i in idx:
        px, py = event_point(ds['fut'][i, :, :2], ds['t_to_event'][i])
        ce, ne = D.remove_obstacle(ds['crop'][i], px, py, H8A_R)
        v = cvm(ds['hist'][i:i + 1], 4)[0, 0]
        n = np.array([-v[1], v[0]]) / max(np.linalg.norm(v), 1e-6)
        opts = [D.remove_obstacle(ds['crop'][i], px + sg * H8A_SIDE * n[0], py + sg * H8A_SIDE * n[1], H8A_R) for sg in (-1, 1)]
        cc, nc = max(opts, key=lambda o: o[1])                     # control: same removal 2.5 m to the side (more cells side)
        if ne >= 10 and nc >= 10:
            crops_e.append(ce)
            crops_c.append(cc)
            keep.append(i)
    keep = np.array(keep, dtype=int)
    if len(keep) < 10:
        lines.append(f'H8a: only {len(keep)} usable samples (stop point with obstacle cells on both sides): not evaluated')
    else:
        h, f = ds['hist'][keep], ds['fold'][keep]
        p0 = ens.predict(h, ds['crop'][keep], f).mean(0)
        pe = ens.predict(h, np.stack(crops_e), f).mean(0)
        pc = ens.predict(h, np.stack(crops_c), f).mean(0)
        cv = cvm(h, 4)
        mv = lambda pa, pb: np.array([to_cvm_dist(pa[k], cv[k]) - to_cvm_dist(pb[k], cv[k]) for k in range(len(keep))])
        react_e = mv(p0, pe) >= H8A_MOVE            # distance to CVM got smaller by >= 0.05 m
        react_c = mv(p0, pc) >= H8A_MOVE
        se, sc = react_e.mean(), react_c.mean()
        ok = se >= H8A_SHARE and (se - sc) >= H8A_GAP
        lines.append(f'H8a remove the obstacle at the stop point (C near, visible; n={len(keep)}):')
        lines.append(f'    moves toward constant velocity by >= {H8A_MOVE} m:  edit {se * 100:.0f} %   control (2.5 m aside) {sc * 100:.0f} %')
        lines.append(f'    mean change of distance to CVM over 6 s: edit {mv(p0, pe).mean():+.3f} m   control {mv(p0, pc).mean():+.3f} m')
        lines.append(f'    rule: edit >= {H8A_SHARE * 100:.0f} % and edit - control >= {H8A_GAP * 100:.0f} points  ->  {"PASS" if ok else "FAIL"}')
    # ---- H8b: put a wall in the way (A and E, walking)
    idx = np.nonzero(masks['A all'] | masks['E all'])[0]
    sp = np.linalg.norm(cvm(ds['hist'][idx], 4)[:, 1] - cvm(ds['hist'][idx], 4)[:, 0], axis=1) / DT
    idx = rng.permutation(idx[sp >= H8B_MIN_SPEED])[:max_n * 2]
    real, ctrl, keep, dirs = [], [], [], []
    cv_all = cvm(ds['hist'][idx], 4)
    for k, i in enumerate(idx):
        v = cv_all[k, 0]
        u = v / np.linalg.norm(v)
        n = np.array([-u[1], u[0]])
        cw, w = D.insert_wall(ds['crop'][i], ds['robot'][i], H8B_DIST, u, WALL_HALF_LEN, WALL_HALF_THICK)
        # control: the same wall 2.5 m to the SIDE, running parallel to the walking direction (it does not block the way)
        cc = None
        for sg in (1, -1):                                           # try both sides, take the first that fits in known cells
            cc, w2 = D.insert_wall(ds['crop'][i], ds['robot'][i], H8B_DIST, sg * n, WALL_HALF_LEN, WALL_HALF_THICK)
            if cc is not None:
                break
        if cw is not None and cc is not None:
            real.append(cw)
            ctrl.append(cc)
            keep.append(i)
            dirs.append(u)
    keep = np.array(keep, dtype=int)
    if len(keep) < 10:
        lines.append(f'H8b: only {len(keep)} usable samples: not evaluated')
        return
    keep, real, ctrl, dirs = keep[:max_n], real[:max_n], ctrl[:max_n], dirs[:max_n]
    h, f = ds['hist'][keep], ds['fold'][keep]
    p0 = ens.predict(h, ds['crop'][keep], f).mean(0)
    pr = ens.predict(h, np.stack(real), f).mean(0)
    pc = ens.predict(h, np.stack(ctrl), f).mean(0)
    base_cross = np.array([crosses(p0[k], H8B_DIST, dirs[k], WALL_HALF_LEN) for k in range(len(keep))])
    edit_cross = np.array([crosses(pr[k], H8B_DIST, dirs[k], WALL_HALF_LEN) for k in range(len(keep))])
    n_base = int(base_cross.sum())
    if n_base < 10:
        lines.append(f'H8b: only {n_base} samples whose normal prediction crosses the wall line: not evaluated')
        return
    share = float((~edit_cross[base_cross]).mean())
    d_real = np.linalg.norm(pr - p0, axis=2).mean()
    d_ctrl = np.linalg.norm(pc - p0, axis=2).mean()
    ok = share >= H8B_SHARE and d_real >= H8B_RATIO * max(d_ctrl, 1e-6)
    lines.append(f'H8b put a wall {H8B_DIST} m ahead (A and E walking; n={len(keep)}, {n_base} whose normal prediction crosses it):')
    lines.append(f'    no longer cross the wall: {share * 100:.0f} %')
    lines.append(f'    mean change of the prediction: wall in the way {d_real:.3f} m   control wall {d_ctrl:.3f} m')
    lines.append(f'    rule: >= {H8B_SHARE * 100:.0f} % and change >= {H8B_RATIO:.0f} x control  ->  {"PASS" if ok else "FAIL"}   (exploratory: outside the training scenes)')


# ------------------------------------------------------------------------------------------------ main
def run(ds, ens, hist_ens, what, outdir, runs_root, max_n=600, lines=None):
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [] if lines is None else lines
    masks = group_masks(ds)
    for g, m in masks.items():
        lines.append(f'{g}: {int(m.sum())} samples')
    boot = Boot(ds)
    idx = np.nonzero(np.any(list(masks.values()), axis=0))[0]
    # clean predictions + self-check against the stored out-of-fold files
    pred = ens.predict(ds['hist'][idx], ds['crop'][idx], ds['fold'][idx])
    v_full = ade3(pred, ds, idx)
    oof = load_oof(runs_root, ens.variant)
    if oof:
        _, arrs = oof_ade3(ds, [f for f in oof if int(f.stem.split('_s')[-1]) in ens.seeds])
        if arrs:
            diff = max(np.nanmax(np.abs(a[idx] - pred[k])) for k, a in enumerate(arrs[:len(pred)]))
            lines.append(f'self-check: largest difference to the stored out-of-fold predictions: {diff:.2e} m '
                         f'({"ok" if diff < 1e-2 else "TOO BIG - wrong models or data"})')
    ref_vals = []
    cv = cvm(ds['hist'], 4)
    d = np.linalg.norm(cv - ds['fut'][:, :, :2], axis=2)[:, :H3].mean(1)
    ref_vals.append(('cvm4 (reference)', np.where(np.isin(np.arange(len(d)), idx), d, np.nan)))
    v_hist = None
    hf = load_oof(runs_root, 'hist')
    if hf:
        dh, _ = oof_ade3(ds, hf)
        v_hist = np.where(np.isin(np.arange(len(dh)), idx), dh, np.nan)
        ref_vals.append(('hist, no map (reference)', v_hist))
    if 'pics' in what:
        pictures(ds, ens, hist_ens, v_full, v_hist, outdir)
    if 'damage' in what:
        damage_table(ds, ens, boot, masks, idx, lines, ref_vals, outdir, hist_ens=hist_ens)
    if 'h8' in what:
        h8(ds, ens, masks, lines, max_n)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=None, help='packed .npz (default packed/mvp_v1_packed.npz)')
    ap.add_argument('--models', default='runs', help='folder searched (recursively) for <variant>_s*_f*.pt')
    ap.add_argument('--variant', default='full')
    ap.add_argument('--seeds', default=None, help='e.g. 0 or 0,1,2 (default: all found)')
    ap.add_argument('--what', default='pics,damage,h8')
    ap.add_argument('--out', default='reports/inspect')
    ap.add_argument('--max-n', type=int, default=600, help='samples per counterfactual test')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    data = args.data or ('packed/mvp_v1_packed.npz' if (here / 'packed' / 'mvp_v1_packed.npz').exists()
                         else '../mvp_collection/datasets/mvp_v1')
    ds = load_dataset(data)
    seeds = None if args.seeds is None else {int(s) for s in args.seeds.split(',')}
    ens = Ensemble(args.models, args.variant, seeds, args.device)
    hist_ens = None
    if 'pics' in args.what or 'damage' in args.what:
        try:
            hist_ens = Ensemble(args.models, args.variant.replace('full', 'hist') if args.variant.endswith(('_dmg', '_hn')) else 'hist', seeds, args.device)
        except SystemExit:
            print('[models] no hist models found: pictures without the hist line')
    lines = run(ds, ens, hist_ens, set(args.what.split(',')), Path(args.out), args.models, args.max_n)
    text = '\n'.join(lines)
    print(text)
    (Path(args.out) / 'inspect_report.txt').write_text(text + '\n')
    print(f'\n[done] {args.out}/inspect_report.txt')


if __name__ == '__main__':
    main()