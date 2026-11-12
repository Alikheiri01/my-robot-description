#!/usr/bin/env python3
"""
make_demo.py -- M6: replay demo. GIFs of real simulated episodes, frame by frame, with the prediction of constant velocity, of
the no-map network and of the map network. Laptop, CPU, no training. Run from mvp_model/ :

    python3 make_demo.py                       # one episode per scenario B, C, D and E (auto-chosen), GIFs in reports/demo/
    python3 make_demo.py --list B_corner       # list episodes of a scenario (with number of usable frames)
    python3 make_demo.py --episodes env_B_003_ep1,env_C_010_ep0

Every frame is a sample (a moment of the episode). Each sample is predicted by the fold model that NEVER saw its environment
(the same rule as the evaluation), so this is a replay of held-out data, not of training data.
Left panel: the pedestrian-centred view the model gets (grey = unknown, white = free, black = occupied; the pedestrian is at
(0,0) looking right), history (grey dots), truth (green), constant velocity (purple dashed), no-map network (orange), map network
(pink), robot (blue triangle). Right panel: ADE@3s of the three predictors over the episode so far.
Needs the trained models (default: the noise-trained pair full_hn / hist_hn found under runs/).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion.baselines import cvm  # noqa: E402
from motion.data import load_dataset  # noqa: E402

COL = {'truth': '#1b9e77', 'cvm': '#7570b3', 'nomap': '#d95f02', 'map': '#e7298a'}
H3 = 6


def ade3(pred, fut):
    return np.linalg.norm(pred - fut[:, :, :2], axis=2)[:, :H3].mean(1)


def pick_episodes(ds, scenarios=('B', 'C', 'D', 'E'), per=1):
    """The episode with the most usable frames that contain a near event (or the most usable frames for E)."""
    first = np.array([s[0] for s in ds['scen']])
    out = []
    for S in scenarios:
        m = ds['usable'] & (first == S)
        score = {}
        for i in np.nonzero(m)[0]:
            near = bool(ds['ev_future'][i] and np.nan_to_num(ds['t_to_event'][i], nan=99) <= 4.0)
            sc = score.setdefault(ds['ep'][i], [0, 0])
            sc[0] += 1
            sc[1] += int(near)
        ranked = sorted(score, key=lambda e: (score[e][1], score[e][0]), reverse=True)
        out += ranked[:per]
    return out


def episode_frames(ds, ep):
    idx = np.nonzero(ds['usable'] & (ds['ep'] == ep))[0]
    if len(idx) == 0:
        raise SystemExit(f'episode {ep} has no usable samples (use --list)')
    t = ds['t_rel'][idx]
    order = np.lexsort((ds['ids'][idx], np.where(np.isfinite(t), t, 1e9)))
    return idx[order]


def render_gif(ds, ep, idx, ens_map, ens_nomap, path, fps=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#c8c8c8', '#ffffff', '#202020'])
    h, c, f = ds['hist'][idx], ds['crop'][idx], ds['fold'][idx]
    p_map = ens_map.predict(h, c, f).mean(0)
    p_no = ens_nomap.predict(h, c, f).mean(0)
    p_cv = cvm(h, 4)
    err = {k: ade3(p, ds['fut'][idx]) for k, p in (('cvm', p_cv), ('nomap', p_no), ('map', p_map))}
    t = ds['t_rel'][idx]
    tx = t if np.isfinite(t).all() else np.arange(len(idx)) * 0.5
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={'width_ratios': [1.15, 1]})
    ymax = max(0.5, float(np.nanmax([e.max() for e in err.values()])) * 1.1)

    def draw(k):
        i = idx[k]
        ax.clear()
        ax2.clear()
        arr = np.where(ds['crop'][i] == -1, 0, np.where(ds['crop'][i] == 0, 1, 2))
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4), interpolation='nearest')
        hh, ff = ds['hist'][i], ds['fut'][i, :, :2]
        ax.plot(hh[:, 0], hh[:, 1], 'o-', color='#555555', ms=3, lw=1, label='history')
        ax.plot(np.r_[0, ff[:, 0]], np.r_[0, ff[:, 1]], '-', color=COL['truth'], lw=2.5, label='truth (6 s)')
        for key, p, lab, ls in (('cvm', p_cv[k], 'constant velocity', '--'), ('nomap', p_no[k], 'no-map network', '-'),
                                ('map', p_map[k], 'map network', '-')):
            ax.plot(np.r_[0, p[:, 0]], np.r_[0, p[:, 1]], ls, color=COL[key], lw=2, label=lab)
        ax.plot(*ds['robot'][i], '^', color='#1f78b4', ms=8, label='robot')
        ax.set_xlim(-2, 6)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        tte = ds['t_to_event'][i]
        ax.set_title(f'{ep}   t = {tx[k]:.1f} s' + (f'   event in {tte:.1f} s' if np.isfinite(tte) else '   no event ahead'), fontsize=10)
        ax.legend(fontsize=7, loc='lower right')
        for key, lab in (('cvm', 'constant velocity'), ('nomap', 'no-map network'), ('map', 'map network')):
            ax2.plot(tx[:k + 1], err[key][:k + 1], '-o', ms=3, color=COL[key], label=lab)
        ax2.axvline(tx[k], color='#999999', lw=1)
        ax2.set_xlim(tx[0], tx[-1] if tx[-1] > tx[0] else tx[0] + 1)
        ax2.set_ylim(0, ymax)
        ax2.set_xlabel('time since the walk started [s]')
        ax2.set_ylabel('ADE@3s of the prediction made at that moment [m]')
        ax2.set_title('prediction error so far (lower is better)', fontsize=10)
        ax2.legend(fontsize=8, loc='upper left')

    anim = FuncAnimation(fig, draw, frames=len(idx), interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[demo] {path}  ({len(idx)} frames; mean ADE@3s: constant velocity {err["cvm"].mean():.2f}, '
          f'no-map {err["nomap"].mean():.2f}, map {err["map"].mean():.2f} m)')
    return {k: float(v.mean()) for k, v in err.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=None)
    ap.add_argument('--models', default='runs')
    ap.add_argument('--map-variant', default='full_hn')
    ap.add_argument('--nomap-variant', default='hist_hn')
    ap.add_argument('--episodes', default=None, help='comma list of episode ids (default: one per scenario B, C, D, E)')
    ap.add_argument('--list', default=None, help='scenario name (e.g. B_corner): list its episodes and exit')
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--out', default='reports/demo')
    ap.add_argument('--fps', type=int, default=3)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    data = args.data or ('packed/mvp_v1_packed.npz' if (here / 'packed' / 'mvp_v1_packed.npz').exists()
                         else '../mvp_collection/datasets/mvp_v1')
    ds = load_dataset(data)
    if args.list:
        for ep in sorted({e for e, s in zip(ds['ep'], ds['scen']) if s == args.list}):
            m = ds['usable'] & (ds['ep'] == ep)
            near = (m & ds['ev_future'] & (np.nan_to_num(ds['t_to_event'], nan=99) <= 4.0)).sum()
            print(f'{ep:24s} usable frames {int(m.sum()):3d}   near-event frames {int(near):3d}')
        return
    from inspect_model import Ensemble
    seeds = None if args.seeds is None else {int(s) for s in args.seeds.split(',')}
    ens_map = Ensemble(args.models, args.map_variant, seeds)
    ens_no = Ensemble(args.models, args.nomap_variant, seeds)
    eps = args.episodes.split(',') if args.episodes else pick_episodes(ds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in eps:
        render_gif(ds, ep, episode_frames(ds, ep), ens_map, ens_no, out / f'demo_{ep}.gif', args.fps)


if __name__ == '__main__':
    main()