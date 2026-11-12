#!/usr/bin/env python3
"""
make_video.py -- M6: presentation video. One simulated walk shown on the WHOLE environment map (walls, furniture), in real time:
the pedestrian, the robot with its camera cone, and at every moment what each method expects the pedestrian to do next 6 s.
    python3 make_video.py                          # one video per scenario B, C, D, E (auto-chosen) in reports/video/
    python3 make_video.py --episodes env_B_001_ep3 --data ../mvp_collection/datasets/mvp_v1
Needs: the dataset FOLDER (for the walls, environment.json) and the packed file (fast) or the folder; the trained models in runs/.
Output: .mp4 if ffmpeg is installed (conda: `conda install ffmpeg` is NOT needed if `ffmpeg -version` works), else .gif.
Replay of held-out data: every moment is predicted by the fold model that never saw that environment.
Predictions refresh every 0.5 s (the sample rate) and start where the person was when the prediction was made.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo import ade3, episode_frames, pick_episodes  # noqa: E402
from motion.baselines import cvm  # noqa: E402
from motion.data import _scan, load_dataset  # noqa: E402
from motion.wedge import FOV, RANGE  # noqa: E402

COL = {'truth': '#1b9e77', 'cvm': '#d62728', 'nomap': '#ff7f0e', 'map': '#1f4fd8'}
LAB = {'truth': 'Where the person really walks next',
       'cvm': 'Constant-velocity guess (keeps going straight)',
       'nomap': 'Our model WITHOUT the map',
       'map': 'Our model WITH the map'}
STORY = {'A': 'An obstacle blocks the path: will the person walk around it?',
         'B': 'The person walks toward a corner. The robot sees the walls and can guess where the person will turn.',
         'C': 'The person walks toward a place where they will stop. Can the model see it coming from the map?',
         'D': 'A person walks past other objects in the scene.',
         'E': 'Control: nothing special happens, the person just walks.'}
EVENT_NAME = {'corner': 'corner', 'obstacle': 'obstacle', 'stop': 'stop point'}


def load_world(root, ds, idx):
    """World (odom) poses of pedestrian anchor and robot for each sample, walls and the episode's event."""
    root = Path(root)
    runs = {(sc, ed.name): (ed, run) for sc, ed, run in _scan(root)}
    ped, rob = [], []
    for i in idx:
        sc, env, fname = str(ds['ids'][i]).split('/')
        ed, run = runs[(sc, env)]
        d = np.load(run / 'samples' / fname, allow_pickle=True)
        ped.append(np.asarray(d['ped_anchor_odom'], float)[:3])
        rob.append(np.asarray(d['robot_pose_odom'], float)[:3])
    sc, env, _ = str(ds['ids'][idx[0]]).split('/')
    ed = runs[(sc, env)][0]
    e = json.loads((ed / 'environment.json').read_text())
    return np.array(ped), np.array(rob), e


def to_world(anchor, pts):
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    return np.stack([anchor[0] + c * pts[:, 0] - s * pts[:, 1], anchor[1] + s * pts[:, 0] + c * pts[:, 1]], 1)


def box_poly(o):
    c, s = np.cos(o.get('yaw', 0.0)), np.sin(o.get('yaw', 0.0))
    hx, hy = o['size_x'] / 2, o['size_y'] / 2
    pts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return np.stack([o['x'] + c * pts[:, 0] - s * pts[:, 1], o['y'] + s * pts[:, 0] + c * pts[:, 1]], 1)


def render_video(ds, root, ep, idx, ens_map, ens_nomap, path, fps=10, view=(20.0, 11.0)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon, Wedge
    ped, rob, env = load_world(root, ds, idx)
    h, c, f = ds['hist'][idx], ds['crop'][idx], ds['fold'][idx]
    p_map = ens_map.predict(h, c, f).mean(0)
    p_no = ens_nomap.predict(h, c, f).mean(0)
    p_cv = cvm(h, 4)
    err = {k: ade3(p, ds['fut'][idx]) for k, p in (('cvm', p_cv), ('nomap', p_no), ('map', p_map))}
    t = ds['t_rel'][idx]
    t = t if np.isfinite(t).all() else np.arange(len(idx)) * 0.5
    world = {k: np.stack([to_world(ped[j], p[j]) for j in range(len(idx))]) for k, p in
             (('cvm', p_cv), ('nomap', p_no), ('map', p_map))}
    world['truth'] = np.stack([to_world(ped[j], ds['fut'][idx[j], :, :2]) for j in range(len(idx))])
    epinfo = next((e for e in env.get('episodes', []) if e.get('episode_id') == ep), {})
    ev = epinfo.get('event') or {}
    polys = [(o.get('role', 'wall'), box_poly(o)) for o in env.get('obstacles', []) if 'size_x' in o]
    scen = str(ds['scen'][idx[0]])[0]
    cmap = ListedColormap(['#c8c8c8', '#ffffff', '#202020'])
    tt = np.arange(t[0], t[-1], 1.0 / fps)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.5, 1], height_ratios=[1.2, 1.1, 1.0], left=0.03, right=0.98, top=0.83,
                          bottom=0.05, wspace=0.22, hspace=0.35)
    axm = fig.add_subplot(gs[:, 0])
    axc = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1])
    axt = fig.add_subplot(gs[2, 1])

    def interp(a, tq):
        return np.array([np.interp(tq, t, a[:, k]) for k in range(a.shape[1])])

    def draw(fr):
        tq = tt[fr]
        k = int(np.searchsorted(t, tq, side='right') - 1)
        k = max(0, min(k, len(idx) - 1))
        pos = interp(ped[:, :2], tq)
        rpos = interp(rob[:, :2], tq)
        ryaw = float(np.interp(tq, t, np.unwrap(rob[:, 2])))
        trail = np.stack([np.interp(np.r_[t[t <= tq], tq], t, ped[:, 0]), np.interp(np.r_[t[t <= tq], tq], t, ped[:, 1])], 1)
        axm.clear()
        axm.set_facecolor('#f4f1ea')
        for role, poly in polys:
            axm.add_patch(Polygon(poly, closed=True, fc='#3a3a3a' if role == 'wall' else '#9c7a4f', ec='none', zorder=2))
        axm.add_patch(Wedge(tuple(rpos), RANGE, np.degrees(ryaw - FOV / 2), np.degrees(ryaw + FOV / 2), fc='#1f78b4',
                            alpha=0.10, ec='none', zorder=1))
        axm.plot(trail[:, 0], trail[:, 1], '-', color='#777777', lw=2, zorder=3)
        if ev.get('position') is not None:
            axm.plot(*ev['position'], marker='*', ms=16, color='#e6b800', mec='k', zorder=4)
            axm.annotate(EVENT_NAME.get(ev.get('type'), 'event'), ev['position'], xytext=(6, 10), textcoords='offset points',
                         fontsize=9, weight='bold', zorder=6)
        for key, lw, ls in (('truth', 4.5, '-'), ('cvm', 2.5, '--'), ('nomap', 2.5, '-'), ('map', 3.0, '-')):
            w = np.r_[world[key][k][:1] * 0 + ped[k, :2], world[key][k]]
            axm.plot(w[:, 0], w[:, 1], ls, color=COL[key], lw=lw, alpha=0.9 if key != 'truth' else 0.55, zorder=5,
                     solid_capstyle='round')
            axm.plot(*w[-1], 'o', color=COL[key], ms=6, zorder=5)
        axm.plot(*pos, 'o', color='k', ms=11, mfc='white', mew=2.5, zorder=7)
        axm.plot(*rpos, '^', color='#1f78b4', ms=13, mec='k', zorder=7)
        axm.text(rpos[0], rpos[1] - 0.55, 'robot', ha='center', va='top', fontsize=9, color='#1f78b4', weight='bold', zorder=7)
        axm.text(pos[0], pos[1] + 0.5, 'person', ha='center', fontsize=9, weight='bold', zorder=7)
        w, hh_ = view
        axm.set_xlim(pos[0] - w / 2, pos[0] + w / 2)
        axm.set_ylim(pos[1] - hh_ / 2, pos[1] + hh_ / 2)
        axm.set_aspect('equal')
        axm.set_xticks([])
        axm.set_yticks([])
        for lab in LAB:
            axm.plot([], [], '--' if lab == 'cvm' else '-', color=COL[lab], lw=3, label=LAB[lab])
        axm.legend(loc='upper left', fontsize=9, framealpha=0.92)
        axm.text(0.99, 0.015, f'time {tq - t[0]:4.1f} s   (predictions refresh every 0.5 s)', transform=axm.transAxes, ha='right',
                 fontsize=9, color='#444444')
        # what the model sees
        axc.clear()
        arr = np.where(ds['crop'][idx[k]] == -1, 0, np.where(ds['crop'][idx[k]] == 0, 1, 2))
        axc.imshow(arr, cmap=cmap, vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4), interpolation='nearest')
        axc.plot(*[0], *[0], 'o', color='k', ms=6, mfc='white')
        for key in ('truth', 'cvm', 'map'):
            q = np.r_[[[0, 0]], ds['fut'][idx[k], :, :2] if key == 'truth' else (p_cv if key == 'cvm' else p_map)[k]]
            axc.plot(q[:, 0], q[:, 1], '--' if key == 'cvm' else '-', color=COL[key], lw=1.8)
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_title('What the model sees: the map around the person\n(black = wall, grey = not seen)', fontsize=9)
        # bars
        axb.clear()
        keys = ('cvm', 'nomap', 'map')
        vals = [err[q][k] for q in keys]
        axb.barh([0, 1, 2], vals[::-1], color=[COL[q] for q in keys[::-1]], height=0.6)
        axb.set_yticks([0, 1, 2])
        axb.set_yticklabels(['With map', 'Without map', 'Constant velocity'], fontsize=9)
        axb.set_xlim(0, max(1.0, max(float(e.max()) for e in err.values()) * 1.1))
        for yv, v in zip([0, 1, 2], vals[::-1]):
            axb.text(v + 0.01, yv, f'{v:.2f} m', va='center', fontsize=9)
        axb.set_title('Error of the latest prediction (next 3 s; shorter = better)', fontsize=9)
        # running mean
        axt.clear()
        axt.axis('off')
        msg = [f'Average error so far (next 3 s):']
        for q in keys:
            msg.append(f'  {LAB[q].split(" (")[0]:30s} {err[q][:k + 1].mean():.2f} m')
        axt.text(0, 1, '\n'.join(msg), va='top', fontsize=10, family='monospace')
        if err['cvm'][k] > 2 * max(err['map'][k], 0.05) and err['cvm'][k] > 0.3:
            axt.text(0, 0.28, 'Constant velocity misses here.\nThe map model does not.', fontsize=11, color=COL['cvm'], weight='bold')
        return []

    import textwrap
    fig.text(0.03, 0.965, textwrap.fill(STORY.get(scen, ''), 105), fontsize=13, weight='bold', va='top')
    fig.text(0.03, 0.885, f'episode {ep}   |   replay of simulated data the model never trained on', fontsize=9, color='#555555', va='top')
    path = Path(path)
    if path.suffix == '.mp4' and animation.writers.is_available('ffmpeg'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=3500)
    else:
        path = path.with_suffix('.gif')
        writer = animation.PillowWriter(fps=min(fps, 10))
    with writer.saving(fig, str(path), 100):
        for fr in range(len(tt)):
            draw(fr)
            writer.grab_frame()
    plt.close(fig)
    print(f'[video] {path}  ({len(tt)} frames; mean ADE@3s: constant velocity {err["cvm"].mean():.2f}, '
          f'no-map {err["nomap"].mean():.2f}, map {err["map"].mean():.2f} m)')
    return {k: float(v.mean()) for k, v in err.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='../mvp_collection/datasets/mvp_v1', help='dataset FOLDER (walls are read from it)')
    ap.add_argument('--packed', default='packed/mvp_v1_packed.npz')
    ap.add_argument('--models', default='runs')
    ap.add_argument('--map-variant', default='full_hn')
    ap.add_argument('--nomap-variant', default='hist_hn')
    ap.add_argument('--episodes', default=None)
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--out', default='reports/video')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--ext', default='mp4' if shutil.which('ffmpeg') else 'gif')
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    ds = load_dataset(here / args.packed if (here / args.packed).exists() else args.data)
    from inspect_model import Ensemble
    seeds = None if args.seeds is None else {int(s) for s in args.seeds.split(',')}
    ens_map = Ensemble(args.models, args.map_variant, seeds)
    ens_no = Ensemble(args.models, args.nomap_variant, seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ep in (args.episodes.split(',') if args.episodes else pick_episodes(ds)):
        render_video(ds, args.data, ep, episode_frames(ds, ep), ens_map, ens_no, out / f'video_{ep}.{args.ext}', args.fps)


if __name__ == '__main__':
    main()