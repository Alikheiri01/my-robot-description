#!/usr/bin/env python3
"""
figures/make_figures.py -- ALL presentation figures in this one file. Run from mvp_model/ :

    python3 figures/make_figures.py                    # the 6 figures that need only numbers (no data, no models)
    python3 figures/make_figures.py --only results robustness      # just some of them
    python3 figures/make_figures.py --only tokens      # needs packed/mvp_v1_packed.npz (numpy only)
    python3 figures/make_figures.py --only attention   # needs torch + the trained models in runs/
    python3 figures/make_figures.py --all              # everything (tokens and attention need their inputs)

Pictures go to reports/figures/ (the folder git ignores). Names: pipeline, architecture, results, robustness, limits, checks,
tokens, attention. Numbers come from the NUMBERS block below (copied from docs/results.md): change them there if a result changes.
Videos are made by make_demo.py and make_video.py (parent folder).
"""
import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
OUT = HERE / 'reports' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = '#1f2937', '#6b7280', '#e5e7eb'
COL = {'cvm': '#d62728', 'nomap': '#ff7f0e', 'map': '#1f4fd8', 'truth': '#1b9e77', 'hist': '#dbeafe', 'scene': '#dcfce7',
       'fuse': '#fef3c7', 'out': '#fee2e2', 'grey': '#9ca3af'}
LABEL = {'cvm': 'Constant velocity', 'nomap': 'Model without map', 'map': 'Model with map'}
plt.rcParams.update({'font.size': 13, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 100, 'savefig.dpi': 200, 'axes.titleweight': 'bold'})


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[figure]', p)
    return p


# ======================================================================================================= NUMBERS
# (copied from docs/results.md; ADE@3s on near events [m] unless a figure says otherwise)
# ---- headline (clean-trained family): cvm4 / hist / full, ADE@3s near events, 95 environments, 3 seeds x 5 folds
HEAD = {'B (corner)': {'cvm': 0.74, 'nomap': 0.51, 'map': 0.19},
        'C (stop)': {'cvm': 0.85, 'nomap': 0.45, 'map': 0.19}}
ALL_SAMPLES = {'cvm': 0.30, 'nomap': 0.17, 'map': 0.11}
# ---- final model full_hn under tracker noise, near events (B, C) and walking controls (A, E)
NOISE_X = [0.0, 0.05, 0.1, 0.2]
FINAL = {'B': [0.20, 0.22, 0.27, 0.39], 'C': [0.19, 0.21, 0.25, 0.36], 'A': [0.11, 0.14, 0.18, 0.32], 'E': [0.05, 0.08, 0.13, 0.27]}
CVM_N = {'B': {0.0: 0.74, 0.1: 0.84, 0.2: 1.00}, 'C': {0.0: 0.85, 0.1: 0.92, 0.2: 1.06},
         'A': {0.1: 0.33, 0.2: 0.56}, 'E': {0.1: 0.30, 0.2: 0.54}}
NOMAP_N = {'B': {0.1: 0.59, 0.2: 0.71}, 'C': {0.1: 0.78, 0.2: 0.89}, 'A': {0.1: 0.22, 0.2: 0.39}, 'E': {0.1: 0.15, 0.2: 0.34}}
# ---- final model with damaged map (clean tracker)
DAMAGE = [('clean', 0.20, 0.19), ('1 % wrong cells', None, None), ('5 % wrong cells', 0.32, 0.24), ('map shifted 0.2 m', 0.38, 0.22),
          ('map shifted 0.5 m', 0.46, 0.27), ('map rotated 5 deg', 0.29, 0.20), ('map rotated 10 deg', 0.40, 0.22),
          ('no map at all', 0.66, 0.97)]
DAMAGE = [d for d in DAMAGE if d[1] is not None]
REF = {'B': {'cvm': 0.74, 'nomap': 0.51}, 'C': {'cvm': 0.85, 'nomap': 0.45}}
# ---- transfer (held-out scenario type), full_hn vs hist_hn vs constant velocity, and the same when the type is in training
TRANSFER = {'B (corner)': dict(cvm=0.74, nomap=0.74, held=0.66, trained=0.20),
            'C (stop)': dict(cvm=0.85, nomap=0.95, held=0.78, trained=0.19),
            'D (doorway)': dict(cvm=0.37, nomap=0.54, held=0.44, trained=0.25)}
# ---- gain by time to the event, ADE@6s, B/C/D pooled (clean-trained family)
TTE = {'0-2 s': (1.47, 0.42), '2-4 s': (1.24, 0.63), '4-6 s': (0.40, 0.34)}
# ---- hypotheses (rules fixed before the runs)
HYP = [('H1  map helps at corners (B)', 'PASS', '62 % lower error'), ('H1  map helps at stops (C)', 'PASS', '59 % lower error'),
       ('H2  no harm on controls', 'PASS', 'better, not worse'), ('H3  gain where the cause is visible', 'PASS', '0.32 m vs 0.09 m'),
       ('H4  reads THIS scene, not just the type', 'PASS', 'wrong map: 0.19 -> 0.61 m'),
       ('H8a obstacle removed -> prediction reacts', 'PASS', '86 % react (control 29-36 %)'),
       ('H8b wall inserted -> prediction reacts', 'FAIL', 'learned scene types, not "walls stop people"'),
       ('H11 not a robot-placement leak', 'PASS', 'camera-cone only: 3.6 % (< 5 %)'),
       ('H10 transfer to unseen scenario type', 'PARTIAL', 'keeps 11 % of 62 % (B)')]


# ---------------------------------------------------------------------------------------- fig_pipeline
EDGE = {'hist': '#2563eb', 'scene': '#16a34a', 'fuse': '#d97706', 'out': '#dc2626', 'grey': '#6b7280'}


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 52)
    ax.axis('off')
    steps = [('1  Simulation', 'Gazebo + pedestrian\nsimulator (HuNavSim)\n95 scenes, 5 situation types', 'grey'),
             ('2  What the robot sees', 'where the person was\n(last 3.5 s) and an\noccupancy map around him', 'hist'),
             ('3  Small transformer', '282k parameters\nreads the motion and\nlooks at the map', 'scene'),
             ('4  Prediction', 'where the person will be\nfor the next 6 s\n(one path)', 'fuse'),
             ('5  Later: robot', 'planning / MPC uses the\npath to avoid the person\n(not in the MVP)', 'out')]
    w, h, gap = 25, 24, 6.2
    for i, (t, d, c) in enumerate(steps):
        x = 2 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 14), w, h, boxstyle='round,pad=0.4,rounding_size=1.5', fc=COL[c] if c in COL else '#f3f4f6',
                                    ec=EDGE[c], lw=2, ls='--' if i == 4 else '-'))
        ax.text(x + w / 2, 14 + h - 4, t, ha='center', va='center', fontsize=13, weight='bold')
        ax.text(x + w / 2, 14 + h / 2 - 3, d, ha='center', va='center', fontsize=11, color=INK, linespacing=1.4)
        if i < 4:
            ax.add_patch(FancyArrowPatch((x + w + 0.8, 26), (x + w + gap - 0.8, 26), arrowstyle='-|>', mutation_scale=18, lw=2, color=INK))
    ax.text(75, 6, 'We compare the same predictor with and without the map, and against "keep walking straight".',
            ha='center', fontsize=12, color=MUTED)
    save(fig, 'pipeline.png')


# ---------------------------------------------------------------------------------------- fig_architecture_detail
EDGE = {'hist': '#2563eb', 'scene': '#16a34a', 'fuse': '#d97706', 'out': '#dc2626', 'grey': '#6b7280'}


def fig_architecture_detail():
    fig, ax = plt.subplots(figsize=(16, 9.6))
    ax.set_xlim(0, 160)
    ax.set_ylim(-12, 96)
    ax.axis('off')

    def box(x, y, w, h, title, sub, c, size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1', fc=COL[c], ec=EDGE[c], lw=1.8))
        ax.text(x + w / 2, y + h - 2.4, title, ha='center', va='center', fontsize=size, weight='bold')
        ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha='center', va='center', fontsize=size - 2, color=INK, linespacing=1.35)

    def arr(x1, y1, x2, y2, c=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14, lw=1.8, color=c))

    ax.text(2, 93, 'The transformer, layer by layer  (d = 64 features per token, 4 attention heads, 282,584 parameters)', fontsize=15, weight='bold')
    import textwrap
    ax.text(2, -2.5, textwrap.fill('Tensor shapes are per sample; parameter counts in brackets. One attention layer = attention + a small 64-128-64 network, '
            'each with a residual connection and normalisation (pre-norm, GELU, dropout 0.1).', 62), fontsize=9.5, color=MUTED, va='top')
    # history path
    ax.text(2, 87, 'HISTORY PATH', color=EDGE['hist'], fontsize=12, weight='bold')
    box(2, 70, 42, 14, '8 past poses', 'x, y, speed x/y, cos/sin heading\n(8 x 6 numbers)', 'hist')
    box(2, 50, 42, 16, 'Linear 6->64 + position  (960)', 'every pose becomes a token of 64 numbers;\na learned vector marks "how long ago"\n(8 x 64)', 'hist')
    box(2, 24, 42, 22, 'History encoder x 2  (66,944)', 'self-attention: the 8 tokens look at each\nother (8 x 8 weights per head):\n"am I speeding up, slowing, turning?"\n(8 x 64)', 'hist')
    arr(23, 70, 23, 66.5, EDGE['hist']); arr(23, 50, 23, 46.5, EDGE['hist'])
    # scene path
    ax.text(116, 87, 'SCENE PATH', color=EDGE['scene'], fontsize=12, weight='bold')
    box(116, 70, 42, 14, 'Map crop 80 x 80 cells', 'unknown / free / occupied as 3 channels\n(3 x 80 x 80), 8 m x 8 m', 'scene')
    box(116, 52, 42, 14, 'Conv stem  (56,320)', '3 convolutions, each halves the size\n80 -> 40 -> 20 -> 10:  10 x 10 x 64\n= 100 scene tokens, 0.8 m each', 'scene')
    box(116, 38, 42, 11, '+ token position  (4,352)', 'a small network turns each token\'s (x, y)\nin metres into 64 numbers', 'scene')
    box(116, 14, 42, 21, 'Scene encoder x 1  (33,472)', 'self-attention among 100 tokens\n(100 x 100 weights per head):\n"which patches belong to the same wall,\nopening or obstacle?"   (100 x 64)', 'scene')
    arr(137, 70, 137, 66.5, EDGE['scene']); arr(137, 52, 137, 49.5, EDGE['scene']); arr(137, 38, 137, 35.5, EDGE['scene'])
    # cross
    box(55, 36, 50, 38, 'Cross-attention decoder x 2  (100,480)', 'each layer, in this order:\n1. history tokens look at each other  (8 x 8)\n2. CROSS-ATTENTION: each of the 8 history\n    tokens asks the 100 scene tokens\n    (8 x 100 weights per head):\n    "is there a wall, corner or obstacle\n    that explains my motion?"\n3. small network 64-128-64\n(output 8 x 64)', 'fuse', size=11)
    arr(44, 35, 56, 48, EDGE['hist']); ax.text(41, 41, 'queries', color=EDGE['hist'], fontsize=10, rotation=60)
    arr(116, 28, 106, 48, EDGE['scene']); ax.text(110.5, 47, 'keys /\nvalues', color=EDGE['scene'], fontsize=9, ha='center')
    # head
    box(55, 8, 50, 22, 'Pool + MLP head  (19,864)', 'last token + average of the 8 tokens (128 numbers)\n-> LayerNorm -> 128 -> 128 -> 24\n= 12 corrections (dx, dy)', 'out')
    arr(80, 36, 80, 30.5, EDGE['fuse'])
    box(2, 2, 42, 14, 'Constant-velocity guess', 'last 4 poses -> speed -> 12 points\n(no parameters, nothing to learn)', 'out')
    box(55, -10, 50, 11, 'Prediction = guess + correction', '12 x (x, y), every 0.5 s up to 6 s', 'out')
    arr(80, 8, 80, 1.5, EDGE['out'])
    arr(44, 7, 55, -2, EDGE['out'])
    ax.text(47, 6.5, '+', fontsize=24, ha='center', va='center', weight='bold')
    ax.text(116, 9, '(+ 192 unused parameters: robot input,\noff in the final model)', fontsize=8.5, color=MUTED, va='top')
    save(fig, 'architecture.png')


# ---------------------------------------------------------------------------------------- fig_results
def fig_results():
    fig, ax = plt.subplots(figsize=(11, 6))
    groups = list(HEAD)
    x = np.arange(len(groups))
    w = 0.26
    for i, k in enumerate(('cvm', 'nomap', 'map')):
        v = [HEAD[g][k] for g in groups]
        b = ax.bar(x + (i - 1) * w, v, w, color=COL[k], label=LABEL[k])
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.015, f'{val:.2f}', ha='center', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g}\nperson is about to turn' if g.startswith('B') else f'{g}\nperson is about to stop' for g in groups])
    ax.set_ylabel('average error of the next 3 s  [m]   (lower is better)')
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color='#e5e7eb')
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('With the map, the error near corners and stops drops to about a quarter', loc='left')
    fig.text(0.01, -0.03, 'Simulation, 95 environments, held-out environments only (5-fold, 3 seeds). Walking-straight control: no harm '
             '(error 0.11 vs 0.30 constant velocity over all samples).', fontsize=9, color=MUTED)
    save(fig, 'results.png')


# ---------------------------------------------------------------------------------------- fig_robustness
def fig_robustness():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.1]})
    for scen, ls in (('B', '-'), ('C', '--')):
        a.plot(NOISE_X, FINAL[scen], ls, marker='o', color=COL['map'], lw=2.5, label=f'our model, {"corner" if scen == "B" else "stop"}')
        xs = sorted(CVM_N[scen])
        a.plot(xs, [CVM_N[scen][x] for x in xs], ls, marker='s', color=COL['cvm'], lw=2, label=f'constant velocity, {"corner" if scen == "B" else "stop"}')
    a.set_xlabel('noise of the tracked positions  [m]')
    a.set_ylabel('average error of the next 3 s  [m]')
    a.set_xticks(NOISE_X)
    a.set_title('Noisy tracker: still clearly better', loc='left')
    a.legend(frameon=False, fontsize=11)
    a.yaxis.grid(True, color='#e5e7eb')
    a.set_ylim(0, 1.15)
    names = [d[0] for d in DAMAGE][::-1]
    y = np.arange(len(names))
    bb = [d[1] for d in DAMAGE][::-1]
    cc = [d[2] for d in DAMAGE][::-1]
    b.barh(y + 0.19, bb, 0.36, color=COL['map'], label='corners (B)')
    b.barh(y - 0.19, cc, 0.36, color='#7aa2f7', label='stops (C)')
    for yy, v in zip(y + 0.19, bb):
        b.text(v + 0.01, yy, f'{v:.2f}', va='center', fontsize=10)
    for yy, v in zip(y - 0.19, cc):
        b.text(v + 0.01, yy, f'{v:.2f}', va='center', fontsize=10)
    b.axvline(REF['B']['cvm'], color=COL['cvm'], ls=':', lw=2)
    b.text(REF['B']['cvm'] + 0.01, len(names) - 0.45, 'constant velocity (corners)', color=COL['cvm'], fontsize=10)
    b.set_yticks(y)
    b.set_yticklabels(names)
    b.set_xlabel('average error of the next 3 s  [m]')
    b.set_xlim(0, 1.15)
    b.set_title('Damaged map: it needs an accurate map', loc='left')
    b.legend(frameon=False, loc='center right', fontsize=11)
    save(fig, 'robustness.png')


# ---------------------------------------------------------------------------------------- fig_limits
def fig_limits():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(TTE))
    a.bar(x - 0.18, [v[0] for v in TTE.values()], 0.36, color=COL['nomap'], label='without map')
    a.bar(x + 0.18, [v[1] for v in TTE.values()], 0.36, color=COL['map'], label='with map')
    for i, (n, m) in enumerate(TTE.values()):
        a.text(i - 0.18, n + 0.03, f'{n:.2f}', ha='center', fontsize=11)
        a.text(i + 0.18, m + 0.03, f'{m:.2f}', ha='center', fontsize=11)
    a.set_xticks(x)
    a.set_xticklabels(list(TTE))
    a.set_xlabel('time until the turn / stop happens')
    a.set_ylabel('average error of the next 6 s  [m]')
    a.set_title('The map helps when the cause is near', loc='left')
    a.legend(frameon=False)
    a.yaxis.grid(True, color='#e5e7eb')
    a.set_axisbelow(True)
    groups = list(TRANSFER)
    xx = np.arange(len(groups))
    w = 0.2
    series = (('cvm', 'constant velocity', COL['cvm']), ('nomap', 'without map', COL['nomap']),
              ('held', 'with map, type NEVER seen', '#7aa2f7'), ('trained', 'with map, type in training', COL['map']))
    for i, (k, lab, c) in enumerate(series):
        v = [TRANSFER[g][k] for g in groups]
        b.bar(xx + (i - 1.5) * w, v, w, color=c, label=lab)
        for px, val in zip(xx + (i - 1.5) * w, v):
            b.text(px, val + 0.015, f'{val:.2f}', ha='center', fontsize=9)
    b.set_xticks(xx)
    b.set_xticklabels(groups)
    b.set_ylabel('average error of the next 3 s  [m]')
    b.set_title('On an unseen situation type most of the gain is lost', loc='left')
    b.legend(frameon=False, fontsize=10, loc='upper right')
    b.yaxis.grid(True, color='#e5e7eb')
    b.set_axisbelow(True)
    b.set_ylim(0, 1.15)
    save(fig, 'limits.png')


# ---------------------------------------------------------------------------------------- fig_checks
COLS = {'PASS': '#16a34a', 'FAIL': '#dc2626', 'PARTIAL': '#d97706'}


def fig_checks():
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.axis('off')
    n = len(HYP)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, n + 1)
    ax.text(0, n + 0.35, 'Check (rule written BEFORE the run)', weight='bold', fontsize=12)
    ax.text(52, n + 0.35, 'Result', weight='bold', fontsize=12)
    ax.text(62, n + 0.35, 'What it means', weight='bold', fontsize=12)
    for i, (name, verdict, note) in enumerate(HYP):
        y = n - 0.5 - i
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((-1, y - 0.45), 102, 0.9, fc='#f3f4f6', ec='none'))
        ax.text(0, y, name, va='center', fontsize=12)
        ax.text(52, y, verdict, va='center', fontsize=12, weight='bold', color=COLS[verdict])
        ax.text(62, y, note, va='center', fontsize=11, color=INK)
    save(fig, 'checks.png')


# ---------------------------------------------------------------------------------------- tokens
def draw_tokens(crop, hist, fut, title='', name='tokens.png'):
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={'width_ratios': [1.25, 1]})
    cmap = ListedColormap(['#c8c8c8', '#ffffff', '#202020'])
    arr = np.where(crop == -1, 0, np.where(crop == 0, 1, 2))
    a.imshow(arr, cmap=cmap, vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4), interpolation='nearest')
    for k in range(11):
        a.axvline(-2 + 0.8 * k, color='#2563eb', lw=0.8, alpha=0.7)
        a.axhline(-4 + 0.8 * k, color='#2563eb', lw=0.8, alpha=0.7)
    a.plot(hist[:, 0], hist[:, 1], 'o-', color='#f59e0b', ms=5, lw=2, label='where the person was (8 poses)')
    a.plot(np.r_[0, fut[:, 0]], np.r_[0, fut[:, 1]], '-', color=COL['truth'], lw=3, label='where the person goes next')
    a.plot(0, 0, 'o', color='k', mfc='white', ms=10, mew=2)
    a.set_xlim(-2, 6)
    a.set_ylim(-4, 4)
    a.set_aspect('equal')
    a.set_xlabel('metres ahead of the person')
    a.set_ylabel('metres to the left')
    a.set_title('Map crop with the 10 x 10 token grid' + (f'  ({title})' if title else ''), loc='left', fontsize=12)
    a.legend(loc='lower right', fontsize=9)
    # summary of what each token "sees": share of occupied / free / unknown in the patch
    t = crop.reshape(10, 8, 10, 8).transpose(0, 2, 1, 3).reshape(10, 10, 64)
    occ = (t == 100).mean(2)
    b.imshow(occ, origin='lower', extent=(-2, 6, -4, 4), cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    unk = (t == -1).mean(2)
    for i in range(10):
        for j in range(10):
            b.text(-2 + 0.8 * (j + 0.5), -4 + 0.8 * (i + 0.5), f'{int(round(100 * unk[i, j]))}' if unk[i, j] > 0.5 else '',
                   ha='center', va='center', fontsize=7, color='#2563eb')
    b.set_title('100 tokens: darkness = share of occupied cells\nblue number = % unknown (when more than half)', loc='left', fontsize=11)
    b.set_aspect('equal')
    b.set_xlabel('metres ahead of the person')
    save(fig, name)


def fig_tokens(args):
    from motion.data import load_dataset
    ds = load_dataset(HERE / args.data)
    i = args.sample
    if i is None:
        m = np.nonzero(ds['usable'] & ds['ev_future'] & ds['ev_visible'] & (np.char.startswith(ds['scen'], 'B')))[0]
        i = int(m[len(m) // 2])
    draw_tokens(ds['crop'][i], ds['hist'][i], ds['fut'][i, :, :2], str(ds['ids'][i]).split('/')[1])


# ---------------------------------------------------------------------------------------- attention
def plot_attention(samples, name='attention.png'):
    """samples: list of dicts(crop (80,80), hist (8,2), fut (12,2), pred (12,2), w (100,) attention over the scene tokens, title)."""
    n = len(samples)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    cmap = ListedColormap(['#c8c8c8', '#ffffff', '#202020'])
    for ax, s in zip(axes.ravel(), samples):
        arr = np.where(s['crop'] == -1, 0, np.where(s['crop'] == 0, 1, 2))
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4), interpolation='nearest')
        w = np.asarray(s['w']).reshape(10, 10)
        ax.imshow(w, origin='lower', extent=(-2, 6, -4, 4), cmap='autumn_r', alpha=np.clip(w / (w.max() + 1e-9), 0, 1) * 0.75,
                  interpolation='nearest')
        ax.plot(s['hist'][:, 0], s['hist'][:, 1], 'o-', color='#555555', ms=3, lw=1.2)
        ax.plot(np.r_[0, s['fut'][:, 0]], np.r_[0, s['fut'][:, 1]], '-', color=COL['truth'], lw=2.5)
        ax.plot(np.r_[0, s['pred'][:, 0]], np.r_[0, s['pred'][:, 1]], '-', color=COL['map'], lw=2.5)
        ax.plot(0, 0, 'o', color='k', mfc='white', ms=8, mew=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(s['title'], fontsize=10, loc='left')
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.suptitle('Where the model looks (red/orange = attention).  green = truth, blue = prediction, grey dots = history',
                 fontsize=12, weight='bold')
    return save(fig, name)


def capture_attention(model, torch):
    """Make the cross-attention layers of a MotionTransformer return their weights; returns the list that is filled on each forward."""
    store = []
    for layer in model.cross.layers:
        mha = layer.multihead_attn
        orig = mha.forward

        def f(q, k, v, *a, _o=orig, **kw):
            kw['need_weights'] = True
            kw['average_attn_weights'] = True
            out, w = _o(q, k, v, *a, **kw)
            store.append(w.detach())
            return out, w
        mha.forward = f
    return store


def fig_attention(args):
    import torch
    from motion.data import load_dataset
    from motion.models import VARIANTS
    ds = load_dataset(HERE / args.data)
    models = {}
    for k in range(int(ds['fold'].max()) + 1):
        fs = sorted(Path(args.models).rglob(f'{args.variant}_s{args.seed}_f{k}.pt'))
        if not fs:
            sys.exit(f'missing {args.variant}_s{args.seed}_f{k}.pt under {args.models}')
        m = VARIANTS[args.variant]()
        m.load_state_dict(torch.load(fs[0], map_location='cpu'))
        m.eval()
        models[k] = (m, capture_attention(m, torch))
    out = []
    for letter in ('B', 'C'):
        cand = np.nonzero(ds['usable'] & ds['ev_future'] & ds['ev_visible'] & (np.char.startswith(ds['scen'], letter))
                          & (np.nan_to_num(ds['t_to_event'], nan=99) <= 3.0))[0]
        for i in cand[np.linspace(0, len(cand) - 1, args.n).astype(int)]:
            m, store = models[int(ds['fold'][i])]
            store.clear()
            with torch.no_grad():
                pred = m(torch.tensor(ds['hist'][i:i + 1]), torch.tensor(ds['crop'][i:i + 1]))[0].numpy()
            w = torch.stack(store).mean(0)[0, -1].numpy()                # layers averaged, last history token -> (100,)
            out.append(dict(crop=ds['crop'][i], hist=ds['hist'][i][:, :2], fut=ds['fut'][i][:, :2], pred=pred, w=w,
                            title=f'{str(ds["ids"][i]).split("/")[1]}, event in {ds["t_to_event"][i]:.1f} s'))
    plot_attention(out)


# ============================================================================================================== main
NUMBER_ONLY = {'pipeline': fig_pipeline, 'architecture': fig_architecture_detail, 'results': fig_results,
               'robustness': fig_robustness, 'limits': fig_limits, 'checks': fig_checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(list(NUMBER_ONLY) + ['tokens', 'attention']), default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--data', default='packed/mvp_v1_packed.npz')
    ap.add_argument('--models', default='runs')
    ap.add_argument('--variant', default='full_hn')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=3, help='attention: samples per scenario (B and C)')
    ap.add_argument('--sample', type=int, default=None, help='tokens: sample index (default: a corner sample)')
    args = ap.parse_args()
    want = args.only or (sorted(NUMBER_ONLY) + ['tokens', 'attention'] if args.all else sorted(NUMBER_ONLY))
    for k in want:
        if k in NUMBER_ONLY:
            NUMBER_ONLY[k]()
        elif k == 'tokens':
            fig_tokens(args)
        else:
            fig_attention(args)


if __name__ == '__main__':
    main()