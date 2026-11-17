#!/usr/bin/env python3
"""
figures/make_figures.py -- ALL presentation figures in this one file. Run from mvp_model/ :

    python3 figures/make_figures.py                    # the 6 figures that need only numbers (no data, no models)
    python3 figures/make_figures.py --only results robustness      # just some of them
    python3 figures/make_figures.py --only tokens      # needs packed/mvp_v1_packed.npz (numpy only)
    python3 figures/make_figures.py --only scenarios randomization    # need the dataset folder (--root)
    python3 figures/make_figures.py --only input       # folder + packed file
    python3 figures/make_figures.py --only attention counterfactual   # need torch + the trained models in runs/
    python3 figures/make_figures.py --all              # everything (tokens and attention need their inputs)

Pictures go to reports/figures/ (the folder git ignores). Names: pipeline, architecture, results, robustness, limits, checks,
scenarios, randomization_B, input, tokens, attention, counterfactual. Numbers come from the NUMBERS block below (copied from docs/results.md): change them there if a result changes.
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


# ======================================================================================== scenario / environment drawings
# These need the dataset FOLDER (environment.json of every environment): --root ../mvp_collection/datasets/mvp_v1
SCEN_NAME = {'A': 'A   Detour', 'B': 'B   Corner', 'C': 'C   Stop', 'D': 'D   Doorway', 'E': 'E   Free walk'}
ROLE_COL = {'wall': '#2b2b2b', 'furniture': '#a8865a', 'event': '#e08a1e', 'column': '#6b7280'}
PATH_COL = '#1f4fd8'


def env_files(root, letter):
    fs = sorted(Path(root).glob(f'{letter}_*/env_{letter}_*/environment.json'))
    if not fs:
        sys.exit(f'no environment.json for scenario {letter} under {root} (use --root <dataset folder>)')
    return fs


def load_env(path):
    import json
    return json.loads(Path(path).read_text())


def box_poly(o):
    c, s = np.cos(o.get('yaw', 0.0)), np.sin(o.get('yaw', 0.0))
    hx, hy = o['size_x'] / 2, o['size_y'] / 2
    pts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return np.stack([o['x'] + c * pts[:, 0] - s * pts[:, 1], o['y'] + s * pts[:, 0] + c * pts[:, 1]], 1)


def to_world(anchor, pts):
    pts = np.asarray(pts, float)
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    return np.stack([anchor[0] + c * pts[:, 0] - s * pts[:, 1], anchor[1] + s * pts[:, 0] + c * pts[:, 1]], 1)


def draw_env(ax, env, center, size=(18.0, 12.0), floor='#ece8de'):
    """Top-down drawing of one environment: floor (walkable rectangles), walls, furniture, the event obstacle, columns."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    for r in env.get('rects', []):
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc=floor, ec='none', zorder=1))
    for o in env['obstacles']:
        col = ROLE_COL.get(o['role'], '#888888')
        if o['type'] == 'box':
            ax.add_patch(Polygon(box_poly(o), closed=True, fc=col, ec='none', zorder=2))
        else:
            ax.add_patch(Circle((o['x'], o['y']), o['radius'], fc=col, ec='none', zorder=2))
    ax.set_xlim(center[0] - size[0] / 2, center[0] + size[0] / 2)
    ax.set_ylim(center[1] - size[1] / 2, center[1] + size[1] / 2)
    ax.set_aspect('equal')
    ax.set_facecolor('#fbfbf9')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#d4d4d4')


def draw_route(ax, ep, color=PATH_COL, lw=3.0, label=None):
    r = np.array(ep['route'])
    ax.plot(r[:, 0], r[:, 1], '-', color=color, lw=lw, zorder=5, solid_capstyle='round', label=label)
    ax.plot(*r[0], 'o', color=color, mfc='white', ms=9, mew=2.5, zorder=6)
    if len(r) > 1:
        d = r[-1] - r[-2]
        ax.annotate('', xy=r[-1] + 0.01 * d / (np.linalg.norm(d) + 1e-9), xytext=r[-1] - 0.9 * d / (np.linalg.norm(d) + 1e-9),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=18), zorder=6)


def route_point(route, s):
    r = np.asarray(route, float)
    seg = np.linalg.norm(np.diff(r, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    s = float(np.clip(s, 0, cum[-1]))
    return np.array([np.interp(s, cum, r[:, 0]), np.interp(s, cum, r[:, 1])])


def event_center(env, ep, span=6.0):
    """Centre of the drawing window: halfway between the route point `span` m before the event and `span` m after it
    (so the event and the manoeuvre are in the middle, whichever way the corner turns)."""
    r = np.asarray(ep['route'], float)
    total = float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())
    s_ev = (ep.get('event') or {}).get('s')
    if s_ev is None:
        return route_point(r, total / 2)
    return 0.5 * (route_point(r, s_ev - span) + route_point(r, s_ev + span))


def fig_scenarios(args):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.4), gridspec_kw={'hspace': 0.12, 'wspace': 0.05})
    for ax, letter in zip(axes.ravel(), 'ABCDE'):
        fs = env_files(args.root, letter)
        f = fs[min(args.pick, len(fs) - 1)]
        if letter == 'E' and args.pick == 0:                       # prefer a curved walk: it differs most from the hallway ones
            arcs = [g for g in fs if load_env(g)['params'].get('shape') == 'arc']
            f = arcs[0] if arcs else f
        env = load_env(f)
        ep = env['episodes'][0]
        draw_env(ax, env, event_center(env, ep), size=(16, 12.5))
        draw_route(ax, ep)
        ax.set_title(SCEN_NAME[letter], fontsize=18, weight='bold', loc='left')
    ax = axes[1, 2]
    ax.axis('off')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=ROLE_COL['wall'], label='wall'), Patch(fc=ROLE_COL['furniture'], label='furniture'),
                       Patch(fc=ROLE_COL['event'], label='obstacle on the path'),
                       Line2D([0], [0], color=PATH_COL, lw=3, marker='o', mfc='white', mew=2.5, label="person's walk (start o)")],
              loc='center', frameon=False, fontsize=15)
    save(fig, 'scenarios.png')


def param_text(letter, env):
    p = env['params']
    if letter == 'B':
        return f"hall {p['hall_width']:.1f} m   legs {p['leg1']:.0f} / {p['leg2']:.0f} m   turns {p['turn']}"
    if letter == 'D':
        return f"rooms {p['room_width']:.1f} m wide   door {p['door_width']:.1f} m"
    if letter == 'A':
        o = p['event_obstacle']
        return f"hall {p['hall_width']:.1f} m   obstacle {o.get('along', 2 * o.get('r', 0.2)):.1f} x {o.get('across', 2 * o.get('r', 0.2)):.1f} m"
    return f"hall {p['hall_width']:.1f} m"


def schematic_corner(width, leg1, leg2, turn, bays=(), furniture=()):
    """An exaggerated corner template (not from the dataset): returns an environment dict in the same format as environment.json.
    The hallway runs along +x to the corner, then leg 2 goes down (turn 'right') or up (turn 'left')."""
    sg = -1.0 if turn == 'right' else 1.0
    w = width
    rects = [[-leg1, -w / 2, w, w / 2], [0.0, -leg2 if sg < 0 else -w / 2, w, w / 2 if sg < 0 else leg2]]
    obst = []
    t = 0.2

    def wall(x0, y0, x1, y1):
        obst.append(dict(role='wall', type='box', x=(x0 + x1) / 2, y=(y0 + y1) / 2, yaw=0.0, size_x=abs(x1 - x0), size_y=abs(y1 - y0)))
    wall(-leg1, w / 2, w + t, w / 2 + t) if sg < 0 else wall(-leg1, w / 2, 0, w / 2 + t)
    wall(-leg1, -w / 2 - t, 0, -w / 2) if sg < 0 else wall(-leg1, -w / 2 - t, w + t, -w / 2)
    if sg < 0:
        wall(w, -leg2, w + t, w / 2); wall(-t, -leg2, 0, -w / 2)
    else:
        wall(w, -w / 2, w + t, leg2); wall(-t, w / 2, 0, leg2)
    for bx, side in bays:                                              # alcoves on the first leg
        y0, y1 = (w / 2, w / 2 + 1.4) if side > 0 else (-w / 2 - 1.4, -w / 2)
        rects.append([bx, y0, bx + 1.8, y1])
        obst.append(dict(role='wall', type='box', x=bx - t / 2, y=(y0 + y1) / 2, yaw=0.0, size_x=t, size_y=y1 - y0))
        obst.append(dict(role='wall', type='box', x=bx + 1.8 + t / 2, y=(y0 + y1) / 2, yaw=0.0, size_x=t, size_y=y1 - y0))
        obst.append(dict(role='wall', type='box', x=bx + 0.9, y=y1 + t / 2 if side > 0 else y0 - t / 2, yaw=0.0, size_x=1.8 + 2 * t, size_y=t))
    for fx, fy in furniture:
        obst.append(dict(role='furniture', type='box', x=fx, y=fy, yaw=0.0, size_x=1.0, size_y=0.5))
    route = [[-leg1 + 1.0, 0.0], [w / 2, 0.0], [w / 2, sg * (leg2 - 1.0)]]
    ep = dict(route=route, event=dict(position=[w / 2, 0.0], s=leg1 - 1.0 + w / 2), speed=1.2)
    return dict(rects=rects, obstacles=obst, params={}, episodes=[ep]), sg


def fig_randomization(args):
    """Row 1: one corner template with its settings pushed to extremes (an ILLUSTRATION, drawn here, exaggerated).
    Row 2: the same idea for real: four environments of the dataset, from the narrowest to the widest hallway.
    Row 3: one real environment, four episodes (route, start, goal, speed, robot)."""
    letter = args.scenario
    fs = env_files(args.root, letter)
    envs = [load_env(f) for f in fs]
    wkey = 'hall_width' if 'hall_width' in envs[0]['params'] else 'room_width'
    order = sorted(range(len(envs)), key=lambda i: envs[i]['params'][wkey])
    pos = sorted({0, len(order) // 3, 2 * len(order) // 3, len(order) - 1})
    chosen = [order[k] for k in pos][:4]
    size = (14, 14) if letter == 'B' else (17, 11.5)
    fig = plt.figure(figsize=(17, 15.2 if letter == 'B' else 11.5))
    gs = fig.add_gridspec(3, 4, hspace=0.2, wspace=0.04, top=0.94, bottom=0.015, left=0.02, right=0.98)
    rows_ax = {}
    # row 1: exaggerated schematic (corner template; for other scenarios the corner picture is still used as the illustration)
    variants = [('narrow hallway', schematic_corner(1.1, 16, 14, 'right')),
                ('very wide hallway', schematic_corner(4.8, 16, 14, 'right')),
                ('other turn side, short first leg', schematic_corner(2.2, 7, 17, 'left')),
                ('alcoves and furniture', schematic_corner(2.2, 16, 14, 'right', bays=[(-12, -1), (-7, 1), (-2.5, 1)],
                                                           furniture=[(-4.5, 0.75), (-10, -0.75)]))]
    for k, (lab, (env, sg)) in enumerate(variants):
        ax = fig.add_subplot(gs[0, k])
        rows_ax[0] = ax
        ep = env['episodes'][0]
        draw_env(ax, env, event_center(env, ep, 7.0), size=size)
        draw_route(ax, ep)
        ax.set_title(lab, fontsize=14, loc='left', weight='bold')
    for k, i in enumerate(chosen):
        ax = fig.add_subplot(gs[1, k])
        rows_ax[1] = ax
        env = envs[i]
        ep = env['episodes'][0]
        draw_env(ax, env, event_center(env, ep, 7.0), size=size)
        draw_route(ax, ep)
        ax.set_title(param_text(letter, env), fontsize=11, loc='left')
    base = envs[order[len(order) // 2]]
    cmap = plt.get_cmap('viridis')
    sp = [e['speed'] for e in base['episodes']]
    for k in range(4):
        ax = fig.add_subplot(gs[2, k])
        rows_ax[2] = ax
        ep = base['episodes'][k]
        draw_env(ax, base, event_center(base, base['episodes'][0], 7.0), size=size)
        draw_route(ax, ep, color=cmap((ep['speed'] - min(sp)) / (max(sp) - min(sp) + 1e-9) * 0.8))
        rp = (ep.get('robot') or {}).get('start_pose')
        if rp is not None:
            ax.plot(rp[0], rp[1], '^', color='#1f78b4', mec='k', ms=12, zorder=7)
        ax.set_title(f"episode {k + 1}:  {ep['speed']:.1f} m/s", fontsize=11, loc='left')
    name = SCEN_NAME[letter].split()[1]
    heads = [f'1.  The {name} template with its settings pushed to the extreme (illustration)',
             '2.  Real environments of the dataset: same type, different geometry',
             '3.  One real environment, different episodes (start, goal, speed; blue triangle = robot)']
    for r in range(3):
        y = rows_ax[r].get_position().y1 + 0.032
        fig.text(0.02, y, heads[r], fontsize=16, weight='bold')
    save(fig, f'randomization_{letter}.png')


# ------------------------------------------------------------------------------------------------ how the input is made
def sample_world(root, ds, i):
    """Pedestrian anchor pose and robot pose of sample i in the world (odom) frame, and the environment dict."""
    from motion.data import _scan
    runs = {(sc, ed.name): (ed, run) for sc, ed, run in _scan(Path(root))}
    sc, env, fname = str(ds['ids'][i]).split('/')
    ed, run = runs[(sc, env)]
    d = np.load(run / 'samples' / fname, allow_pickle=True)
    return (np.asarray(d['ped_anchor_odom'], float)[:3], np.asarray(d['robot_pose_odom'], float)[:3],
            load_env(ed / 'environment.json'))


def draw_input(ped, rob, env, crop, hist, fut, name='input.png'):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon, Wedge
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1.5, 1]})
    draw_env(a, env, ped[:2], size=(22, 14))
    box = to_world(ped, [[-2, -4], [6, -4], [6, 4], [-2, 4]])
    a.add_patch(Polygon(box, closed=True, fc='none', ec='#2563eb', lw=3, zorder=6))
    a.plot(*to_world(ped, [[0, 0]])[0], 'o', color='k', mfc='white', ms=12, mew=2.5, zorder=8)
    h = to_world(ped, hist[:, :2])
    a.plot(h[:, 0], h[:, 1], '-', color='#f59e0b', lw=3, zorder=7)
    f = to_world(ped, np.r_[[[0, 0]], fut[:, :2]])
    a.plot(f[:, 0], f[:, 1], '-', color=COL['truth'], lw=3, zorder=7)
    a.plot(rob[0], rob[1], '^', color='#1f78b4', mec='k', ms=14, zorder=8)
    a.add_patch(Wedge((rob[0], rob[1]), 8.0, np.degrees(rob[2] - 1.274 / 2), np.degrees(rob[2] + 1.274 / 2), fc='#1f78b4', alpha=0.10,
                      ec='none', zorder=3))
    a.set_title('The world, seen from above   (blue frame = what is cut out around the person)', fontsize=13, loc='left')
    arr = np.where(crop == -1, 0, np.where(crop == 0, 1, 2))
    b.imshow(arr, cmap=ListedColormap(['#c8c8c8', '#ffffff', '#202020']), vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4),
             interpolation='nearest')
    b.plot(hist[:, 0], hist[:, 1], 'o-', color='#f59e0b', ms=5, lw=2.5)
    b.plot(0, 0, 'o', color='k', mfc='white', ms=11, mew=2.5)
    b.set_xticks([])
    b.set_yticks([])
    for s in b.spines.values():
        s.set_edgecolor('#2563eb')
        s.set_linewidth(3)
    b.set_title('What the model gets: the map around the person,\nturned so that he looks to the right\n(grey = not seen, white = free, black = wall)', fontsize=12, loc='left')
    save(fig, name)


def fig_input(args):
    from motion.data import load_dataset
    ds = load_dataset(HERE / args.data)
    i = pick_samples(ds, 'B', 1.0, 2.5, 1)[0]
    ped, rob, env = sample_world(args.root, ds, i)
    draw_input(ped, rob, env, ds['crop'][i], ds['hist'][i], ds['fut'][i])


# ======================================================================================= attention and counterfactual
TOKEN_X = -2 + 0.8 * (np.arange(10) + 0.5)          # token centres in the pedestrian frame (metres)
TOKEN_Y = -4 + 0.8 * (np.arange(10) + 0.5)


def pick_samples(ds, letter, t_lo, t_hi, n):
    """Deterministic, NOT best-looking: usable samples of scenario `letter` whose event is visible and t_lo <= time to event <= t_hi;
    returns n of them taken from the middle of the list (sorted by id)."""
    tte = np.nan_to_num(ds['t_to_event'], nan=99.0)
    m = (ds['usable'] & ds['ev_future'] & ds['ev_visible'] & np.char.startswith(ds['scen'], letter) & (tte >= t_lo) & (tte <= t_hi))
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        sys.exit(f'no usable sample with a visible event for scenario {letter}')
    mid = len(idx) // 2
    return [int(j) for j in idx[max(0, mid - n // 2):max(0, mid - n // 2) + n]]


class TorchAttend:
    """Fold models of one variant (one seed). attend(hist, crop, folds) -> (prediction (n,12,2), attention (n,100) or None).
    attention = weights of the NEWEST history token over the 100 scene tokens, averaged over heads and over the cross-attention layers."""

    def __init__(self, models_root, variant, seed=0, capture=True):
        import torch
        from motion.models import VARIANTS
        self.torch = torch
        self.models = {}
        k = 0
        while True:
            fs = sorted(Path(models_root).rglob(f'{variant}_s{seed}_f{k}.pt'))
            if not fs:
                break
            m = VARIANTS[variant]()
            m.load_state_dict(torch.load(fs[0], map_location='cpu'))
            m.eval()
            self.models[k] = (m, self._capture(m) if capture else None)
            k += 1
        if not self.models:
            sys.exit(f'no {variant}_s{seed}_f<fold>.pt under {models_root}')

    @staticmethod
    def _capture(model):
        store = []
        layers = getattr(getattr(model, 'cross', None), 'layers', None)
        if layers is None:
            return None
        for layer in layers:
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

    def attend(self, hist, crop, folds, bs=256):
        torch = self.torch
        pred = np.zeros((len(hist), 12, 2), np.float32)
        att = np.zeros((len(hist), 100), np.float32)
        have = True
        with torch.no_grad():
            for f in sorted(set(int(x) for x in folds)):
                m, store = self.models[f]
                idx = np.nonzero(folds == f)[0]
                for s in range(0, len(idx), bs):
                    j = idx[s:s + bs]
                    if store is not None:
                        store.clear()
                    pred[j] = m(torch.as_tensor(hist[j]), torch.as_tensor(crop[j])).numpy()
                    if store:
                        a = torch.stack(store).mean(0)[:, -1, :].numpy()         # (layers,b,8,keys) -> (b,keys)
                        if a.shape[1] == 100:
                            att[j] = a
                        else:
                            have = False                                         # e.g. the no-map model attends to the history (8 keys)
                    else:
                        have = False
        return pred, (att if have else None)


def cvm_pred(hist):
    from motion.baselines import cvm
    return cvm(hist, 4)


def draw_attention_panel(ax, s, label=True):
    """s: dict(crop, hist (8,2), fut (12,2), pred (12,2), cvm (12,2) or None, w (100,), event (x,y) or None, title)."""
    arr = np.where(s['crop'] == -1, 0, np.where(s['crop'] == 0, 1, 2))
    ax.imshow(arr, cmap=ListedColormap(['#c8c8c8', '#ffffff', '#202020']), vmin=0, vmax=2, origin='lower', extent=(-2, 6, -4, 4),
              interpolation='nearest')
    w = np.asarray(s['w']).reshape(10, 10)
    ax.imshow(w, origin='lower', extent=(-2, 6, -4, 4), cmap='autumn_r', alpha=np.clip(w / (w.max() + 1e-9), 0, 1) * 0.8,
              interpolation='nearest')
    ax.plot(s['hist'][:, 0], s['hist'][:, 1], 'o-', color='#555555', ms=3, lw=1.2)
    ax.plot(np.r_[0, s['fut'][:, 0]], np.r_[0, s['fut'][:, 1]], '-', color=COL['truth'], lw=3)
    if s.get('cvm') is not None:
        ax.plot(np.r_[0, s['cvm'][:, 0]], np.r_[0, s['cvm'][:, 1]], '--', color=COL['cvm'], lw=2.2)
    ax.plot(np.r_[0, s['pred'][:, 0]], np.r_[0, s['pred'][:, 1]], '-', color=COL['map'], lw=2.8)
    if s.get('event') is not None:
        ax.plot(*s['event'], '*', color='#e6b800', mec='k', ms=17, zorder=6)
    ax.plot(0, 0, 'o', color='k', mfc='white', ms=8, mew=2)
    ax.set_xlim(-2, 6)
    ax.set_ylim(-4, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(s['title'], fontsize=11, loc='left')


def plot_attention(samples, name='attention.png'):
    """Fake-data friendly: a row of attention panels."""
    n = len(samples)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    for ax, s in zip(axes.ravel(), samples):
        draw_attention_panel(ax, s)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    return save(fig, name)


def env_bootstrap(vals, groups, n=1000, seed=0):
    """Mean and 95 % interval of vals, resampling whole environments (groups)."""
    rng = np.random.RandomState(seed)
    keys = sorted(set(groups))
    per = {k: vals[groups == k] for k in keys}
    mean = float(np.mean(vals))
    bs = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        bs.append(np.concatenate([per[keys[j]] for j in pick]).mean())
    return mean, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0)]


def attention_evidence(ds, att_map, pred_map, pred_no, radius):
    """Per time-to-event bin: share of the cross-attention that falls on the scene tokens within `radius` of the event point
    (+ what a uniform attention would give), and ADE@3s of constant velocity / without map / with map.
    Only usable samples of B, C, D whose event is inside the crop and visible."""
    from motion.data import event_point
    tte = np.nan_to_num(ds['t_to_event'], nan=99.0)
    m = ds['usable'] & ds['ev_future'] & ds['ev_visible'] & np.isin(np.array([s[0] for s in ds['scen']]), list('BCD'))
    idx = np.nonzero(m)[0]
    share, uni, err = {}, {}, {}
    cv = cvm_pred(ds['hist'])
    for lo, hi in BINS:
        sel = [i for i in idx if lo < tte[i] <= hi or (lo == 0 and tte[i] <= hi)]
        sh, un = [], []
        for i in sel:
            ex, ey = event_point(ds['fut'][i, :, :2], tte[i])
            mask = ((TOKEN_X[None, :] - ex) ** 2 + (TOKEN_Y[:, None] - ey) ** 2 <= radius ** 2).ravel()
            sh.append(float(att_map[i][mask].sum()))
            un.append(float(mask.mean()))
        sel = np.array(sel, int)
        g = ds['env_key'][sel]
        share[(lo, hi)] = env_bootstrap(np.array(sh), g) if len(sel) else (np.nan,) * 3
        uni[(lo, hi)] = float(np.mean(un)) if len(sel) else np.nan
        e = {}
        for k, p in (('cvm', cv), ('nomap', pred_no), ('map', pred_map)):
            d = np.linalg.norm(p[sel] - ds['fut'][sel, :, :2], axis=2)[:, :6].mean(1)
            e[k] = env_bootstrap(d, g) if len(sel) else (np.nan,) * 3
        err[(lo, hi)] = (e, len(sel))
    return share, uni, err


def fig_attention(args):
    from motion.data import event_point, load_dataset
    ds = load_dataset(HERE / args.data)
    tm = TorchAttend(args.models, args.variant, args.seed)
    tn = TorchAttend(args.models, args.nomap_variant, args.seed, capture=False)
    pred_map, att_map = tm.attend(ds['hist'], ds['crop'], ds['fold'])
    pred_no, _ = tn.attend(ds['hist'], ds['crop'], ds['fold'])
    if att_map is None:
        sys.exit('this model has no cross-attention layers (model.cross.layers)')
    draw_attention_figure(ds, att_map, pred_map, pred_no, args.radius)


def draw_attention_figure(ds, att_map, pred_map, pred_no, radius, name='attention.png', letters='BCD'):
    from motion.data import event_point
    share, uni, err = attention_evidence(ds, att_map, pred_map, pred_no, radius)
    cv = cvm_pred(ds['hist'])
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 6, hspace=0.38, wspace=0.55, top=0.9, bottom=0.115, left=0.05, right=0.985, height_ratios=[1.05, 1])
    names = {'B': 'corner', 'C': 'stop', 'D': 'doorway'}
    for k, letter in enumerate(letters):
        i = pick_samples(ds, letter, 1.0, 2.5, 1)[0]
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        s = dict(crop=ds['crop'][i], hist=ds['hist'][i][:, :2], fut=ds['fut'][i][:, :2], pred=pred_map[i], cvm=cv[i],
                 w=att_map[i], event=event_point(ds['fut'][i, :, :2], ds['t_to_event'][i]),
                 title=f"{names[letter]}: event in {ds['t_to_event'][i]:.1f} s")
        draw_attention_panel(ax, s)
    fig.text(0.05, 0.935, 'Where the model looks: red/orange = attention on the map.   star = the corner / stop / door.   '
             'green = what happens, red dashed = constant velocity, blue = our model.', fontsize=12.5)
    labs = [f'{lo:g}-{hi:g} s' for lo, hi in BINS]
    x = np.arange(len(BINS))
    a = fig.add_subplot(gs[1, 0:3])
    m = np.array([share[b][0] for b in BINS])
    a.bar(x, 100 * m, 0.55, color='#ef6c00', label='attention on the map around the event')
    a.errorbar(x, 100 * m, yerr=[100 * (m - np.array([share[b][1] for b in BINS])), 100 * (np.array([share[b][2] for b in BINS]) - m)],
               fmt='none', ecolor='k', capsize=4)
    a.plot(x, [100 * uni[b] for b in BINS], 'k--', marker='_', ms=22, label='if the model looked everywhere equally')
    a.set_xticks(x)
    a.set_xticklabels(labs)
    a.set_xlabel('time until the corner / stop / door')
    a.set_ylabel(f'share of attention within {radius:g} m of the event  [%]')
    a.set_title('The closer the event, the more the model reads it', loc='left')
    a.legend(frameon=False, fontsize=10)
    b = fig.add_subplot(gs[1, 3:6])
    for key, c, lab in (('cvm', COL['cvm'], LABEL['cvm']), ('nomap', COL['nomap'], LABEL['nomap']), ('map', COL['map'], LABEL['map'])):
        v = np.array([err[bb][0][key][0] for bb in BINS])
        lo = np.array([err[bb][0][key][1] for bb in BINS])
        hi = np.array([err[bb][0][key][2] for bb in BINS])
        b.plot(x, v, '-o', color=c, lw=2.5, label=lab)
        b.fill_between(x, lo, hi, color=c, alpha=0.15)
    b.set_xticks(x)
    b.set_xticklabels(labs)
    b.set_xlabel('time until the corner / stop / door')
    b.set_ylabel('error of the next 3 s  [m]')
    b.set_title('...and that is where the error drops', loc='left')
    b.legend(frameon=False, fontsize=10)
    n_tot = sum(err[bb][1] for bb in BINS)
    fig.text(0.05, 0.008, textwrap.fill(f'Held-out environments only (each sample is predicted by the model that never saw its environment), '
             f'{n_tot} samples where the event is visible in the map. Attention shows what the model reads, not proof of why it '
             f'predicts what it does: the proof is the obstacle-removal test (next figure).', 190), fontsize=9, color=MUTED)
    return save(fig, name)


def fig_counterfactual(args):
    from motion.data import event_point, load_dataset
    ds = load_dataset(HERE / args.data)
    tm = TorchAttend(args.models, args.variant, args.seed)
    draw_counterfactual(ds, lambda h, c, f: tm.attend(h, c, f)[0])


def draw_counterfactual(ds, predict, name='counterfactual.png'):
    """Take samples whose stop is explained by an obstacle (C, visible, event within 4 s), delete the obstacle around the stop
    point from the map, predict again. Shows the typical case (the MEDIAN reaction among all edited samples), plus the rule result."""
    from motion.damage import remove_obstacle
    from motion.data import event_point
    tte = np.nan_to_num(ds['t_to_event'], nan=99.0)
    m = ds['usable'] & ds['ev_future'] & ds['ev_visible'] & np.char.startswith(ds['scen'], 'C') & (tte <= 4.0)
    idx = np.nonzero(m)[0]
    edited, keep = [], []
    for i in idx:
        px, py = event_point(ds['fut'][i, :, :2], tte[i])
        c2, n = remove_obstacle(ds['crop'][i], px, py, 1.5)
        if n >= 10:
            edited.append(c2)
            keep.append(i)
    if not keep:
        sys.exit('no sample with an obstacle at the stop point')
    keep = np.array(keep)
    edited = np.stack(edited)
    p0 = predict(ds['hist'][keep], ds['crop'][keep], ds['fold'][keep])
    p1 = predict(ds['hist'][keep], edited, ds['fold'][keep])
    cv = cvm_pred(ds['hist'][keep])
    react = np.linalg.norm(p0 - cv, axis=2).mean(1) - np.linalg.norm(p1 - cv, axis=2).mean(1)   # > 0: moves toward constant velocity
    j = int(np.argsort(react)[len(react) // 2])
    i = keep[j]
    ev = event_point(ds['fut'][i, :, :2], tte[i])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    for ax, crop, pr, ttl in ((a, ds['crop'][i], p0[j], 'Map as it is: obstacle at the stop point'),
                              (b, edited[j], p1[j], 'Same map, obstacle removed')):
        draw_attention_panel(ax, dict(crop=crop, hist=ds['hist'][i][:, :2], fut=ds['fut'][i][:, :2], pred=pr, cvm=cv[j],
                                      w=np.zeros(100), event=ev, title=ttl))
    c.axis('off')
    txt = ('A typical sample (median reaction of %d edited samples).\n\nWithout the obstacle the model no longer expects the stop:\n'
           'its prediction moves toward "keep walking".\n\nAcross the test (rule fixed beforehand):\n   obstacle removed: 86 %% of predictions\n   '
           'move toward constant velocity\n   control (an edit 2.5 m aside): 36 %%\n\ngreen = what happens, red dashed = constant\nvelocity, '
           'blue = our model, star = stop point') % len(keep)
    c.text(0, 1, txt, va='top', fontsize=13)
    save(fig, name)




# ============================================================================================================== main
NUMBER_ONLY = {'pipeline': fig_pipeline, 'architecture': fig_architecture_detail, 'results': fig_results,
               'robustness': fig_robustness, 'limits': fig_limits, 'checks': fig_checks}
FOLDER_ONLY = {'scenarios': fig_scenarios, 'randomization': fig_randomization}      # need the dataset folder (environment.json)
NEED_DATA = {'tokens': fig_tokens, 'input': fig_input}                              # + packed dataset
NEED_MODELS = {'attention': fig_attention, 'counterfactual': fig_counterfactual}    # + torch and the trained models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    allnames = sorted(list(NUMBER_ONLY) + list(FOLDER_ONLY) + list(NEED_DATA) + list(NEED_MODELS))
    ap.add_argument('--only', nargs='+', choices=allnames, default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--root', default='../mvp_collection/datasets/mvp_v1', help='dataset FOLDER (environment.json files)')
    ap.add_argument('--data', default='packed/mvp_v1_packed.npz')
    ap.add_argument('--models', default='runs')
    ap.add_argument('--variant', default='full_hn')
    ap.add_argument('--nomap-variant', default='hist_hn')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scenario', default='B', help='randomization: which scenario letter to show')
    ap.add_argument('--pick', type=int, default=0, help='scenarios: which environment of each scenario (0 = first)')
    ap.add_argument('--radius', type=float, default=1.2, help='attention: radius around the event [m]')
    ap.add_argument('--sample', type=int, default=None, help='tokens: sample index (default: a corner sample)')
    args = ap.parse_args()
    table = {**NUMBER_ONLY, **FOLDER_ONLY, **NEED_DATA, **NEED_MODELS}
    want = args.only or (allnames if args.all else sorted(NUMBER_ONLY))
    for k in want:
        fn = table[k]
        fn(args) if (k in FOLDER_ONLY or k in NEED_DATA or k in NEED_MODELS) else fn()


if __name__ == '__main__':
    main()