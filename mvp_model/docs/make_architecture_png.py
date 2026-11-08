#!/usr/bin/env python3
"""Draws docs/architecture.png (run from mvp_model/:  python3 docs/make_architecture_png.py). Needs matplotlib only."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / 'architecture.png'
INK, MUTED = '#1f2937', '#6b7280'
C_HIST, C_SCENE, C_FUSE, C_OUT = '#dbeafe', '#dcfce7', '#fef3c7', '#fee2e2'
EDGE = {C_HIST: '#2563eb', C_SCENE: '#16a34a', C_FUSE: '#d97706', C_OUT: '#dc2626'}

fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 150)
ax.set_ylim(0, 86)
ax.axis('off')
ax.set_aspect('auto')


def box(x, y, w, h, text, color, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=color, ec=EDGE[color], lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK,
            fontweight='bold' if bold else 'normal', linespacing=1.35)


def arrow(x1, y1, x2, y2, color=INK, style='-|>', lw=1.6, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=lw, color=color, linestyle=ls))


ax.text(75, 83, 'How the model turns "what I saw" into "where the pedestrian goes"', ha='center', fontsize=15,
        fontweight='bold', color=INK)

# ---------------------------------------------------------------- inputs
ax.text(14, 76, 'INPUT 1: history', ha='center', fontsize=10.5, fontweight='bold', color=EDGE[C_HIST])
box(1, 58, 26, 15, 'last 8 positions\nof the pedestrian\n(every 0.5 s = 3.5 s)\nx, y, heading', C_HIST)
ax.text(14, 54, 'pedestrian frame: he stands at (0,0)\nand looks along +x', ha='center', fontsize=8.5, color=MUTED)

ax.text(14, 44, 'INPUT 2: the crop the robot saw', ha='center', fontsize=10.5, fontweight='bold', color=EDGE[C_SCENE])
# little crop picture, drawn inside the main axes
img = np.full((80, 80), 0.55)                    # unknown = grey
img[22:58, 0:70] = 1.0                           # free corridor
img[22:25, 0:70] = 0.0
img[55:58, 0:70] = 0.0
img[25:58, 52:70] = 0.0                          # a corner: wall at the end, opening to the left
img[25:44, 52:70] = 1.0
CX0, CX1, CY0, CY1 = 3, 25, 22, 42
ax.imshow(img, cmap='gray', vmin=0, vmax=1, origin='lower', extent=[CX0, CX1, CY0, CY1], aspect='auto', zorder=1)
ax.add_patch(Rectangle((CX0, CY0), CX1 - CX0, CY1 - CY0, fc='none', ec=EDGE[C_SCENE], lw=2, zorder=2))
px, py = CX0 + (CX1 - CX0) * 2 / 8, CY0 + (CY1 - CY0) * 0.5
ax.plot([px], [py], 'o', color='#16a34a', ms=8, zorder=3)
ax.annotate('', xy=(px + 4, py), xytext=(px + 0.5, py), arrowprops=dict(arrowstyle='->', color='#16a34a', lw=2), zorder=3)
ax.text(14, 16.0, '80 x 80 cells = 8 m x 8 m\nblack = occupied, white = free, grey = never seen\ngreen dot = the pedestrian', ha='center',
        fontsize=8.5, color=MUTED)

# ---------------------------------------------------------------- branches
box(36, 58, 24, 15, 'History encoder\n2 self-attention layers\n"how is he moving?"', C_HIST)
ax.text(48, 55.5, '8 history tokens', ha='center', fontsize=8.5, color=MUTED)
arrow(27.8, 65.5, 35.4, 65.5)

box(36, 26, 24, 15, 'Conv stem  80x80 -> 10x10\n+ 1 self-attention layer\n"what is around him?"', C_SCENE)
ax.text(48, 22.3, '100 scene tokens (each = a 0.8 m patch)\nkeep their (x, y) position on the map', ha='center', fontsize=8.5, color=MUTED)
arrow(25.8, 33.5, 35.4, 33.5)
ax.text(31.6, 36, '', fontsize=8)

# ---------------------------------------------------------------- fusion
box(70, 40, 30, 20, 'Cross-attention  x 2\n\nevery history token asks the\nscene tokens:\n"is there a wall, corner or\nobstacle that explains my motion?"', C_FUSE)
arrow(60.6, 65.5, 69.4, 54, color=EDGE[C_HIST])
arrow(60.6, 33.5, 69.4, 46, color=EDGE[C_SCENE])
ax.text(65, 70, 'queries', fontsize=8.5, color=EDGE[C_HIST], ha='center')
ax.text(70, 31, 'keys / values', fontsize=8.5, color=EDGE[C_SCENE], ha='center')

# ---------------------------------------------------------------- output
box(110, 46, 18, 12, 'Small MLP head\n12 x (dx, dy)\n"corrections"', C_OUT)
arrow(100.6, 50, 109.4, 52)
box(110, 24, 18, 12, 'Constant-velocity\nguess (last 4 steps)\n"keep going as now"', C_OUT)
ax.text(119, 20.5, 'always computed, needs no learning', ha='center', fontsize=8.5, color=MUTED)
ax.text(119, 41, '+', fontsize=24, ha='center', color=INK, fontweight='bold')
arrow(128.6, 52, 134.5, 44)
arrow(128.6, 30, 134.5, 38)
box(134, 36, 15, 12, 'OUTPUT\n12 points\n0.5 ... 6 s', C_OUT, bold=True)

# ---------------------------------------------------------------- variants strip
ax.add_patch(Rectangle((1, 1), 148, 13.5, fc='#f9fafb', ec='#d1d5db', lw=1))
ax.text(3, 11.6, 'The same network is trained in other versions, to find out what really helps:', fontsize=9.5,
        fontweight='bold', color=INK)
ax.text(3, 7.2, 'hist  = no crop: the cross-attention looks at the history itself (same size).  If  full  beats  hist,  the scene helps.', fontsize=9, color=INK)
ax.text(3, 3.4, 'noocc = obstacles drawn as free.   concat = no attention, just  GRU(history) + CNN(crop)  joined.   '
        'Shuffle test = give the model the crop of another environment.', fontsize=9, color=INK)

fig.savefig(OUT, dpi=130, bbox_inches='tight', facecolor='white')
print('wrote', OUT)
