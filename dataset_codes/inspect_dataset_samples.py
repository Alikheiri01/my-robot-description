#!/usr/bin/env python3
"""
Independent checker for dataset_recorder.py's .npz output -- same role in
this stage as check_occupancy_grid_accuracy.py / check_pedestrian_crop_accuracy.py
play for the earlier stages: verify the artifact is structurally sound and
let a human eyeball a sample visually, rather than trusting the recorder's
own belief that what it wrote is correct.

TWO MODES
------------------------------------------------------------------
Default (no --view): SCAN mode. Loads every .npz in the given directory,
checks each one structurally (shapes, dtypes, the anchor row being ~zero,
no NaN/Inf, crop values in the -1/0/100 set, no absurd per-step jumps), and
prints one summary report -- pass/fail counts plus aggregate stats. This is
the fast first check across the whole batch.

--view: VIEWER mode. Opens one sample at a time in a matplotlib window,
crop drawn exactly like pedestrian_crop_view_dynamic.py's own convention,
with the trajectory_history (orange, trailing) and future_target (blue,
leading) overlaid in the SAME local frame the crop image itself uses --
so you can visually confirm history sits behind the anchor and future
extends ahead of it, roughly following the walkable (white) cells. n/p
step through samples, r jumps to a random one, q quits.

WHY THE ANCHOR ROW SHOULD BE ~ZERO
------------------------------------------------------------------
dataset_recorder.py expresses every step relative to the anchor step's own
position+heading (see that file's docstring) -- so trajectory_history's
LAST row (the anchor itself) should transform to (0, 0, 0) by construction.
If it doesn't, that's a real bug in the recorder's frame math, not
sensor noise -- this check exists specifically to catch that class of bug.
"""
import argparse
import math
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent / 'hunav_codes'))
sys.path.insert(0, str(_THIS_DIR.parent / 'scene codes'))

import numpy as np

from hunav_config import TRAJECTORY_HISTORY_LENGTH, FUTURE_HORIZON_LENGTH

EXPECTED_HISTORY_SHAPE = (TRAJECTORY_HISTORY_LENGTH, 3)
EXPECTED_FUTURE_SHAPE = (FUTURE_HORIZON_LENGTH, 3)
EXPECTED_CROP_SHAPE = (80, 80)
VALID_CROP_VALUES = {-1, 0, 100}
ANCHOR_ZERO_TOL = 1e-4

# Heuristic only, not a hard physical limit -- flags a per-step jump that
# would require an implausible speed, which is far more likely to be a
# frame-math bug (e.g. a stale/mismatched robot pose) than real motion.
# Comfortably above the teleop tool's own MAX_LINEAR_SPEED=2.0 m/s.
MAX_REASONABLE_SPEED_MPS = 3.5


def validate_sample(data) -> list:
    """Returns a list of human-readable issue strings; empty means clean."""
    issues = []

    for key in ('trajectory_history', 'crop', 'future_target'):
        if key not in data:
            issues.append(f'missing key: {key}')
    if issues:
        return issues  # can't check shapes etc. without the arrays

    history = data['trajectory_history']
    future = data['future_target']
    crop = data['crop']

    if history.shape != EXPECTED_HISTORY_SHAPE:
        issues.append(f'trajectory_history shape {history.shape}, expected {EXPECTED_HISTORY_SHAPE}')
    if future.shape != EXPECTED_FUTURE_SHAPE:
        issues.append(f'future_target shape {future.shape}, expected {EXPECTED_FUTURE_SHAPE}')
    if crop.shape != EXPECTED_CROP_SHAPE:
        issues.append(f'crop shape {crop.shape}, expected {EXPECTED_CROP_SHAPE}')

    for name, arr in (('trajectory_history', history), ('future_target', future), ('crop', crop)):
        if np.isnan(arr.astype(np.float64)).any() or np.isinf(arr.astype(np.float64)).any():
            issues.append(f'{name} contains NaN/Inf')

    if crop.shape == EXPECTED_CROP_SHAPE:
        bad_values = set(np.unique(crop).tolist()) - VALID_CROP_VALUES
        if bad_values:
            issues.append(f'crop contains values outside {{-1,0,100}}: {sorted(bad_values)}')

    if history.shape == EXPECTED_HISTORY_SHAPE:
        anchor = history[-1]
        if np.abs(anchor).max() > ANCHOR_ZERO_TOL:
            issues.append(f'anchor row (history[-1]) is not ~zero: {anchor} -- frame math bug')

    period = float(data['sample_period_sec']) if 'sample_period_sec' in data else None
    if period:
        max_step = MAX_REASONABLE_SPEED_MPS * period
        for name, arr in (('trajectory_history', history), ('future_target', future)):
            if arr.shape[0] < 2:
                continue
            deltas = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
            worst = deltas.max() if len(deltas) else 0.0
            if worst > max_step:
                issues.append(
                    f'{name} has a {worst:.2f}m step (> {max_step:.2f}m implied by '
                    f'{MAX_REASONABLE_SPEED_MPS} m/s heuristic) -- possible frame/continuity bug')

    return issues


def scan(samples_dir: Path):
    files = sorted(samples_dir.glob('*.npz'))
    if not files:
        print(f'No .npz files found in {samples_dir}')
        return

    all_step_dists = []
    n_clean = 0
    failures = []

    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                issues = validate_sample(data)
                if 'trajectory_history' in data and 'future_target' in data:
                    combined = np.concatenate([data['trajectory_history'][:, :2], data['future_target'][:, :2]])
                    if len(combined) > 1:
                        all_step_dists.extend(np.linalg.norm(np.diff(combined, axis=0), axis=1).tolist())
        except Exception as e:  # corrupt/unreadable file
            issues = [f'failed to load: {e}']

        if issues:
            failures.append((f.name, issues))
        else:
            n_clean += 1

    print(f'Scanned {len(files)} samples in {samples_dir}')
    print(f'  clean: {n_clean}')
    print(f'  with issues: {len(failures)}')
    if all_step_dists:
        arr = np.array(all_step_dists)
        print(
            f'  per-step displacement (history+future combined): '
            f'mean={arr.mean():.3f}m  max={arr.max():.3f}m  '
            f'median={np.median(arr):.3f}m'
        )
    if failures:
        print('\nFirst 15 samples with issues:')
        for name, issues in failures[:15]:
            print(f'  {name}:')
            for issue in issues:
                print(f'    - {issue}')
        if len(failures) > 15:
            print(f'  ... and {len(failures) - 15} more')


# --- interactive viewer ----------------------------------------------------
def _to_display(crop):
    disp = np.zeros_like(crop, dtype=np.int8)
    disp[crop == -1] = 0
    disp[crop == 0] = 1
    disp[crop == 100] = 2
    return disp


def view(samples_dir: Path, start_index: int = 0):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from extract_pedestrian_crop import CROP_FORWARD, CROP_BEHIND, CROP_SIDE

    files = sorted(samples_dir.glob('*.npz'))
    if not files:
        print(f'No .npz files found in {samples_dir}')
        return

    cmap = ListedColormap(['#b0b0b0', '#ffffff', '#1a1a1a'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    state = {'idx': max(0, min(start_index, len(files) - 1))}

    fig, ax = plt.subplots(num='Dataset sample viewer')

    def draw():
        ax.clear()
        f = files[state['idx']]
        with np.load(f, allow_pickle=True) as data:
            issues = validate_sample(data)
            crop = data['crop']
            history = data['trajectory_history']
            future = data['future_target']

        im = ax.imshow(
            _to_display(crop), cmap=cmap, norm=norm, origin='lower',
            extent=[-CROP_BEHIND, CROP_FORWARD, -CROP_SIDE, CROP_SIDE])
        # BoundaryNorm has no inverse mapping, so matplotlib's mouse-hover
        # cursor-data tooltip crashes trying to compute it -- same cosmetic
        # issue already suppressed in pedestrian_crop_view_dynamic.py.
        im.format_cursor_data = lambda data: ''
        ax.plot(history[:, 0], history[:, 1], 'o-', color='orange', label='history', markersize=4)
        ax.plot(future[:, 0], future[:, 1], 'o-', color='dodgerblue', label='future', markersize=4)
        ax.plot(0, 0, 'r+', markersize=14, markeredgewidth=2, label='anchor')
        ax.annotate('', xy=(1.0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.set_xlabel('meters ahead of pedestrian (heading-aligned)')
        ax.set_ylabel('meters left(+) / right(-) of pedestrian')
        ax.legend(loc='upper right', fontsize=8)

        status = 'CLEAN' if not issues else f'{len(issues)} ISSUE(S)'
        color = 'green' if not issues else 'red'
        ax.set_title(
            f'[{state["idx"] + 1}/{len(files)}] {f.name}  --  {status}\n'
            f'n=next  p=prev  r=random  q=quit', fontsize=9, color=color)

        print(f'\n{f.name}: {"clean" if not issues else issues}')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'n':
            state['idx'] = (state['idx'] + 1) % len(files)
        elif event.key == 'p':
            state['idx'] = (state['idx'] - 1) % len(files)
        elif event.key == 'r':
            state['idx'] = np.random.randint(len(files))
        elif event.key == 'q':
            plt.close(fig)
            return
        draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples_dir', nargs='?', default=str(_THIS_DIR / 'samples'))
    parser.add_argument('--view', action='store_true', help='open the interactive visual viewer instead of scanning')
    parser.add_argument('--start', type=int, default=0, help='--view only: starting sample index')
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir).expanduser().resolve()
    if args.view:
        view(samples_dir, args.start)
    else:
        scan(samples_dir)


if __name__ == '__main__':
    main()