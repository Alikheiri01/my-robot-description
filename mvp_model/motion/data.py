"""
motion/data.py -- loads a collected dataset (e.g. mvp_collection/datasets/mvp_v1) into plain arrays for the baselines,
the evaluation and the training. numpy only (no torch).

One sample = one .npz of the NEWEST PASSED run of an environment (same rule as check_dataset.py).

load_dataset() returns a dict `ds` (N = all samples; ds['usable'] selects the ones for training/evaluation):
    hist (N,8,3) float32   pedestrian history in the pedestrian frame at the anchor (last row = 0,0,0)
    fut  (N,12,3) float32  the label (0.5 s spacing, 6 s)
    crop (N,80,80) int8    -1 unknown, 0 free, 100 occupied; rows = y -4..4 m, cols = x -2..6 m, 0.1 m cells
    robot (N,2) float32    robot position in the pedestrian frame (NOT a model input by default)
    robot_yaw (N,) float32 robot heading in the pedestrian frame [rad] (only for the wedge control, motion/wedge.py)
    scen, env, tmpl, ep, mode, ids   strings: 'B_corner', 'env_B_003', 'T2_corner', 'env_B_003_ep1', 'follow', unique id
    env_key (N,) str       'B_corner/env_B_003'  (the unit for splits and for the bootstrap)
    t_rel (N,) float       anchor time after walk start [s]
    ev_future (N,) bool    the scenario event happens inside the 6 s future
    t_to_event (N,) float  seconds from the anchor to the event (nan if none in the future)
    ev_visible (N,) bool   event point lies in the crop and >= 50 % of the 1 m disk around it is known (not grey)
    usable (N,) bool       anchor seen and >= 6 of 8 history steps seen
    speed_plan (N,) float
    fold (N,) int          environment-grouped fold (stratified by scenario), see make_folds
"""
import json
import re
from pathlib import Path

import numpy as np

DT = 0.5
N_HIST, N_FUT = 8, 12
X0, Y0, RES = -2.0, -4.0, 0.1
MIN_HIST_SEEN = 6
DISK_R = 1.0
VIS_FRAC = 0.5
SCHEMA = 'time_aligned_v2'


def run_passed(run_dir):
    try:
        return bool(re.search(r'OVERALL: PASS', (Path(run_dir) / 'report.txt').read_text()))
    except OSError:
        return False


def event_point(fut_xy, t_to_event):
    """Label position at the event time (linear interpolation; the anchor itself is (0, 0))."""
    t = np.arange(0, N_FUT + 1) * DT
    x = np.concatenate([[0.0], fut_xy[:, 0]])
    y = np.concatenate([[0.0], fut_xy[:, 1]])
    return float(np.interp(t_to_event, t, x)), float(np.interp(t_to_event, t, y))


def disk_known_fraction(crop, px, py, r=DISK_R):
    """Share of the disk cells (radius r around the point) that are not unknown. 0 if the point is outside the crop."""
    c0 = int(np.floor((px - X0) / RES))
    r0 = int(np.floor((py - Y0) / RES))
    if not (0 <= c0 < 80 and 0 <= r0 < 80):
        return 0.0
    k = int(round(r / RES))
    rr, cc = np.mgrid[-k:k + 1, -k:k + 1]
    m = rr ** 2 + cc ** 2 <= k * k
    R, Cc = rr[m] + r0, cc[m] + c0
    inside = (R >= 0) & (R < 80) & (Cc >= 0) & (Cc < 80)
    if not inside.any():
        return 0.0
    return float((crop[R[inside], Cc[inside]] != -1).sum()) / float(m.sum())   # cells outside the crop count as unknown


def flip_arrays(hist, crop, fut):
    """Mirror the world left/right (y -> -y) for numpy arrays; hist (...,8,3), fut (...,12,3), crop (...,80,80).
    Flips y and yaw of history and label, and the ROWS of the crop (rows are y, the crop is symmetric around y = 0).
    The torch version in motion/models.py (flip_batch) must do exactly the same; tests/test_all.py checks that."""
    h = np.array(hist, copy=True)
    h[..., 1] *= -1
    h[..., 2] *= -1
    f = np.array(fut, copy=True)
    f[..., 1] *= -1
    if f.shape[-1] > 2:
        f[..., 2] *= -1
    return h, np.flip(crop, axis=-2).copy(), f


def make_folds(env_key, scen, k=5, seed=0):
    """Environment-grouped folds, stratified by scenario: every scenario's environments are spread over the k folds.
    Returns (N,) int fold per sample. Never split by sample: neighbouring samples are almost the same scene."""
    rng = np.random.RandomState(seed)
    fold_of, counter = {}, 0
    for sc in sorted(set(scen)):
        envs = sorted({e for e, s in zip(env_key, scen) if s == sc})
        order = rng.permutation(len(envs))
        for j in order:
            fold_of[envs[j]] = counter % k
            counter += 1
    return np.array([fold_of[e] for e in env_key], dtype=np.int64)


def _scan(root):
    """-> list of (scenario, env_dir, run_dir) for every environment with a passed run."""
    out = []
    for sc in sorted(p for p in root.iterdir() if p.is_dir()):
        for ed in sorted(p for p in sc.iterdir() if (p / 'environment.json').exists()):
            runs = [r for r in sorted((ed / 'runs').glob('*')) if r.is_dir() and run_passed(r)] \
                if (ed / 'runs').is_dir() else []
            if runs:
                out.append((sc.name, ed, runs[-1]))
    return out


def load_dataset(root, cache=True, cache_dir='runs', folds=5, seed=0, verbose=True):
    """Loads all samples from a dataset folder (or from a packed .npz file, see pack_dataset.py).
    The cache (a single .npz inside cache_dir, NOT inside the dataset) makes reloading a folder fast."""
    root = Path(root)
    if root.is_file():                                   # a packed dataset made by pack_dataset.py (one .npz file)
        z = np.load(root, allow_pickle=False)
        ds = {k: z[k] for k in z.files if k != 'sig'}
        if verbose:
            print(f'[data] loaded packed dataset {root} ({len(ds["ids"])} samples, '
                  f'{int(ds["usable"].sum())} usable, {int(ds["fold"].max()) + 1} folds)')
        if int(ds['fold'].max()) + 1 != folds:
            print(f'[data] WARNING: the file was packed with {int(ds["fold"].max()) + 1} folds, you asked for {folds}')
        return ds
    found = _scan(root)
    if not found:
        raise SystemExit(f'no environment with a passed run under {root}')
    files = [(sc, ed, run, f) for sc, ed, run in found for f in sorted((run / 'samples').glob('*.npz'))]
    sig = f'v2:{root.resolve()}:{len(files)}:{max(f.stat().st_mtime_ns for *_, f in files)}:{folds}:{seed}'
    cpath = Path(cache_dir) / f'cache_{root.name}.npz'
    if cache and cpath.exists():
        try:
            z = np.load(cpath, allow_pickle=False)
            if str(z['sig']) == sig:
                ds = {k: z[k] for k in z.files if k != 'sig'}
                if verbose:
                    print(f'[data] loaded cache {cpath} ({len(ds["ids"])} samples)')
                return ds
        except Exception:  # noqa: BLE001
            pass

    cols = {k: [] for k in ('hist', 'fut', 'crop', 'robot', 'robot_yaw', 'scen', 'env', 'tmpl', 'ep', 'mode', 'ids', 't_rel',
                            'ev_future', 't_to_event', 'ev_visible', 'usable', 'speed_plan')}
    env_cache = {}
    skipped = 0
    for sc, ed, run, f in files:
        if ed not in env_cache:
            env = json.loads((ed / 'environment.json').read_text())
            meas = {}
            for ef in sorted((run / 'episodes').glob('*.json')):
                try:
                    meas[ef.stem] = json.loads(ef.read_text()).get('measured', {})
                except (OSError, ValueError):
                    pass
            env_cache[ed] = (env, {e['episode_id']: e for e in env['episodes']}, meas)
        env, eps, meas = env_cache[ed]
        try:
            d = np.load(f, allow_pickle=True)
            if str(d['schema']) != SCHEMA:
                skipped += 1
                continue
            h = np.asarray(d['trajectory_history'], np.float32)
            fu = np.asarray(d['future_target'], np.float32)
            crop = np.asarray(d['crop'], np.int8)
            if h.shape != (N_HIST, 3) or fu.shape != (N_FUT, 3) or crop.shape != (80, 80):
                skipped += 1
                continue
            eid = str(d['episode_id'])
            m = meas.get(eid, {})
            t0 = m.get('t_walk_start_sim')
            t_rel = float(d['anchor_time_sim']) - t0 if t0 is not None else np.nan
            hs = np.asarray(d['ped_seen_history']).astype(int).ravel()
            usable = int(d['ped_seen_anchor']) == 1 and int((hs == 1).sum()) >= MIN_HIST_SEEN
            t_ev = m.get('t_event_after_walk_start_sec')
            tte = (t_ev - t_rel) if (t_ev is not None and np.isfinite(t_rel)) else np.nan
            ev_future = bool(np.isfinite(tte) and 0 < tte <= N_FUT * DT)
            vis = False
            if ev_future:
                px, py = event_point(fu[:, :2], tte)
                vis = disk_known_fraction(crop, px, py) >= VIS_FRAC
            ax, ay, ayaw = [float(v) for v in d['ped_anchor_odom'][:3]]
            rb = np.asarray(d['robot_pose_odom'], float)
            dx, dy = rb[0] - ax, rb[1] - ay
            c, s = np.cos(ayaw), np.sin(ayaw)
            robot = (c * dx + s * dy, -s * dx + c * dy)
            ryaw = float((rb[2] - ayaw + np.pi) % (2 * np.pi) - np.pi)       # robot heading relative to the pedestrian's x axis
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        cols['hist'].append(h)
        cols['fut'].append(fu)
        cols['crop'].append(crop)
        cols['robot'].append(np.array(robot, np.float32))
        cols['robot_yaw'].append(ryaw)
        cols['scen'].append(sc)
        cols['env'].append(ed.name)
        cols['tmpl'].append(str(env.get('template', '?')))
        cols['ep'].append(eid)
        cols['mode'].append(str((eps.get(eid) or {}).get('robot', {}).get('mode', '?')))
        cols['ids'].append(f'{sc}/{ed.name}/{f.name}')
        cols['t_rel'].append(t_rel)
        cols['ev_future'].append(ev_future)
        cols['t_to_event'].append(tte if ev_future else np.nan)
        cols['ev_visible'].append(vis)
        cols['usable'].append(usable)
        cols['speed_plan'].append(float((eps.get(eid) or {}).get('speed', np.nan)))
    ds = {}
    for k, v in cols.items():
        if k in ('hist', 'fut', 'robot', 'robot_yaw'):
            ds[k] = np.stack(v).astype(np.float32)
        elif k == 'crop':
            ds[k] = np.stack(v).astype(np.int8)
        elif k in ('ev_future', 'ev_visible', 'usable'):
            ds[k] = np.array(v, dtype=bool)
        elif k in ('t_rel', 't_to_event', 'speed_plan'):
            ds[k] = np.array(v, dtype=np.float64)
        else:
            ds[k] = np.array(v, dtype=str)
    ds['env_key'] = np.array([f'{s}/{e}' for s, e in zip(ds['scen'], ds['env'])], dtype=str)
    ds['fold'] = make_folds(ds['env_key'], ds['scen'], k=folds, seed=seed)
    order = np.argsort(ds['ids'], kind='stable')                # fixed sample order => predictions align by position
    ds = {k: v[order] for k, v in ds.items()}
    if verbose:
        print(f'[data] {len(ds["ids"])} samples from {len(found)} environments, {int(ds["usable"].sum())} usable'
              + (f', {skipped} skipped' if skipped else ''))
    if cache:
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, sig=np.array(sig), **ds)
        except OSError:
            pass
    return ds