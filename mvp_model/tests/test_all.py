#!/usr/bin/env python3
"""
tests/test_all.py -- step M1: does the data layer + evaluation work, and can it tell a real effect from none?
No simulator, no torch.  Run from mvp_model/ :
    python3 tests/test_all.py            # must end with ALL TESTS PASSED  (about 30 s)

It builds two SYNTHETIC datasets with the real folder layout and sample format:
    "determined"  the crop shows a marker that fully decides what the pedestrian does
                  (B: the turn side and the distance to the turn; C: the distance to the wall where it stops)
    "noise"       the same markers, but placed at random: the crop carries NO information about the label
A small numpy learner (k nearest neighbours; test-only) is trained through the same 5 environment-grouped folds, once
with and once without the crop features. The evaluation must then say
    determined: H1 B PASS and H4 B PASS (shuffling the crops inside the scenario hurts)
    noise:      H1 B FAIL
It also tests: the mirror augmentation (flip y, yaw, label AND the crop rows consistently; CVM of the mirrored history is
the mirrored CVM), the folds (an environment never in two folds), the cache, and the baselines.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from motion.baselines import baseline_predictions, cvm  # noqa: E402
from motion.data import DT, N_FUT, X0, Y0, RES, flip_arrays, load_dataset  # noqa: E402
from motion.metrics import all_metrics  # noqa: E402


# ----------------------------------------------------------------------------------------------- synthetic world
def make_walk(kind, v, side, t_ev, n=48):
    t = np.arange(n) * DT
    psi = np.zeros(n)
    sp = np.full(n, v)
    if kind == 'B':
        psi = np.clip((t - t_ev) / 2.0, 0, 1) * (np.pi / 2) * side
    if kind == 'C':
        sp = v * (1 - np.clip((t - t_ev) / 2.0, 0, 1))
    if kind == 'A':
        psi = np.where((t > t_ev) & (t < t_ev + 1.0), 0.25 * side, 0.0)
    if kind == 'E':
        psi = 0.02 * t * side
    xy = np.zeros((n, 2))
    for i in range(1, n):
        xy[i] = xy[i - 1] + sp[i] * DT * np.array([np.cos(psi[i]), np.sin(psi[i])])
    return t, xy, psi


def cell(x, y):
    return int(np.floor((y - Y0) / RES)), int(np.floor((x - X0) / RES))      # (row, col): rows are y, cols are x


def make_crop(markers):
    """Hallway of free cells with two walls; `markers` = list of (x, y, half_x, half_y) occupied blocks (ped frame)."""
    crop = np.full((80, 80), -1, np.int8)
    crop[28:52, 0:70] = 0
    crop[26:28, 0:70] = 100
    crop[52:54, 0:70] = 100
    for x, y, hx, hy in markers:
        r, c = cell(x, y)
        r0, r1 = max(r - int(hy / RES), 0), min(r + int(hy / RES) + 1, 80)
        c0, c1 = max(c - int(hx / RES), 0), min(c + int(hx / RES) + 1, 80)
        crop[r0:r1, c0:c1] = 100
    return crop


def build(root, planted, n_env=12, seed=0):
    rng = np.random.RandomState(seed)
    for sc, kind in (('A_detour', 'A'), ('B_corner', 'B'), ('C_stop', 'C'), ('E_walk', 'E')):
        for e in range(n_env):
            name = f'env_{kind}_{e:03d}'
            ed = Path(root) / sc / name
            run = ed / 'runs' / 'r1'
            (run / 'samples').mkdir(parents=True)
            (run / 'episodes').mkdir()
            (run / 'report.txt').write_text('OVERALL: PASS\n')
            eps = []
            for k in range(4):
                eid = f'{name}_ep{k}'
                v = float(rng.uniform(0.8, 1.5))
                side = 1 if rng.rand() < 0.5 else -1
                t_ev = 12.0
                t, xy, psi = make_walk(kind, v, side, t_ev)
                eps.append({'episode_id': eid, 'speed': v, 'robot': {'mode': 'follow' if k % 2 == 0 else 'watch'}})
                t_end = 22.0
                (run / 'episodes' / f'{eid}.json').write_text(json.dumps({'measured': {
                    't_walk_start_sim': 100.0, 't_end_after_walk_start_sec': t_end, 'outcome': 'ok',
                    **({'t_event_after_walk_start_sec': t_ev} if kind != 'E' else {})}}))
                # where the marker is (world frame): decides the label in "determined", random in "noise"
                if planted == 'determined':
                    m_side, m_x = side, v * t_ev                       # B: turn side and turn position
                    wall_x = xy[-1, 0] + 0.3                           # C: where the walk ends
                else:
                    m_side, m_x = (1 if rng.rand() < 0.5 else -1), rng.uniform(5, 15)
                    wall_x = rng.uniform(5, 25)
                for ia in range(8, 40):
                    ta = t[ia]
                    if ta < 4.0 or ta > t_end - 6.5 or (ta * 2) % 2 != 0:
                        continue
                    yaw_a = psi[ia]
                    R = np.array([[np.cos(yaw_a), np.sin(yaw_a)], [-np.sin(yaw_a), np.cos(yaw_a)]])
                    hh = (xy[ia - 7:ia + 1] - xy[ia]) @ R.T
                    ff = (xy[ia + 1:ia + 1 + N_FUT] - xy[ia]) @ R.T
                    hist = np.concatenate([hh, (psi[ia - 7:ia + 1] - yaw_a)[:, None]], 1)
                    fut = np.concatenate([ff, (psi[ia + 1:ia + 1 + N_FUT] - yaw_a)[:, None]], 1)
                    markers = []
                    if kind == 'B':
                        markers.append(tuple(R @ (np.array([m_x, m_side * 2.0]) - xy[ia])) + (0.25, 0.25))
                    if kind == 'C':
                        markers.append(tuple(R @ (np.array([wall_x, 0.0]) - xy[ia])) + (0.15, 1.0))
                    seen = (rng.rand(8) > 0.08).astype(np.uint8)
                    np.savez(run / 'samples' / f'sample_{k}_{ia:03d}.npz',
                             trajectory_history=hist.astype(np.float32), future_target=fut.astype(np.float32),
                             crop=make_crop(markers), ped_anchor_odom=np.array([xy[ia, 0], xy[ia, 1], yaw_a]),
                             robot_pose_odom=np.array([xy[ia, 0] - 5.0, 0.0, 0.0]),
                             history_times=t[ia - 7:ia + 1] + 100, future_times=t[ia + 1:ia + 1 + N_FUT] + 100,
                             anchor_time_sim=np.float64(100.0 + ta), ped_seen_anchor=np.uint8(1),
                             ped_seen_history=seen, ped_seen_future=np.ones(N_FUT, np.uint8), gated=np.bool_(True),
                             schema=np.array('time_aligned_v2'), episode_id=np.array(eid))
            (ed / 'environment.json').write_text(json.dumps({'template': 'T1', 'obstacles': [], 'episodes': eps,
                                                             'params': {}}))


# ----------------------------------------------------------------------------------- test-only learner (no torch)
XX = X0 + RES * (np.arange(80) + 0.5)[None, :] * np.ones((80, 1))
YY = Y0 + RES * (np.arange(80) + 0.5)[:, None] * np.ones((1, 80))


def scene_feats(crop):
    """[side of the lateral marker (+1 left, -1 right, 0 none), its x, x of the wall across the corridor]"""
    occ = crop == 100
    lat = occ & (np.abs(YY) > 1.6) & (XX > -1)
    side, lx = (np.sign(YY[lat].mean()), XX[lat].mean()) if lat.sum() >= 6 else (0.0, 8.0)
    wall = occ & (np.abs(YY) < 1.0) & (XX > 0)
    wx = XX[wall].min() if wall.sum() >= 6 else 8.0
    return np.array([side, lx, wx])


def hist_feats(h):
    v = (h[:, -1, :2] - h[:, -5, :2]) / (DT * 4)
    return np.stack([np.linalg.norm(v, axis=1), v[:, 1]], 1)


def knn_predict(Xtr, Ytr, Xte, k=10):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    a, b = (Xtr - mu) / sd, (Xte - mu) / sd
    d = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * b @ a.T
    idx = np.argpartition(d, k, axis=1)[:, :k]
    return Ytr[idx].mean(1)


def oof_knn(ds, use_scene, scene_f):
    """Out-of-fold predictions with the dataset's own folds; scene_f (N,3) are the crop features to use."""
    n = len(ds['ids'])
    pred = np.full((n, N_FUT, 2), np.nan)
    H = hist_feats(ds['hist'])
    X = np.concatenate([H, 1.5 * scene_f], 1) if use_scene else H
    base = cvm(ds['hist'], 4)
    Y = (ds['fut'][:, :, :2] - base).reshape(n, -1)
    for k in sorted(set(ds['fold'])):
        te = ds['usable'] & (ds['fold'] == k)
        tr = ds['usable'] & (ds['fold'] != k)
        pred[te] = base[te] + knn_predict(X[tr], Y[tr], X[te]).reshape(-1, N_FUT, 2)
    return pred


def shuffled_features(ds, scene_f, seed=0):
    """crop features of another environment of the same scenario (the within-scenario shuffle of H4)"""
    rng = np.random.RandomState(seed)
    out = scene_f.copy()
    for sc in set(ds['scen']):
        idx = np.nonzero(ds['scen'] == sc)[0]
        for i in idx:
            other = idx[ds['env_key'][idx] != ds['env_key'][i]]
            out[i] = scene_f[other[rng.randint(len(other))]]
    return out


# ----------------------------------------------------------------------------------------------------- the tests
H1B = r'H1 B \(PRIMARY\)'
H4B = r'H4 B \(PRIMARY\)'


def fail(msg):
    print('TEST FAILED:', msg)
    sys.exit(1)


def run_eval(tmp, root, ds, scene_pred, ref_pred, shuf_pred, tag):
    m = ds['usable']
    files = {}
    for name, arr in (('scene', scene_pred), ('ref', ref_pred), ('shuf', shuf_pred)):
        files[name] = tmp / f'{tag}_{name}.npz'
        np.savez(files[name], ids=ds['ids'][m], pred=arr[m])
    rep = tmp / f'{tag}_report.txt'
    r = subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), str(root), '--ref', 'ref', '--test', 'scene',
                        '--preds', f'scene={files["scene"]}', '--preds', f'ref={files["ref"]}',
                        '--preds', f'scn_shuf={files["shuf"]}', '--shuf-scn', 'scn_shuf', '--out', str(rep)],
                       capture_output=True, text=True, cwd=tmp)
    if r.returncode != 0:
        fail('evaluate.py crashed:\n' + r.stderr[-1500:])
    return rep.read_text()


def verdict(txt, tag):
    m = re.search(tag + r'\s+(PASS|FAIL)', txt)
    if not m:
        fail(f'"{tag}" not in the report:\n' + txt[-2500:])
    return m.group(1)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='mm_test_'))
    try:
        results = {}
        for planted in ('determined', 'noise'):
            root = tmp / planted
            build(root, planted)
            ds = load_dataset(root, cache=True, cache_dir=tmp / f'cache_{planted}')

            if planted == 'determined':
                n = len(ds['ids'])
                u = ds['usable']
                # --- data layer
                assert ds['hist'].shape == (n, 8, 3) and ds['fut'].shape == (n, 12, 3) and ds['crop'].shape == (n, 80, 80)
                if not np.allclose(ds['hist'][:, -1], 0):
                    fail('last history row must be 0')
                if ds['ev_future'][ds['scen'] == 'E_walk'].any() or not (0.2 < ds['ev_future'][ds['scen'] == 'B_corner'].mean() < 0.9):
                    fail('event flags')
                tte = ds['t_to_event'][ds['ev_future']]
                if not ((tte > 0).all() and (tte <= 6).all()):
                    fail('time to event range')
                for ek in set(ds['env_key']):
                    if len(set(ds['fold'][ds['env_key'] == ek])) != 1:
                        fail('an environment is split over folds')
                for sc in set(ds['scen']):
                    if len(set(ds['fold'][ds['scen'] == sc])) != 5:
                        fail(f'scenario {sc} is not spread over all 5 folds')
                ds2 = load_dataset(root, cache=True, cache_dir=tmp / f'cache_{planted}')
                if not (np.array_equal(ds['ids'], ds2['ids']) and np.array_equal(ds['crop'], ds2['crop'])):
                    fail('cache differs')
                # --- baselines sanity: CVM is good on straight walking and bad on turns
                a = all_metrics(baseline_predictions(ds)['cvm4'], ds)['ade6']
                e_err, b_err = np.nanmean(a[u & (ds['scen'] == 'E_walk')]), np.nanmean(a[u & (ds['scen'] == 'B_corner') & ds['ev_future']])
                if not (e_err < 0.5 and b_err > 3 * e_err):
                    fail(f'cvm4 sanity: E {e_err:.2f}, B event {b_err:.2f}')
                # --- MIRROR tests
                B = np.nonzero(u & (ds['scen'] == 'B_corner'))[0][::7]
                h, c, f = ds['hist'][B], ds['crop'][B], ds['fut'][B]
                h2, c2, f2 = flip_arrays(h, c, f)
                h3, c3, f3 = flip_arrays(h2, c2, f2)
                if not (np.allclose(h3, h) and np.array_equal(c3, c) and np.allclose(f3, f)):
                    fail('mirror twice must be the identity')
                m_cvm = cvm(h2, 4)
                r_cvm = cvm(h, 4).copy()
                r_cvm[..., 1] *= -1
                if not np.allclose(m_cvm, r_cvm, atol=1e-6):
                    fail('CVM of the mirrored history must be the mirrored CVM')
                if not (np.allclose(h2[..., 1], -h[..., 1]) and np.allclose(h2[..., 2], -h[..., 2])
                        and np.allclose(f2[..., 1], -f[..., 1]) and np.allclose(h2[..., 0], h[..., 0])):
                    fail('mirror must flip y and yaw of history and label and nothing else')
                # the planted rule "marker side == turn side" must survive the mirror (this tests the crop-row flip)
                s1 = np.array([scene_feats(x)[0] for x in c])
                s2 = np.array([scene_feats(x)[0] for x in c2])
                if not np.array_equal(s2, -s1):
                    fail('mirror: the marker side seen in the crop must flip sign')
                turn = np.sign(f[:, -1, 1])
                ok1 = np.mean((s1 * turn)[(s1 != 0) & (np.abs(f[:, -1, 1]) > 0.5)] > 0)
                ok2 = np.mean((s2 * np.sign(f2[:, -1, 1]))[(s2 != 0) & (np.abs(f2[:, -1, 1]) > 0.5)] > 0)
                if not (ok1 > 0.9 and abs(ok1 - ok2) < 1e-9):
                    fail(f'mirror: crop side and label side must agree before ({ok1:.2f}) and after ({ok2:.2f})')
                print('mirror: twice = identity, CVM(mirror) = mirror(CVM), crop rows flip together with the label: ok')

            # --- planted effect through the whole evaluation
            n = len(ds['ids'])
            sf = np.array([scene_feats(c) for c in ds['crop']])
            ref = oof_knn(ds, False, sf)
            scene = oof_knn(ds, True, sf)
            shuf = oof_knn(ds, True, shuffled_features(ds, sf))
            txt = run_eval(tmp, root, ds, scene, ref, shuf, planted)
            results[planted] = txt
            v_h1b, v_h1c = verdict(txt, H1B), verdict(txt, 'H1 C')
            v_h4b, v_h2 = verdict(txt, H4B), verdict(txt, 'H2 no harm')
            print(f'[{planted}] H1 B: {v_h1b}   H1 C: {v_h1c}   H4 B: {v_h4b}   H2: {v_h2}')

        if verdict(results['determined'], H1B) != 'PASS':
            fail('determined crop: H1 B should PASS')
        if verdict(results['determined'], H4B) != 'PASS':
            fail('determined crop: H4 B should PASS (shuffling the crops inside the scenario must hurt)')
        if verdict(results['noise'], H1B) != 'FAIL':
            fail('noise crop: H1 B should FAIL')
        print(results['determined'][results['determined'].find('HYPOTHESES'):][:2600])
        print('ALL TESTS PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
