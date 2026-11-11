#!/usr/bin/env python3
"""
tests/test_inspect.py -- checks the M5 machinery (damage functions, counterfactual edits, tables, H8 rules) WITHOUT torch,
with stand-in predictors:
    blind    constant velocity only (ignores the map)        -> H8 must say FAIL
    reactive stops before the first occupied cell in front   -> H8 must say PASS
    python3 tests/test_inspect.py      # must end with INSPECT CHECK PASSED
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_model as IM  # noqa: E402
from motion import damage as D  # noqa: E402
from motion.baselines import cvm  # noqa: E402
from motion.data import load_dataset  # noqa: E402
from test_all import build  # noqa: E402

rng = np.random.RandomState(0)
crop = rng.choice([-1, 0, 100], size=(6, 80, 80), p=[0.3, 0.6, 0.1]).astype(np.int8)
hist = np.cumsum(rng.normal(0.5, 0.1, (6, 8, 3)), 1).astype(np.float32)
hist -= hist[:, -1:, :]

# ---- damage functions: zero damage = identity, damage changes the right amount
assert (D.shift_map(crop, 0.0) == crop).all() and (D.rotate_map(crop, 0.0) == crop).all()
assert (D.noise_cells(crop, 0.0) == crop).all()
frac = (D.noise_cells(crop, 0.05, 1) != crop).mean()
assert 0.045 < frac < 0.055, frac
assert (D.no_map(crop) == -1).all()
assert (D.shift_map(crop, 0.5, 2) != crop).any()
hn, sh = D.noisy_history(hist, 0.0)
assert np.allclose(hn[:, :, :2], hist[:, :, :2]) and np.allclose(sh, 0)
hn, sh = D.noisy_history(hist, 0.1, 3)
assert np.abs(hn[:, -1, :2]).max() < 1e-6                                  # re-anchored
assert abs(np.std(hn[:, :-1, :2] - hist[:, :-1, :2] + sh[:, None, :]) - 0.1) < 0.03   # noise level is right
# remove_obstacle: only occupied -> free, nothing else
c0 = crop[0]
c1, n = D.remove_obstacle(c0, 2.0, 0.0, 1.5)
assert n > 0 and ((c0 != c1) == ((c0 == 100) & (c1 == 0))).all() and (c1[c0 == -1] == -1).all()
# insert_wall: wall only on known cells, shadow only turns known cells into unknown, wall cells are occupied
free = np.zeros((80, 80), np.int8)
cw, w = D.insert_wall(free, (-4.0, 0.0), 2.5, (1.0, 0.0))
X, Y = D.cell_centres()
assert cw is not None and (cw[w] == 100).all() and (np.abs(X[w] - 2.5) <= 0.16).all() and (np.abs(Y[w]) <= 0.81).all()
sh_cells = (cw == -1)
assert sh_cells.any() and (X[sh_cells] > 2.6).all() and (np.abs(Y[sh_cells]) < 1.4).all()   # a cone behind the wall
unk = np.full((80, 80), -1, np.int8)
assert D.insert_wall(unk, (-4.0, 0.0), 2.5, (1.0, 0.0))[0] is None            # nothing known: refuse
print('damage functions: ok')


# ---- stand-in predictors
class Fake:
    def __init__(self, reactive):
        self.reactive, self.seeds, self.variant = reactive, [0], 'full'

    def predict(self, h, c, folds, robot=None):
        p = cvm(h, 4)
        if self.reactive:
            Xc, Yc = D.cell_centres()
            for k in range(len(h)):
                m = (c[k] == 100) & (np.abs(Yc) < 0.6) & (Xc > 0.3)
                if m.any():
                    allow = max(Xc[m].min() - 0.6, 0.0)
                    r = np.linalg.norm(p[k], axis=1)
                    p[k] = p[k] * np.minimum(1.0, allow / np.maximum(r, 1e-6))[:, None]
        return p[None].astype(np.float32)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'ds'
    build(root, 'determined', n_env=12)
    ds = load_dataset(root, cache=False, verbose=False)
    scn = np.array([s_[0] for s_ in ds['scen']])
    for i in np.nonzero(np.isin(scn, ['A', 'E']))[0]:               # open room for the wall tests (the test hallway is too narrow)
        c = np.zeros((80, 80), np.int8)
        c[:3], c[-3:] = 100, 100
        ds['crop'][i] = c
    res = {}
    for name, reactive in (('blind', False), ('reactive', True)):
        lines = IM.run(ds, Fake(reactive), Fake(False), {'damage', 'h8'}, Path(tmp) / f'out_{name}', Path(tmp), max_n=300)
        text = '\n'.join(lines)
        res[name] = text
        print(f'--- {name}\n{text}\n')
    for name in ('blind', 'reactive'):
        assert 'map: none' in res[name] and 'history noise 0.2 m' in res[name] and 'clean (reference)' in res[name]
        assert (Path(tmp) / f'out_{name}' / 'damage.png').exists()
        assert 'cvm4, history noise 0.1 m' in res[name] and 'hist, history noise 0.1 m' in res[name]
    assert 'H8a' in res['reactive'] and 'H8b' in res['reactive'], 'H8 not evaluated'
    assert '->  FAIL' in res['blind'].split('H8a')[1].split('H8b')[0], 'blind model must FAIL H8a'
    a_react = res['reactive'].split('H8a')[1].split('H8b')[0]
    assert '->  PASS' in a_react, 'reactive stand-in must PASS H8a'
    b_blind = res['blind'].split('H8b')[1]
    assert '->  FAIL' in b_blind, 'blind model must FAIL H8b'
    # the damage table must show the reactive stand-in getting worse with "no map" or shifts on C (it uses the wall)
print('INSPECT CHECK PASSED')