#!/usr/bin/env python3
"""tests/test_figures.py -- figures/make_figures.py: every number-only figure draws; token and attention plots draw with fake data.
Ends with FIGURES CHECK PASSED."""
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / 'figures'))
import make_figures as MF  # noqa: E402

for fn in MF.NUMBER_ONLY.values():
    fn()
rng = np.random.RandomState(0)
crop = np.full((80, 80), -1, np.int8)
crop[20:60, 0:60] = 0
crop[20:24, :] = 100
hist = np.stack([np.linspace(-2, 0, 8), np.zeros(8)], 1)
fut = np.stack([np.linspace(0.3, 3, 12), np.linspace(0, 1.5, 12)], 1)
MF.draw_tokens(crop, hist, fut, 'test', 'test_tokens.png')
MF.plot_attention([dict(crop=crop, hist=hist, fut=fut, pred=fut * 0.9, w=rng.rand(100), title=f't{i}') for i in range(4)],
                  'test_attention.png')
for f in ('pipeline', 'architecture', 'results', 'robustness', 'limits', 'checks', 'test_tokens', 'test_attention'):
    p = MF.OUT / f'{f}.png'
    assert p.exists() and p.stat().st_size > 10000, f
for f in ('test_tokens', 'test_attention'):
    (MF.OUT / f'{f}.png').unlink()

# ---- dataset-based figures with fake predictors (the torch parts are NOT tested here)
import tempfile
from motion.data import load_dataset
from test_all import build
sys.path.insert(0, str(root))
with tempfile.TemporaryDirectory() as tmp:
    droot = Path(tmp) / 'ds'
    build(droot, 'determined', n_env=4)
    ds = load_dataset(droot, cache=False, verbose=False)
    cv = MF.cvm_pred(ds['hist']).astype(np.float32)
    N = len(ds['hist'])
    att = rng.dirichlet(np.ones(100), N).astype(np.float32)
    MF.draw_attention_figure(ds, att, cv * 0.95, cv * 1.05, 1.2, 'test_att_fig.png', letters='BC')
    MF.draw_counterfactual(ds, lambda h, c, f: (MF.cvm_pred(h) * 0.9).astype(np.float32), 'test_cf.png')
    i = MF.pick_samples(ds, 'B', 0.0, 6.0, 1)[0]
    ped, rob, env = MF.sample_world(droot, ds, i)
    MF.draw_input(ped, rob, env, ds['crop'][i], ds['hist'][i], ds['fut'][i], 'test_input.png')
    mp = MF.OUT
    for f in ('test_att_fig', 'test_cf', 'test_input'):
        p = mp / f'{f}.png'
        assert p.exists() and p.stat().st_size > 10000, f
        p.unlink()
# ---- environment drawings: real dataset folder if it is there (mvp_v1, else dev)
for cand in ('../mvp_collection/datasets/mvp_v1', '../mvp_collection/datasets/dev'):
    r = (root / cand).resolve()
    if r.exists():
        class A:
            pass
        a = A()
        a.root, a.pick, a.scenario = str(r), 0, 'B'
        MF.fig_scenarios(a)
        MF.fig_randomization(a)
        assert (MF.OUT / 'scenarios.png').stat().st_size > 10000 and (MF.OUT / 'randomization_B.png').stat().st_size > 10000
        break
else:
    print('(no dataset folder found: scenarios / randomization not tested)')
print('FIGURES CHECK PASSED')