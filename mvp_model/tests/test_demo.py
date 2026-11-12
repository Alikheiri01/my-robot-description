#!/usr/bin/env python3
"""tests/test_demo.py -- make_demo.py renders a GIF with stand-in predictors (no torch).  Must end with DEMO CHECK PASSED."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_demo as MD  # noqa: E402
from motion.baselines import cvm  # noqa: E402
from motion.data import load_dataset  # noqa: E402
from test_all import build  # noqa: E402


class Fake:
    seeds = [0]
    variant = 'full'

    def __init__(self, noise):
        self.noise = noise

    def predict(self, h, c, f, robot=None):
        return (cvm(h, 4) + self.noise)[None].astype(np.float32)


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'ds'
    build(root, 'determined', n_env=4)
    ds = load_dataset(root, cache=False, verbose=False)
    eps = MD.pick_episodes(ds)
    assert len(eps) == 3 and all(isinstance(e, str) for e in eps), eps            # B, C, E exist in the test data
    for ep in eps:
        idx = MD.episode_frames(ds, ep)
        assert len(idx) >= 2 and (np.diff(ds['t_rel'][idx]) >= 0).all(), 'frames in time order'
        res = MD.render_gif(ds, ep, idx, Fake(0.0), Fake(0.1), Path(tmp) / f'{ep}.gif')
        assert (Path(tmp) / f'{ep}.gif').stat().st_size > 5000 and res['map'] > 0
print('DEMO CHECK PASSED')