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
print('FIGURES CHECK PASSED')