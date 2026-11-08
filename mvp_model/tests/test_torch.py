#!/usr/bin/env python3
"""
tests/test_torch.py -- step M2: do the networks run?  (needs torch; run from mvp_model/)
    python3 tests/test_torch.py        # must end with TORCH CHECK PASSED

Checks: output shapes and parameter counts of every variant; at initialisation every model equals the constant-velocity
prediction; the torch mirror equals the numpy mirror (motion.data.flip_arrays); a mirrored input gives a mirrored
CVM; the coordinate encoding of the 100 scene tokens has the right layout.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motion.baselines import cvm as cvm_np  # noqa: E402
from motion.data import flip_arrays  # noqa: E402
from motion.models import VARIANTS, count_params, cvm_baseline, flip_batch, token_centres, N_FUT  # noqa: E402

torch.manual_seed(0)
hist = torch.randn(4, 8, 3).cumsum(1) * 0.2
hist = hist - hist[:, -1:, :]
crop = torch.tensor([-1, 0, 100], dtype=torch.int8)[torch.randint(0, 3, (4, 80, 80))]
fut = torch.randn(4, N_FUT, 3)

for name, make in VARIANTS.items():
    m = make().eval()
    out = m(hist, crop, torch.randn(4, 2))
    assert out.shape == (4, N_FUT, 2), (name, out.shape)
    if name.endswith('_direct'):
        assert torch.isfinite(out).all(), name
        assert not torch.allclose(out, cvm_baseline(hist), atol=1e-3), f'{name}: direct output must not be CVM-based'
        print(f'{name:14s} parameters {count_params(m):7d}   output {tuple(out.shape)}   finite, not CVM-based: ok')
    else:
        assert torch.allclose(out, cvm_baseline(hist), atol=1e-5), f'{name}: must equal CVM at initialisation'
        print(f'{name:14s} parameters {count_params(m):7d}   output {tuple(out.shape)}   equals CVM at init: ok')

# the torch mirror == the numpy mirror
h_t, c_t, f_t = flip_batch(hist, crop, fut)
h_n, c_n, f_n = flip_arrays(hist.numpy(), crop.numpy(), fut.numpy())
assert np.allclose(h_t.numpy(), h_n) and np.array_equal(c_t.numpy(), c_n) and np.allclose(f_t.numpy(), f_n)
# flipping twice is the identity
h2, c2, f2 = flip_batch(h_t, c_t, f_t)
assert torch.allclose(h2, hist) and torch.equal(c2, crop) and torch.allclose(f2, fut)
# CVM of the mirrored history == mirrored CVM
c_mir = cvm_baseline(h_t)
c_ref = cvm_baseline(hist).clone()
c_ref[..., 1] *= -1
assert torch.allclose(c_mir, c_ref, atol=1e-6)
assert np.allclose(cvm_np(h_n, 4), c_mir.numpy(), atol=1e-5)
print('mirror: torch == numpy, twice = identity, CVM(mirror) = mirror(CVM): ok')

# token layout: token 0 = top-left in the array = (x = -2 + 0.4, y = -4 + 0.4)
tc = token_centres(10)
assert tc.shape == (100, 2)
assert abs(tc[0, 0].item() - (-1.6 / 4)) < 1e-6 and abs(tc[0, 1].item() - (-3.6 / 4)) < 1e-6
assert abs(tc[99, 0].item() - (5.6 / 4)) < 1e-6 and abs(tc[99, 1].item() - (3.6 / 4)) < 1e-6
assert abs(tc[1, 0].item() - tc[0, 0].item() - 0.2) < 1e-6          # next token = next COLUMN = +0.8 m in x
print('scene token coordinates: ok')
print('TORCH CHECK PASSED')
