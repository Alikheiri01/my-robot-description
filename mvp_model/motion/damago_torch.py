"""
motion/damage_torch.py -- random damage of the INPUT during training (torch). Used by train.py for the variants
full_dmg / hist_dmg ("trained with damage"). The same kinds of damage that inspect_model.py tests (numpy version in
motion/damage.py), at levels about the size of a good real system:
    15 % of the samples: no map at all (all unknown)
    the others: 50 % get a map shift (up to 0.3 m, random direction) and a rotation (up to 5 degrees);
                every sample gets 0-3 % wrong cells
    every sample: tracker noise on the history positions, sigma = 0 (40 %), 0.05 m (30 %), 0.1 m (30 %); the heading is
                recomputed from the noisy points; the history is re-anchored at the noisy last point and the label
                is expressed in that same frame (label - noisy anchor), exactly what a real system would see.
The validation set used for early stopping stays CLEAN.
"""
import math

import torch
import torch.nn.functional as F

UNK, FREE, OCC = -1, 0, 100
P_MAP_DROP, P_GEOM = 0.15, 0.5
MAX_SHIFT, MAX_ROT_DEG, MAX_WRONG = 0.3, 5.0, 0.03
NOISE_LEVELS, NOISE_P = (0.0, 0.05, 0.1), (0.4, 0.3, 0.3)
X0, Y0, RES, N = -2.0, -4.0, 0.1, 80


def geometric(crop, dx, dy, deg):
    """crop (B,80,80) int8; dx, dy (metres) and deg (B,) tensors. The map content is rotated by `deg` around the
    pedestrian and then moved by (dx, dy). Cells that come in from outside are unknown. Zeros = identity."""
    B, dev = crop.shape[0], crop.device
    cx = X0 + (torch.arange(N, device=dev) + 0.5) * RES
    cy = Y0 + (torch.arange(N, device=dev) + 0.5) * RES
    Y, X = torch.meshgrid(cy, cx, indexing='ij')                       # (80,80), X along columns, Y along rows
    a = deg * math.pi / 180.0
    ca, sa = torch.cos(a)[:, None, None], torch.sin(a)[:, None, None]
    xd, yd = X[None] - dx[:, None, None], Y[None] - dy[:, None, None]  # undo the shift
    xs, ys = ca * xd + sa * yd, -sa * xd + ca * yd                      # undo the rotation: source position
    grid = torch.stack([2 * (xs - X0) / (N * RES) - 1, 2 * (ys - Y0) / (N * RES) - 1], dim=-1)   # (B,80,80,2)
    src = (crop.float() + 1.0)[:, None]                                 # unknown -> 0, free -> 1, occupied -> 101
    out = F.grid_sample(src, grid, mode='nearest', padding_mode='zeros', align_corners=False)[:, 0]
    return (out.round() - 1.0).to(crop.dtype)                           # outside -> 0 - 1 = -1 = unknown


def wrong_cells(crop, rate):
    """Every cell becomes a DIFFERENT random state with probability rate (B,)."""
    hit = torch.rand(crop.shape, device=crop.device) < rate[:, None, None]
    idx = torch.where(crop == UNK, 0, torch.where(crop == FREE, 1, 2))
    new = (idx + torch.randint(1, 3, crop.shape, device=crop.device)) % 3
    states = torch.tensor([UNK, FREE, OCC], dtype=crop.dtype, device=crop.device)
    return torch.where(hit, states[new], crop)


def noisy_history(h, sigma):
    """h (B,8,3), sigma (B,). Returns (h_noisy re-anchored with the last row = 0, shift (B,2) = noisy anchor in the true frame)."""
    p = h[:, :, :2] + torch.randn_like(h[:, :, :2]) * sigma[:, None, None]
    shift = p[:, -1, :].clone()
    p = p - shift[:, None, :]
    d = torch.gradient(p, dim=1)[0]
    yaw_new = torch.atan2(d[..., 1], d[..., 0])
    use = (sigma > 0)[:, None] & (d.norm(dim=2) >= 1e-3)
    yaw = torch.where(use, yaw_new, h[:, :, 2])
    return torch.cat([p, yaw[..., None]], dim=2), shift


def damage_batch(h, c, f, r):
    """h (B,8,3), c (B,80,80) int8, f (B,12,3), r (B,2) -> damaged copies (same shapes)."""
    B, dev = h.shape[0], h.device
    u = lambda: torch.rand(B, device=dev)
    drop = u() < P_MAP_DROP
    geo = (~drop) & (u() < P_GEOM)
    mag, ang = u() * MAX_SHIFT, u() * 2 * math.pi
    dx, dy = mag * torch.cos(ang) * geo, mag * torch.sin(ang) * geo
    deg = (u() * 2 - 1) * MAX_ROT_DEG * geo
    c2 = torch.where(geo[:, None, None], geometric(c, dx, dy, deg), c)
    c2 = wrong_cells(c2, u() * MAX_WRONG * (~drop))
    c2 = torch.where(drop[:, None, None], torch.full_like(c2, UNK), c2)
    levels = torch.tensor(NOISE_LEVELS, device=dev, dtype=h.dtype)
    sigma = levels[torch.multinomial(torch.tensor(NOISE_P, device=dev), B, replacement=True)]
    h2, shift = noisy_history(h, sigma)
    f2 = f.clone()
    f2[:, :, :2] = f2[:, :, :2] - shift[:, None, :]
    return h2, c2, f2, r - shift