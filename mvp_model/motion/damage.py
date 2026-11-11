"""
motion/damage.py -- ways to damage the INPUT of a trained model, and the counterfactual map edits. numpy only.

Used by inspect_model.py (M5). Everything works on whole arrays: crop (N,80,80) int8 (-1 unknown, 0 free, 100 occupied),
hist (N,8,3) in the pedestrian frame. Every function takes a seed, so a table can be reproduced exactly.
Crop geometry: cell (row r, col c) has its centre at x = X0 + (c+0.5)*RES, y = Y0 + (r+0.5)*RES; the pedestrian is at (0,0).
"""
import numpy as np

from .data import DT, N_FUT, RES, X0, Y0

UNK, FREE, OCC = -1, 0, 100


def cell_centres():
    c = X0 + (np.arange(80) + 0.5) * RES
    r = Y0 + (np.arange(80) + 0.5) * RES
    X, Y = np.meshgrid(c, r)                                     # (80,80): X varies along columns, Y along rows
    return X, Y


# ----------------------------------------------------------------------------------------------- map damage
def no_map(crop):
    """No map at all: every cell unknown."""
    return np.full_like(crop, UNK)


def shift_map(crop, metres, seed=0):
    """The map is misplaced by `metres` in a random direction (per sample). Cells that move in from outside are unknown."""
    rng = np.random.RandomState(seed)
    ang = rng.uniform(0, 2 * np.pi, len(crop))
    dr = np.rint(metres / RES * np.sin(ang)).astype(int)
    dc = np.rint(metres / RES * np.cos(ang)).astype(int)
    out = np.full_like(crop, UNK)
    for i in range(len(crop)):
        r0, r1 = max(0, dr[i]), min(80, 80 + dr[i])               # destination rows
        c0, c1 = max(0, dc[i]), min(80, 80 + dc[i])
        if r1 > r0 and c1 > c0:
            out[i, r0:r1, c0:c1] = crop[i, r0 - dr[i]:r1 - dr[i], c0 - dc[i]:c1 - dc[i]]
    return out


def noise_cells(crop, p, seed=0):
    """A share p of all cells is replaced by a different, random state (free / occupied / unknown)."""
    rng = np.random.RandomState(seed)
    hit = rng.rand(*crop.shape) < p
    states = np.array([UNK, FREE, OCC], dtype=crop.dtype)
    idx = np.where(crop == UNK, 0, np.where(crop == FREE, 1, 2))
    pick = states[(idx + rng.randint(1, 3, crop.shape)) % 3]       # always a DIFFERENT state
    out = crop.copy()
    out[hit] = pick[hit]
    return out


def rotate_map(crop, degrees, seed=0):
    """The map is turned by +-degrees (sign random per sample) around the pedestrian; empty corners are unknown."""
    rng = np.random.RandomState(seed)
    X, Y = cell_centres()
    out = np.full_like(crop, UNK)
    sign = rng.choice([-1.0, 1.0], len(crop))
    for s in (-1.0, 1.0):
        idx = np.nonzero(sign == s)[0]
        if len(idx) == 0:
            continue
        a = np.deg2rad(degrees) * s
        c, sn = np.cos(a), np.sin(a)
        xs, ys = c * X + sn * Y, -sn * X + c * Y                  # source position of every destination cell (inverse rotation)
        sc = np.floor((xs - X0) / RES).astype(int)
        sr = np.floor((ys - Y0) / RES).astype(int)
        ok = (sc >= 0) & (sc < 80) & (sr >= 0) & (sr < 80)
        scc, srr = np.clip(sc, 0, 79), np.clip(sr, 0, 79)
        tmp = crop[idx][:, srr, scc]
        tmp[:, ~ok] = UNK
        out[idx] = tmp
    return out


# --------------------------------------------------------------------------------------------- history damage
def noisy_history(hist, sigma, seed=0):
    """A tracker with position noise sigma (m) per history point. Returns (hist_in, anchor_shift):
    hist_in  = noisy positions re-anchored so the last row is (0,0) (as the real system would do), yaw recomputed from
               the noisy positions (central differences), because a tracker has no better heading than that;
    anchor_shift = the noisy position of the anchor in the TRUE frame: add it to the predictions to compare with the label.
    Limitation: the frame axes (the crop orientation) stay at the true heading; heading error of the frame is covered
    by the map-rotation test."""
    if sigma <= 0:
        return hist.copy(), np.zeros((len(hist), 2), np.float32)
    rng = np.random.RandomState(seed)
    p = hist[:, :, :2] + rng.normal(0, sigma, hist[:, :, :2].shape)
    shift = p[:, -1, :].copy()
    p = p - shift[:, None, :]
    d = np.gradient(p, axis=1)
    yaw = np.arctan2(d[:, :, 1], d[:, :, 0])
    still = np.linalg.norm(d, axis=2) < 1e-3
    yaw = np.where(still, hist[:, :, 2], yaw)
    out = np.concatenate([p, yaw[:, :, None]], axis=2).astype(np.float32)
    return out, shift.astype(np.float32)


# -------------------------------------------------------------------------------- counterfactual map edits (H8)
def disk_mask(px, py, r):
    X, Y = cell_centres()
    return (X - px) ** 2 + (Y - py) ** 2 <= r * r


def remove_obstacle(crop_i, px, py, r=1.5):
    """H8a: occupied cells within r of (px,py) become free. Unknown cells stay unknown (we never invent what is behind).
    Returns (new_crop, number_of_cells_changed)."""
    m = disk_mask(px, py, r) & (crop_i == OCC)
    out = crop_i.copy()
    out[m] = FREE
    return out, int(m.sum())


def insert_wall(crop_i, robot_xy, d, direction, half_len=0.8, half_thick=0.15, min_known=0.6):
    """H8b: a thin wall (2*half_len long, 2*half_thick thick) across the walking direction `direction` (unit vector) at
    distance d from the pedestrian. Only KNOWN cells can become wall. Cells behind the wall, seen from the robot, become
    unknown (the grey shadow moves with the wall). Returns (new_crop, wall_mask) or (None, None) if the place is
    mostly unknown or outside the crop."""
    X, Y = cell_centres()
    u = np.asarray(direction, float)
    n = np.array([-u[1], u[0]])
    cx, cy = d * u[0], d * u[1]
    a = (X - cx) * u[0] + (Y - cy) * u[1]                         # along the walking direction
    b = (X - cx) * n[0] + (Y - cy) * n[1]                         # across it
    wall = (np.abs(a) <= half_thick) & (np.abs(b) <= half_len)
    if wall.sum() < 10:
        return None, None
    known = (crop_i != UNK) & wall
    if known.sum() < min_known * wall.sum():
        return None, None
    out = crop_i.copy()
    wall = known                                                  # only known cells become wall
    # shadow: segment robot -> cell crosses the wall box (slab test in wall coordinates)
    rx, ry = robot_xy
    ra, rb = (rx - cx) * u[0] + (ry - cy) * u[1], (rx - cx) * n[0] + (ry - cy) * n[1]
    da, db = a - ra, b - rb                                       # direction of the segment, to the cell
    with np.errstate(divide='ignore', invalid='ignore'):
        t_lo = np.zeros_like(a)
        t_hi = np.ones_like(a)
        for r0, dd, h in ((ra, da, half_thick), (rb, db, half_len)):
            t1, t2 = (-h - r0) / dd, (h - r0) / dd
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            par = np.abs(dd) < 1e-9                               # parallel to the slab: inside or never
            inside = np.abs(r0) <= h
            lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
            hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
            t_lo, t_hi = np.maximum(t_lo, lo), np.minimum(t_hi, hi)
    hit = t_lo <= t_hi
    behind = hit & (t_lo > 0) & (np.abs(a) > half_thick) & ~wall # the cell lies beyond the wall box as seen from the robot
    out[behind & (out != UNK)] = UNK
    out[wall] = OCC
    return out, wall