"""
motion/metrics.py -- the per-sample error numbers. numpy only.

For every sample and predictor (pred is (N,12,2): positions at +0.5 ... +6.0 s in the pedestrian frame):
    ade6  mean distance to the label over all 12 steps (6 s)
    fde6  distance at the last step (6 s)
    ade3  mean distance over the first 6 steps (3 s)        <- PRIMARY: this is where the crop can still see the cause
    fde3  distance at step 6 (3 s)
    occ   share of the 12 predicted points that lie in an OCCUPIED crop cell (0..1). The label itself is almost never
          in an occupied cell, so a high value means the prediction goes through a wall/obstacle (what an MPC fears)
"""
import numpy as np

from .data import X0, Y0, RES, N_FUT

H3 = 6                       # steps that make 3 s


def occupied_share(pred, crop):
    x, y = np.nan_to_num(pred[:, :, 0], nan=1e6), np.nan_to_num(pred[:, :, 1], nan=1e6)   # nan = not predicted = outside
    c = np.floor((x - X0) / RES).astype(int)
    r = np.floor((y - Y0) / RES).astype(int)
    ok = (c >= 0) & (c < 80) & (r >= 0) & (r < 80)
    cc, rr = np.clip(c, 0, 79), np.clip(r, 0, 79)
    v = crop[np.arange(len(pred))[:, None], rr, cc]
    return ((v == 100) & ok).sum(1) / float(N_FUT)


def all_metrics(pred, ds):
    d = np.linalg.norm(pred - ds['fut'][:, :, :2], axis=2)       # (N,12)
    return {'ade6': d.mean(1), 'fde6': d[:, -1], 'ade3': d[:, :H3].mean(1), 'fde3': d[:, H3 - 1],
            'occ': occupied_share(pred, ds['crop'])}


def label_occ(ds):
    return occupied_share(ds['fut'][:, :, :2], ds['crop'])
