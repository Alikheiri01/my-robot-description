"""
motion/baselines.py -- predictors that need no deep learning. numpy only.

    stand       "stays where it is"
    cvm2, cvm4  constant velocity from the last 2 / 4 history steps
    ridge_hist  ridge regression on the history only (a correction over cvm4), trained on the other folds.
                The simplest honest "history-only" predictor: the transformer has to beat it to be worth anything.
"""
import numpy as np

from .data import DT, N_FUT, flip_arrays


def cvm(hist, k):
    v = (hist[:, -1, :2] - hist[:, -1 - k, :2]) / (DT * k)
    return v[:, None, :] * (DT * np.arange(1, N_FUT + 1))[None, :, None]


def hist_features(hist):
    """Features that do not use the scene: relative positions, velocities, heading (cos/sin)."""
    p = hist[:, :-1, :2]                                           # the last row is the origin
    v = np.diff(hist[:, :, :2], axis=1) / DT
    yaw = hist[:, :-1, 2]
    return np.concatenate([p.reshape(len(hist), -1), v.reshape(len(hist), -1),
                           np.cos(yaw), np.sin(yaw)], axis=1)


def ridge_fit_predict(h_tr, f_tr, h_te, alpha=10.0):
    dummy = np.zeros((len(h_tr), 80, 80), np.int8)
    h2, _, f2 = flip_arrays(h_tr, dummy, f_tr)                    # mirrored copies double the training data
    H = np.concatenate([h_tr, h2])
    F = np.concatenate([f_tr, f2])
    X = hist_features(H)
    Y = (F - cvm(H, 4)).reshape(len(H), -1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    Ym = Y.mean(0)
    W = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ (Y - Ym))
    Xt = (hist_features(h_te) - mu) / sd
    return cvm(h_te, 4) + (Xt @ W + Ym).reshape(len(h_te), N_FUT, 2)


def baseline_predictions(ds):
    """-> dict name -> (N,12,2) predictions (nan for samples that are not usable). Ridge is out-of-fold."""
    n = len(ds['ids'])
    u = ds['usable']
    h = ds['hist']
    out = {'stand': np.zeros((n, N_FUT, 2)), 'cvm2': cvm(h, 2), 'cvm4': cvm(h, 4)}
    rp = np.full((n, N_FUT, 2), np.nan)
    for k in sorted(set(ds['fold'])):
        te = u & (ds['fold'] == k)
        tr = u & (ds['fold'] != k)
        if te.any() and tr.any():
            rp[te] = ridge_fit_predict(h[tr], ds['fut'][tr, :, :2], h[te])
    out['ridge_hist'] = rp
    for k in out:
        out[k] = np.where(u[:, None, None], out[k], np.nan)
    return out
