"""
motion/stats.py -- confidence intervals by bootstrapping over ENVIRONMENTS. numpy only.

Why environments: samples of one environment are almost the same scene, so 8000 samples are really ~95 scenes.
Each bootstrap round draws environments with replacement (inside each scenario, so every scenario keeps its size),
recomputes the number, and the 2.5 % / 97.5 % percentiles of 2000 rounds give the 95 % interval.
"""
import numpy as np

B_BOOT = 2000


class Boot:
    def __init__(self, ds, seed=0):
        self.envs, self.env_idx = np.unique(ds['env_key'], return_inverse=True)
        self.env_scen = np.array([e.split('/')[0] for e in self.envs])
        self.rng = np.random.RandomState(seed)

    def agg(self, vals, mask):
        ok = mask & np.isfinite(vals)
        s = np.bincount(self.env_idx[ok], weights=vals[ok], minlength=len(self.envs))
        c = np.bincount(self.env_idx[ok], minlength=len(self.envs)).astype(float)
        return s, c

    def draws(self, present):
        cols = []
        for sc in sorted(set(self.env_scen[present])):
            grp = present[self.env_scen[present] == sc]
            cols.append(grp[self.rng.randint(0, len(grp), (B_BOOT, len(grp)))])
        return np.concatenate(cols, axis=1)

    def ci(self, stat, parts):
        """stat(sel) -> (B,) values; parts: list of (sum, count) per environment.
        Returns (point, lo, hi, n_samples, n_envs)."""
        c_all = sum(c for _, c in parts)
        present = np.nonzero(c_all > 0)[0]
        if len(present) < 2:
            return (np.nan,) * 3 + (int(c_all.sum()), len(present))
        point = stat(present[None, :])[0]
        with np.errstate(all='ignore'):
            d = stat(self.draws(present))
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return point, lo, hi, int(c_all.sum()), len(present)

    def mean(self, vals, mask):
        s, c = self.agg(vals, mask)
        return self.ci(lambda sel: s[sel].sum(1) / c[sel].sum(1), [(s, c)])

    def rel_diff(self, v_test, v_ref, mask):
        """(mean(test) - mean(ref)) / mean(ref) on the same samples: the relative change, e.g. -0.10 = 10 % better."""
        st, c = self.agg(v_test, mask)
        sr, _ = self.agg(v_ref, mask)
        return self.ci(lambda sel: (st[sel].sum(1) - sr[sel].sum(1)) / sr[sel].sum(1), [(st, c)])

    def diff_of_means(self, va, ma, vb, mb):
        sa, ca = self.agg(va, ma)
        sb, cb = self.agg(vb, mb)
        return self.ci(lambda sel: sa[sel].sum(1) / ca[sel].sum(1) - sb[sel].sum(1) / cb[sel].sum(1),
                       [(sa, ca), (sb, cb)])


def fmt(r, nd=2):
    p, lo, hi = r[:3]
    return 'n/a' if not np.isfinite(p) else f'{p:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]'
