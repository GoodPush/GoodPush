"""Banister impulse-response (fitness-fatigue) model fitted to your own data.

p(t) = p0 + k1 * sum_s w(s) e^{-(t-s)/tau1} - k2 * sum_s w(s) e^{-(t-s)/tau2}

w = daily TSS, p = performance marker (e.g. 20-min-equivalent power on days
with a near-maximal effort).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter


class Banister:
    """Least-squares fit (soft-L1 loss, a grid of starting time constants) of the five parameters."""
    MIN_DAYS = 15
    STARTS = [(t1, t2) for t1 in (30, 45, 60) for t2 in (5, 10, 15)]
    BOUNDS = ([0, 0, 0, 10, 2], [2000, 5, 10, 90, 30])

    @staticmethod
    def impulse(load: np.ndarray, tau: float) -> np.ndarray:
        """acc[i] = acc[i-1] * exp(-1/tau) + load[i], as a C-level IIR filter (no Python loop)."""
        return lfilter([1.0], [1.0, -np.exp(-1.0 / tau)], np.asarray(load, float))

    @classmethod
    def predict(cls, params, load: np.ndarray):
        """performance, fitness, fatigue per day; the response on day t uses load up to day t-1."""
        p0, k1, k2, t1, t2 = params
        f = np.concatenate([[0.0], cls.impulse(load, t1)[:-1]])
        g = np.concatenate([[0.0], cls.impulse(load, t2)[:-1]])
        return p0 + k1 * f - k2 * g, f, g

    @classmethod
    def fit_arrays(cls, load: np.ndarray, perf: np.ndarray):
        """the best least_squares result on the days where perf is finite (None if too few or none converged)."""
        m = np.isfinite(perf)
        if m.sum() < cls.MIN_DAYS:
            return None
        yv = perf[m]
        best = None
        for t1, t2 in cls.STARTS:                      # 9 starts: the fit has local minima
            try:
                r = least_squares(lambda x: cls.predict(x, load)[0][m] - yv, [np.nanmin(yv), 0.1, 0.2, t1, t2],
                                  bounds=cls.BOUNDS, loss="soft_l1", f_scale=10.0)
            except ValueError:
                continue
            if r.success and (best is None or r.cost < best.cost):
                best = r
        return best

    @classmethod
    def fit(cls, daily_tss: pd.Series, perf: pd.Series) -> dict:
        """Fit on days where `perf` is not NaN. Returns params + diagnostics."""
        load = daily_tss.fillna(0).to_numpy(float)
        y = perf.reindex(daily_tss.index).to_numpy(float)
        m = np.isfinite(y)
        res = {"n": int(m.sum()), "ok": False}
        best = cls.fit_arrays(load, y)
        if best is None:
            return res
        p0, k1, k2, t1, t2 = best.x
        pred, _, _ = cls.predict(best.x, load)
        ss = ((pred[m] - y[m]) ** 2).sum()
        st = ((y[m] - y[m].mean()) ** 2).sum()
        res.update(ok=True, p0=p0, k1=k1, k2=k2, tau_fitness=t1, tau_fatigue=t2,
                   r2=1 - ss / st if st > 0 else np.nan, rmse=float(np.sqrt(ss / m.sum())))
        # days after a single training impulse until net effect is maximal / turns positive
        if t1 > t2 and k1 > 0 and k2 > 0:
            r = (k2 * t1) / (k1 * t2)
            res["days_to_peak_after_load"] = float(np.log(r) * t1 * t2 / (t1 - t2)) if r > 1 else 0.0
            z = k2 / k1
            res["days_to_positive_effect"] = float(np.log(z) * t1 * t2 / (t1 - t2)) if z > 1 else 0.0
        res["fitted"] = pd.Series(pred, index=daily_tss.index)
        return res
