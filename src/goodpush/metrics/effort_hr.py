"""Did a ride's best effort go all-out? Heart rate around the end of its best window says.

A best power is only a measurement of your limit when you were at it: a 20-minute all-out effort ends with heart
rate at (or within a few beats of) your maximum, a 5-minute one nearly so. A 1-minute effort is over before heart
rate catches up, so its peak is read in the 45 s after and allowed further below your maximum. Efforts that pass
are "maxed" in the physio model (physio KINDS: all-out with high probability); the rest stay "maybe all-out"
(their ride kind's chance), and quantities only the power curve gives (FTP, CP, W') stay inferred.

All rides at once from the 1 Hz streams (io.streamstore.read_all): best windows by prefix sums, the best per ride
by one lexsort, peak heart rate per window by np.maximum.reduceat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class EffortHeartRate:
    DURATIONS = (60, 180, 300, 600, 1200)
    # where the peak heart rate is read, in s from the effort's last second: heart rate lags the power, the more
    # the shorter the effort; a long effort's own end (a sprint after it doesn't count)
    WINDOW = {60: (0, 45), 180: (-15, 20), 300: (-30, 15), 600: (-45, 10), 1200: (-60, 5)}
    # how far below your maximum heart rate an all-out effort's peak may be (bpm)
    MARGIN = {60: 10, 180: 7, 300: 5, 600: 5, 1200: 5}
    MIN_POWER_W = 50.0                     # a "best" window below this is no effort (no power, coasting)
    MAX_COAST = 0.05                       # an effort of 3 min or more with more of its seconds unpowered (descents,
    STEADY_FROM_S = 180                    # stops) is no all-out effort, whatever heart rate did

    @classmethod
    def table(cls, processed: Path) -> pd.DataFrame:
        """id, then per duration: effort_w_<d> (best power), effort_hr_<d> (peak heart rate around its end),
        effort_coast_<d> (share of its seconds without power)."""
        from ..io import streamstore
        try:
            ids, n, data = streamstore.read_all(processed, "ride", ["power", "heart_rate"])
        except (OSError, KeyError, ValueError):
            return pd.DataFrame(columns=["id"])
        if not len(ids):
            return pd.DataFrame(columns=["id"])
        return cls.from_streams(ids, n, data["power"], data["heart_rate"])

    @classmethod
    def from_streams(cls, ids, n, power, hr) -> pd.DataFrame:
        """the same from rides' streams laid end to end (`n` samples each)."""
        n = np.asarray(n, np.int64)
        P = np.nan_to_num(np.asarray(power, float))
        H = np.asarray(hr, float)
        H = np.where(np.isfinite(H) & (H > 30), H, -np.inf)          # no reading: never the peak
        start, end = np.r_[0, np.cumsum(n)[:-1]], np.cumsum(n)
        ride = np.repeat(np.arange(len(n)), n)
        cs = np.r_[0.0, np.cumsum(P)]
        cz = np.r_[0, np.cumsum(P <= 0)]                                   # unpowered seconds
        out = {"id": [str(i) for i in ids]}
        for d in cls.DURATIONS:
            w = np.full(len(n), np.nan)
            peak = np.full(len(n), np.nan)
            coast = np.full(len(n), np.nan)
            if P.size > d:
                mean = (cs[d:] - cs[:-d]) / d                             # window starting at each sample
                i = np.arange(mean.size)
                inside = ride[i] == ride[i + d - 1]                        # within one ride
                r = ride[i][inside]
                v = mean[inside]
                order = np.lexsort((-v, r))                                # each ride's best first
                first = order[np.r_[True, r[order][1:] != r[order][:-1]]]
                rides, best = r[first], i[inside][first]
                w[rides] = v[first]
                coast[rides] = (cz[best + d] - cz[best]) / d
                a, b = cls.WINDOW[d]
                last = best + d - 1
                lo = np.clip(last + a, start[rides], end[rides] - 1)
                hi = np.clip(last + b, start[rides], end[rides] - 1)
                edges = np.c_[lo, hi + 1].ravel()                          # [lo, hi] per ride, in order
                m = np.maximum.reduceat(np.r_[H, -np.inf], edges)[::2]
                peak[rides] = np.where(np.isfinite(m), m, np.nan)
            ok = w >= cls.MIN_POWER_W
            out[f"effort_w_{d}"] = np.where(ok, w, np.nan)
            out[f"effort_hr_{d}"] = np.where(ok, peak, np.nan)
            out[f"effort_coast_{d}"] = np.where(ok, coast, np.nan)
        return pd.DataFrame(out)

    @classmethod
    def maxed(cls, efforts: pd.DataFrame, max_hr: float) -> pd.DataFrame:
        """per ride and duration: its best effort reached your maximum heart rate then (within MARGIN) - all-out."""
        ref = cls.max_hr_then(efforts, max_hr)
        nan = pd.Series(np.nan, index=efforts.index)
        return pd.DataFrame({d: (efforts.get(f"effort_hr_{d}", nan) >= ref - cls.MARGIN[d])
                             & ((d < cls.STEADY_FROM_S) | (efforts.get(f"effort_coast_{d}", nan) <= cls.MAX_COAST))
                             for d in cls.DURATIONS}, index=efforts.index)

    MAX_HR_DAYS = 365                      # what a year of rides shows: the year around a ride
    MAX_HR_QUANTILE = 0.99                 # of the rides' highest heart rates then (a strap's spike doesn't set it)

    @classmethod
    def max_hr_then(cls, rides: pd.DataFrame, max_hr: float) -> pd.Series:
        """each ride's maximum heart rate at the time: the MAX_HR_QUANTILE of the rides' highest heart rates in the
        MAX_HR_DAYS around it (at least 30 rides), else `max_hr`; then the highest of those on that day or later.
        Rides only show a floor (a year without hard efforts reads low) and maximum heart rate falls with age, never
        rises: a maximum seen later was already there before."""
        if not {"date", "max_hr"} <= set(rides.columns):
            return pd.Series(float(max_hr), index=rides.index)
        day = pd.to_datetime(rides["date"]).dt.normalize()
        hr = pd.Series(pd.to_numeric(rides["max_hr"], errors="coerce").to_numpy(float), index=day)
        hr = hr[(hr > 100) & (hr < 225)].groupby(level=0).max().sort_index()
        if hr.empty:
            return pd.Series(float(max_hr), index=rides.index)
        q = hr.rolling(f"{cls.MAX_HR_DAYS}D", center=True, min_periods=30).quantile(cls.MAX_HR_QUANTILE)
        q = q.reindex(q.index.union(pd.DatetimeIndex(day.unique()))).ffill().bfill()
        q = pd.Series(np.fmax.accumulate(q.to_numpy(float)[::-1])[::-1], index=q.index)   # never rises later
        ref = q.reindex(day).to_numpy(float)
        return pd.Series(np.where(np.isfinite(ref), ref, float(max_hr)), index=rides.index)
