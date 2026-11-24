"""Heart-rate evidence that a second of riding was severe (above CP): the physio model's z2 / z3 split (VERSION 16).

The model splits the band LT1 - MAP into heavy (z2) and severe (z3) by a severe share per second,
s = logistic((log P - log CP) / EDGE_WIDTH + e): the soft CP edge plus heart-rate evidence e, added as log-odds.
Power alone can't tell steady from severe within a few % of CP; heart rate alone (v15) counted long hot rides and
climbs and booked history by one rule and planned sessions by another. Here e = (corrected HR - threshold HR) / width,
from the data only, per second and all rides at once from the 1 Hz streams:

1. lag: effective HR = smoothed HR + tau x its slope, tau the rider's own first-order time constant - the one at which
   lag-filtered power lines up best with heart rate within rides (`lag`);
2. drift: per ride, the upward trend over ride time of (HR - what lag-filtered power explains), fitted on seconds
   below CP only (above CP the rise is the slow component the evidence is about), slopes >= 0, taken off (`drift`);
3. threshold HR then (an input: the athlete's LTHR scaled by maximum heart rate then) stands for heart rate at CP;
4. width = WIDTH_SHARE x the scatter of corrected heart rate around a straight line (heart rate / LTHR on log power /
   CP) in steady riding between LT1 and CP (`width`).

Seconds without heart rate (and every planned session) have e = 0: the plain power edge. `fractions`: per ride, power
bin and evidence bin (physio EVIDENCE), the share of the bin's seconds with heart rate in it; `SevereSplit` keeps them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import lfilter

from .power import POWER_BIN_W


class SevereHeartRate:
    TAUS = np.arange(15.0, 125.0, 5.0)     # lag time constants tried (s)
    HR_SMOOTH_S = 15                       # heart rate smoothed over this (s)
    SLOPE_SMOOTH_S = 20                    # ... and over this for its slope
    POWER_SMOOTH_S = 30                    # power for steady riding
    SETTLE_S = 300                         # a ride's first seconds: heart rate still settling, filters filling
    HIGH_PASS_S = 600                      # the lag is fitted on what changes faster than this
    DRIFT_FROM_S = 600                     # drift fitted on seconds after this
    DRIFT_MIN_S = 1200                     # ... with at least this many below CP (else no drift)
    DRIFT_BAND = (0.5, 0.98)               # lag-filtered power / CP of those seconds
    STEADY_S = 180                         # steady riding (width): power within STEADY_SPREAD of its mean this long
    STEADY_SPREAD = 0.10
    STEADY_TOP = 0.97                      # ... between LT1 and this x CP
    # the evidence's unit: this share of the scatter. Half separated session types (VO2max, threshold above sweet spot
    # and long rides); the whole scatter blurred them (notes/findings.md 2026-11-12/13)
    WIDTH_SHARE = 0.5
    MIN_HR = 30.0
    RIDER = ("tau_s", "hr_per_cp_share", "scatter_bpm")  # what is learned from all rides (`rider`)

    def __init__(self, n, power, hr, lt1, cp, lthr, rider: dict | None = None):
        """streams of rides laid end to end (`n` samples each); per ride: LT1 and CP (W) on its day, NaN where unknown,
        and threshold HR then (lthr: heart rate at CP). rider: RIDER values learned before (new rides only); None:
        learned from these rides."""
        self.n = np.asarray(n, np.int64)
        self.ride = np.repeat(np.arange(len(self.n)), self.n)
        self.start = np.r_[0, np.cumsum(self.n)[:-1]]
        self.k = np.arange(self.n.sum()) - np.repeat(self.start, self.n)         # seconds into the ride
        self.P = np.nan_to_num(np.asarray(power, float))
        H = np.asarray(hr, float)
        self.has = np.isfinite(H) & (H > self.MIN_HR)
        Hf = pd.Series(np.where(self.has, H, np.nan)).ffill().bfill().to_numpy()
        self.Hs = uniform_filter1d(Hf, self.HR_SMOOTH_S) if Hf.size else Hf
        self.slope = np.gradient(uniform_filter1d(Hf, self.SLOPE_SMOOTH_S)) if Hf.size > 1 else np.zeros(Hf.size)
        self.P30 = uniform_filter1d(self.P, self.POWER_SMOOTH_S) if self.P.size else self.P
        self.lt1, self.cp, self.lthr = (np.repeat(np.asarray(v, float), self.n) for v in (lt1, cp, lthr))
        self.known = self.has & np.isfinite(self.cp) & np.isfinite(self.lt1) & np.isfinite(self.lthr)
        self.rider = dict(rider or {})
        self.info = {}

    # --- lag --------------------------------------------------------------------------------------
    def lagged(self, tau: float) -> np.ndarray:
        """power through a first-order lag of time constant tau, restarted at each ride's start."""
        a = 1.0 - np.exp(-1.0 / tau)
        y = lfilter([a], [1.0, a - 1.0], self.P)
        before = np.r_[0.0, y][self.start]                                    # the filter's value before each ride
        return y - np.repeat(before, self.n) * (1.0 - a) ** (self.k + 1)

    def lag(self) -> float:
        """the time constant at which lag-filtered power explains most of heart rate's variance within rides, both
        high-passed (their HIGH_PASS_S means taken off: drift and slow trends would favour the longest lag)."""
        if "tau_s" in self.rider:
            return float(self.rider["tau_s"])
        use = self.has & (self.k >= self.SETTLE_S)
        h = (self.Hs - uniform_filter1d(self.Hs, self.HIGH_PASS_S))[use]
        r2 = []
        for tau in self.TAUS:                                                 # a handful of candidate constants
            p = self.lagged(tau)
            p = (p - uniform_filter1d(p, self.HIGH_PASS_S))[use]
            r2.append((h @ p) ** 2 / max((h @ h) * (p @ p), 1e-12))
        r2 = np.asarray(r2)
        self.info["lag_r2"] = dict(zip(self.TAUS.tolist(), np.round(r2, 4).tolist()))
        self.rider["tau_s"] = float(self.TAUS[int(np.argmax(r2))])
        return self.rider["tau_s"]

    # --- drift ------------------------------------------------------------------------------------
    @staticmethod
    def ride_lines(r: np.ndarray, k: np.ndarray, v: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """per ride (index r) the least-squares line of v over time k: residuals, slopes, sample counts."""
        sn, sk, sv, skk, skv = (np.bincount(r, w, N) for w in (np.ones_like(k), k, v, k * k, k * v))
        den = sn * skk - sk ** 2
        slope = np.where(den > 0, (sn * skv - sk * sv) / np.where(den > 0, den, 1), 0.0)
        icpt = (sv - slope * sk) / np.maximum(sn, 1)
        return v - icpt[r] - slope[r] * k, slope, sn

    def drift(self, Pf: np.ndarray) -> np.ndarray:
        """per ride: the upward trend (bpm / s, >= 0) of heart rate less what lag-filtered power explains, over ride
        time, from its seconds below CP (DRIFT_BAND) after DRIFT_FROM_S; 0 with under DRIFT_MIN_S of them. Heart rate
        = a per ride + b x power / CP + a slope per ride, fitted together (b from the per-ride detrended values: power
        that falls over a ride would otherwise take some of the drift)."""
        rel = Pf / self.cp
        sub = self.known & (rel > self.DRIFT_BAND[0]) & (rel < self.DRIFT_BAND[1]) & (self.k >= self.DRIFT_FROM_S)
        N = len(self.n)
        r, x, h, k = self.ride[sub], rel[sub], self.Hs[sub], self.k[sub].astype(float)
        if "hr_per_cp_share" not in self.rider:
            xd, hd = self.ride_lines(r, k, x, N)[0], self.ride_lines(r, k, h, N)[0]
            self.rider["hr_per_cp_share"] = float((xd @ hd) / max(xd @ xd, 1e-12))   # HR per CP share
        _, slope, sn = self.ride_lines(r, k, h - self.rider["hr_per_cp_share"] * x, N)
        fitted = sn >= self.DRIFT_MIN_S
        slope = np.where(fitted, np.clip(slope, 0.0, None), 0.0)
        if fitted.any():
            self.info["drift_bpm_h"] = {"rides": int(fitted.sum()), "median": float(np.median(slope[fitted]) * 3600),
                                        "p90": float(np.quantile(slope[fitted], 0.9) * 3600)}
        return slope

    def corrected(self) -> np.ndarray:
        """lag- and drift-corrected heart rate per second."""
        tau = self.lag()
        eff = np.where(self.k < self.SLOPE_SMOOTH_S, self.Hs, self.Hs + tau * self.slope)
        return eff - np.repeat(self.drift(self.lagged(tau)), self.n) * self.k

    # --- width ------------------------------------------------------------------------------------
    def steady(self) -> tuple[np.ndarray, np.ndarray]:
        """seconds of steady riding between LT1 and STEADY_TOP x CP (the 30-s power within STEADY_SPREAD of its mean
        over the STEADY_S before), and that mean power."""
        S = self.STEADY_S
        mean = uniform_filter1d(self.P30, S, origin=S // 2 - 1)               # the S seconds up to here
        lo = minimum_filter1d(self.P30, S, origin=S // 2 - 1)
        hi = maximum_filter1d(self.P30, S, origin=S // 2 - 1)
        ok = (self.known & (self.k >= self.SETTLE_S + S) & (lo >= (1 - self.STEADY_SPREAD) * mean)
              & (hi <= (1 + self.STEADY_SPREAD) * mean) & (mean > self.lt1) & (mean < self.STEADY_TOP * self.cp))
        return ok, mean

    def width(self, hr: np.ndarray) -> float:
        """the evidence's unit (bpm): WIDTH_SHARE x the scatter (robust sd) of corrected heart rate around a straight
        line - heart rate / LTHR on log power / CP - in steady riding below CP. The line's heart rate at CP / LTHR goes
        in `info` (how far threshold HR sits from it)."""
        if "scatter_bpm" not in self.rider:
            s, mean = self.steady()
            if s.sum() < 2:
                self.rider["scatter_bpm"] = np.nan
            else:
                x, y = np.log(mean[s] / self.cp[s]), hr[s] / self.lthr[s]
                b, a = np.polyfit(x, y, 1)
                r = (y - a - b * x) * self.lthr[s]
                self.rider["scatter_bpm"] = float(1.4826 * np.median(np.abs(r - np.median(r))))
                self.info |= {"line_at_cp_lthr": float(a), "steady_hours": float(s.sum() / 3600)}
        return self.WIDTH_SHARE * float(self.rider["scatter_bpm"])

    # --- evidence ---------------------------------------------------------------------------------
    def evidence(self) -> np.ndarray:
        """each second's heart-rate evidence (log-odds, clipped to +-EVIDENCE_MAX); NaN without heart rate or edges."""
        from ..physio.model import EVIDENCE_MAX
        hr = self.corrected()
        w = self.width(hr)
        self.info["width_bpm"] = w
        if not np.isfinite(w) or w <= 0:
            return np.full(self.P.size, np.nan)
        return np.where(self.known, np.clip((hr - self.lthr) / w, -EVIDENCE_MAX, EVIDENCE_MAX), np.nan)

    def fractions(self, bins: int) -> np.ndarray:
        """(rides, bins, E): of each ride's seconds in a raw power bin (as metrics.power's histogram; the top bin takes
        everything above), the share with heart rate in each evidence bin (physio EVIDENCE); the rest have none."""
        from ..physio.model import EVIDENCE, EVIDENCE_MAX
        e = self.evidence()
        E = len(EVIDENCE)
        has = np.isfinite(e)
        k = np.clip(((np.nan_to_num(e) + EVIDENCE_MAX) / (2 * EVIDENCE_MAX) * E).astype(np.int64), 0, E - 1)
        b = np.minimum((self.P / POWER_BIN_W).astype(np.int64), bins - 1)
        N = len(self.n) * bins
        total = np.bincount(self.ride * bins + b, minlength=N).reshape(-1, bins).astype(float)
        part = np.bincount(((self.ride * bins + b) * E + k)[has], minlength=N * E).reshape(-1, bins, E)
        self.info["hr_hours"] = float(has.sum() / 3600)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nan_to_num(part / total[:, :, None])


class SevereSplit:
    """Each ride's evidence shares (SevereHeartRate.fractions) for the physio model's inputs, kept in FILE: a ride is
    done once, with LT1 / CP per day of the stored fit at the time (any version's; they pick the steady and drift
    seconds), and the rider's lag, HR-per-power and scatter learned from all rides the first time. Without a stored fit
    nothing has evidence (the power edges alone) until there is one."""
    FILE = "physio_severe.npz"
    STATES = "physio_states.parquet"
    VERSION = 2                            # bumped (or the model's) -> every ride done again

    def __init__(self, processed):
        self.processed = Path(processed)
        self.info = {}

    def tag(self) -> str:
        from ..physio.model import EVIDENCE, VERSION
        return f"{self.VERSION}|v{VERSION}|e{len(EVIDENCE)}|w{SevereHeartRate.WIDTH_SHARE}"

    def edges(self) -> pd.DataFrame | None:
        """lt1 / cp per day of the stored fit; None without one."""
        try:
            return pd.read_parquet(self.processed / self.STATES)[["lt1", "cp"]]
        except (OSError, KeyError, ValueError):
            return None

    def stored(self) -> dict:
        """the rides done {ids, share (rides, B, E)} and the rider's values, if made with this tag."""
        try:
            with np.load(self.processed / self.FILE, allow_pickle=False) as f:
                if str(f["tag"]) != self.tag():
                    return {}
                return {"ids": [str(x) for x in f["ids"]], "share": f["share"],
                        "rider": dict(zip([str(x) for x in f["rider_keys"]], f["rider"].tolist()))}
        except (OSError, KeyError, ValueError):
            return {}

    def fractions(self, rides: pd.DataFrame, lthr, bins: int) -> np.ndarray:
        """(rides, bins, E): of each ride's power-bin seconds, the share with heart rate in each evidence bin; zeros
        (no evidence) without a stored fit or streams. lthr: threshold HR then, per ride (rows of `rides`)."""
        from ..io import streamstore
        from ..physio.model import EVIDENCE
        E = len(EVIDENCE)
        out = np.zeros((len(rides), bins, E))
        states = self.edges()
        if states is None or not len(rides):
            return out
        ids = rides["id"].astype(str).str.removesuffix(".0").to_numpy()
        st = self.stored()
        known = dict(zip(st.get("ids", []), range(len(st.get("ids", [])))))
        have = set(streamstore.ids(self.processed, "ride"))
        new = [i for i in dict.fromkeys(ids) if i in have and i not in known]
        share, rider = st.get("share", np.zeros((0, bins, E))), st.get("rider")
        if new:
            if rider is None:                                                 # first time: every ride, rider learned
                got, n, data = streamstore.read_all(self.processed, "ride", ["power", "heart_rate"])
                got = [str(i) for i in got]
                P, H = data["power"], data["heart_rate"]
            else:
                many = streamstore.read_many(self.processed, new, "ride", ["power", "heart_rate"])
                got = [i for i in new if i in many and "power" in many[i]]
                n = np.array([many[i]["power"].size for i in got], np.int64)
                P = np.concatenate([many[i]["power"] for i in got]) if got else np.zeros(0)
                H = np.concatenate([many[i].get("heart_rate", np.full(many[i]["power"].size, np.nan)) for i in got]) \
                    if got else np.zeros(0)
            row = pd.Series(np.arange(len(rides)), index=ids)
            row = row[~row.index.duplicated()].reindex(got).to_numpy(float)
            ok = np.isfinite(row)
            r = np.where(ok, row, 0).astype(int)
            day = pd.DatetimeIndex(pd.to_datetime(rides["date"]).dt.normalize().to_numpy()[r])
            e = states.reindex(states.index.union(day.unique())).ffill().reindex(day)   # after the fit: its last day
            lt1, cp, th = (np.where(ok, np.asarray(v, float), np.nan)
                           for v in (e["lt1"].to_numpy(), e["cp"].to_numpy(), np.asarray(lthr, float)[r]))
            m = SevereHeartRate(n, P, H, lt1, cp, th, rider)
            f = m.fractions(bins)
            self.info, rider = m.info | m.rider, m.rider
            keep = [i for i, o in zip(got, ok) if o and i not in known]       # rides of this table only
            pos = {i: j for j, i in enumerate(got)}
            share = np.concatenate([share, f[np.array([pos[i] for i in keep], int)]])
            known |= {i: len(known) + j for j, i in enumerate(keep)}
            self.save(list(known), share, rider)
        if rider is not None:
            self.info = self.info or dict(rider)
        idx = np.array([known.get(i, -1) for i in ids])
        hit = idx >= 0
        out[hit] = share[idx[hit]]
        return out

    def save(self, ids: list, share: np.ndarray, rider: dict) -> None:
        keys = list(SevereHeartRate.RIDER)
        tmp = self.processed / (self.FILE + ".tmp.npz")
        np.savez_compressed(tmp, ids=np.array(ids), share=share, tag=self.tag(), rider_keys=np.array(keys),
                            rider=np.array([float(rider.get(k, np.nan)) for k in keys]))
        tmp.replace(self.processed / self.FILE)
