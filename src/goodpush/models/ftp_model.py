"""FTP fitted to Garmin's own FTP estimates (and your [ftp.history] entries).

Model, per day t (rides before t only, so a ride never sets its own FTP):

    FTP(t) = (1 - w) * sum_d k_d * B_d(t) + w * H(t)

* B_d(t): your best d-second power (d = 3, 5, 10, 20, 40, 60 min), older efforts fading:
  max over rides r of MMP_d(r) * exp(-(t - t_r) / tau). One running maximum in log space per
  duration (cummax of log MMP + t / tau), so the whole history is a few numpy operations.
* H(t): power at threshold heart rate, 42-day median over steady rides (>= 30 min, VI <= 1.15):
  P_avg * (LTHR - RHR) / (HR_avg - RHR) - heart rate rises about linearly from resting HR with power.
  LTHR is an input, never fitted: your [athlete] lthr, else goodpush's estimate (98th percentile of
  your best-20-min heart rates).
* k_d, tau and w are fitted (scipy differential evolution, Huber loss) to the days Garmin's
  auto-detected FTP changed, plus your own entries, plus your ramp tests (`tests`: the physio model's LT2 that
  day, counting TEST_WEIGHT; only a ramp LT2 that passes RampTest.lt2_check). The fit is cached in
  processed/ftp_model.json and redone only when those labels, their weights or LTHR change.

auto_method "model" skips this for FTP itself: FTP = the physio model's LT2 (CP) from its last fit (`physio_cp`).

Checked by forward validation by year (fit on the years before, test on the next) against the critical-power
estimate and H(t) alone. No fitted fading (tau ~ years): best efforts alone hardly say when fitness drops; the
heart-rate part does.
Garmin's FTP is itself an estimate: this agrees with Garmin, it doesn't prove either right.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

DURATIONS = (180, 300, 600, 1200, 2400, 3600)
HR_WINDOW = 42
HR_REP_S, HR_REP_CV = 300, 0.12           # steady reps for power at threshold HR: >= 5 min, power within 12 %
MIN_LABELS = 10
CACHE = "ftp_model.json"
STATES = "physio_states.parquet"          # PhysioRun.STATES (not imported here: that loads jax)
TEST_WEIGHT = 3.0                         # a ramp test counts this many Garmin estimates


@dataclass
class FtpModel:
    weights: list            # k_d per duration
    tau: float               # days for an old best effort to fade to 1/e
    hr_weight: float         # w
    lthr: float              # threshold heart rate (bpm) it was fitted with: an input, not fitted
    labels_key: str = ""     # which labels it was fitted to (refit when they change)
    mae: float = float("nan")

    # --- inputs ----------------------------------------------------------------------------
    @staticmethod
    def log_bests(rides: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
        """log of each day's best power per duration (days x durations), -inf on days without rides."""
        m = rides.groupby("date")[[f"mmp_{d}" for d in DURATIONS]].max().reindex(days).to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(m)
        out[~np.isfinite(out)] = -np.inf
        return out

    @staticmethod
    def faded(log_bests: np.ndarray, tau: float) -> np.ndarray:
        """best power so far per day and duration, older efforts fading with `tau` (days); before today."""
        t = np.arange(len(log_bests), dtype=float)[:, None]
        cum = np.maximum.accumulate(log_bests + t / tau, axis=0) - t / tau
        prev = np.vstack([np.full((1, cum.shape[1]), -np.inf), cum[:-1]])        # rides before the day
        return np.exp(prev)

    @staticmethod
    def power_at_lthr(p, h, rhr, lthr: float) -> np.ndarray:
        """power at threshold HR from steady power p at heart rate h: heart-rate reserve scaled linearly
        to LTHR; NaN when h is within 15 bpm of resting."""
        ok = h > rhr + 15
        return np.where(ok, p * (lthr - rhr) / np.where(ok, h - rhr, 1.0), np.nan)

    @classmethod
    def hr_intervals(cls, log: pd.DataFrame, days: pd.DatetimeIndex, rhr: pd.Series, lthr: float) -> pd.Series:
        """power at threshold HR per day from steady reps (models.intervals.IntervalLog: >= HR_REP_S long,
        power within HR_REP_CV, heart rate over the rep's second half, once it has settled); index = day
        number, median over the day's reps."""
        st = log[(log["seconds"] >= HR_REP_S) & (log["cv"] <= HR_REP_CV) & log["hr_late"].notna()]
        d = days.get_indexer(pd.to_datetime(st["date"]))
        st, d = st[d >= 0], d[d >= 0]
        r = rhr.reindex(days).ffill().bfill().fillna(50.0).to_numpy(float)[d]
        v = cls.power_at_lthr(st["watts"].to_numpy(float), st["hr_late"].to_numpy(float), r, lthr)
        return pd.Series(v, index=d).groupby(level=0).median().dropna()

    @staticmethod
    def hr_rides(rides: pd.DataFrame, days: pd.DatetimeIndex, rhr: pd.Series, lthr: float) -> pd.Series:
        """power at threshold HR per day with steady rides (index = day number, median over the day's rides)."""
        st = rides[(rides["duration_s"] >= 1800) & rides["avg_hr"].notna() & (rides["avg_power"] > 80)
                   & (rides["np_used"] / rides["avg_power"] <= 1.15) & (rides["power_coverage"] > 0.9)]
        r = rhr.reindex(days).ffill().bfill().fillna(50.0).reindex(st["date"]).to_numpy(float)
        v = FtpModel.power_at_lthr(st["avg_power"].to_numpy(float), st["avg_hr"].to_numpy(float), r, lthr)
        return pd.Series(v, index=days.get_indexer(st["date"])).groupby(level=0).median().dropna()

    @classmethod
    def hr_threshold(cls, rides: pd.DataFrame, days: pd.DatetimeIndex, rhr: pd.Series, lthr: float) -> np.ndarray:
        """power at threshold HR per day: 42-day median over the steady rides before each day."""
        s = cls.hr_rides(rides, days, rhr, lthr).reindex(np.arange(len(days)))
        return s.rolling(HR_WINDOW, min_periods=3).median().shift(1).ffill().to_numpy()     # rides before the day

    # --- model -----------------------------------------------------------------------------
    def predict_from(self, log_bests: np.ndarray, hr: np.ndarray) -> np.ndarray:
        b = (self.faded(log_bests, self.tau) * np.asarray(self.weights)).sum(axis=1)
        out = (1 - self.hr_weight) * b + self.hr_weight * np.where(np.isfinite(hr), hr, b)
        return np.where(np.isfinite(out) & (out > 0), out, np.nan)

    @classmethod
    def fit(cls, log_bests: np.ndarray, hr: np.ndarray, days: pd.DatetimeIndex, labels: pd.Series,
            lthr: float, seed: int = 1, weights: pd.Series | None = None) -> FtpModel:
        """Robust fit to `labels` (date -> FTP), each counting `weights` (default 1); labels are compared with the
        next day's estimate, which is the first to know that day's rides. `hr` was computed with `lthr`."""
        idx = days.get_indexer(labels.index + pd.Timedelta(days=1))
        keep = idx >= 0
        idx, y = idx[keep], labels.to_numpy(float)[keep]
        w = (np.ones(len(labels)) if weights is None else weights.reindex(labels.index).fillna(1.0).to_numpy(float))[keep]
        nd = len(DURATIONS)

        def loss(p):
            m = cls(list(p[:nd]), p[nd], p[nd + 1], lthr)
            r = m.predict_from(log_bests, hr)[idx] - y
            a = np.abs(np.where(np.isfinite(r), r, 50.0))
            return np.average(np.where(a <= 6.0, 0.5 * a * a, 6.0 * (a - 3.0)), weights=w)    # Huber, 6 W
        bounds = [(0.0, 1.0)] * nd + [(7.0, 3000.0), (0.0, 1.0)]
        res = differential_evolution(loss, bounds, seed=seed, tol=1e-8, maxiter=300, polish=False)
        p = res.x
        m = cls([float(x) for x in p[:nd]], float(p[nd]), float(p[nd + 1]), float(lthr))
        m.mae = float(np.nanmean(np.abs(m.predict_from(log_bests, hr)[idx] - y)))
        return m

    # --- labels and the cached fit ----------------------------------------------------------
    @staticmethod
    def labels(daily_wellness: pd.DataFrame, history: dict) -> pd.Series:
        """Garmin's auto-detected FTP on the days it changed, plus your [ftp.history] entries."""
        out = pd.Series(dtype=float)
        if {"ftp_garmin", "ftp_garmin_auto"} <= set(daily_wellness.columns):
            g = daily_wellness.loc[daily_wellness["ftp_garmin_auto"].eq(1), "ftp_garmin"].dropna()
            out = g[g.ne(g.shift())].astype(float)
        mine = pd.Series({pd.Timestamp(k): float(v) for k, v in (history or {}).items()}, dtype=float)
        return pd.concat([out[~out.index.isin(mine.index)], mine]).sort_index()

    @staticmethod
    def tests(rides: pd.DataFrame, cp: pd.Series | None = None) -> pd.Series:
        """FTP from your ramp tests (date -> W): the physio model's LT2 (CP) on the test day - it corrects the
        ramp's own alpha1 LT2, which reads high - else that reading. Weighted TEST_WEIGHT in the fit."""
        from .ramp import RampTest
        if "ramp_lt2_w" not in rides:
            return pd.Series(dtype=float)
        r = rides.dropna(subset=["ramp_lt2_w"])
        r = r[(r["ramp_lt2_w"] > 0) & RampTest.lt2_check(r).eq("")]    # only a ramp LT2 that can be a threshold
        day = pd.to_datetime(r["date"]).dt.normalize()
        w = pd.Series(r["ramp_lt2_w"].to_numpy(float), index=day)
        if cp is not None and len(cp):
            w = cp.reindex(day).fillna(w)
        return w.groupby(level=0).last()

    @staticmethod
    def last_ramps(processed: Path) -> pd.DataFrame:
        """your ramp tests as the last analysis found them (FTP is set before this analysis reads them)"""
        cols = ["date", "ramp_top_w", "ramp_max_hr", "ramp_lt2_w", "ramp_lt2_hr", "ramp_hr_lt2_w", "ramp_hr_lt2_hr"]
        try:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(Path(processed) / "activities.parquet").names)
            return pd.read_parquet(Path(processed) / "activities.parquet", columns=[c for c in cols if c in have])
        except (OSError, ValueError, KeyError):
            return pd.DataFrame(columns=["date"])

    @staticmethod
    def physio_cp(processed: Path) -> pd.Series | None:
        """the physio model's LT2 (CP) per day, from its last fit (processed/physio_states.parquet), or None"""
        try:
            s = pd.read_parquet(Path(processed) / STATES, columns=["cp"])["cp"]
        except (OSError, ValueError, KeyError):
            return None
        s = s[s > 0].dropna()
        return s if len(s) else None

    @classmethod
    def cached(cls, processed: Path, labels: pd.Series, fit_args: tuple, lthr: float, progress=print,
               weights: pd.Series | None = None) -> FtpModel | None:
        """The fitted model for these labels (and weights) and LTHR: from processed/ftp_model.json, else fitted now
        (and saved)."""
        if len(labels) < MIN_LABELS:
            return None
        wsum = 0.0 if weights is None else float(weights.reindex(labels.index).fillna(1.0).sum())
        key = f"{len(labels)}:{labels.index.max().date()}:{labels.sum():.0f}:lthr{lthr:.0f}:w{wsum:.0f}"
        p = Path(processed) / CACHE
        try:
            m = cls(**json.loads(p.read_text()))
            if m.labels_key == key:
                return m
        except (OSError, ValueError, TypeError):
            pass
        progress(f"fitting the FTP model to {len(labels)} FTP values (Garmin's and yours) ...")
        m = cls.fit(*fit_args, labels, lthr, weights=weights)
        m.labels_key = key
        p.write_text(json.dumps(asdict(m), indent=1))
        progress(f"FTP model: {m.mae:.1f} W from the labels on average")
        return m
