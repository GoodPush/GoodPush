"""Training stress, zones and the performance-management (fitness/fatigue) model."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .power import POWER_BIN_W, POWER_BINS, HR_BINS

# Coggan power zones as fraction of FTP (upper bounds)
POWER_ZONES = [("Z1 recovery", 0.55), ("Z2 endurance", 0.75), ("Z3 tempo", 0.90),
               ("Z4 threshold", 1.05), ("Z5 VO2max", 1.20), ("Z6 anaerobic", 1.50),
               ("Z7 neuromuscular", np.inf)]
# Friel HR zones as fraction of LTHR (upper bounds)
HR_ZONES = [("Z1", 0.81), ("Z2", 0.90), ("Z3", 0.94), ("Z4", 1.00), ("Z5a", 1.03),
            ("Z5b", 1.06), ("Z5c", np.inf)]

_PCENTERS = (np.arange(POWER_BINS) + 0.5) * POWER_BIN_W
_HCENTERS = np.arange(HR_BINS) + 0.5


def _hist(v) -> np.ndarray | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
        return None
    if isinstance(v, str):
        v = json.loads(v)
    return np.asarray(v, dtype=float)


def power_zone_seconds(hist, ftp: float) -> list[float]:
    h = _hist(hist)
    if h is None or not ftp or np.isnan(ftp):
        return [np.nan] * len(POWER_ZONES)
    frac = _PCENTERS / ftp
    out, lo = [], -np.inf
    for _, hi in POWER_ZONES:
        out.append(float(h[(frac > lo) & (frac <= hi)].sum()))
        lo = hi
    # exclude zero-power seconds (coasting/stopped) from Z1? keep them: they are time on bike
    return out


def hr_zone_seconds(hist, lthr: float) -> list[float]:
    h = _hist(hist)
    if h is None or not lthr:
        return [np.nan] * len(HR_ZONES)
    frac = _HCENTERS / lthr
    out, lo = [], -np.inf
    for _, hi in HR_ZONES:
        out.append(float(h[(frac > lo) & (frac <= hi)].sum()))
        lo = hi
    return out


def power_tss(duration_s: float, np_w: float, ftp: float) -> float:
    if not ftp or np.isnan(np_w) or np.isnan(duration_s):
        return np.nan
    return duration_s * (np_w / ftp) ** 2 / 3600.0 * 100.0


def hr_tss_from_hist(hist, rhr: float, lthr: float) -> float:
    """hrTSS: sum over seconds of ((HR-rest)/(LTHR-rest))^2, capped at IF 1.3."""
    h = _hist(hist)
    if h is None or not lthr or lthr <= rhr:
        return np.nan
    f = np.clip((_HCENTERS - rhr) / (lthr - rhr), 0, 1.3)
    return float((h * f ** 2).sum() / 3600.0 * 100.0)


def hr_tss_from_avg(duration_s: float, avg_hr: float, rhr: float, lthr: float) -> float:
    if np.isnan(avg_hr) or not lthr or lthr <= rhr:
        return np.nan
    f = np.clip((avg_hr - rhr) / (lthr - rhr), 0, 1.3)
    # average-HR underestimates sum of squares for variable efforts; +5 % fudge
    return float(duration_s / 3600.0 * f ** 2 * 100.0 * 1.05)


def ewma_load(tss: pd.Series, days: int) -> pd.Series:
    """Classic exponentially weighted load (TrainingPeaks CTL/ATL style)."""
    from scipy.signal import lfilter
    k = 1 - np.exp(-1.0 / days)
    vals = np.nan_to_num(tss.to_numpy(dtype=float))
    # acc[i] = acc[i-1] + k * (v[i] - acc[i-1]) as a C-level IIR filter
    out = lfilter([k], [1.0, -(1.0 - k)], vals)
    return pd.Series(out, index=tss.index)


def pmc(daily_tss: pd.Series, ctl_days=42, atl_days=7) -> pd.DataFrame:
    tss = daily_tss.fillna(0)
    ctl = ewma_load(tss, ctl_days)
    atl = ewma_load(tss, atl_days)
    df = pd.DataFrame({"tss": tss, "ctl": ctl, "atl": atl})
    df["tsb"] = ctl.shift(1) - atl.shift(1)   # form going *into* the day
    df["acwr"] = (atl / ctl.replace(0, np.nan)).clip(upper=5)
    r7 = tss.rolling(7, min_periods=7)
    df["monotony"] = r7.mean() / r7.std().replace(0, np.nan)
    df["strain"] = r7.sum() * df["monotony"]
    df["ramp"] = ctl - ctl.shift(7)            # CTL points per week
    return df


# --- vectorised versions (all activities at once) -------------------------------------
def hist_matrix(col, nbins: int) -> np.ndarray:
    """JSON histogram strings -> (n, nbins) array, NaN rows where missing. One json call."""
    vals = list(col)
    ok = np.array([isinstance(v, str) and v.startswith("[") for v in vals])
    out = np.full((len(vals), nbins), np.nan)
    if ok.any():
        parsed = json.loads("[" + ",".join(v for v, k in zip(vals, ok) if k) + "]")
        rows = np.full((int(ok.sum()), nbins), 0.0)
        for i, r in enumerate(parsed):           # ragged-safe copy (normally all nbins long)
            rows[i, :min(len(r), nbins)] = r[:nbins]
        out[ok] = rows
    return out


def zone_seconds_matrix(H: np.ndarray, centers: np.ndarray, ref: np.ndarray, zones) -> np.ndarray:
    """Seconds per zone for each row: H (n, bins), ref (n,) = FTP or LTHR."""
    with np.errstate(all="ignore"):
        frac = centers[None, :] / ref[:, None]
    out, lo = [], -np.inf
    for _, hi in zones:                        # 7 zones, not rows
        out.append(np.where((frac > lo) & (frac <= hi), H, 0.0).sum(1))
        lo = hi
    Z = np.stack(out, 1)
    bad = ~np.isfinite(H).all(1) | ~(np.isfinite(ref) & (ref > 0))
    Z[bad] = np.nan
    return Z


def hr_tss_matrix(H: np.ndarray, rhr: float, lthr: float) -> np.ndarray:
    if not lthr or lthr <= rhr:
        return np.full(H.shape[0], np.nan)
    f = np.clip((_HCENTERS - rhr) / (lthr - rhr), 0, 1.3)
    return (H * f ** 2).sum(1) / 3600.0 * 100.0
