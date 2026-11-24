"""Critical power / W' modelling and rolling best-effort envelopes."""
from __future__ import annotations

import numpy as np
import pandas as pd

CP_DURATIONS = [180, 240, 300, 420, 480, 600, 720, 900, 1200]


def fit_cp(durations, powers):
    """2-parameter CP model via the linear work-time relation W = CP*t + W'.

    Returns (cp_w, wprime_j, r2) or (nan, nan, nan) when not identifiable.
    """
    t = np.asarray(durations, float)
    p = np.asarray(powers, float)
    m = np.isfinite(p) & (p > 0)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    t, p = t[m], p[m]
    w = p * t
    A = np.vstack([t, np.ones_like(t)]).T
    (cp, wp), *_ = np.linalg.lstsq(A, w, rcond=None)
    pred = cp * t + wp
    ss = ((w - pred) ** 2).sum()
    st = ((w - w.mean()) ** 2).sum()
    r2 = 1 - ss / st if st > 0 else np.nan
    if not (0 < cp < 700) or not (2000 < wp < 60000):
        return np.nan, np.nan, np.nan
    return float(cp), float(wp), float(r2)


def daily_envelope(acts: pd.DataFrame, durations, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily best mean-max power per duration (NaN on days without power)."""
    cols = [f"mmp_{d}" for d in durations]
    pw = acts[acts["has_power"].fillna(False).astype(bool)]
    day = pw.groupby("date")[cols].max()
    return day.reindex(index)


def rolling_envelope(day_env: pd.DataFrame, window_days: int, exclude_today=False) -> pd.DataFrame:
    env = day_env.rolling(window_days, min_periods=1).max()
    return env.shift(1) if exclude_today else env


def cp_series(env: pd.DataFrame) -> pd.DataFrame:
    """CP, W' and R^2 for each row of an envelope frame (columns mmp_<d>).

    Same model as fit_cp (W = CP*t + W'), solved for every row at once with the closed-form
    least-squares sums - no Python loop."""
    ds = [d for d in CP_DURATIONS if f"mmp_{d}" in env.columns]
    P = env[[f"mmp_{d}" for d in ds]].to_numpy(float)
    t = np.asarray(ds, float)[None, :]
    m = np.isfinite(P) & (P > 0)
    W = np.where(m, P * t, 0.0)
    T = np.where(m, t, 0.0)
    n = m.sum(1).astype(float)
    St, Sw = T.sum(1), W.sum(1)
    Stt, Stw, Sww = (T * T).sum(1), (T * W).sum(1), (W * W).sum(1)
    with np.errstate(all="ignore"):
        den = n * Stt - St * St
        cp = (n * Stw - St * Sw) / den
        wp = (Sw - cp * St) / n
        ss_res = Sww - 2 * cp * Stw - 2 * wp * Sw + cp * cp * Stt + 2 * cp * wp * St + n * wp * wp
        ss_tot = Sww - Sw * Sw / n
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    ok = (n >= 3) & (np.abs(den) > 0) & (cp > 0) & (cp < 700) & (wp > 2000) & (wp < 60000)
    out = np.where(ok[:, None], np.c_[cp, wp, r2], np.nan)
    return pd.DataFrame(out, index=env.index, columns=["cp", "wprime", "cp_r2"])


def estimate_ftp_series(day_env: pd.DataFrame, window_days=90, method="cp",
                        fallback=200.0, floor_days=180) -> pd.DataFrame:
    """Daily auto-FTP from the previous `window_days` (today excluded).

    A 90-day max-effort model drops whenever you stop doing hard efforts, even if
    fitness hasn't changed. `floor_days` stops that: FTP never goes below 95 % of
    the best 20-min power of the previous `floor_days` (0 disables the floor)."""
    env = rolling_envelope(day_env, window_days, exclude_today=True)
    cps = cp_series(env)
    p20 = env.get("mmp_1200")
    if method == "p20" or p20 is None:
        ftp = 0.95 * p20 if p20 is not None else pd.Series(np.nan, index=env.index)
    else:
        ftp = cps["cp"].copy()
        # guard against unrealistic CP when only short efforts exist
        if p20 is not None:
            alt = 0.95 * p20
            ftp = ftp.where(ftp.notna(), alt)
            ftp = ftp.clip(upper=1.05 * p20.where(p20.notna(), np.inf))
    if floor_days and "mmp_1200" in day_env:
        p20_long = rolling_envelope(day_env, floor_days, exclude_today=True)["mmp_1200"]
        ftp = pd.concat([ftp, 0.95 * p20_long], axis=1).max(axis=1, skipna=True) \
            .where(ftp.notna() | p20_long.notna())
    ftp = ftp.ffill()
    first = ftp.first_valid_index()
    if first is not None:
        ftp = ftp.fillna(ftp.loc[first])  # back-fill the period before data
    ftp = ftp.fillna(fallback)
    out = pd.DataFrame({"ftp_auto": ftp, "cp": cps["cp"], "wprime": cps["wprime"],
                        "cp_r2": cps["cp_r2"]})
    return out
