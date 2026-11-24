"""Stream processing and FTP-independent per-activity metrics."""
from __future__ import annotations

import numpy as np

# Durations (s) for the mean-maximal power curve
MMP_DURATIONS = [1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 480,
                 600, 720, 900, 1200, 1800, 2400, 3600, 5400, 7200]
POWER_BIN_W = 10
POWER_BINS = 200          # 0..2000 W
HR_BINS = 230             # 0..229 bpm, 1 bpm bins

SMART_GAP_S = 10          # gaps <= this are forward-filled (smart recording)
STOP_GAP_S = 300          # gaps > this are removed entirely (long stop); between -> zeros


def ride_seconds(t) -> np.ndarray:
    """Whole seconds from the first sample on the ride's timeline: stops longer than STOP_GAP_S are cut
    out (as in to_1hz), so other per-ride series (alpha1) line up with the 1 Hz streams. `t` sorted."""
    t = np.asarray(t, float)
    ti = np.floor(t - t[0]).astype(np.int64)
    gaps = np.diff(ti)
    return ti - np.r_[0, np.cumsum(np.where(gaps > STOP_GAP_S, gaps - 1, 0))]


def to_1hz(records: dict, max_power: float = 2000):
    """Resample raw records to a 1 Hz grid.

    Returns dict of numpy arrays (power, heart_rate, cadence, speed, altitude)
    with NaN where missing, plus 'start' (unix s) and 'moving' mask.
    Gaps <= 10 s are forward filled; 10-300 s gaps are filled with power 0 /
    HR NaN (stopped); longer gaps are cut out of the timeline.
    """
    t = np.asarray(records["time"], dtype=float)
    if t.size < 2:
        return None
    order = np.argsort(t, kind="stable")
    t = t[order]
    ti = np.floor(t - t[0]).astype(np.int64)
    # unique seconds: keep last sample in each second
    keep = np.ones(ti.size, bool)
    keep[:-1] = ti[1:] != ti[:-1]
    idx_sorted = order[keep]
    tc = ride_seconds(t[keep])          # long gaps cut out
    n = int(tc[-1]) + 1

    # fill index: for each grid second, the most recent sample index, and the age
    pos = np.searchsorted(tc, np.arange(n), side="right") - 1
    age = np.arange(n) - tc[pos]
    # identify which grid seconds belong to a "stop" gap (10<gap<=300)
    next_t = np.append(tc[1:], tc[-1])
    gap_len = (next_t - tc)[pos]
    stopped = (age > 0) & (gap_len > SMART_GAP_S)

    out = {"start": float(t[0]), "n": n}
    for k in ("power", "heart_rate", "cadence", "speed", "altitude", "distance"):
        raw = records.get(k)
        if raw is None:
            out[k] = np.full(n, np.nan)
            continue
        arr = np.array([np.nan if v is None else v for v in raw], dtype=float)[idx_sorted]
        # forward-fill missing raw values within short windows
        vals = arr[pos]
        if k in ("power", "cadence", "speed"):
            vals = np.where(stopped, 0.0, vals)
        elif k == "heart_rate":
            vals = np.where(stopped, np.nan, vals)
        out[k] = vals
    p = out["power"]
    p[(p < 0) | (p > max_power)] = np.nan
    hr = out["heart_rate"]
    hr[(hr < 30) | (hr > 230)] = np.nan
    sp = out["speed"]
    moving = ~stopped
    if np.isfinite(sp).any():
        moving &= ~(np.nan_to_num(sp) < 0.5) | (np.nan_to_num(p) > 0)
    out["moving"] = moving
    return out


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if x.size < w:
        return np.empty(0)
    c = np.concatenate([[0.0], np.cumsum(x)])
    return (c[w:] - c[:-w]) / w


def normalized_power(p: np.ndarray) -> float:
    r = rolling_mean(p, 30)
    if r.size == 0:
        return float(np.mean(p)) if p.size else np.nan
    return float(np.mean(r ** 4) ** 0.25)


def mean_max(x: np.ndarray, durations=MMP_DURATIONS) -> list:
    out = []
    for d in durations:
        r = rolling_mean(x, d)
        out.append(float(r.max()) if r.size else np.nan)
    return out


def power_histogram(p: np.ndarray) -> np.ndarray:
    """seconds in each POWER_BIN_W bin of 1-Hz power (POWER_BINS bins; coasting in the first)."""
    return np.histogram(p, bins=POWER_BINS, range=(0, POWER_BINS * POWER_BIN_W))[0].astype(int)


def summarize_stream(s: dict) -> dict:
    """FTP-independent metrics from a 1 Hz stream (output of to_1hz)."""
    n = s["n"]
    p = s["power"]
    hr = s["heart_rate"]
    moving = s["moving"]
    res = {"n_seconds": int(n), "moving_s": int(moving.sum())}

    has_p = np.isfinite(p)
    pz = np.nan_to_num(p)
    res["power_coverage"] = float(has_p.mean()) if n else 0.0
    power_ok = res["power_coverage"] > 0.5 and np.nanmax(pz, initial=0) > 0 and pz.mean() > 20
    res["has_power"] = bool(power_ok)
    if power_ok:
        res["avg_power"] = float(pz.mean())
        res["np"] = normalized_power(pz)
        res["work_kj"] = float(pz.sum() / 1000.0)
        res["max_power"] = float(pz.max())
        res["mmp"] = mean_max(pz)
        res["power_hist"] = power_histogram(pz).tolist()
    else:
        res.update(avg_power=np.nan, np=np.nan, work_kj=np.nan, max_power=np.nan,
                   mmp=[np.nan] * len(MMP_DURATIONS), power_hist=None)

    has_hr = np.isfinite(hr)
    res["hr_coverage"] = float(has_hr.mean()) if n else 0.0
    if has_hr.sum() > 60:
        hv = hr[has_hr]
        res["avg_hr"] = float(hv.mean())
        res["max_hr"] = float(np.percentile(hv, 99.9))
        h, _ = np.histogram(hv, bins=HR_BINS, range=(0, HR_BINS))
        res["hr_hist"] = h.astype(int).tolist()
        hr_ff = _ffill(hr)
        r = rolling_mean(np.nan_to_num(hr_ff, nan=float(hv.mean())), 1200)
        res["best20_hr"] = float(r.max()) if r.size else np.nan
    else:
        res.update(avg_hr=np.nan, max_hr=np.nan, hr_hist=None, best20_hr=np.nan)

    # Efficiency factor and aerobic decoupling (Pw:HR), moving seconds only
    res["ef"] = np.nan
    res["decoupling"] = np.nan
    if power_ok and has_hr.sum() > 600:
        m = moving & has_hr & has_p
        if m.sum() > 600:
            pm, hm = pz[m], hr[m]
            res["ef"] = normalized_power(pm) / hm.mean()
            if m.sum() >= 2400:
                half = m.sum() // 2
                ef1 = normalized_power(pm[:half]) / hm[:half].mean()
                ef2 = normalized_power(pm[half:]) / hm[half:].mean()
                res["decoupling"] = float((ef1 - ef2) / ef1 * 100.0)
    cad = s["cadence"]
    c = cad[np.isfinite(cad) & (cad > 0)]
    res["avg_cadence"] = float(c.mean()) if c.size else np.nan
    return res


def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(x), np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]
