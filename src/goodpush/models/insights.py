"""What drove progress, and when was it too much?

All functions operate on the daily / weekly tables from `goodpush.analyze`.
These are *observational* analyses of one athlete: treat them as evidence and
hypotheses, not proof of causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = {
    "hours_wk": "Avg hours / week",
    "tss_wk": "Avg TSS / week",
    "ctl_start": "Load (CTL) at start",
    "ctl_change": "CTL change over window",
    "f_z12": "Share of time in Z1-2 (endurance)",
    "f_z3": "Share of time in Z3 (tempo)",
    "f_z4": "Share of time in Z4 (sweet-spot/threshold)",
    "f_z5p": "Share of time in Z5+ (VO2max and above)",
    "f_hz12": "Share of HR time in Z1-2 (HR-based, FTP-independent)",
    "f_hz34": "Share of HR time in Z3-4 (tempo/threshold HR)",
    "f_hz5p": "Share of HR time in Z5 (above LTHR)",
    "hard_wk": "Hard sessions / week",
    "long_wk": "Long rides (3h+) / week",
    "rest_days_wk": "Rest days / week",
    "consistency": "Consistency (1 - CV of weekly TSS)",
    "zero_weeks": "Weeks with <1 h training",
    "acwr_max": "Max acute:chronic ratio",
    "indoor_frac": "Indoor share of riding",
    "sleep_h": "Avg sleep (h)",
    "sleep_z": "Sleep vs baseline (z)",
    "stress": "Garmin all-day stress",
    "rhr_z": "Resting HR vs baseline (z)",
    "hrv_z": "HRV vs baseline (z)",
    "perf_start": "Performance level at start",
}


def window_features(W: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Features summarising the n weeks ending at each week (inclusive)."""
    r = W.rolling(n, min_periods=n)
    F = pd.DataFrame(index=W.index)
    F["hours_wk"] = r["hours"].mean()
    F["tss_wk"] = r["tss"].mean()
    F["ctl_start"] = W["ctl"].shift(n)
    F["ctl_change"] = W["ctl"] - W["ctl"].shift(n)
    zs = {z: W[f"pz{z}_s"].rolling(n, min_periods=n).sum() for z in range(1, 8)}
    tot = sum(zs.values()).replace(0, np.nan)
    F["f_z12"] = (zs[1] + zs[2]) / tot
    F["f_z3"] = zs[3] / tot
    F["f_z4"] = zs[4] / tot
    F["f_z5p"] = (zs[5] + zs[6] + zs[7]) / tot
    hz = {z: W[f"hz{z}_s"].rolling(n, min_periods=n).sum() for z in range(1, 8)}
    htot = sum(hz.values()).replace(0, np.nan)
    F["f_hz12"] = (hz[1] + hz[2]) / htot
    F["f_hz34"] = (hz[3] + hz[4]) / htot
    F["f_hz5p"] = (hz[5] + hz[6] + hz[7]) / htot
    F["hard_wk"] = r["hard"].mean()
    F["long_wk"] = r["long"].mean()
    F["rest_days_wk"] = r["rest_days"].mean()
    F["consistency"] = 1 - r["tss"].std() / r["tss"].mean().replace(0, np.nan)
    F["zero_weeks"] = (W["hours"] < 1).astype(float).rolling(n, min_periods=n).sum()
    F["acwr_max"] = r["acwr_max"].max()
    F["indoor_frac"] = r["indoor_h"].sum() / r["ride_hours"].sum().replace(0, np.nan)
    F["sleep_h"] = W["sleep_h"].rolling(n, min_periods=max(2, n // 2)).mean()
    F["sleep_z"] = W["sleep_z"].rolling(n, min_periods=max(2, n // 2)).mean()
    F["stress"] = W["stress"].rolling(n, min_periods=max(2, n // 2)).mean()
    F["rhr_z"] = W["rhr_z"].rolling(n, min_periods=max(2, n // 2)).mean()
    F["hrv_z"] = W["hrv_z"].rolling(n, min_periods=max(2, n // 2)).mean()
    F["perf_start"] = W["perf"].shift(n)
    return F


def targets(W: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    T = pd.DataFrame(index=W.index)
    T["d_perf_pct"] = (W["perf"] / W["perf"].shift(n) - 1) * 100
    ef = W["ef"].rolling(3, min_periods=1).mean()
    T["d_ef_pct"] = (ef / ef.shift(n) - 1) * 100
    return T


def driver_analysis(W: pd.DataFrame, n: int = 8) -> dict:
    """Rank-correlate each training feature with the subsequent change in
    performance, and fit a small gradient-boosted model as a cross-check."""
    F = window_features(W, n)
    T = targets(W, n)
    active = (F["hours_wk"] > 1)
    out = {"window_weeks": n, "rows": []}
    for tcol in ("d_perf_pct", "d_ef_pct"):
        y = T[tcol]
        for f in FEATURES:
            x = F[f]
            m = active & x.notna() & y.notna()
            if m.sum() < 20 or x[m].nunique() < 4:
                continue
            rho, p = spearmanr(x[m], y[m])
            # effective sample size: overlapping windows -> roughly n_rows / n
            n_eff = max(3, m.sum() / n)
            out["rows"].append({"target": tcol, "feature": f, "label": FEATURES[f],
                                "rho": float(rho), "n_weeks": int(m.sum()),
                                "n_eff": float(n_eff),
                                # rough significance from effective n
                                "strong": bool(abs(rho) * np.sqrt(n_eff - 2) > 2.0)})
    tab = pd.DataFrame(out["rows"])
    out["table"] = tab
    out["model"] = _gbm(F, T["d_perf_pct"], active)
    out["F"], out["T"] = F, T
    return out


def _gbm(F, y, active) -> dict:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    except Exception:  # pragma: no cover
        return {"ok": False, "why": "scikit-learn not installed"}
    cols = [c for c in FEATURES if F[c].notna().mean() > 0.6]
    m = active & y.notna() & F[cols].notna().all(axis=1)
    if m.sum() < 60:
        return {"ok": False, "why": f"only {int(m.sum())} complete weeks"}
    X, yy = F.loc[m, cols], y[m]
    mdl = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    cv = cross_val_score(mdl, X, yy, cv=TimeSeriesSplit(5), scoring="r2")
    mdl.fit(X, yy)
    pi = permutation_importance(mdl, X, yy, n_repeats=5, random_state=0)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)
    return {"ok": True, "cv_r2": float(np.mean(cv)), "cv_r2_folds": [float(c) for c in cv],
            "importance": imp, "n": int(m.sum())}


# ---------------------------------------------------------------------------
def smoothed(W: pd.DataFrame, col: str = "perf") -> pd.Series:
    """the weekly marker with short gaps filled and a 3-week running median (what periods are measured on)."""
    return W[col].interpolate(limit=8, limit_area="inside").rolling(3, center=True, min_periods=1).median()


def marked_periods(W: pd.DataFrame, spans: list[dict], col: str = "perf") -> pd.DataFrame:
    """Periods you marked ([{start, end}], e.g. a gain on the load chart) in find_periods' shape: the weeks they
    fall in, the marker there and its change; `yours` True."""
    vals, rows = smoothed(W, col).to_numpy(), []
    for p in spans:                                    # a handful
        i = max(int(W.index.searchsorted(pd.Timestamp(p["start"]), "right")) - 1, 0)
        j = max(int(W.index.searchsorted(pd.Timestamp(p["end"]), "right")) - 1, i)
        rows.append({"start": W.index[i], "end": W.index[j], "weeks": j - i, "perf_start": vals[i], "perf_end": vals[j],
                     "change_pct": (vals[j] / vals[i] - 1) * 100, "yours": True})
    return pd.DataFrame(rows, columns=["start", "end", "weeks", "perf_start", "perf_end", "change_pct", "yours"])


def find_periods(W: pd.DataFrame, col="perf", min_gain=0.04, min_weeks=4, max_weeks=20,
                 direction=+1, max_periods=8) -> pd.DataFrame:
    """Non-overlapping periods of largest rise (direction=+1) or fall (-1)."""
    vals = smoothed(W, col).to_numpy()
    n = len(vals)
    cands = []
    for i in range(n):
        if not np.isfinite(vals[i]) or vals[i] <= 0:
            continue
        seg = vals[i + min_weeks: i + max_weeks + 1]
        if seg.size == 0 or not np.isfinite(seg).any():
            continue
        j = int(np.nanargmax(seg) if direction > 0 else np.nanargmin(seg)) + i + min_weeks
        g = vals[j] / vals[i] - 1
        if direction * g >= min_gain:
            cands.append((direction * g, i, j))
    cands.sort(reverse=True)
    used = np.zeros(n, bool)
    rows = []
    for g, i, j in cands:
        if used[i:j + 1].any():
            continue
        used[i:j + 1] = True
        rows.append({"start": W.index[i], "end": W.index[j], "weeks": j - i,
                     "perf_start": vals[i], "perf_end": vals[j],
                     "change_pct": direction * g * 100})
        if len(rows) >= max_periods:
            break
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True) if rows else \
        pd.DataFrame(columns=["start", "end", "weeks", "perf_start", "perf_end", "change_pct"])


def describe_periods(periods: pd.DataFrame, W: pd.DataFrame, lead_weeks: int = 0) -> pd.DataFrame:
    """Training characteristics during each period vs the athlete's typical week."""
    if periods.empty:
        return periods
    act = W[W["hours"] > 1]
    typical = {
        "hours": act["hours"].median(), "tss": act["tss"].median(), "hard": act["hard"].median(),
        "long": act["long"].mean(), "f_z12": act["f_z12"].median(), "f_z4": act["f_z4"].median(),
        "f_z5p": act["f_z5p"].median(), "sleep_h": act["sleep_h"].median(),
    }
    rows = []
    for p in periods.itertuples():
        s = p.start - pd.Timedelta(weeks=lead_weeks)
        seg = W.loc[s:p.end]
        tot = seg[[f"pz{z}_s" for z in range(1, 8)]].sum()
        ts = tot.sum() or np.nan
        r = {
            "start": p.start.date(), "end": p.end.date(), "weeks": p.weeks,
            "perf_start_w": round(p.perf_start) if np.isfinite(p.perf_start) else None,
            "perf_end_w": round(p.perf_end) if np.isfinite(p.perf_end) else None, "yours": bool(getattr(p, "yours", False)),
            "change_pct": round(p.change_pct, 1) if np.isfinite(p.change_pct) else None,
            "hours_wk": seg["hours"].mean(), "tss_wk": seg["tss"].mean(),
            "ctl_start": seg["ctl"].iloc[0], "ctl_end": seg["ctl"].iloc[-1],
            "hard_wk": seg["hard"].mean(), "long_wk": seg["long"].mean(),
            "f_z12": (tot["pz1_s"] + tot["pz2_s"]) / ts, "f_z3": tot["pz3_s"] / ts,
            "f_z4": tot["pz4_s"] / ts, "f_z5p": (tot["pz5_s"] + tot["pz6_s"] + tot["pz7_s"]) / ts,
            "rest_days_wk": seg["rest_days"].mean(), "sleep_h": seg["sleep_h"].mean(),
            "indoor_frac": seg["indoor_h"].sum() / max(seg["ride_hours"].sum(), 1e-9),
        }
        notes = []
        if r["hours_wk"] > 1.2 * typical["hours"]:
            notes.append(f"more volume ({r['hours_wk']:.1f} vs {typical['hours']:.1f} h/wk)")
        elif r["hours_wk"] < 0.8 * typical["hours"]:
            notes.append(f"less volume ({r['hours_wk']:.1f} vs {typical['hours']:.1f} h/wk)")
        if r["hard_wk"] > typical["hard"] + 0.75:
            notes.append(f"more hard sessions ({r['hard_wk']:.1f}/wk)")
        elif r["hard_wk"] < typical["hard"] - 0.75:
            notes.append(f"fewer hard sessions ({r['hard_wk']:.1f}/wk)")
        if r["f_z4"] > 1.4 * typical["f_z4"]:
            notes.append(f"more sweet-spot/threshold ({r['f_z4']:.0%} of time)")
        if r["f_z5p"] > 1.4 * typical["f_z5p"]:
            notes.append(f"more VO2max+ work ({r['f_z5p']:.0%} of time)")
        if r["f_z12"] > typical["f_z12"] + 0.08:
            notes.append(f"more polarised/endurance-heavy ({r['f_z12']:.0%} Z1-2)")
        if r["long_wk"] > typical["long"] + 0.3:
            notes.append(f"more long rides ({r['long_wk']:.1f}/wk)")
        if r["ctl_end"] - r["ctl_start"] > 10:
            notes.append(f"CTL built {r['ctl_start']:.0f}->{r['ctl_end']:.0f}")
        if r["ctl_end"] - r["ctl_start"] < -10:
            notes.append(f"CTL fell {r['ctl_start']:.0f}->{r['ctl_end']:.0f}")
        r["what_was_different"] = "; ".join(notes) or "similar to a typical block"
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def overreach_episodes(d: pd.DataFrame, W: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag days/weeks of excessive load and classify what happened next.

    Returns (weekly_flags, episodes).
    """
    f = pd.DataFrame(index=d.index)
    rel_tsb = d["tsb"] / d["ctl"].replace(0, np.nan)
    f["load_spike"] = (d["acwr"] > 1.5) & (d["ctl"] > 15)
    f["deep_fatigue"] = (d["tsb"] < -30) | ((rel_tsb < -0.45) & (d["ctl"] > 20))
    f["fast_ramp"] = d["ramp"] > 8
    f["monotony"] = (d["monotony"] > 2.0) & (d["strain"] > d["strain"].quantile(0.9))
    f["rhr_up"] = d["rhr_z"] > 1.5
    f["hrv_down"] = d["hrv_z"] < -1.5
    f["sleep_short"] = (d["sleep_h"].rolling(3, min_periods=2).mean()
                        < d["sleep_h_base"] - 1.0)
    f["mood"] = d["mood_bad"] > 0
    f["ef_down"] = d["ef"] < 0.93 * d["ef_base"]
    wk = f.resample("W-MON", label="left", closed="left").sum()
    wk = wk.reindex(W.index).fillna(0)
    load_cols = ["load_spike", "deep_fatigue", "fast_ramp", "monotony"]
    rec_cols = ["rhr_up", "hrv_down", "sleep_short", "mood", "ef_down"]
    wk["load_flags"] = (wk[load_cols] >= 2).sum(axis=1)
    wk["recovery_flags"] = (wk[rec_cols] >= 2).sum(axis=1) + (wk["mood"] >= 1) * 0
    hot = wk["load_flags"] >= 1

    # group consecutive hot weeks into episodes
    eps = []
    idx = list(W.index)
    i = 0
    while i < len(idx):
        if not hot.iloc[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(idx) and (hot.iloc[j + 1] or (j + 2 < len(idx) and hot.iloc[j + 2])):
            j += 1
        while j > i and not hot.iloc[j]:
            j -= 1
        s, e = idx[i], idx[j]
        seg = W.loc[s:e]
        pre = W.loc[s - pd.Timedelta(weeks=6): s - pd.Timedelta(weeks=1)]
        post = W.loc[e + pd.Timedelta(weeks=1): e + pd.Timedelta(weeks=6)]
        post2 = W.loc[e + pd.Timedelta(weeks=1): e + pd.Timedelta(weeks=2)]
        perf_before = pre["perf"].median() if len(pre) else np.nan
        perf_after = post["perf"].max() if len(post) else np.nan
        perf_low = post["perf"].min() if len(post) else np.nan
        pre_h = pre["hours"].mean() if len(pre) else np.nan
        flags_l = [c for c in load_cols if wk.loc[s:e, c].sum() >= 2]
        # recovery markers during the block and the 2 weeks after; daily markers need >= 3 days
        need = {"rhr_up": 3, "hrv_down": 3, "sleep_short": 3, "mood": 1, "ef_down": 2}
        flags_r = [c for c in rec_cols
                   if wk.loc[s:e + pd.Timedelta(weeks=2), c].sum() >= need.get(c, 2)]
        forced_break = bool(len(post2) and pre_h and post2["hours"].mean() < 0.4 * pre_h)
        gain = (perf_after / perf_before - 1) * 100 if perf_before else np.nan
        drop = (perf_low / perf_before - 1) * 100 if perf_before else np.nan
        events = "; ".join(sorted(set(v for v in W.loc[s:e + pd.Timedelta(weeks=2), "event"]
                                      if v)))
        gained = np.isfinite(gain) and gain >= 2
        if forced_break or (np.isfinite(drop) and drop <= -4 and not gained) \
                or (len(flags_r) >= 2 and not gained):
            verdict = "too much"
        elif gained and len(flags_r) >= 2:
            verdict = "productive but costly"
        elif gained:
            verdict = "productive overload"
        else:
            verdict = "neutral / unclear"
        eps.append({
            "start": s.date(), "end": (e + pd.Timedelta(days=6)).date(), "weeks": len(seg),
            "peak_atl": seg["atl"].max(), "ctl": seg["ctl"].max(), "min_tsb": seg["tsb_min"].min(),
            "max_acwr": seg["acwr_max"].max(), "hours_wk": seg["hours"].mean(),
            "tss_wk": seg["tss"].mean(), "load_flags": ", ".join(flags_l),
            "recovery_flags": ", ".join(flags_r), "forced_break_after": forced_break,
            "perf_before": perf_before, "perf_best_after": perf_after,
            "change_after_pct": gain, "worst_after_pct": drop, "events": events,
            "verdict": verdict,
        })
        i = j + 1
    return wk, pd.DataFrame(eps)


def banister_performance(acts: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Daily performance marker: best '20-min equivalent' power on days with a
    near-maximal effort (>= 92 % of the previous-90-day best)."""
    r = acts[acts["sport"].eq("ride") & acts["has_power"]]
    eq = pd.concat([r["mmp_1200"], 0.88 * r["mmp_480"], 0.82 * r["mmp_300"]], axis=1).max(axis=1)
    day = eq.groupby(r["date"]).max().reindex(daily.index)
    prior = day.rolling(90, min_periods=1).max().shift(1)
    perf = day.where(day >= 0.92 * prior)
    return perf


def breaks(W: pd.DataFrame, min_weeks=2) -> pd.DataFrame:
    """Runs of >= min_weeks weeks with under 1.5 h training (illness, life, off-season)."""
    low = (W["hours"] < 1.5).to_numpy()
    rows, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            if j - i + 1 >= min_weeks:
                rows.append({"start": W.index[i].date(), "weeks": j - i + 1,
                             "ctl_before": W["ctl"].iloc[max(0, i - 1)],
                             "ctl_after": W["ctl"].iloc[j]})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recovery: how training affects sleep / HRV / resting HR, and how recovery
# state relates to performance. Garmin dates sleep/HRV/RHR by the morning you
# wake up, so training on day D shows up in the metrics dated D+1.
# ---------------------------------------------------------------------------
_MARKERS = [("hrv_z", "HRV vs baseline (z)"), ("rhr_z", "Resting HR vs baseline (z)"),
            ("sleep_h", "Sleep (h)"), ("sleep_score", "Sleep score"), ("readiness", "Readiness")]


def _bin_table(x: pd.Series, bins, labels, cols: dict, min_n=5) -> pd.DataFrame:
    b = pd.cut(x, bins=bins, labels=labels)
    rows = []
    for lab in labels:
        m = b == lab
        r = {"bin": lab, "days": int(m.sum())}
        for c, s in cols.items():
            v = s[m].dropna()
            r[c] = float(v.mean()) if len(v) >= min_n else np.nan
            r[c + "_n"] = int(len(v))
        rows.append(r)
    return pd.DataFrame(rows)


def recovery_analysis(d: pd.DataFrame) -> dict:
    out = {}
    have = [c for c, _ in _MARKERS if c in d and d[c].notna().sum() > 30]
    out["available"] = have
    if not have:
        return out
    nxt = {c: d[c].shift(-1) for c in have}
    out["dose_next_morning"] = _bin_table(
        d["tss"], [-1, 10, 60, 120, 200, 1e9],
        ["rest (<10)", "easy (10-60)", "moderate (60-120)", "hard (120-200)", "very hard (200+)"], nxt)
    out["form_same_day"] = _bin_table(
        d["tsb"], [-1e9, -30, -20, -10, 0, 10, 1e9],
        ["< -30", "-30..-20", "-20..-10", "-10..0", "0..10", "> 10"], {c: d[c] for c in have})
    # recovery curve after very hard days
    hard = d.index[d["tss"] >= 150]
    curve = []
    for k in range(1, 6):
        r = {"days_after": k}
        for c in ("hrv_z", "rhr_z", "sleep_h"):
            if c in have:
                idx = hard + pd.Timedelta(days=k)
                r[c] = float(d[c].reindex(idx).mean())
        curve.append(r)
    out["after_hard_day"] = pd.DataFrame(curve)
    out["n_hard_days"] = int(len(hard))
    # lagged rank correlation of daily load with markers
    lag = []
    for c in have:
        for k in (1, 2, 3):
            y = d[c].shift(-k)
            m = d["tss"].notna() & y.notna()
            if m.sum() > 60:
                rho, _ = spearmanr(d.loc[m, "tss"], y[m])
                lag.append({"marker": c, "lag_days": k, "rho": float(rho), "n": int(m.sum())})
    out["load_lag_corr"] = pd.DataFrame(lag)
    # recovery state -> performance on the day
    prior_best = d["eq20_test"].rolling(90, min_periods=1).max().shift(1)
    perf_ratio = d["eq20_test"] / prior_best
    ef_ratio = d["ef"] / d["ef_base"]
    targets = {"test_vs_90d_best": perf_ratio, "ef_vs_6wk": ef_ratio}
    if "hrv_z" in have:
        out["hrv_vs_performance"] = _bin_table(d["hrv_z"], [-1e9, -1, 0, 1, 1e9],
                                               ["low (< -1)", "below (-1..0)", "above (0..1)",
                                                "high (> 1)"], targets, min_n=4)
    if "sleep_h" in have:
        out["sleep_vs_performance"] = _bin_table(d["sleep_h"], [0, 6, 7, 8, 24],
                                                 ["< 6 h", "6-7 h", "7-8 h", "8+ h"], targets, min_n=4)
    if "readiness" in have:
        out["readiness_vs_performance"] = _bin_table(d["readiness"], [-1, 33, 50, 75, 101],
                                                     ["low (<33)", "moderate (33-50)",
                                                      "high (50-75)", "prime (75+)"], targets, min_n=4)
    # long-term trends
    yr = d.groupby(d.index.year)
    trend = pd.DataFrame({c: yr[c].median() for c in ("sleep_h", "rhr", "hrv", "stress",
                                                       "bedtime_h", "sleep_score")
                          if c in d and d[c].notna().any()})
    trend["ctl"] = yr["ctl"].mean()
    out["yearly"] = trend.dropna(how="all", subset=[c for c in trend.columns if c != "ctl"])
    w = d.resample("W-MON").mean(numeric_only=True)
    lt = []
    for c in ("rhr", "hrv", "sleep_h"):
        if c in w and w[c].notna().sum() > 30:
            m = w[c].notna() & w["ctl"].notna()
            rho, _ = spearmanr(w.loc[m, "ctl"], w.loc[m, c])
            lt.append({"marker": c, "rho_with_ctl": float(rho), "weeks": int(m.sum())})
    out["fitness_vs_markers"] = pd.DataFrame(lt)
    return out
