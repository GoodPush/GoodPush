"""What did the build-up to your best performances look like, and how fast can
you safely ramp?

Uses the weekly table (with n_<workout type> counts from models.workouts) and the
illness episodes from models.illness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .workouts import TYPES

GROUPS = {  # coarse groups for charts / tables
    "easy": ["recovery", "endurance"],
    "long": ["long_endurance"],
    "tempo/sweet spot": ["tempo", "sweetspot"],
    "threshold": ["threshold"],
    "vo2max/anaerobic": ["vo2max", "anaerobic"],
    "race/test": ["race", "test"],
}


RAMP_BINS = [-99, 0, 1.5, 3, 4.5, 6, 99]          # 4-week CTL ramp (points a week) bins of ramp_sustainability
RAMP_LABELS = ["falling (<0)", "0-1.5", "1.5-3", "3-4.5", "4.5-6", "> 6"]


def _n(W, t):
    c = f"n_{t}"
    return W[c] if c in W else pd.Series(0.0, index=W.index)


def peak_week(W: pd.DataFrame) -> pd.Timestamp | None:
    t = W["perf_test"].dropna()
    return t.idxmax() if len(t) else None


def block_table(W: pd.DataFrame, end: pd.Timestamp, weeks=52, block=4) -> pd.DataFrame:
    """The `weeks` before `end`, summarised in `block`-week chunks."""
    seg = W.loc[end - pd.Timedelta(weeks=weeks - 1): end]
    g = np.arange(len(seg)) // block
    hz = seg[[f"hz{i}_s" for i in range(1, 8)]]
    rows = []
    for k, part in seg.groupby(g):
        h = hz.loc[part.index].sum()
        r = {"start": part.index[0].date(), "hours_wk": part["hours"].mean(),
             "tss_wk": part["tss"].mean(), "ctl_end": part["ctl"].iloc[-1],
             "ramp_wk": (part["ctl"].iloc[-1] - W["ctl"].shift(1).loc[part.index[0]]) / len(part),
             "best_test": part["perf_test"].max(), "ef": part["ef"].mean(),
             "hr_z12": (h["hz1_s"] + h["hz2_s"]) / max(h.sum(), 1),
             "rest_days_wk": part["rest_days"].mean(), "sleep_h": part["sleep_h"].mean(),
             "rhr": part["rhr"].mean()}
        for t in TYPES:
            r[t] = _n(part, t).sum() / len(part)
        rows.append(r)
    return pd.DataFrame(rows)


def build_summary(W: pd.DataFrame, end: pd.Timestamp, weeks=16) -> dict:
    s = W.loc[end - pd.Timedelta(weeks=weeks - 1): end]
    hz = s[[f"hz{i}_s" for i in range(1, 8)]].sum()
    out = {"end": end.date(), "hours_wk": s["hours"].mean(), "ctl_end": s["ctl"].iloc[-1],
           "ctl_gain": s["ctl"].iloc[-1] - s["ctl"].iloc[0],
           "max_ramp_4wk": ((s["ctl"] - s["ctl"].shift(4)) / 4).max(),
           "hr_z12": (hz["hz1_s"] + hz["hz2_s"]) / max(hz.sum(), 1),
           "rest_days_wk": s["rest_days"].mean(), "sleep_h": s["sleep_h"].mean(),
           "long_wk": s["long"].mean(), "best_test": s["perf_test"].max()}
    for t in TYPES:
        out[t] = _n(s, t).mean()
    out["quality_wk"] = sum(out[t] for t in ("tempo", "sweetspot", "threshold", "vo2max",
                                             "anaerobic"))
    return out


def compare_builds(W: pd.DataFrame, periods: pd.DataFrame, weeks=16) -> pd.DataFrame:
    rows = []
    for p in periods.itertuples():
        r = build_summary(W, pd.Timestamp(p.end), weeks)
        r["label"] = f"build to {p.end} (+{p.change_pct:.0f}%)"
        r["gain_pct"] = p.change_pct
        rows.append(r)
    r = build_summary(W, W.index[-1], weeks)
    r["label"] = f"last {weeks} weeks"
    r["gain_pct"] = np.nan
    rows.append(r)
    return pd.DataFrame(rows)


def ramp_sustainability(W: pd.DataFrame, onsets) -> dict:
    """Trailing 4-week CTL ramp vs what happened next (illness, performance, efficiency).

    Weeks just after an illness are excluded (the ramp there is a comeback, and the
    illness would otherwise be counted as caused by it)."""
    w = W.copy()
    on = pd.to_datetime(pd.Series(list(onsets))) if len(onsets) else pd.Series([], dtype="datetime64[ns]")
    w["ramp4"] = (w["ctl"] - w["ctl"].shift(4)) / 4
    w["ill_next6"] = [bool(((on > t) & (on <= t + pd.Timedelta(weeks=6))).any()) for t in w.index]
    w["ill_prev5"] = [bool(((on > t - pd.Timedelta(weeks=5)) & (on <= t)).any()) for t in w.index]
    w["gain"] = (w["perf"].shift(-6) / w["perf"].shift(4) - 1) * 100
    ef = w["ef"].rolling(3, min_periods=1).mean()
    w["ef_gain"] = (ef.shift(-4) / ef.shift(4) - 1) * 100
    w["rhr_z_next4"] = w["rhr_z"].shift(-1).rolling(4, min_periods=2).mean().shift(-3)
    act = w[(w["hours"] > 1) & w["ramp4"].notna() & ~w["ill_prev5"]]
    b = pd.cut(act["ramp4"], RAMP_BINS, labels=RAMP_LABELS)
    tab = act.groupby(b, observed=False).agg(
        weeks=("ramp4", "size"), illness_next_6wk=("ill_next6", "mean"),
        power_change_pct=("gain", "median"), ef_change_pct=("ef_gain", "median"),
        rhr_z_after=("rhr_z_next4", "mean")).reset_index().rename(columns={"ramp4": "ramp"})
    tab = tab.rename(columns={tab.columns[0]: "ctl_ramp_per_week"})
    # sustained ramps: runs of >= 4 weeks with trailing 4-wk ramp >= 3
    runs, idx, flag = [], list(w.index), (w["ramp4"] >= 3).to_numpy()
    i = 0
    while i < len(idx):
        if flag[i]:
            j = i
            while j + 1 < len(idx) and flag[j + 1]:
                j += 1
            s, e = idx[max(0, i - 4)], idx[j]
            runs.append({
                "start": s.date(), "end": e.date(), "weeks": (e - s).days // 7,
                "ctl_from": w["ctl"].get(s), "ctl_to": w["ctl"].get(e),
                "avg_ramp": (w["ctl"].get(e) - w["ctl"].get(s)) / max((e - s).days / 7, 1),
                "illness_within_3wk": bool(((on >= s) & (on <= e + pd.Timedelta(weeks=3))).any()),
                "test_before": w["perf"].loc[:s].dropna().iloc[-1] if w["perf"].loc[:s].notna().any() else np.nan,
                "best_test_8wk_after": w["perf_test"].loc[e: e + pd.Timedelta(weeks=8)].max(),
            })
            i = j + 1
        else:
            i += 1
    return {"table": tab, "runs": pd.DataFrame(runs),
            "illness_base_rate": float(act["ill_next6"].mean()) if len(act) else np.nan, "illness_episodes": len(on)}


def type_drivers(W: pd.DataFrame, n=8) -> pd.DataFrame:
    """Rank correlation of sessions/week of each type (n-week windows) with the change in
    tested power and aerobic efficiency over the same window."""
    y_p = (W["perf"] / W["perf"].shift(n) - 1) * 100
    ef = W["ef"].rolling(3, min_periods=1).mean()
    y_e = (ef / ef.shift(n) - 1) * 100
    feats = {f"{t}": _n(W, t).rolling(n).mean() for t in TYPES}
    feats["quality (tempo..vo2)"] = sum(_n(W, t) for t in ("tempo", "sweetspot", "threshold",
                                                           "vo2max", "anaerobic")).rolling(n).mean()
    feats["hours"] = W["hours"].rolling(n).mean()
    rows = []
    for k, x in feats.items():
        for name, y in (("power", y_p), ("efficiency", y_e)):
            m = x.notna() & y.notna() & (W["hours"] > 1)
            if m.sum() < 30 or x[m].nunique() < 3:
                continue
            rho, _ = spearmanr(x[m], y[m])
            n_eff = m.sum() / n
            rows.append({"feature": k, "target": name, "rho": float(rho), "weeks": int(m.sum()),
                         "notable": bool(abs(rho) * np.sqrt(max(n_eff - 2, 1)) > 2)})
    return pd.DataFrame(rows)


def buildup_analysis(W: pd.DataFrame, periods: pd.DataFrame, onsets) -> dict:
    pk = peak_week(W)
    return {"peak_week": pk, "peak_value": float(W["perf_test"].max()) if pk is not None else np.nan,
            "peak_blocks": block_table(W, pk) if pk is not None else pd.DataFrame(),
            "builds": compare_builds(W, periods) if len(periods) else pd.DataFrame(),
            "ramp": ramp_sustainability(W, onsets), "type_drivers": type_drivers(W)}
