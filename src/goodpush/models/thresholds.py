"""Threshold estimates side by side.

* FTP in use (your [ftp.history] entry or the critical-power model)
* Garmin's FTP: what your Garmin uses (its export history, then each ride's FIT file); Garmin's own
  estimate when Auto Detect is on. Shown for comparison, the plan's targets don't use it
* critical power and W' from your best 3-20 min efforts: only as high as the efforts you have
  actually ridden all-out, so without a recent max effort it is a floor
* LT1 / LT2 from DFA alpha1 (alphaHRV): HRVT1 at alpha1 = 0.75, HRVT2 at 0.5, read from a recent ramp
  test (reads high: alpha1 lags the rising power) and from steady riding of the last 4 / 6 weeks (reads low:
  alpha1 also falls with time into a ride). Each is a reading; what's shown is the physio model's LT1 / CP,
  which weighs them all (and Garmin's, the heart-rate ones, ...) with each kind's own fitted bias
* glycolytic indicators: VLamax needs blood lactate - there is no validated way to get it from
  power alone - so this shows what moves with it: W' and 15 / 30 / 60 s power relative to CP
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ramp import RampTest

RAMP_FRESH_DAYS = 56     # a ramp test counts for 8 weeks
LT1_FRESH_DAYS = 28
LT2_FRESH_DAYS = 42


def _f(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _last(daily: pd.DataFrame, col: str, today, days: int | None = None):
    """Latest finite value of a daily column (within `days`) and its date."""
    if col not in daily:
        return None, None
    s = daily[col].loc[:today].dropna()
    if days is not None:
        s = s[s.index > today - pd.Timedelta(days=days)]
    return (float(s.iloc[-1]), s.index[-1].date().isoformat()) if len(s) else (None, None)


def ramp_tests(acts: pd.DataFrame) -> list[dict]:
    """your ramp tests: alpha1's thresholds where it was recorded, else the power at your LT1 / LT2 heart rate on
    the ramp (`by_hr`; models.ramp)."""
    if "ramp_top_w" not in acts:
        return []
    r = acts[acts["ramp_top_w"].notna()].sort_values("date")
    r = r.assign(lt2_loose=RampTest.lt2_check(r))       # an LT2 at exhaustion: only a loose cap
    col = lambda x, k, c: x.get(f"ramp_{k}_{c}") if pd.notna(x.get(f"ramp_{k}_w")) else x.get(f"ramp_hr_{k}_{c}")  # noqa: E731
    return [{"date": x["date"].date().isoformat(), "id": str(x["id"]).removesuffix(".0"), "name": str(x.get("name") or ""),
             "lt1_w": _f(col(x, "lt1", "w")), "lt1_hr": _f(col(x, "lt1", "hr")),
             "lt2_w": _f(col(x, "lt2", "w")), "lt2_hr": _f(col(x, "lt2", "hr")),
             "by_hr": bool(pd.isna(x.get("ramp_lt1_w")) and pd.isna(x.get("ramp_lt2_w"))),
             "top_w": _f(x.get("ramp_top_w")), "max_hr": _f(x.get("ramp_max_hr")), "minutes": _f(x.get("ramp_min")),
             "hrr_60": _f(x.get("ramp_hrr_60")), "lt2_loose": x["lt2_loose"] or None}
            for _, x in r.iterrows()]


MEASURED_S = (60, 300, 1200)              # the efforts shown as measured (Rider page)
MEASURED_DAYS = 365


def measured(acts: pd.DataFrame, today: pd.Timestamp) -> dict:
    """{seconds: your best effort that long in the last MEASURED_DAYS that heart rate confirmed as all-out (W, day,
    peak heart rate, your maximum then) or None} - measured, where the rest of the curve (FTP, CP, W') is inferred."""
    from ..metrics.effort_hr import EffortHeartRate
    if acts is None or not len(acts) or "max_hr" not in acts:
        return {str(s): None for s in MEASURED_S}
    rides = acts[(acts["sport"].eq("ride") if "sport" in acts else True)
                 & (pd.to_datetime(acts["date"]) > today - pd.Timedelta(days=MEASURED_DAYS))
                 & (pd.to_datetime(acts["date"]) <= today)]
    allr = acts[acts["sport"].eq("ride")] if "sport" in acts else acts
    hit = EffortHeartRate.maxed(allr, float(allr["max_hr"].quantile(0.995) or 190)).reindex(rides.index)
    ref = EffortHeartRate.max_hr_then(allr, float(allr["max_hr"].quantile(0.995) or 190)).reindex(rides.index)
    out = {}
    for s in MEASURED_S:
        w = rides.get(f"effort_w_{s}", pd.Series(np.nan, index=rides.index)).where(hit[s].fillna(False).astype(bool))
        if w.notna().any():
            i = w.idxmax()
            out[str(s)] = {"w": _f(w[i]), "date": pd.Timestamp(rides.at[i, "date"]).date().isoformat(),
                           "id": str(rides.at[i, "id"]).removesuffix(".0") if "id" in rides else None,
                           "hr": _f(rides.at[i, f"effort_hr_{s}"]), "max_hr": _f(ref[i])}
        else:
            out[str(s)] = None
    return out


def summary(daily: pd.DataFrame, acts: pd.DataFrame, bests_90d: dict, bests_all: dict, ftp_cfg: dict,
            today: pd.Timestamp, states: pd.DataFrame | None = None) -> dict:
    """`states`: the physio model's daily capacities (PhysioRun.STATES), whose LT1 / CP lead LT1 / LT2."""
    d = daily.loc[:today]
    ftp = float(d["ftp"].iloc[-1])
    hist = {pd.Timestamp(k): float(v) for k, v in (ftp_cfg.get("history") or {}).items()}
    set_on = max((k for k in hist if k <= today), default=None) if ftp_cfg.get("mode") == "manual" else None
    auto = {"model": "the model's LT2 (your tests weigh most)", "fitted": "fitted to Garmin's FTP, yours and your tests",
            "cp": "critical power", "p20": "95 % of your best 20 min"}.get(ftp_cfg.get("auto_method"), "model")
    out = {"ftp": {"w": _f(ftp), "source": "your setting" if set_on is not None else auto,
                   "since": set_on.date().isoformat() if set_on is not None else None,
                   "model_w": _f(d["ftp_auto"].iloc[-1]) if "ftp_auto" in d else None}}
    out["measured"] = measured(acts, today)
    g = d["ftp_garmin"].dropna() if "ftp_garmin" in d else pd.Series(dtype=float)
    if len(g):                                    # the latest value and since when it has been that
        since = g.index[g.ne(g.shift()).to_numpy()][-1]
        out["garmin"] = {"w": _f(g.iloc[-1]), "date": g.index[-1].date().isoformat(),
                         "since": since.date().isoformat(),
                         "note": "The FTP your Garmin uses: its own estimate when Auto Detect FTP is on, else what "
                                 "you entered. For comparison only; the plan's targets use the FTP above."}
    cp90, _ = _last(d, "cp", today)
    wp90, _ = _last(d, "wprime", today)
    cp42, _ = _last(d, "cp42", today)
    wp42, _ = _last(d, "wprime42", today)
    b = lambda k, src=bests_90d: (src.get(str(k)) or {}).get("w")  # noqa: E731
    out["cp"] = {"cp_90d": _f(cp90), "wprime_90d_kj": _f(wp90 / 1000 if wp90 else None, 1),
                 "cp_42d": _f(cp42), "wprime_42d_kj": _f(wp42 / 1000 if wp42 else None, 1),
                 "best_20min_90d": b(1200), "best_5min_90d": b(300),
                 "note": "From your best 3-20 min efforts of the last 90 days. Only as high as efforts you "
                         "rode all-out: without a recent max effort it's a floor, not your limit."}
    ramps = ramp_tests(acts)
    fresh = [x for x in ramps if not x["by_hr"]           # alpha1's; HR-based ones only go to the physio model
             and pd.Timestamp(x["date"]) > today - pd.Timedelta(days=RAMP_FRESH_DAYS)]
    rp = fresh[-1] if fresh else None
    model = states.loc[:today].iloc[-1] if states is not None and len(states.loc[:today]) else None
    for key, col, wcol, hcol, days in (("lt1", "lt1", "hrvt1_28d", "hrvt1_hr_28d", LT1_FRESH_DAYS),
                                       ("lt2", "cp", "hrvt2_42d", "hrvt2_hr_42d", LT2_FRESH_DAYS)):
        w, dt = _last(d, wcol, today, days)
        hr, _ = _last(d, hcol, today, days)
        readings = ([{"w": rp[f"{key}_w"], "hr": rp[f"{key}_hr"], "source": "ramp test", "date": rp["date"],
                      "ride": rp["id"], **({"loose": rp["lt2_loose"]} if key == "lt2" and rp.get("lt2_loose") else {})}]
                    if rp and rp.get(f"{key}_w") else []) + \
            ([{"w": _f(w), "hr": _f(hr), "date": dt, "source": f"steady riding, last {days // 7} weeks"}] if w else [])
        if model is not None and np.isfinite(model.get(col, np.nan)):
            out[key] = {"w": _f(model[col]), "lo": _f(model.get(f"{col}_p05")), "hi": _f(model.get(f"{col}_p95")),
                        "hr": next((r["hr"] for r in readings[::-1] if r["hr"]), None),   # steady riding's first
                        "date": today.date().isoformat(), "source": "all your evidence (physio model)"}
        else:
            out[key] = dict(readings[0]) if readings else {"w": None, "hr": None, "date": None, "source": None}
        out[key]["readings"] = readings
    out["lt1"]["what"] = "alpha1 drops through 0.75 (HRVT1, about the first lactate / ventilatory threshold)"
    out["lt2"]["what"] = "alpha1 drops through 0.5 (HRVT2, about the second threshold / MLSS)"
    if not rp:
        out["lt2"]["note"] = ("From ordinary rides alpha1 also falls the longer you ride, so this tends to read low. "
                              "A ramp test with alphaHRV gives the validated number.")
    out["ramps"] = ramps[-6:]
    p15, p30, p60 = b(15), b(30), b(60)
    ratio = lambda p: _f(p / cp90, 2) if p and cp90 else None  # noqa: E731
    out["glycolytic"] = {"p15": p15, "p30": p30, "p60": p60, "p30_best": b(30, bests_all),
                         "p30_cp": ratio(p30), "p60_cp": ratio(p60), "wprime_kj": out["cp"]["wprime_90d_kj"],
                         "note": "VLamax needs blood lactate; there's no validated way to get it from power alone. "
                                 "These go with it: higher 15-60 s power and W' relative to CP = a more glycolytic, "
                                 "punchier profile (higher VLamax); lower = more diesel."}
    return out
