"""Assemble the athlete profile and all findings into one results dict."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics import load as L
from ..metrics.power import MMP_DURATIONS
from ..models import insights as I
from ..models.banister import Banister
from ..models.cp import fit_cp, CP_DURATIONS
from ..models.profile import category, phenotype, profile_scores


def _f(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, nd)


def _best(acts, d, since=None):
    r = acts[acts["sport"].eq("ride") & acts["has_power"]]
    if since is not None:
        r = r[r["date"] >= since]
    col = f"mmp_{d}"
    v = r[col].dropna().sort_values(ascending=False)
    if v.empty:
        return None, None
    # drop isolated outliers (power-meter spikes): top value >25 % (short) / 15 % above the next
    tol = 1.25 if d <= 60 else 1.15
    while len(v) > 2 and v.iloc[0] > tol * v.iloc[1]:
        v = v.iloc[1:]
    i = v.index[0]
    return float(v.iloc[0]), r.at[i, "date"].date().isoformat()


def build_results(proj, r: dict) -> dict:
    acts, daily, W, info = r["acts"], r["daily"], r["weekly"], dict(r["info"])
    sex = proj.athlete.get("sex", "male")
    today = acts["date"].max()  # "now" = the 90 days up to the latest activity
    since90 = today - pd.Timedelta(days=90)
    weight = info.get("weight_kg")
    res = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
           "athlete": {"name": proj.athlete.get("name", ""), "sex": sex,
                       "weight_kg": _f(weight), "weight_source": info.get("weight_source"),
                       "max_hr": _f(info["max_hr"], 0), "lthr": _f(info["lthr"], 0),
                       "lthr_source": info.get("lthr_source"),
                       "resting_hr": _f(info["resting_hr"], 0),
                       "resting_hr_source": info.get("resting_hr_source")}}

    # ---- data coverage --------------------------------------------------------
    rides = acts[acts["sport"].eq("ride")]
    res["data"] = {
        "first": acts["date"].min().date().isoformat(), "last": acts["date"].max().date().isoformat(),
        "activities": int(len(acts)), "rides": int(len(rides)),
        "rides_with_power": int(rides["has_power"].sum()),
        "tss_sources": acts["tss_source"].value_counts().to_dict(),
        "wellness_days": info.get("wellness_days", 0),
        "parse_errors": int((acts.get("error", pd.Series(dtype=str)).fillna("").astype(str)
                             .replace("nan", "") != "").sum()),
    }

    # ---- power profile ---------------------------------------------------------
    cur, peak = {}, {}
    for d in MMP_DURATIONS:
        cur[d] = _best(acts, d, since90)
        peak[d] = _best(acts, d)
    env90 = {d: cur[d][0] for d in CP_DURATIONS}
    cp, wp, r2 = fit_cp(CP_DURATIONS, [env90[d] if env90[d] else np.nan for d in CP_DURATIONS])
    last_ride = acts["date"].max() if len(acts) else daily.index[-1]
    ftp_now = float(daily["ftp"].loc[:last_ride].iloc[-1])   # same model number the plan uses
    # best 90-day window ever: the max of rolling-90 CP
    # from your first power data on, as the physio model (before it FTP is only a starting guess)
    measured = daily["ftp_auto"][daily["day_env_ok"].cummax()] if "day_env_ok" in daily else daily["ftp_auto"]
    measured = measured if measured.notna().any() else daily["ftp_auto"]
    ftp_peak = float(measured.max())
    ftp_peak_date = measured.idxmax().date().isoformat()

    def wkg(v):
        return v / weight if v and weight and np.isfinite(weight) else np.nan

    def prof(src, ftp):
        return {"5s": wkg(src[5][0]), "1min": wkg(src[60][0]), "5min": wkg(src[300][0]),
                "ftp": wkg(ftp)}
    sc_cur = profile_scores(prof(cur, ftp_now), sex)
    sc_peak = profile_scores(prof(peak, ftp_peak), sex)
    pheno = phenotype(sc_peak)
    res["power"] = {
        "ftp_now": _f(ftp_now, 0), "ftp_method": proj.cfg["ftp"].get("mode"),
        "ftp_peak": _f(ftp_peak, 0), "ftp_peak_date": ftp_peak_date,
        "cp_90d": _f(cp, 0), "wprime_90d_kj": _f(wp / 1000 if wp == wp else np.nan, 1),
        "cp_fit_r2": _f(r2, 3),
        "wkg_now": _f(wkg(ftp_now), 2), "wkg_peak": _f(wkg(ftp_peak), 2),
        "bests_90d": {str(d): {"w": _f(cur[d][0], 0), "date": cur[d][1]} for d in MMP_DURATIONS},
        "bests_all": {str(d): {"w": _f(peak[d][0], 0), "date": peak[d][1]} for d in MMP_DURATIONS},
        "profile_scores_now": {k: _f(v, 0) for k, v in sc_cur.items()},
        "profile_scores_peak": {k: _f(v, 0) for k, v in sc_peak.items()},
        "profile_category_peak": {k: category(v) for k, v in sc_peak.items()},
        "phenotype": pheno[0], "phenotype_reason": pheno[1],
    }
    from ..models import thresholds as TH
    from ..models.physio import PhysioRun
    try:
        states = pd.read_parquet(proj.processed / PhysioRun.STATES)
    except (OSError, ValueError):
        states = None
    res["thresholds"] = TH.summary(daily, acts, res["power"]["bests_90d"], res["power"]["bests_all"],
                                   proj.cfg["ftp"], last_ride, states)
    res["zones"] = {
        "power": [{"zone": n, "upper_w": None if not np.isfinite(h) else round(h * ftp_now)}
                  for n, h in L.POWER_ZONES],
        "hr": [{"zone": n, "upper_bpm": None if not np.isfinite(h) else round(h * info["lthr"])}
               for n, h in L.HR_ZONES],
    }

    # ---- training history per year ----------------------------------------------
    y = daily.groupby(daily.index.year)
    zt = daily[[f"pz{i}_s" for i in range(1, 8)]].groupby(daily.index.year).sum()
    ztot = zt.sum(axis=1).replace(0, np.nan)
    hist = pd.DataFrame({
        "hours": y["hours"].sum(), "ride_hours": y["ride_hours"].sum(), "tss": y["tss"].sum(),
        "km": y["ride_km"].sum(), "avg_ctl": y["ctl"].mean(), "peak_ctl": y["ctl"].max(),
        "hard_per_wk": y["hard"].sum() / (y.size() / 7), "long_rides": y["long"].sum(),
        "z12_share": (zt["pz1_s"] + zt["pz2_s"]) / ztot,
        "z4_share": zt["pz4_s"] / ztot,
        "z5p_share": (zt["pz5_s"] + zt["pz6_s"] + zt["pz7_s"]) / ztot,
        "hr_z12_share": (lambda h: (h["hz1_s"] + h["hz2_s"]) / h.sum(axis=1).replace(0, np.nan))(
            daily[[f"hz{i}_s" for i in range(1, 8)]].groupby(daily.index.year).sum()),
        "indoor_share": y["indoor_h"].sum() / y["ride_hours"].sum().replace(0, np.nan),
        "best_perf": W.groupby(W.index.year)["perf_test"].max(),
        "peak_ftp": y["ftp_auto"].max(),
    })
    res["years"] = hist.round(2).reset_index(names="year").to_dict("records")
    res["load"] = {
        "ctl_now": _f(daily["ctl"].iloc[-1]), "atl_now": _f(daily["atl"].iloc[-1]),
        "tsb_now": _f(daily["tsb"].iloc[-1]),
        "ctl_peak": _f(daily["ctl"].max()), "ctl_peak_date": daily["ctl"].idxmax().date().isoformat(),
        "max_weekly_ramp": _f(W["ramp"].max()),
        "typical_week_hours": _f(W.loc[W["hours"] > 1, "hours"].median()),
        "typical_week_tss": _f(W.loc[W["hours"] > 1, "tss"].median(), 0),
    }

    # ---- efficiency / device sanity ------------------------------------------------
    aer = acts[acts["aerobic_ride"]]
    ef_in = aer.loc[aer["indoor"], "ef"].median()
    ef_out = aer.loc[~aer["indoor"], "ef"].median()
    res["efficiency"] = {"ef_indoor": _f(ef_in, 3), "ef_outdoor": _f(ef_out, 3),
                         "indoor_vs_outdoor_pct": _f((ef_in / ef_out - 1) * 100, 1)
                         if ef_in and ef_out else None,
                         "decoupling_median": _f(aer["decoupling"].median(), 1)}

    # ---- models -------------------------------------------------------------------
    n = proj.cfg["analysis"].get("driver_window_weeks", 8)
    drv = I.driver_analysis(W, n)
    res["drivers"] = {"window_weeks": n,
                      "table": drv["table"].round(3).to_dict("records") if len(drv["table"]) else [],
                      "model": {k: (v.round(4).to_dict() if isinstance(v, pd.Series) else v)
                                for k, v in drv["model"].items()}}
    gain = proj.cfg["analysis"].get("progress_min_gain", 0.04)
    from ..models.dips import BreakLabels
    found = I.find_periods(W, min_gain=gain).assign(yours=False)   # beside the gains you marked on the load chart
    P = I.describe_periods(pd.concat([found, I.marked_periods(W, BreakLabels(proj.processed).periods("gain"))])
                           .sort_values("start").reset_index(drop=True), W)
    N = I.describe_periods(I.find_periods(W, min_gain=gain, direction=-1), W)
    wkflags, eps = I.overreach_episodes(daily, W)
    br = I.breaks(W)
    perf = I.banister_performance(acts, daily)
    ban = Banister.fit(daily["tss"], perf)
    res["progress_periods"] = _records(P)
    res["decline_periods"] = _records(N)
    res["episodes"] = _records(eps)
    res["breaks"] = _records(br)
    from ..models import illness as IL
    ill = IL.illness_analysis(acts, daily, N, proj.cfg.get("events"))
    ep_ill = ill["episodes"]
    if len(eps) and len(ep_ill):
        on = pd.to_datetime(ep_ill["onset"])
        eps["illness_after"] = [", ".join(str(o.date()) for o in on
                                          if pd.Timestamp(s) <= o <= pd.Timestamp(e) + pd.Timedelta(weeks=3))
                                for s, e in zip(eps["start"], eps["end"])]
        res["episodes"] = _records(eps)
    prof = ill["profile"]
    res["illness"] = {
        "notes_found": int(len(ill["notes"])),
        "episodes": _records(ep_ill.drop(columns=["notes"]) if len(ep_ill) else ep_ill),
        "episode_notes": ep_ill["notes"].tolist() if len(ep_ill) else [],
        "profile": _records(prof.reset_index()) if len(prof) else [],
        "pre_signals": _records(ill["pre_signals"]),
        "warning": ill["warning"],
        "declines": _records(ill["declines"]),
    }
    from ..models.buildup import GROUPS, buildup_analysis
    conf = ep_ill[~ep_ill["severity"].str.startswith("unconfirmed")] if len(ep_ill) else ep_ill
    bu = buildup_analysis(W, P, conf["onset"] if len(conf) else [])
    # sessions per week by coarse group, in 4-week blocks, for the dashboard
    blocks = W.resample("4W-MON", label="left", closed="left").sum(numeric_only=True)
    nweeks = W["hours"].resample("4W-MON", label="left", closed="left").size().replace(0, 1)
    grp = {g: sum(blocks.get(f"n_{t}", 0) for t in ts) / nweeks for g, ts in GROUPS.items()}
    res["buildup"] = {
        "peak_week": bu["peak_week"].date().isoformat() if bu["peak_week"] is not None else None,
        "peak_value": _f(bu["peak_value"], 0),
        "peak_blocks": _records(bu["peak_blocks"]),
        "builds": _records(bu["builds"]),
        "ramp_table": _records(bu["ramp"]["table"]),
        "ramp_runs": _records(bu["ramp"]["runs"]),
        "illness_base_rate": bu["ramp"]["illness_base_rate"],
        "illness_episodes": bu["ramp"]["illness_episodes"],
        "type_drivers": _records(bu["type_drivers"]),
        "sessions_by_group": {"block": [d.date().isoformat() for d in blocks.index],
                              **{g: [round(float(x), 2) for x in v] for g, v in grp.items()}},
    }
    rec = I.recovery_analysis(daily)
    res["recovery"] = {k: (_records(v) if isinstance(v, pd.DataFrame) else v)
                       for k, v in rec.items() if k != "yearly"}
    if isinstance(rec.get("yearly"), pd.DataFrame):
        res["recovery"]["yearly"] = _records(rec["yearly"].reset_index(names="year"))
    res["wellness"] = {"days": info.get("wellness_days", 0),
                       "coverage": (info.get("wellness_inventory") or {}).get("coverage", {})}
    res["banister"] = {k: (_f(v, 3) if isinstance(v, (float, np.floating)) else v)
                       for k, v in ban.items() if k != "fitted"}
    r.update(buildup=bu, illness=ill, recovery=rec, drivers=drv, periods=P, declines=N, episodes=eps, week_flags=wkflags,
             banister=ban, banister_perf=perf)
    res["notes"] = data_quality_notes(res, acts, daily)
    return res


def _records(df):
    if df is None or df.empty:
        return []
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(3)
    return out.astype(object).where(out.notna(), None).to_dict("records")


def data_quality_notes(res, acts, daily) -> list[str]:
    notes = []
    if res["data"]["wellness_days"] < 30:
        notes.append("Little or no Garmin wellness data was found (sleep / resting HR / HRV), so "
                     "recovery markers are not yet part of the analysis. Export your full Garmin "
                     "data (Account > Data management > Export your data) and unzip it into the "
                     "garmin folder.")
    w = acts["athlete_weight"].dropna()
    if res["athlete"]["weight_source"] == "garmin":
        last = daily["weight_garmin"].last_valid_index() if "weight_garmin" in daily else None
        notes.append(f"Body weight comes from the Garmin weight log (last entry "
                     f"{last.date() if last is not None else '?'}); between entries the last value "
                     "is carried forward. Log weight occasionally (or set weight_kg) to keep W/kg right.")
    elif res["athlete"]["weight_source"] != "config" and (w.nunique() <= 1):
        notes.append("Body weight comes from a single constant Strava profile value "
                     f"({res['athlete']['weight_kg']} kg); W/kg and profile scores are only as "
                     "good as that number - set weight_kg (or a history) in goodpush.toml.")
    ef = res["efficiency"].get("indoor_vs_outdoor_pct")
    if ef is not None and abs(ef) >= 5:
        notes.append(f"Indoor rides show {abs(ef):.0f}% {'higher' if ef > 0 else 'lower'} power per "
                     "heartbeat than outdoor rides. Part of this is physiology (cooling, cadence), "
                     "but differences >5% often mean the trainer and the outdoor power meter read "
                     "differently - compare FTP tests on the same device.")
    if res["athlete"]["resting_hr_source"] == "default":
        notes.append("Resting HR unknown - assumed 55 bpm for heart-rate TSS. Set resting_hr in "
                     "goodpush.toml or add Garmin data.")
    return notes
