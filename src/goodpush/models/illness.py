"""Illness episodes from ride notes + Garmin recovery markers.

Your sickness notes are usually written on the comeback ride, *after* the
illness. So each note is used as an anchor and the episode is reconstructed
backwards from the data: the training gap before the note and the resting-HR
rise that marks the onset.

Outputs
* episodes: one row per illness with onset, days off, how hard it hit the
  markers, what training looked like before, and the performance cost.
* profile: markers averaged over all episodes, aligned on onset day
  (day 0) - the "what does getting sick look like for me" curve.
* warning: how often the early-warning pattern fires on normal days
  (false alarms) vs before real illnesses (hits).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ILLNESS_RE = re.compile(
    r"\b(?:sick|sickness|illness|lurg\w*|flu|covid|virus|viral|fever|cough\w*|"
    r"chest (?:thing|cough|infection)|sore throat|throat|sinus\w*|bronchitis|pneumonia|"
    r"infection|phlegm\w*|mucus\w*|congest\w*|head cold|a cold(?! (?:cold|wet|day|morning|start|ride|one))|this cold|got a cold|"
    r"unwell|poorly|bug|food poisoning|stomach bug|antibiotic\w*)\b", re.I)
EXCLUDE_RE = re.compile(r"\b(?:sick kid|kid(?:s)? (?:is|are|was|were) sick|sick of)\b", re.I)

GAP_LOOKBACK = 28     # a training gap that ends within 5 days of the first note
RHR_LOOKBACK = 14     # resting-HR rise searched in the 14 days before the first note
CLUSTER_DAYS = 21     # notes closer than this belong to the same illness


def find_notes(acts: pd.DataFrame) -> pd.DataFrame:
    txt = (acts["name"].fillna("") + " || " + acts["description"].fillna("") + " || "
           + acts["private_note"].fillna(""))
    hit = txt.str.contains(ILLNESS_RE) & ~txt.str.contains(EXCLUDE_RE)
    out = acts.loc[hit, ["date", "name", "type"]].copy()
    out["text"] = txt[hit].str.replace(r"\s+", " ", regex=True).str.strip(" |")
    out["matched"] = txt[hit].apply(lambda t: ", ".join(sorted(set(
        m.group(0).lower() for m in ILLNESS_RE.finditer(t)))))
    return out.sort_values("date").reset_index(drop=True)


def _rhr_delta(d: pd.DataFrame) -> pd.Series:
    base = d["rhr"].rolling(42, min_periods=15).median().shift(1)
    return d["rhr"] - base


def build_episodes(d: pd.DataFrame, notes: pd.DataFrame, events: list | None = None) -> pd.DataFrame:
    """Cluster notes into illnesses and reconstruct onset / comeback from the data."""
    d = d.copy()
    d["rhr_d"] = _rhr_delta(d)
    low = d["tss"] < 15
    anchors = []
    for dt, grp in notes.groupby((notes["date"].diff().dt.days > CLUSTER_DAYS).cumsum()):
        anchors.append((grp["date"].min(), grp["date"].max(), " / ".join(grp["text"].str[:90]),
                        "strava note"))
    for ev in events or []:
        if str(ev.get("type", "")).lower() == "illness" and "example" not in str(ev.get("note", "")):
            s = pd.Timestamp(ev["date"])
            e = pd.Timestamp(ev.get("end") or ev["date"])
            anchors.append((s, e, ev.get("note", ""), "config event"))
    anchors.sort()
    rows = []
    for first, last, text, src in anchors:
        if src == "config event":
            onset, comeback = first, last
        else:
            win = d.loc[first - pd.Timedelta(days=GAP_LOOKBACK): first]
            # training gap: last run of >= 3 low days ending within 5 days of the first note
            gap_start = None
            lw = low.reindex(win.index).fillna(False).to_numpy()
            j = len(lw) - 1
            while j >= 0 and not lw[j]:
                j -= 1
            if j >= len(lw) - 6:
                k = j
                while k - 1 >= 0 and lw[k - 1]:
                    k -= 1
                if j - k + 1 >= 3:
                    gap_start = win.index[k]
            # resting-HR rise: first run of >= 2 days at >= +3 bpm above baseline
            r = d.loc[first - pd.Timedelta(days=RHR_LOOKBACK): first, "rhr_d"] >= 3
            run = r & r.shift(-1, fill_value=False)
            rhr_on = run[run].index.min() if run.any() else None
            cands = [x for x in (gap_start, rhr_on) if x is not None and not pd.isna(x)]
            onset = min(cands) if cands else first - pd.Timedelta(days=3)
            comeback = last
        # the day you'd have noticed without any marker: training stopped, or (no gap) a few
        # days before the first note. Used to test markers fairly (onset itself uses resting HR).
        noticed = first if src == "config event" else (gap_start if gap_start is not None and not pd.isna(gap_start)
                                                         else first - pd.Timedelta(days=3))
        seg = d.loc[onset: comeback]
        pre = d.loc[onset - pd.Timedelta(days=7): onset - pd.Timedelta(days=1)]
        pre3w = d.loc[onset - pd.Timedelta(days=21): onset - pd.Timedelta(days=1)]
        prev6w = d.loc[onset - pd.Timedelta(days=63): onset - pd.Timedelta(days=22)]
        post = d.loc[comeback: comeback + pd.Timedelta(days=42)]
        perf_before = d.loc[onset - pd.Timedelta(days=42): onset, "eq20_test"].max()
        perf_after = d.loc[comeback: comeback + pd.Timedelta(days=42), "eq20_test"]
        regain = np.nan
        if np.isfinite(perf_before):
            ok = d.loc[comeback:, "eq20_test"]
            ok = ok[ok >= 0.97 * perf_before]
            if len(ok):
                regain = (ok.index[0] - comeback).days
        rows.append({
            "onset": onset.date(), "comeback_note": comeback.date(), "noticed": noticed.date(),
            "days_sick": (comeback - onset).days,
            "days_off": int(low.reindex(seg.index).sum()),
            "rhr_peak_delta": seg["rhr_d"].max(),
            "hrv_min_z": seg["hrv_z"].min() if "hrv_z" in seg else np.nan,
            "sleep_vs_base_h": (seg["sleep_h"] - seg["sleep_h_base"]).mean(),
            "pre_rhr_delta": pre["rhr_d"].mean(), "pre_hrv_z": pre["hrv_z"].mean(),
            "pre_sleep_vs_base_h": (pre["sleep_h"] - pre["sleep_h_base"]).mean(),
            "pre_stress": pre["stress"].mean() - d["stress"].rolling(42, min_periods=10).median()
                .shift(1).reindex(pre.index).mean(),
            "pre3w_tss_wk": pre3w["tss"].sum() / 3, "prev6w_tss_wk": prev6w["tss"].sum() / 6,
            "atl_at_onset": d["atl"].get(onset, np.nan), "tsb_at_onset": d["tsb"].get(onset, np.nan),
            "max_acwr_pre3w": pre3w["acwr"].max(),
            "ctl_drop": d["ctl"].get(onset, np.nan) - post["ctl"].min() if len(post) else np.nan,
            "perf_before": perf_before,
            "perf_after_6wk": perf_after.max() if perf_after.notna().any() else np.nan,
            "days_to_regain_97pct": regain,
            "source": src, "notes": text,
        })
    ep = pd.DataFrame(rows)
    if len(ep):
        conf = (ep["rhr_peak_delta"] >= 3) | (ep["days_off"] >= 3) | (ep["hrv_min_z"] <= -1.5)
        sev = np.where((ep["days_off"] >= 10) | (ep["rhr_peak_delta"] >= 7), "severe",
                       np.where((ep["days_off"] >= 4) | (ep["rhr_peak_delta"] >= 5), "moderate", "mild"))
        ep["severity"] = np.where(conf, sev, "unconfirmed (no sign in data)")
        ep["load_ramp_pre"] = ep["pre3w_tss_wk"] / ep["prev6w_tss_wk"].replace(0, np.nan)
        ep["perf_change_pct"] = (ep["perf_after_6wk"] / ep["perf_before"] - 1) * 100
    return ep


def aligned_profile(d: pd.DataFrame, episodes: pd.DataFrame, before=21, after=35) -> pd.DataFrame:
    """Average marker trajectories aligned on onset (day 0)."""
    d = d.copy()
    d["rhr_d"] = _rhr_delta(d)
    d["sleep_d"] = d["sleep_h"] - d["sleep_h_base"]
    d["stress_d"] = d["stress"] - d["stress"].rolling(42, min_periods=10).median().shift(1)
    cols = ["rhr_d", "hrv_z", "sleep_d", "stress_d", "tss", "tsb"]
    frames = []
    for o in pd.to_datetime(episodes["onset"]):
        w = d.reindex(pd.date_range(o - pd.Timedelta(days=before), o + pd.Timedelta(days=after)))[cols]
        w.index = range(-before, after + 1)
        frames.append(w)
    if not frames:
        return pd.DataFrame()
    stack = pd.concat(frames, keys=range(len(frames)))
    prof = stack.groupby(level=1).mean()
    prof["n"] = stack.groupby(level=1)["rhr_d"].count()
    prof.index.name = "day"
    return prof


def warning_hit_rate(d: pd.DataFrame, episodes: pd.DataFrame, rhr_bpm=3, days=2,
                     horizon=7) -> dict:
    """Early-warning rule: resting HR >= `rhr_bpm` above baseline for `days` days in a row.
    How often does it fire, and how often is it followed by an illness onset within `horizon` days?"""
    rd = _rhr_delta(d)
    fire = (rd >= rhr_bpm).rolling(days).sum() >= days
    fire = fire & ~fire.shift(1, fill_value=False)  # first day of each alarm
    onsets = pd.to_datetime(episodes["onset"]) if len(episodes) else pd.DatetimeIndex([])
    sick = pd.Series(False, index=d.index)
    for o in onsets:  # mark days inside an illness (onset-3 .. onset+7)
        sick.loc[o - pd.Timedelta(days=3): o + pd.Timedelta(days=horizon)] = True
    alarms = fire[fire].index
    true_alarm = [a for a in alarms if sick.get(a, False)]
    caught = [o for o in onsets if fire.loc[o - pd.Timedelta(days=3): o + pd.Timedelta(days=3)].any()]
    years = max(1e-9, d["rhr"].notna().sum() / 365.25)
    return {"rule": f"resting HR >= +{rhr_bpm} bpm over 6-week baseline for {days} days",
            "alarms": int(len(alarms)), "alarms_per_year": len(alarms) / years,
            "alarms_near_illness": len(true_alarm),
            "illnesses_caught": len(caught), "illnesses": int(len(onsets))}


PRE_FEATURES = {
    "tss_14d": "Training load, previous 2 weeks (TSS)",
    "load_ramp": "Load ramp (last 2 wk vs previous 4 wk)",
    "acwr_max": "Max acute:chronic ratio",
    "monotony_max": "Max training monotony",
    "strain_max": "Max training strain (Foster)",
    "hard_14d": "Hard sessions, previous 2 weeks",
    "rhr_d": "Resting HR vs baseline (bpm)",
    "hrv_z": "HRV vs baseline (z)",
    "sleep_d": "Sleep vs baseline (h)",
    "short_nights": "Nights < 6.5 h",
    "stress_d": "All-day stress vs baseline",
    "bedtime_d": "Bedtime vs baseline (h later)",
}


def pre_illness_signals(d: pd.DataFrame, episodes: pd.DataFrame, lo=-14, hi=-2) -> pd.DataFrame:
    """Case-control: the 2 weeks before each onset vs every other 2-week window.

    Day -1 and 0 are excluded so the signs found are *early* warnings, not
    the illness itself.
    """
    d = d.copy()
    d["rhr_d"] = _rhr_delta(d)
    d["sleep_d"] = d["sleep_h"] - d["sleep_h_base"]
    d["stress_d"] = d["stress"] - d["stress"].rolling(42, min_periods=10).median().shift(1)
    d["bedtime_d"] = d["bedtime_h"] - d["bedtime_h"].rolling(42, min_periods=10).median().shift(1)
    n = hi - lo + 1
    r = d.rolling(n, min_periods=max(3, n // 2))
    W = pd.DataFrame({
        "tss_14d": d["tss"].rolling(n).sum(),
        "load_ramp": d["tss"].rolling(n).sum() / (d["tss"].shift(n).rolling(28).sum() / 2).replace(0, np.nan),
        "acwr_max": r["acwr"].max(), "monotony_max": r["monotony"].max(),
        "strain_max": r["strain"].max(),
        "hard_14d": d["hard"].rolling(n).sum(),
        "rhr_d": r["rhr_d"].mean(), "hrv_z": r["hrv_z"].mean(), "sleep_d": r["sleep_d"].mean(),
        "short_nights": (d["sleep_h"] < 6.5).astype(float).where(d["sleep_h"].notna()).rolling(n, min_periods=3).sum(),
        "stress_d": r["stress_d"].mean(), "bedtime_d": r["bedtime_d"].mean(),
    })
    W = W.shift(-hi)  # row t now summarises days t+lo .. t+hi
    onsets = pd.to_datetime(episodes["onset"]) if len(episodes) else pd.DatetimeIndex([])
    near = pd.Series(False, index=d.index)
    for o in onsets:
        near.loc[o - pd.Timedelta(days=30): o + pd.Timedelta(days=45)] = True
    rows = []
    for f, label in PRE_FEATURES.items():
        case = W[f].reindex(onsets).dropna()
        ctrl = W.loc[~near, f].dropna()
        if len(case) < 4 or len(ctrl) < 50:
            continue
        # mid-rank percentile of each pre-illness value within the normal distribution
        srt = np.sort(ctrl.to_numpy())
        pct = [(np.searchsorted(srt, v, "left") + np.searchsorted(srt, v, "right")) / 2 / len(srt) * 100
               for v in case]
        rows.append({"feature": f, "label": label, "before_illness": float(case.mean()),
                     "normal": float(ctrl.mean()), "n_illnesses": int(len(case)),
                     "avg_percentile": float(np.mean(pct)),
                     "margin": float(2 * 28.9 / np.sqrt(len(case)))})
    out = pd.DataFrame(rows)
    if len(out):
        # chance level is the 50th percentile; +/- margin is ~2 standard errors
        dev = out["avg_percentile"] - 50
        out["signal"] = np.where(dev > out["margin"], "higher before illness",
                                 np.where(dev < -out["margin"], "lower before illness",
                                          "no clear difference"))
    return out


def map_declines(declines: pd.DataFrame, episodes: pd.DataFrame, pad_weeks=6) -> pd.DataFrame:
    if declines is None or declines.empty:
        return pd.DataFrame()
    rows = []
    if len(episodes) and "severity" in episodes:
        episodes = episodes[~episodes["severity"].str.startswith("unconfirmed")]
    on = pd.to_datetime(episodes["onset"]) if len(episodes) else pd.Series(dtype="datetime64[ns]")
    for p in declines.itertuples():
        s = pd.Timestamp(p.start) - pd.Timedelta(weeks=pad_weeks)
        e = pd.Timestamp(p.end)
        hit = episodes[(on >= s) & (on <= e)] if len(episodes) else episodes
        rows.append({"start": p.start, "end": p.end, "change_pct": p.change_pct,
                     "hours_wk": p.hours_wk, "hard_wk": p.hard_wk,
                     "illness_onsets": ", ".join(str(x) for x in hit["onset"]) if len(hit) else "",
                     "explained_by_illness": bool(len(hit))})
    return pd.DataFrame(rows)


def illness_analysis(acts, d, declines, events=None) -> dict:
    notes = find_notes(acts)
    ep = build_episodes(d, notes, events)
    if ep.empty:
        return {"notes": notes, "episodes": ep, "profile": pd.DataFrame(),
                "pre_signals": pd.DataFrame(), "warning": {}, "declines": pd.DataFrame()}
    return {"notes": notes, "episodes": ep, "profile": aligned_profile(d, ep),
            "pre_signals": pre_illness_signals(d, ep),
            "warning": warning_hit_rate(d, ep), "declines": map_declines(declines, ep)}
