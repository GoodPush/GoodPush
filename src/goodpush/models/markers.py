"""Personal illness markers: learn which morning signals warned of *your* past illnesses,
then watch them every day.

How it works
1. Illness episodes come from sickness words in your ride notes, [[events]] of type "illness" in
   goodpush.toml, days you marked "sick" in the app and training breaks you labelled illness - see
   models/illness.py and models/dips.py.
2. For each candidate marker (resting HR above baseline, HRV below baseline, short sleep,
   sleep score, all-day stress, body battery, breathing rate, Garmin readiness, training
   monotony, load spikes) and each threshold / "how many days in a row" variant, we replay
   your history: did it fire in the week before you noticed each illness (caught), and how
   often did it fire when no illness followed (false alarms)?
3. The best variant of each marker is kept; markers that caught enough illnesses without
   crying wolf become *your* markers, plus a combined rule (any one, or two together) for the record.
4. Every day the same markers are evaluated on the newest data (`status`): one firing = watch (go easy),
   two or more = warning (red).

Fair test: an episode's "noticed" day is when training stopped or a few days before your first
sick note - not the resting-HR based onset - so resting HR isn't graded against itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW_BEFORE = 7      # a marker "caught" an illness if it fired up to 7 days before it was noticed
WINDOW_AFTER = 1       # ... or the day after
TRUE_AFTER = 3         # alarms up to 3 days after noticing still count as true (not false) alarms
MAX_ALARMS_PER_YEAR = 12
MIN_EPISODES = 4
LEARN_FORMAT = 2       # bumped when learn() changes: stored markers are learned again once (2: one illness, one episode)

# key, label, column (computed below), direction, thresholds, days-in-a-row options, unit
FAMILIES = [
    ("rhr", "Resting HR above your baseline", "rhr_d", ">=", [2, 3, 4, 5], [1, 2, 3], "bpm"),
    ("hrv", "HRV below your baseline", "hrv_z", "<=", [-1.0, -1.5, -2.0], [1, 2], "z"),
    ("sleep", "Sleep shorter than usual", "sleep_d", "<=", [-0.75, -1.0, -1.5], [1, 2, 3], "h"),
    ("sleep_score", "Sleep score below usual", "sleep_score_z", "<=", [-1.0, -1.5, -2.0], [1, 2], "z"),
    ("stress", "All-day stress above usual", "stress_z", ">=", [1.0, 1.5, 2.0], [1, 2], "z"),
    ("battery", "Body battery recharge below usual", "battery_z", "<=", [-1.0, -1.5, -2.0], [1, 2], "z"),
    ("resp", "Breathing rate above usual", "resp_d", ">=", [0.5, 1.0, 1.5], [1, 2], "br/min"),
    ("readiness", "Garmin training readiness below usual", "readiness_z", "<=", [-1.0, -1.5], [1, 2], "z"),
    ("monotony", "Training monotony high", "monotony", ">=", [1.5, 1.8, 2.2], [1, 3], ""),
    ("acwr", "Load spike (acute:chronic ratio)", "acwr", ">=", [1.3, 1.5], [1, 3], ""),
] + []
# DFA alpha1 (alphaHRV) in the warm-up / early session vs your normal at the same power -
# only on days you rode with it. Too low and too high are tested separately and together.
ALPHA_FAMILIES = [
    ("a1_low", "Warm-up alpha1 below your normal", "a1_warm_z", "<=", [-1.5, -2.0, -2.5], [1], "z"),
    ("a1_high", "Warm-up alpha1 above your normal", "a1_warm_z", ">=", [1.5, 2.0, 2.5], [1], "z"),
    ("a1_off", "Warm-up alpha1 unusual (either way)", "a1_warm_absz", ">=", [1.5, 2.0, 2.5], [1], "z"),
    ("a1_hr", "Warm-up heart rate high for the power", "a1_warm_hr_z", ">=", [1.5, 2.0, 2.5], [1], "z"),
    ("a1_sess_off", "Alpha1 unusual as the session gets going", "a1_session_absz", ">=", [1.5, 2.0, 2.5], [1], "z"),
]
FAMILIES = FAMILIES + ALPHA_FAMILIES
# overreaching / burnout builds up over weeks: trends and load, tested 14 days ahead
OVERREACH_FAMILIES = [
    ("hrv_trend", "HRV trending below baseline (7-day average)", "hrv_7z", "<=", [-0.5, -0.75, -1.0], [1, 3], "z"),
    ("rhr_trend", "Resting HR trending up (7-day average)", "rhr_7d", ">=", [1.0, 1.5, 2.0], [1, 3], "bpm"),
    ("sleep_trend", "Sleep below usual (7-day average)", "sleep_7d", "<=", [-0.4, -0.6, -0.8], [1, 3], "h"),
    ("tsb", "Deep fatigue (form / TSB)", "tsb", "<=", [-20, -25, -30], [1, 3], ""),
    ("ramp", "Fast load build (CTL change over 4 weeks)", "ramp28", ">=", [8, 10, 12], [1], ""),
    ("monotony", "Training monotony high", "monotony", ">=", [1.5, 1.8, 2.2], [1, 3], ""),
    ("acwr", "Load spike (acute:chronic ratio)", "acwr", ">=", [1.3, 1.5], [1, 3], ""),
] + ALPHA_FAMILIES
LABELS = {f[0]: f[1] for f in FAMILIES + OVERREACH_FAMILIES}


def _base(x: pd.Series):
    b = x.rolling(42, min_periods=15).median().shift(1)
    sd = x.rolling(42, min_periods=15).std().shift(1)
    return b, sd.replace(0, np.nan)


def features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-day marker values on a common footing (deviation from your rolling 6-week normal)."""
    d = daily
    f = pd.DataFrame(index=d.index)

    def col(c):
        return d[c].astype(float) if c in d else pd.Series(np.nan, index=d.index)
    b, _ = _base(col("rhr"))
    f["rhr_d"] = col("rhr") - b
    for src, out in (("hrv", "hrv_z"), ("sleep_score", "sleep_score_z"), ("stress", "stress_z"),
                     ("body_battery_max", "battery_z"), ("readiness", "readiness_z")):
        b, sd = _base(col(src))
        f[out] = (col(src) - b) / sd
    b, _ = _base(col("sleep_h"))
    f["sleep_d"] = col("sleep_h") - b
    b, _ = _base(col("respiration"))
    f["resp_d"] = col("respiration") - b
    f["monotony"] = col("monotony")
    f["acwr"] = col("acwr")
    f["tsb"] = col("tsb")
    for c in ("a1_warm_z", "a1_warm_hr_z", "a1_session_z"):
        f[c] = col(c)
    f["a1_warm_absz"] = f["a1_warm_z"].abs()
    f["a1_session_absz"] = f["a1_session_z"].abs()
    f["ramp28"] = col("ctl") - col("ctl").shift(28)
    f["hrv_7z"] = f["hrv_z"].rolling(7, min_periods=4).mean()
    f["rhr_7d"] = f["rhr_d"].rolling(7, min_periods=4).mean()
    f["sleep_7d"] = f["sleep_d"].rolling(7, min_periods=4).mean()
    # load markers only make sense on days with training data
    return f


def fire_series(x: pd.Series, op: str, thr: float, days: int) -> pd.Series:
    cond = (x >= thr) if op == ">=" else (x <= thr)
    cond = cond & x.notna()
    return cond.astype(int).rolling(days, min_periods=days).sum().fillna(0) >= days


def _alarm_starts(fire: pd.Series) -> pd.DatetimeIndex:
    start = fire & ~fire.shift(1, fill_value=False)
    return start[start].index


def evaluate(fire: pd.Series, avail: pd.Series, anchors: list, all_anchors: list,
             before: int = WINDOW_BEFORE, min_days: int = 3) -> dict:
    """How a firing series would have done on your history."""
    n_eval, caught, leads = 0, 0, []
    for a in anchors:
        w0, w1 = a - pd.Timedelta(days=before), a + pd.Timedelta(days=WINDOW_AFTER)
        if avail.loc[w0:w1].sum() < min_days:
            continue
        n_eval += 1
        f = fire.loc[w0:w1]
        if f.any():
            caught += 1
            leads.append((a - f[f].index[0]).days)
    alarms = _alarm_starts(fire)
    true = 0
    for t in alarms:
        if any(a - pd.Timedelta(days=before) <= t <= a + pd.Timedelta(days=TRUE_AFTER)
               for a in all_anchors):
            true += 1
    on = avail[avail].index
    years = max(((on[-1] - on[0]).days + 1) / 365.25, 1 / 365.25) if len(on) else 1e-9
    recall = caught / n_eval if n_eval else 0.0
    # chance: how often a random 9-day window (with data) contains a firing day
    win = before + WINDOW_AFTER + 1
    any_fire = fire.astype(float).rolling(win, min_periods=1).max()
    have = avail.astype(float).rolling(win, min_periods=1).sum() >= min_days
    p0 = float(any_fire[have].mean()) if have.any() else 0.0
    p_value = _binom_tail(caught, n_eval, p0)
    precision = true / len(alarms) if len(alarms) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n_eval": n_eval, "caught": caught, "recall": round(float(recall), 3),
            "alarms": int(len(alarms)), "true_alarms": int(true), "precision": round(float(precision), 3),
            "alarms_per_year": round(float(len(alarms) / years), 1), "f1": round(float(f1), 3),
            "lead_days": float(np.median(leads)) if leads else None, "years": round(float(years), 1),
            "chance_recall": round(p0, 3), "p_value": round(p_value, 4)}


def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p): how likely this many catches are by luck."""
    from math import comb
    if n == 0:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def _rule_text(key, op, thr, days, unit):
    sign = "+" if thr > 0 and op == ">=" and unit in ("bpm", "br/min") else ""
    u = f" {unit}" if unit and unit != "z" else (" SD" if unit == "z" else "")
    rel = "at or above" if op == ">=" else "at or below"
    dd = "1 day" if days == 1 else f"{days} days in a row"
    return f"{LABELS[key]}: {rel} {sign}{thr:g}{u}, {dd}"


def discover(daily: pd.DataFrame, episodes: pd.DataFrame, families=None, before: int = WINDOW_BEFORE,
             min_episodes: int = MIN_EPISODES, target: str = "illness") -> dict:
    families = families or FAMILIES
    f = features(daily)
    ep = episodes.copy() if episodes is not None else pd.DataFrame()
    if len(ep):
        ep["anchor"] = pd.to_datetime(ep.get("noticed", ep["onset"]))
        conf = ~ep["severity"].astype(str).str.startswith("unconfirmed")
        anchors = sorted(ep.loc[conf, "anchor"])
        all_anchors = sorted(ep["anchor"])
    else:
        anchors, all_anchors = [], []
    # chance level: share of days that sit inside some "true alarm" window
    inwin = pd.Series(False, index=f.index)
    for a in all_anchors:
        inwin.loc[a - pd.Timedelta(days=before): a + pd.Timedelta(days=TRUE_AFTER)] = True
    base_rate = float(inwin.mean()) if len(inwin) else 0.0
    cands = []
    for key, label, colname, op, thrs, dayss, unit in families:
        x = f[colname]
        avail = x.notna()
        if avail.sum() < (60 if colname.startswith("a1_") else 120):
            cands.append({"key": key, "label": label, "status": "not enough data",
                          "days_with_data": int(avail.sum())})
            continue
        best = None
        for thr in thrs:
            for days in dayss:
                ev = evaluate(fire_series(x, op, thr, days), avail, anchors, all_anchors, before,
                              min_days=1 if colname.startswith("a1_") else 3)
                cand = {"key": key, "label": label, "column": colname, "op": op, "threshold": thr,
                        "days": days, "unit": unit, "rule": _rule_text(key, op, thr, days, unit), **ev}
                ok = ev["alarms_per_year"] <= MAX_ALARMS_PER_YEAR
                score = (ok, ev["f1"], -ev["alarms_per_year"])
                if best is None or score > best[0]:
                    best = (score, cand)
        c = best[1]
        c["days_with_data"] = int(avail.sum())
        c["lift"] = round(c["precision"] / base_rate, 2) if base_rate else None
        c["status"] = "candidate"
        cands.append(c)
    usable = [c for c in cands if c["status"] == "candidate" and c["n_eval"] >= min_episodes
              and c["recall"] >= 0.3 and c["alarms_per_year"] <= MAX_ALARMS_PER_YEAR
              and c["precision"] >= max(1.5 * base_rate, 0.1) and c["p_value"] <= 0.1]
    usable.sort(key=lambda c: (-c["f1"], c["alarms_per_year"]))
    chosen = usable[:4]
    for c in cands:
        if c in chosen:
            c["status"] = "chosen"
        elif c["status"] == "candidate":
            c["status"] = "weak" if c.get("n_eval", 0) >= min_episodes else "too few episodes with this data"
    out = {"target": target, "window_days": before, "min_episodes": min_episodes,
           "analyzed": str(pd.Timestamp.now().date()), "episodes": int(len(ep)),
           "confirmed_episodes": len(anchors), "base_rate": round(base_rate, 3),
           "candidates": cands, "chosen": [_slim(c) for c in chosen], "combined": None,
           "fallback": not chosen}
    if chosen:
        best = None
        for k in sorted({1, 2} & set(range(1, len(chosen) + 1))):
            cnt = sum(fire_series(f[c["column"]], c["op"], c["threshold"], c["days"]).astype(int)
                      for c in chosen)
            fire = cnt >= k
            avail = pd.concat([f[c["column"]].notna() for c in chosen], axis=1).any(axis=1)
            ev = evaluate(fire, avail, anchors, all_anchors, before)
            cand = {"min_markers": k, "rule": "any one of your markers" if k == 1 else
                    "two or more of your markers together", **ev}
            if best is None or (ev["f1"], -ev["alarms_per_year"]) > (best["f1"], -best["alarms_per_year"]):
                best = cand
        out["combined"] = best
    elif target != "illness":   # nothing general to fall back on
        out["chosen"], out["combined"] = [], None
    else:   # not enough history to learn from - fall back to the general resting-HR rule
        out["chosen"] = [{"key": "rhr", "label": LABELS["rhr"], "column": "rhr_d", "op": ">=",
                          "threshold": 3, "days": 2, "unit": "bpm",
                          "rule": _rule_text("rhr", ">=", 3, 2, "bpm"), "default": True}]
        out["combined"] = {"min_markers": 1, "rule": "the general rule (not personalised yet)"}
    return out


def _slim(c):
    keep = ["key", "label", "column", "op", "threshold", "days", "unit", "rule", "n_eval", "caught",
            "recall", "alarms_per_year", "precision", "lead_days", "lift", "chance_recall", "p_value"]
    return {k: c.get(k) for k in keep}


def status(daily: pd.DataFrame, learned: dict, days: int = 14) -> dict:
    """Evaluate your markers on the most recent days."""
    if not learned or not learned.get("chosen"):
        return {}
    f = features(daily).iloc[-(days + 60):]
    chosen = learned["chosen"]
    rows = []
    for c in chosen:
        x = f[c["column"]]
        fire = fire_series(x, c["op"], c["threshold"], c["days"])
        rows.append({"key": c["key"], "label": c["label"], "rule": c["rule"], "unit": c["unit"],
                     "threshold": c["threshold"], "op": c["op"],
                     "values": [None if pd.isna(v) else round(float(v), 2) for v in x.iloc[-days:]],
                     "fired": [bool(v) for v in fire.iloc[-days:]]})
    dates = [str(i.date()) for i in f.index[-days:]]
    count = [sum(r["fired"][i] for r in rows) for i in range(len(dates))]
    last_data = [i for i in range(len(dates)) if any(r["values"][i] is not None for r in rows)]
    li = last_data[-1] if last_data else len(dates) - 1
    n_today = count[li]
    # one learned marker alone is a watch, two together a warning; the general rule (not personalised) warns alone
    level = "clear" if n_today == 0 else ("warning" if n_today >= 2 or learned.get("fallback") else "watch")
    return {"dates": dates, "markers": rows, "count": count,
            "level": level, "as_of": dates[li], "firing": [r["label"] for r in rows if r["fired"][li]]}


def path(processed: Path, target: str = "illness") -> Path:
    return Path(processed) / ("illness_markers.json" if target == "illness" else f"{target}_markers.json")


def load(processed: Path, target: str = "illness") -> dict | None:
    p = path(processed, target)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def save(processed: Path, learned: dict, target: str = "illness") -> Path:
    p = path(processed, target)
    p.write_text(json.dumps(learned, indent=1, default=str), encoding="utf-8")
    return p


def sick_events(processed: Path) -> list[dict]:
    """days you marked "sick" in the app (plan_state overrides, kept for good) as illness events: days up to
    2 apart are one illness."""
    p = Path(processed) / "plan_state.json"
    try:
        ov = json.loads(p.read_text(encoding="utf-8")).get("overrides") or {} if p.exists() else {}
    except (OSError, ValueError):
        ov = {}
    days = pd.DatetimeIndex(sorted(d for d, v in ov.items() if v == "sick"))
    if not len(days):
        return []
    start = np.r_[True, np.diff(days.values.astype("datetime64[D]").astype(np.int64)) > 2]   # a new one after 3+ days
    first, last = np.flatnonzero(start), np.r_[np.flatnonzero(start)[1:] - 1, len(days) - 1]
    return [{"date": str(days[a].date()), "end": str(days[b].date()), "type": "illness", "note": "marked sick in goodpush"}
            for a, b in zip(first, last)]


def illness_events(proj) -> list[dict]:
    """[[events]] from goodpush.toml plus the sick days you marked in the app."""
    return list(proj.cfg.get("events") or []) + sick_events(proj.processed)


def sources(proj, acts: pd.DataFrame) -> str:
    """a fingerprint of everything markers learn from (sick notes, events, sick days, your break labels):
    when it changes, they're learned again."""
    import hashlib
    from . import illness as IL
    from .dips import BreakLabels
    notes = IL.find_notes(acts)
    key = [LEARN_FORMAT, sorted(map(str, notes["date"])), illness_events(proj), BreakLabels(proj.processed).load()]
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def learn_if_due(proj, progress=print) -> dict | None:
    """learn() when nothing's been learned yet or an illness / label was recorded since (see `sources`);
    after a rebuild or a history import, and when you mark a sick day."""
    from ..io.store import load_tables
    try:
        t = load_tables(proj.processed)
    except FileNotFoundError:                     # no analysis yet
        return None
    old = load(proj.processed)
    if old is not None and old.get("sources") == sources(proj, t["acts"]):
        return None
    progress("Learning your illness markers from your history ...")
    return learn(proj, progress, t)


def learn(proj, progress=print, tables: dict | None = None) -> dict:
    """Analyse training breaks and learn your early-warning markers (see models/dips.py and
    models/markers.py). Your own cause labels win over the automatic guesses; only breaks you
    confirmed (plus illnesses from your ride notes and sick days) are used to learn from - never an automatic
    cause, so a planned recovery phase teaches nothing."""
    from ..io.store import load_tables
    from . import illness as IL
    from .dips import BreakLabels, KnownFacts, TrainingBreaks
    t = tables or load_tables(proj.processed)
    daily = t["daily"]
    events = illness_events(proj)
    notes = IL.find_notes(t["acts"])
    progress(f"{len(notes)} ride notes mention being sick")
    labels = BreakLabels(proj.processed).load()      # sick days: a known fact there, not an [[events]] entry
    dips = TrainingBreaks(daily, notes, list(proj.cfg.get("events") or []), labels,
                          KnownFacts.of(proj.root, proj.processed)).analyze()
    progress(f"{len(dips)} training breaks found")
    recs = json.loads(dips.to_json(orient="records", date_format="iso")) if len(dips) else []
    for r in recs:
        r["start"], r["end"] = r["start"][:10], r["end"][:10]
    profile = {}
    if len(dips):
        per, profile = TrainingBreaks.around(daily, dips)
        for r, a in zip(recs, per):
            r["around"] = a
    (proj.processed / "dips.json").write_text(json.dumps({"analyzed": str(pd.Timestamp.now().date()),
                                                          "dips": recs, "profile": profile}),
                                              encoding="utf-8")

    ep = IL.build_episodes(daily, notes, events)
    # what you said: labelled breaks, and the periods you marked yourself (gains aside); a labelled break inside one
    # of yours with the same cause counts once
    yours = [{**p, "label": p["cause"], "days": (pd.Timestamp(p["end"]) - pd.Timestamp(p["start"])).days + 1}
             for p in BreakLabels(proj.processed).periods() if p["cause"] != "gain"]
    told = yours + [r for r in recs if r.get("label") and not any(
        p["label"] == r["label"] and p["start"] <= r["end"] and r["start"] <= p["end"] for p in yours)]
    # a break you said was NOT illness (e.g. travel) removes a note-based episode inside it
    not_ill = [(pd.Timestamp(r["start"]), pd.Timestamp(r["end"])) for r in told if r["label"] != "illness"]
    if len(ep) and not_ill:
        keep = [not any(s - pd.Timedelta(days=3) <= pd.Timestamp(o) <= e for s, e in not_ill)
                for o in ep["onset"]]
        ep = ep[keep]
    ill = Episodes.illness(ep, told)
    progress(f"learning illness markers from {len(ill)} illnesses")
    learned = discover(daily, ill)
    learned["episode_list"] = [{"onset": str(e["onset"]), "noticed": str(e.get("noticed", "")),
                                "severity": e["severity"], "days_off": int(e["days_off"])}
                               for _, e in ill.iterrows()] if len(ill) else []
    learned["sources"] = sources(proj, t["acts"])
    save(proj.processed, learned)

    over = Episodes.overreaching(told)
    progress(f"learning overreaching markers from {len(over)} confirmed breaks")
    lo = discover(daily, over, families=OVERREACH_FAMILIES, before=14, min_episodes=3,
                    target="overreaching")
    save(proj.processed, lo, "overreaching")
    if learned["fallback"]:
        progress("Illness: not enough history yet - using the general resting-HR rule.")
    else:
        progress("Illness markers: " + "; ".join(c["rule"] for c in learned["chosen"]))
    progress("Overreaching markers: " + ("; ".join(c["rule"] for c in lo["chosen"]) if lo["chosen"] else
                                          f"none - nothing came reliably before the {len(over)} periods you marked"
                                          if len(over) >= lo["min_episodes"] else
                                          "none yet - confirm at least 3 training breaks as overreaching"))
    return {"illness": learned, "overreaching": lo, "dips": recs}


class Episodes:
    """The episodes markers learn from, one per illness / overreaching spell: the illnesses from your notes, events
    and sick days merged with the periods you marked and the breaks you labelled, so one illness counts once."""
    COLUMNS = ["onset", "noticed", "severity", "days_off"]
    ILLNESS_NEAR = 10        # a note-based episode starting inside a marked illness, or this many days from its ends, joins it
    OVERREACH_GAP = 21       # overreaching periods starting at most this many days apart are one

    @classmethod
    def illness(cls, ep: pd.DataFrame, told: list[dict]) -> pd.DataFrame:
        """`ep` (IL.build_episodes) with each illness you told (label "illness": start, end, days) absorbing the
        episodes that start inside it or within ILLNESS_NEAR days of either end: one episode, anchored at the
        earliest "noticed" (yours = its start), confirmed whatever the notes said."""
        per = pd.DataFrame([r for r in told if r["label"] == "illness"], columns=["start", "end", "days"])
        ep = ep.reset_index(drop=True) if len(ep) else pd.DataFrame(columns=cls.COLUMNS)
        if not len(per):
            return ep
        s, e = pd.to_datetime(per["start"]).to_numpy(), pd.to_datetime(per["end"]).to_numpy()
        near = np.timedelta64(cls.ILLNESS_NEAR, "D")
        onset = pd.to_datetime(ep["onset"]).to_numpy()[:, None]
        inside = (onset >= s - near) & (onset <= e + near)                 # (episodes, periods)
        owner = np.where(inside.any(axis=1), inside.argmax(axis=1), -1)
        took = ep[owner >= 0]
        sev = took["severity"].astype(str)
        parts = pd.concat([
            pd.DataFrame({"group": owner[owner >= 0], "onset": pd.to_datetime(took["onset"]).to_numpy(),
                          "noticed": pd.to_datetime(took["noticed"]).to_numpy(),
                          "severity": sev.where(~sev.str.startswith("unconfirmed")).to_numpy(),
                          "days_off": took["days_off"].astype(float).to_numpy()}),
            pd.DataFrame({"group": np.arange(len(per)), "onset": s, "noticed": s, "severity": None,
                          "days_off": per["days"].astype(float).to_numpy()})])
        g = parts.sort_values("noticed").groupby("group")
        merged = pd.DataFrame({"onset": g["onset"].min().dt.date, "noticed": g["noticed"].min().dt.date,
                               "severity": g["severity"].first().fillna("moderate"),
                               "days_off": g["days_off"].max().astype(int)})
        return pd.concat([ep[owner < 0], merged], ignore_index=True)

    @classmethod
    def overreaching(cls, told: list[dict]) -> pd.DataFrame:
        """the overreaching you told (label "overreaching"), periods starting at most OVERREACH_GAP days apart as
        one, anchored at the first start."""
        per = pd.DataFrame([r for r in told if r["label"] == "overreaching"], columns=["start", "end"])
        if not len(per):
            return pd.DataFrame(columns=cls.COLUMNS)
        per = per.assign(start=pd.to_datetime(per["start"]), end=pd.to_datetime(per["end"])).sort_values("start")
        g = per.groupby((per["start"].diff() > pd.Timedelta(days=cls.OVERREACH_GAP)).cumsum())
        start, end = g["start"].min(), g["end"].max()
        return pd.DataFrame({"onset": start.dt.date, "noticed": start.dt.date, "severity": "confirmed",
                             "days_off": (end - start).dt.days + 1}).reset_index(drop=True)
