"""Training breaks: find the periods where training (and so CTL) fell away, and say why.

What is known comes first: your own label (the app, data/processed/dip_labels.json) always wins; then what the app
already knows (KnownFacts): a day you marked yourself sick, days away (the Calendar), a planned recovery phase (the
season); then an [[events]] entry in goodpush.toml. Only then a guess from the evidence (BreakCause), kept simple
and honest - there are no labels to score rules against yet:
- illness:      sick note nearby, or resting HR / HRV / breathing rate went off right as training stopped
- overreaching: the weeks before were heavy for you (load spike, monotony, deep fatigue or a fast build against your
                own history) AND the body was already showing it (HRV down or resting HR up beyond what the load
                explains, or slow to bounce back): load alone is a hard block, not overreaching
- travel:       your sleep moved to a different clock (time-zone change), lots of steps
- life:         body readings there, nothing stands out (busy, holiday at home, weather, ...)
- unclear:      no sleep / HRV / resting-HR readings around the break: nothing to judge by
- planned:      a planned recovery phase; injury only from your label / [[events]]
Your labels are what the marker learning uses (models/markers.py); automatic causes, planned ones included, never are.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CAUSES = ["illness", "overreaching", "travel", "life", "injury", "planned", "unclear"]
CAUSE_LABEL = {"illness": "Illness", "overreaching": "Overreaching / burnout", "travel": "Travel",
               "life": "Life / holiday", "injury": "Injury", "planned": "Planned recovery", "unclear": "Unclear"}


class BreakLabels:
    """Your own causes for breaks (the app's "What was it?"), by start day, in processed/dip_labels.json; and the
    periods you marked on the load chart yourself (`periods`: start, end, cause - a cause or a gain), kept beside
    the detected breaks (until detection learns from them)."""
    FILE = "dip_labels.json"
    KINDS = CAUSES + ["gain"]          # what a period you mark can be (gain: a stretch your fitness rose)

    def __init__(self, processed: Path):
        self.path = Path(processed) / self.FILE

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        except (OSError, ValueError):
            return {}

    def set(self, start: str, cause: str) -> dict:
        """label the break starting on `start` ("" = back to the automatic guess)."""
        if cause and cause not in CAUSES:
            raise ValueError(f"cause must be one of {CAUSES}")
        lab = self.load()
        if cause:
            lab[str(start)] = cause
        else:
            lab.pop(str(start), None)
        return self.save(lab)

    def save(self, lab: dict) -> dict:
        self.path.write_text(json.dumps(lab, indent=1), encoding="utf-8")
        return lab

    def periods(self, kind: str | None = None) -> list[dict]:
        """the periods you marked, by start ({start, end, cause}); `kind` = only that cause. Stored as
        {"start/end": cause}: a peer sync merges them one by one."""
        out = [dict(zip(("start", "end"), k.split("/")), cause=c) for k, c in sorted(self.load().get("periods", {}).items())]
        return [p for p in out if kind is None or p["cause"] == kind]

    def set_period(self, start: str, end: str, cause: str, was: tuple[str, str] | None = None) -> dict:
        """mark the days start..end as `cause` ("" = remove the period); `was` = its (start, end) before, when you
        moved its edges or changed it."""
        if cause and cause not in self.KINDS:
            raise ValueError(f"a period is one of {self.KINDS}")
        start, end = sorted((str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))
        lab = self.load()
        per = lab.setdefault("periods", {})
        if was:
            per.pop("/".join(was), None)
        per.pop(f"{start}/{end}", None)
        if cause:
            per[f"{start}/{end}"] = cause
        lab["periods"] = dict(sorted(per.items()))
        return self.save(lab)


class KnownFacts:
    """What the app already knows about days, before any guess: days you marked yourself sick (Today), days away
    (Calendar; with a note), the season's recovery phases. A break overlapping one gets its cause directly."""
    SICK_BEFORE = 3            # a sick day this close before a break still explains it (you stop the day after)

    def __init__(self, sick=(), away: dict | None = None, recovery=()):
        self.sick = pd.DatetimeIndex(sorted(sick))
        self.away = {pd.Timestamp(k): str(v or "") for k, v in (away or {}).items()}
        self.recovery = pd.DatetimeIndex(sorted(recovery))

    @classmethod
    def of(cls, root: Path, processed: Path) -> KnownFacts:
        """from plan_state.json (sick days, days away) and the season (data/plans.json)."""
        from ..planning.season import Season
        try:
            st = json.loads((Path(processed) / "plan_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        sick = [d for d, v in (st.get("overrides") or {}).items() if v == "sick"]
        phases = [p for p in Season.load(root).phases if p.type == "recovery"]
        recovery = [d for p in phases for d in pd.date_range(p.start, p.end)]
        return cls(sick, st.get("away") or {}, recovery)

    def cause(self, s: pd.Timestamp, e: pd.Timestamp, ev: dict) -> tuple[str, str, list[str]] | None:
        """(cause, "high", reasons) when a known fact covers the break s..e; ev: its evidence (travel or not)."""
        if ((self.sick >= s - pd.Timedelta(days=self.SICK_BEFORE)) & (self.sick <= e)).any():
            return "illness", "high", ["you marked yourself sick"]
        notes = [n for d, n in sorted(self.away.items()) if s <= d <= e]
        if notes:
            note = next((n for n in notes if n), "")
            reason = "days away" + (f": {note}" if note else "")
            if ev["bedtime_shift_nights"] >= 2 or ev["steps_high_days"] >= 2:
                return "travel", "high", [reason, "sleep on another clock or lots of walking"]
            return "life", "high", [reason]
        if ((self.recovery >= s) & (self.recovery <= e)).any():
            return "planned", "high", ["a planned recovery phase"]
        return None


class LoadCutoffs:
    """When the weeks before a break count as heavy, for you: a 4-week CTL rise and a form (TSB) low against your own
    history (beyond QUANTILE of your days), the usual fixed numbers while the history is short."""
    RAMP, TSB = 10.0, -25.0    # fallbacks: CTL +10 in 4 weeks, form -25
    QUANTILE = 0.9             # heavy = beyond 9 in 10 of your days
    MIN_DAYS = 180

    def __init__(self, daily: pd.DataFrame):
        ramp = (daily["ctl"] - daily["ctl"].shift(28)).dropna() if "ctl" in daily else pd.Series(dtype=float)
        tsb = daily["tsb"].dropna() if "tsb" in daily else pd.Series(dtype=float)
        self.ramp = float(ramp.quantile(self.QUANTILE)) if len(ramp) >= self.MIN_DAYS else self.RAMP
        self.tsb = float(tsb.quantile(1 - self.QUANTILE)) if len(tsb) >= self.MIN_DAYS else self.TSB


class BreakEvidence:
    """What the body and the training did around a break. The daily signals (each against its own rolling
    BASE_DAYS normal) are computed once for the whole table; `of` reads them for one break."""
    BASE_DAYS = 42
    BASE_MIN = 15

    WELLNESS = ("rhr", "hrv", "sleep_h")
    WELLNESS_DAYS = 14         # body readings looked for this long either side of a break

    def __init__(self, daily: pd.DataFrame, notes: pd.DataFrame | None = None, events: list | None = None):
        from ..planning.readiness import LoadResponse
        self.d, self.notes, self.events = daily, notes, events
        nan = pd.Series(np.nan, index=daily.index)
        self.rhr_d = daily["rhr"] - self.baseline(daily["rhr"]) if "rhr" in daily else nan
        self.hrv_z, self.steps_z = self.zscore("hrv"), self.zscore("steps")
        # the weeks before: HRV / resting HR net of what the load of the days before explains (readiness.LoadResponse),
        # so a hard block's expected dip isn't read as the body struggling
        load = LoadResponse(daily)
        net = {c: daily[c] - load.explained(daily[c]) for c in ("rhr", "hrv") if c in daily}
        self.pre_rhr_d = net["rhr"] - self.baseline(net["rhr"]) if "rhr" in net else nan
        self.pre_hrv_z = self.zscore("hrv", net.get("hrv"))
        have = [c for c in self.WELLNESS if c in daily]
        self.wellness = daily[have].notna().any(axis=1) if have else pd.Series(False, index=daily.index)
        self.resp_d = daily["respiration"] - self.baseline(daily["respiration"]) if "respiration" in daily else nan
        if "bedtime_h" in daily:                    # bedtime on another clock (time zones), either way round midnight
            shift = (daily["bedtime_h"] - self.baseline(daily["bedtime_h"])).abs()
            self.bed_shift = np.minimum(shift, 24 - shift)
        else:
            self.bed_shift = None

    def baseline(self, x: pd.Series) -> pd.Series:
        return x.rolling(self.BASE_DAYS, min_periods=self.BASE_MIN).median().shift(1)

    def zscore(self, col: str, x: pd.Series | None = None) -> pd.Series:
        """col (or x, a version of it) against its rolling normal, in SDs."""
        if col not in self.d:
            return pd.Series(np.nan, index=self.d.index)
        x = self.d[col] if x is None else x
        sd = x.rolling(self.BASE_DAYS, min_periods=self.BASE_MIN).std().shift(1).replace(0, np.nan)
        return (x - self.baseline(x)) / sd

    @staticmethod
    def extreme(x: pd.Series, how: str, need: int = 1) -> float | None:
        """x's max / min / mean where it has at least `need` values, else None."""
        return float(getattr(x, how)()) if x.notna().sum() >= need else None

    def of(self, s: pd.Timestamp, e: pd.Timestamp) -> dict:
        d, day = self.d, pd.Timedelta(days=1)
        early = slice(s - 3 * day, s + 7 * day)
        pre, pre14, during, post = slice(s - 21 * day, s - day), slice(s - 14 * day, s - day), slice(s, e), slice(e + day, e + 14 * day)
        ev = {"rhr_peak": self.extreme(self.rhr_d.loc[early], "max")}
        run = self.rhr_d.loc[early] >= 3
        ev["rhr_2days"] = bool((run & run.shift(1, fill_value=False)).any())
        ev["hrv_min_z"] = self.extreme(self.hrv_z.loc[early], "min")
        ev["resp_peak"] = self.extreme(self.resp_d.loc[early], "max")
        ev["pre_hrv_z"] = self.extreme(self.pre_hrv_z.loc[pre14], "mean", 5)      # net of the load before
        ev["pre_rhr_d"] = self.extreme(self.pre_rhr_d.loc[pre14], "mean", 5)
        ev["wellness_days"] = int(self.wellness.loc[s - self.WELLNESS_DAYS * day:e + self.WELLNESS_DAYS * day].sum())
        for c, k, f in (("acwr", "pre_acwr_max", "max"), ("monotony", "pre_monotony_max", "max"), ("tsb", "pre_tsb_min", "min")):
            ev[k] = self.extreme(d[c].loc[pre], f) if c in d else None
        c0, c1 = d["ctl"].get(s - 29 * day, np.nan), d["ctl"].get(s - day, np.nan)
        ev["pre_ramp_4wk"] = float(c1 - c0) if np.isfinite(c0) and np.isfinite(c1) else None
        ev["bedtime_shift_nights"] = int((self.bed_shift.loc[during] >= 2.5).sum()) if self.bed_shift is not None else 0
        ev["rhr_days_up"] = int((self.rhr_d.loc[during] >= 3).sum())
        ev["post_hrv_z"] = self.extreme(self.hrv_z.loc[post], "mean", 5)
        ev["post_rhr_d"] = self.extreme(self.rhr_d.loc[post], "mean", 5)
        st = self.steps_z.loc[during]
        ev["steps_high_days"] = int((st >= 1.0).sum()) if st.notna().any() else 0
        ev["sick_note"] = False
        if self.notes is not None and len(self.notes):
            nd = pd.to_datetime(self.notes["date"])
            ev["sick_note"] = bool(((nd >= s - 3 * day) & (nd <= e + 14 * day)).any())
        ev["event"] = None
        for x in self.events or []:
            if "example" in str(x.get("note", "")):
                continue
            a, b = pd.Timestamp(x["date"]), pd.Timestamp(x.get("end") or x["date"])
            if a <= e + 3 * day and b >= s - 3 * day:
                ev["event"] = {"type": str(x.get("type", "")).lower(), "note": x.get("note", "")}
        return ev


class BreakCause:
    """The automatic guess from a break's evidence: points per cause from simple rules, the most points wins
    (at least 2), confidence from its lead over the next. Overreaching needs a body sign as well as load; without
    body readings around the break the guess is "unclear", not "life"."""

    def __init__(self, cutoffs: LoadCutoffs | None = None):
        self.cut = cutoffs or LoadCutoffs(pd.DataFrame())

    def body_sign(self, ev: dict) -> bool:
        """the body showed it before or after: HRV down / resting HR up beyond the load, or slow to bounce back."""
        slow = self.at_most(ev["post_hrv_z"], -0.6) and self.at_most(ev["pre_hrv_z"], -0.3)
        return self.at_most(ev["pre_hrv_z"], -0.6) or self.at_least(ev["pre_rhr_d"], 1.5) or slow

    def classify(self, ev: dict) -> tuple[str, str, list[str]]:
        """(cause, confidence, reasons)."""
        why = {"illness": [], "overreaching": [], "travel": [], "life": []}
        sc = {"illness": 0.0, "overreaching": 0.0, "travel": 0.0}
        if ev.get("event") and ev["event"]["type"] in CAUSES:
            return ev["event"]["type"], "high", [f"goodpush.toml event: {ev['event']['type']} {ev['event']['note']}".strip()]
        rules = [
            ("illness", 3, ev["sick_note"], "sickness mentioned in a ride note"),
            ("illness", 2, ev["rhr_2days"], f"resting HR up to +{ev['rhr_peak'] or 0:.0f} bpm as training stopped"),
            ("illness", 1, not ev["rhr_2days"] and (ev["rhr_peak"] or 0) >= 3, f"resting HR +{ev['rhr_peak'] or 0:.0f} bpm for a day"),
            ("illness", 1, self.at_most(ev["hrv_min_z"], -1.5), f"HRV dropped to {ev['hrv_min_z'] or 0:.1f} SD below normal"),
            ("illness", 1, self.at_least(ev["resp_peak"], 1.5), f"breathing rate +{ev['resp_peak'] or 0:.1f}/min"),
            ("overreaching", 1, self.at_least(ev["pre_acwr_max"], 1.4), f"load spike before (acute:chronic {ev['pre_acwr_max'] or 0:.2f})"),
            ("overreaching", 1, self.at_least(ev["pre_monotony_max"], 1.8), f"monotonous training before (monotony {ev['pre_monotony_max'] or 0:.1f})"),
            ("overreaching", 1, self.at_most(ev["pre_tsb_min"], self.cut.tsb), f"deep fatigue before (form {ev['pre_tsb_min'] or 0:.0f}, lower than 9 in 10 of your days)"),
            ("overreaching", 1, self.at_least(ev["pre_ramp_4wk"], self.cut.ramp), f"fast build before (CTL +{ev['pre_ramp_4wk'] or 0:.0f} in 4 weeks, faster than 9 in 10 of yours)"),
            ("overreaching", 1, self.at_most(ev["pre_hrv_z"], -0.6), f"HRV already down beyond what the load explains ({ev['pre_hrv_z'] or 0:.1f} SD, 2 weeks before)"),
            ("illness", 1, ev["rhr_days_up"] >= 3, f"resting HR stayed up {ev['rhr_days_up']} days during the break"),
            ("overreaching", 1, self.at_least(ev["pre_rhr_d"], 1.5), f"resting HR already up beyond what the load explains (+{ev['pre_rhr_d'] or 0:.1f} bpm)"),
            ("overreaching", 1, self.at_most(ev["post_hrv_z"], -0.6) and self.at_most(ev["pre_hrv_z"], -0.3),
             f"HRV still low 2 weeks after ({ev['post_hrv_z'] or 0:.1f} SD): slow to bounce back"),
            ("travel", 3, ev["bedtime_shift_nights"] >= 2, f"sleep on a different clock for {ev['bedtime_shift_nights']} nights (time-zone change)"),
            ("travel", 1, ev["steps_high_days"] >= 2, f"lots of walking ({ev['steps_high_days']} high-step days)")]
        for cause, pts, hit, text in rules:
            if hit:
                sc[cause] += pts
                why[cause].append(text)
        if not self.body_sign(ev):                      # load alone: a hard block, not overreaching
            sc["overreaching"] = 0.0
        if sc["travel"] >= 3 and not ev["sick_note"]:   # flights + jet lag also nudge HR / HRV
            sc["illness"] -= 1
        best = max(sc, key=sc.get)
        if sc[best] < 2:
            if not ev["wellness_days"]:
                return "unclear", "low", ["no sleep / HRV / resting-HR data around this break"]
            return "life", "low", ["no physiological stress signs and no time-zone change"]
        second = sorted(sc.values())[-2]
        conf = "high" if sc[best] >= 3 and sc[best] - second >= 2 else "medium" if sc[best] - second >= 1 else "low"
        return best, conf, why[best]

    @staticmethod
    def at_least(v, cut) -> bool:
        return v is not None and v >= cut

    @staticmethod
    def at_most(v, cut) -> bool:
        return v is not None and v <= cut


class TrainingBreaks:
    """Runs of (mostly) no training long enough to cost fitness, each with its cause: your label, else the
    automatic guess (BreakCause) from the evidence around it (BreakEvidence)."""
    MIN_DAYS = 5               # a break: at least this many days ...
    MIN_CTL_DROP = 4.0         # ... costing at least this much CTL (and 8 % of it)
    MIN_CTL_SHARE = 0.08
    MIN_RUN = 3                # runs shorter than this are rest days: they don't chain into a break
    JOIN_DAYS = 4              # one illness or trip split by a test ride or two: runs this close are one break
    AROUND = (-28, 28)         # days around the start of a break shown / averaged
    PROFILE_METRICS = {"rhr_d": "Resting HR vs baseline (bpm)", "hrv_z": "HRV vs baseline (SD)",
                       "sleep_d": "Sleep vs usual (h)", "load7": "Training load, last 7 days (TSS)",
                       "tsb": "Form (TSB)"}

    def __init__(self, daily: pd.DataFrame, notes: pd.DataFrame | None = None, events: list | None = None,
                 labels: dict | None = None, known: KnownFacts | None = None):
        self.daily, self.notes, self.events, self.labels = daily, notes, events, labels or {}
        self.known = known or KnownFacts()

    def runs(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """(first, last) day of each run of low training: days far under the usual load (a single short ride inside
        is allowed), runs of MIN_RUN+ days joined when JOIN_DAYS or fewer apart."""
        d = self.daily
        base = d["tss"].rolling(42, min_periods=14).mean().shift(1)
        low = (d["tss"] < np.maximum(0.25 * base, 10)) & base.gt(20)
        low = low | (low.shift(1, fill_value=False) & low.shift(-1, fill_value=False) & (d["tss"] < 0.6 * base))
        edge = np.diff(np.r_[0, low.to_numpy(np.int8), 0])
        days = d.index
        first, last = days[np.flatnonzero(edge == 1)], days[np.flatnonzero(edge == -1) - 1]
        keep = (last - first).days + 1 >= self.MIN_RUN
        first, last = first[keep], last[keep]
        if not len(first):
            return []
        new = np.r_[True, (first[1:] - last[:-1]).days > self.JOIN_DAYS]
        g = np.flatnonzero(new)
        return list(zip(first[g], last[np.r_[g[1:] - 1, len(first) - 1]]))

    def detect(self) -> pd.DataFrame:
        """the runs long enough and costing enough CTL: start, end, days, CTL before / lowest / drop."""
        d, rows = self.daily, []
        for s, e in self.runs():
            n = (e - s).days + 1
            if n < self.MIN_DAYS:
                continue
            before = float(d["ctl"].get(s - pd.Timedelta(days=1), np.nan))
            low = float(d["ctl"].loc[s:e + pd.Timedelta(days=3)].min())
            drop = before - low
            if not np.isfinite(drop) or drop < self.MIN_CTL_DROP or drop < self.MIN_CTL_SHARE * before:
                continue
            rows.append({"start": s.date(), "end": e.date(), "days": n, "ctl_before": round(before, 1),
                         "ctl_min": round(low, 1), "ctl_drop": round(drop, 1),
                         "ctl_drop_pct": round(drop / before * 100, 0) if before else np.nan})
        return pd.DataFrame(rows)

    def analyze(self) -> pd.DataFrame:
        """each break with its automatic cause (a known fact, else the guess), confidence, reasons and evidence, your
        label and the cause in use (your label, else the automatic one)."""
        breaks = self.detect()
        if breaks.empty:
            return breaks
        evidence, guess, out = BreakEvidence(self.daily, self.notes, self.events), BreakCause(LoadCutoffs(self.daily)), []
        for r in breaks.to_dict("records"):
            s, e = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
            ev = evidence.of(s, e)
            cause, conf, why = self.known.cause(s, e, ev) or guess.classify(ev)
            lab = self.labels.get(str(r["start"]))
            r.update({"auto_cause": cause, "confidence": conf, "reasons": why, "evidence": ev,
                      "label": lab or "", "cause": lab or cause})
            out.append(r)
        return pd.DataFrame(out)

    @classmethod
    def around(cls, daily: pd.DataFrame, breaks: pd.DataFrame) -> tuple[list, dict]:
        """Each break's metrics from 4 weeks before to 4 weeks after it started, and the average
        pattern per cause - what illness, overreaching and travel look like for you."""
        from .markers import features
        f = features(daily)
        f["load7"] = daily["tss"].rolling(7, min_periods=1).sum()
        f["tsb"] = daily["tsb"] if "tsb" in daily else np.nan
        offs = list(range(cls.AROUND[0], cls.AROUND[1] + 1))
        metrics = list(cls.PROFILE_METRICS)
        per, groups = [], {}
        for r in breaks.to_dict("records"):
            s0 = pd.Timestamp(r["start"])
            w = f.reindex([s0 + pd.Timedelta(days=o) for o in offs])
            per.append({m: [None if pd.isna(v) else round(float(v), 2) for v in w[m]] for m in metrics})
            groups.setdefault(r["cause"], []).append(w[metrics].to_numpy(dtype=float))
        prof = {"offset": offs, "metrics": cls.PROFILE_METRICS, "causes": {}}
        for c, arrs in groups.items():
            a = np.stack(arrs)
            n = np.sum(np.isfinite(a), axis=0)
            m = np.where(n > 0, np.nansum(a, axis=0) / np.maximum(n, 1), np.nan)   # mean where measured (no "empty slice" warning)
            prof["causes"][c] = {"n": len(arrs), **{k: [None if (not np.isfinite(v) or nn < 2) else round(float(v), 2)
                                                        for v, nn in zip(m[:, j], n[:, j])]
                                                    for j, k in enumerate(metrics)}}
        return per, prof
