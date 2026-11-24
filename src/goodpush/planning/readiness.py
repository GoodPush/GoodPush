"""Today's readiness (green / amber / red) from Garmin wellness, training load, your markers and
how you say you feel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

EASY_DAYS = {"red": 2, "amber": 1}   # days kept easy from today (readiness["easy_days"])
STREAK = 4                  # non-green mornings of the last 7 (this one too) that keep tomorrow easy as well
LATE_BED_H = 0.75           # bedtime this much later than your usual, two nights running: a sign
BED_BASE_DAYS = 42
OVERRIDES = {"off": ("amber", "you said you feel off today"),
             "sick": ("red", "you said you're sick / coming down with something")}


class LoadResponse:
    """What the training of the days before does to a morning marker, learned from the rider's own mornings, so it
    can be taken out before the marker is judged: a raised resting HR or a low HRV the morning after a big day is
    the expected response to load (the point of a loading block), the same reading after an easy day is news
    (illness, stress, not recovering). On one rider's history (2026-11-10) resting HR, HRV and body battery followed
    the load of the day before (correlation 0.24-0.36 within months) and not the week's (0.05).
    strain: each morning's training stress of the days before, yesterday most (TAU_DAYS), less its usual level."""
    TAU_DAYS = 1.0              # a day's weight falls to 1 / e a day later
    USUAL_DAYS = 42             # the strain's usual level: its mean over the mornings before
    DAYS = 730                  # mornings the response is learned from
    MIN_DAYS = 90               # ... at least this many with the marker, else nothing is taken out
    MIN_T = 3.0                 # ... and a slope this many standard errors from 0: a clear response only

    def __init__(self, d: pd.DataFrame):
        tss = d["tss"].fillna(0.0).to_numpy(float) if "tss" in d else np.zeros(len(d))
        strain = pd.Series(lfilter([0.0, 1.0], [1.0, -np.exp(-1.0 / self.TAU_DAYS)], tss), index=d.index)
        self.strain = strain - strain.rolling(self.USUAL_DAYS, min_periods=self.USUAL_DAYS // 3).mean().shift(1)

    def slope(self, marker: pd.Series) -> float:
        """the marker's change per unit of strain over the mornings before its latest one; 0 without a clear one"""
        m = marker.dropna().iloc[:-1].iloc[-self.DAYS:]
        x = self.strain.reindex(m.index)
        ok = x.notna().to_numpy()
        x, y = x.to_numpy(float)[ok], m.to_numpy(float)[ok]
        if len(x) < self.MIN_DAYS or x.std() == 0 or y.std() == 0:
            return 0.0
        b, a = np.polyfit(x, y, 1)
        se = (y - a - b * x).std() / (x.std() * np.sqrt(len(x)))
        return float(b) if se > 0 and abs(b) >= self.MIN_T * se else 0.0

    def explained(self, marker: pd.Series) -> pd.Series:
        """the part of each morning's marker the training before it explains (0 where there is no strain yet)"""
        return (self.slope(marker) * self.strain.reindex(marker.index)).fillna(0.0)


class RecoveryTrend:
    """Are you recovering? Judged on the trend, not on one morning: the mean over the last DAYS mornings of each
    marker with the training before taken out (LoadResponse). A loading block is meant to load you, and a marker
    off after a hard day comes back after an easy one; one that stays off through the easy days - a week's mean
    OFF from your usual and this morning still STILL off - is not coming back (HRV-guided training judges a 7-day
    mean for the same reason: single mornings are mostly noise and the day before). On one rider's 3 years this
    came up about once a month, 2-3 mornings at a time (2026-11-10)."""
    DAYS = 7
    MIN_MORNINGS = 4            # of the last DAYS with the marker
    OFF, STILL = 1.0, 0.5       # in SDs of your usual (the markers are z-scores)
    MARKERS = (("hrv_z", -1.0, "HRV", "below"), ("rhr_z", 1.0, "resting HR", "above"))   # column, the strained sign
    LOADED_TSB = -10            # the opposite pattern (HRV up, resting HR down) is only worth a note under load

    def __init__(self, d: pd.DataFrame, today: pd.Timestamp, load: LoadResponse | None = None):
        load = load or LoadResponse(d)
        self.today, self.mean, self.now = today, {}, {}
        for col, _, _, _ in self.MARKERS:
            m = d[col].dropna() if col in d else pd.Series(dtype=float)
            if not len(m) or (today - m.index[-1]).days > 1:
                continue
            res = (m - load.explained(m)).loc[today - pd.Timedelta(days=self.DAYS - 1):]
            if len(res) >= self.MIN_MORNINGS:
                self.mean[col], self.now[col] = float(res.mean()), float(res.iloc[-1])
        tsb = d["tsb"].dropna() if "tsb" in d else pd.Series(dtype=float)
        self.loaded = bool(len(tsb) and tsb.iloc[-1] < self.LOADED_TSB)

    def off(self) -> list[str]:
        """the markers not coming back, in words (none: [])"""
        return [f"{name} has averaged {abs(self.mean[col]):.1f} SD {word} your usual over the last {self.DAYS} mornings, "
                "beyond what your training explains, and still is today"
                for col, sign, name, word in self.MARKERS
                if col in self.mean and sign * self.mean[col] >= self.OFF and sign * self.now[col] >= self.STILL]

    def held_down(self) -> bool:
        """HRV up and resting HR down for a week while loaded: the picture of an overreached rider when power is
        down too (Le Meur et al. 2013) - and of a fit, fresh one when it isn't. A note, never a flag."""
        return (self.loaded and len(self.mean) == len(self.MARKERS)
                and all(-sign * self.mean[col] >= self.OFF for col, sign, _, _ in self.MARKERS))


class MorningSigns:
    """This morning's signs, each 0 (fine), 1 (a bit off) or 2 (well off) points. One sign on its own is only a
    note; AMBER_POINTS together make the day amber, RED_POINTS red (flag when several agree, e.g.
    a low HRV with a low resting HR, not on HRV alone)."""
    AMBER_POINTS, RED_POINTS = 2, 4
    MILD_Z = 1.5                # sleep score / body battery / stress this many SD off your usual: a sign
    RHR_UP_BPM, RHR_WAY_UP_BPM = 3, 5
    RHR_LOW_BPM = 3             # resting HR this far below usual: a sign only together with a low HRV
    VERY_SHORT_H, TWO_NIGHTS_H = 5.5, 6.5
    DEEP_TSB = -30

    EXPLAINED = ("hrv_z", "body_battery_max_z")     # judged with the load of the days before taken out (LoadResponse)
    WORTH_SAYING = 0.3          # that much of a z (or 1 bpm) explained by load: said in the sign's text

    def __init__(self, d: pd.DataFrame, today: pd.Timestamp, load: LoadResponse | None = None):
        self.d, self.today, self.signs = d, today, []
        self.load = load

    def unexplained(self, col: str) -> tuple[float | None, str]:
        """this morning's value of a marker with what the training before explains taken out (what it is judged
        on), and words for the text when that is worth saying"""
        v = self.last(col)
        if v is None or self.load is None or col not in self.EXPLAINED:
            return v, ""
        e = float(self.load.explained(self.d[col].dropna()).iloc[-1])
        return v - e, (f"; measured {v:+.1f}, {e:+.1f} of it the training before" if abs(e) >= self.WORTH_SAYING else "")

    def last(self, col: str) -> float | None:
        """this morning's value (or yesterday's), else None"""
        v = self.d[col].dropna() if col in self.d else pd.Series(dtype=float)
        return float(v.iloc[-1]) if len(v) and (self.today - v.index[-1]).days <= 1 else None

    def add(self, name: str, points: int, text: str) -> None:
        self.signs.append({"name": name, "points": int(points), "text": text})

    def wellness(self, rhr_delta: float | None, rhr_load: float = 0.0) -> None:
        """HRV, resting HR, sleep, sleep score, body battery, stress, bedtime. rhr_delta: resting HR vs usual with
        rhr_load, the bpm the training before explains, already taken out."""
        hz, said = self.unexplained("hrv_z")
        if hz is not None:
            self.add("HRV", 2 if hz <= -2 else 1 if hz <= -1 else 0,
                     f"HRV {'very low' if hz <= -2 else 'low' if hz <= -1 else 'normal'} (z {hz:+.1f}{said})")
        if rhr_delta is not None:
            low_both = rhr_delta <= -self.RHR_LOW_BPM and hz is not None and hz <= -1
            pts = 2 if rhr_delta >= self.RHR_WAY_UP_BPM else 1 if rhr_delta >= self.RHR_UP_BPM or low_both else 0
            self.add("Resting HR", pts, f"resting HR {rhr_delta:+.0f} bpm vs usual"
                     + (f" (measured {rhr_delta + rhr_load:+.0f}, {rhr_load:+.0f} of it the training before)"
                        if abs(rhr_load) >= 1 else "")
                     + (" - low together with a low HRV (can mean deep fatigue)" if low_both else ""))
        self.sleep()
        for col, name, high in (("sleep_score_z", "Sleep score", False), ("body_battery_max_z", "Body battery", False),
                                ("stress_z", "Stress", True)):
            z, said = self.unexplained(col)
            if z is not None:
                off = z >= self.MILD_Z if high else z <= -self.MILD_Z
                self.add(name, off, f"{name.lower()} {('high' if high else 'low') if off else 'normal'} (z {z:+.1f}{said})")
        if "bedtime_h" in self.d and self.d["bedtime_h"].notna().any():
            late = late_bedtimes(self.d, self.today)
            self.add("Bedtime", bool(late), f"bedtime {late} min later than usual two nights running (tiredness "
                                            "pushes your bedtime)" if late else "bedtime as usual")

    def sleep(self) -> None:
        sl = self.d["sleep_h"].dropna()
        if not len(sl) or (self.today - sl.index[-1]).days > 1:
            return
        two = sl.iloc[-2:].mean() if len(sl) >= 2 and (sl.index[-1] - sl.index[-2]).days == 1 else None
        base = self.d["sleep_h_base"].dropna() if "sleep_h_base" in self.d else pd.Series(dtype=float)
        three = sl.iloc[-3:].mean()
        if sl.iloc[-1] < self.VERY_SHORT_H:
            self.add("Sleep", 2, f"very short sleep ({sl.iloc[-1]:.1f} h)")
        elif two is not None and two < self.TWO_NIGHTS_H:      # it's two short nights that cost you, not one
            self.add("Sleep", 2, f"two short nights ({two:.1f} h a night)")
        elif len(base) and three < base.iloc[-1] - 0.75:
            self.add("Sleep", 1, f"3-night sleep {three:.1f} h vs usual {base.iloc[-1]:.1f} h")
        else:
            self.add("Sleep", 0, f"sleep {sl.iloc[-1]:.1f} h")

    def riding(self) -> None:
        """DFA alpha1 and heart rate in your most recent warm-up (vs your normal at that power), and load"""
        az = self.d["a1_warm_z"].dropna() if "a1_warm_z" in self.d else pd.Series(dtype=float)
        if len(az) and (self.today - az.index[-1]).days <= 1:
            zv, when = az.iloc[-1], "today's" if az.index[-1] == self.today else "yesterday's"
            word = "unusually low" if zv <= -2 else "unusually high" if zv >= 2 else "normal"
            self.add("Warm-up alpha1", abs(zv) >= 2, f"alpha1 in {when} warm-up {word} for the power ({zv:+.1f} SD)")
        whz = self.last("a1_warm_hr_z")
        if whz is not None and whz >= 2:
            self.add("Warm-up HR", 1, f"heart rate high for the power in the last warm-up ({whz:+.1f} SD)")
        tsb = self.d["tsb"].dropna() if "tsb" in self.d else pd.Series(dtype=float)
        if len(tsb):
            deep = tsb.iloc[-1] < self.DEEP_TSB
            self.add("Load", deep, f"{'deep fatigue' if deep else 'form'} (TSB {tsb.iloc[-1]:.0f})")

    def level(self) -> tuple[str | None, list[str], str | None]:
        """("red" | "amber" | None, the texts of the signs that are off, a note when one sign is off on its own)"""
        off = [s for s in self.signs if s["points"]]
        points = sum(s["points"] for s in off)
        texts = [s["text"] for s in off]
        if points >= self.RED_POINTS:
            return "red", texts, None
        if points >= self.AMBER_POINTS:
            return "amber", texts, None
        return None, [], (f"{texts[0]} - one sign on its own, the rest normal: no change" if off else None)


def readiness(daily: pd.DataFrame, today: pd.Timestamp, recent_illness: bool = False,
              override: str | None = None, markers: dict | None = None,
              markers_learned: bool = False, overreach: dict | None = None) -> dict:
    """Green / amber / red from the most recent Garmin + load data (+ how you say you feel): the morning signs
    (MorningSigns) count together; your learned illness / overreaching markers, the resting-HR illness rule, a
    sickness note and how you feel decide on their own. "signs": every sign checked (name, points, text)."""
    d = daily.loc[:today]
    out = {"status": "green", "reasons": [], "notes": [], "signs": [], "data_age_days": None}
    well = d[["rhr", "hrv", "sleep_h"]].dropna(how="all")
    if len(well):
        out["data_age_days"] = int((today - well.index[-1]).days)
    stale = out["data_age_days"] is None or out["data_age_days"] > 2
    reds, ambers = [], []
    load = LoadResponse(d)
    signs = MorningSigns(d, today, load)
    if not stale:
        rb = d["rhr"].rolling(42, min_periods=15).median().shift(1)
        rd = (d["rhr"] - rb).dropna()
        rhr_load = load.explained(rd)         # resting HR up after a big day is the load, not an illness signal
        rd = rd - rhr_load
        if not markers_learned and len(rd) >= 2 and (rd.iloc[-2:] >= 3).all():
            reds.append(f"resting HR +{rd.iloc[-1]:.0f} bpm two days running beyond what your training explains "
                        "(your illness signal)")
        if markers and markers.get("as_of") and (today - pd.Timestamp(markers["as_of"])).days <= 1:
            lv, names = markers.get("level"), ", ".join(markers.get("firing") or [])
            if lv == "warning":
                reds.append(f"illness warning - your markers: {names}")
            elif lv == "watch":
                ambers.append(f"illness watch - {names}")
        if overreach and overreach.get("as_of") and (today - pd.Timestamp(overreach["as_of"])).days <= 1 \
                and overreach.get("level") in ("warning", "watch"):
            ambers.append("signs of overreaching (your markers): " + ", ".join(overreach.get("firing") or []))
        two_days = len(rd) >= 2 and (rd.iloc[-2:] >= 3).all()
        rhr_delta = float(rd.iloc[-1]) if len(rd) and (today - rd.index[-1]).days <= 1 and not \
            (two_days and not markers_learned) else None        # the illness rule above already counted it
        signs.wellness(rhr_delta, float(rhr_load.iloc[-1]) if len(rhr_load) else 0.0)
    signs.riding()
    out["signs"] = signs.signs
    lvl, texts, note = signs.level()
    (reds if lvl == "red" else ambers).extend(texts)
    if note:
        out["notes"].append(note)
    if recent_illness:
        ambers.append("sickness mentioned in a ride note in the last 10 days")
    if override in OVERRIDES:
        lvl, why = OVERRIDES[override]
        (reds if lvl == "red" else ambers).insert(0, why)
    if reds:
        out["status"], out["reasons"] = "red", reds + ambers
    elif ambers:
        out["status"], out["reasons"] = "amber", ambers
    out["easy_days"] = EASY_DAYS.get(out["status"], 0)
    if stale:
        out["notes"].append("no recent Garmin sleep/HRV/RHR - adapting on training load only "
                            "(run `goodpush sync garmin`)")
    return out


NOT_RECOVERING_DAYS = 2     # days kept easy when a marker's trend isn't coming back (RecoveryTrend)
RECOVER_ON_DAYS = 3         # ... and when that is so in a recovery week: the recovery goes on this long


def sustained(daily: pd.DataFrame, today: pd.Timestamp, ready: dict, recovery_week: bool = False) -> dict:
    """`ready` (today's readiness) with the last 7 mornings ("mornings": date, status, reasons; the earlier ones
    from Garmin + load only) and tomorrow kept easy too when an amber morning ends a run of STREAK of them not
    green: a decline, not one bad night. Checked every morning, so the first green morning ends it. (4 of 7 is
    rare enough to mean something; 3 of 7 came up about every other week.)
    Also the trend (RecoveryTrend): a marker that isn't coming back makes the day at least amber and keeps
    NOT_RECOVERING_DAYS easy - intensity waits, easy riding goes on. recovery_week (today is in one, or it ended
    yesterday): the recovery week is the test of a block, so then the recovery goes on for RECOVER_ON_DAYS before
    the next block's hard sessions."""
    ready = recovering(daily, today, ready, recovery_week)
    past = [(today - pd.Timedelta(days=k), readiness(daily, today - pd.Timedelta(days=k))) for k in range(6, 0, -1)]
    mornings = [{"date": str(d.date()), "status": r["status"], "reasons": r["reasons"]} for d, r in past] + \
        [{"date": str(today.date()), "status": ready.get("status", "green"), "reasons": ready.get("reasons", [])}]
    ready = {**ready, "mornings": mornings}
    n = sum(m["status"] != "green" for m in mornings)
    if ready.get("status") != "amber" or n < STREAK:
        return ready
    return {**ready, "easy_days": max(ready.get("easy_days", 0), 2), "streak": n,
            "reasons": ready["reasons"] + [f"{n} of the last 7 mornings not green: easy today and tomorrow"]}


def recovering(daily: pd.DataFrame, today: pd.Timestamp, ready: dict, recovery_week: bool = False) -> dict:
    """`ready` with the recovery trend applied (see `sustained`); "trend": the markers not coming back."""
    if ready.get("data_age_days") is None or ready["data_age_days"] > 2:
        return ready
    trend = RecoveryTrend(daily.loc[:today], today)
    notes = ready.get("notes", []) + (["HRV up and resting HR down for a week while you're loaded: fine if your power "
                                       "is there - with power down it can mean deep fatigue"] if trend.held_down() else [])
    off = trend.off()
    if not off:
        return {**ready, "notes": notes, "trend": []}
    days = RECOVER_ON_DAYS if recovery_week else NOT_RECOVERING_DAYS
    why = ("not recovering yet, so the recovery week goes on: " if recovery_week else "not recovering: ") + "; ".join(off)         + f" - easy riding only for {days} days"
    return {**ready, "notes": notes, "trend": off, "status": "red" if ready.get("status") == "red" else "amber",
            "reasons": ready.get("reasons", []) + [why], "easy_days": max(ready.get("easy_days", 0), days)}


def bedtime_hours(bed: pd.Series) -> pd.Series:
    """Garmin bedtime as hours on one evening scale: after midnight counts as 24 + (00:30 -> 24.5)."""
    return bed.where(bed >= 12, bed + 24)


def late_bedtimes(d: pd.DataFrame, today: pd.Timestamp) -> int:
    """Minutes later than usual (median of the last BED_BASE_DAYS nights) if the last two nights both
    were at least LATE_BED_H late, else 0."""
    if "bedtime_h" not in d:
        return 0
    bed = bedtime_hours(d["bedtime_h"].dropna())
    if len(bed) < 10 or (today - bed.index[-1]).days > 1 or (bed.index[-1] - bed.index[-2]).days != 1:
        return 0
    usual = bed.iloc[:-2].iloc[-BED_BASE_DAYS:].median()
    late = bed.iloc[-2:] - usual
    return int(round(late.min() * 60)) if (late >= LATE_BED_H).all() else 0
