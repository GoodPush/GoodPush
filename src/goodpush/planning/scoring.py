"""How a planned session went: the ride that did it, scored against the session's targets.

The ride is the one you linked on the calendar, otherwise that day's ride with the most work that
no other session of the day took. Rep by rep: the session's work blocks (consecutive work steps
>= 76 % FTP: an over-under block is one) with their targets in your watts that day (the intensity map,
as the plan shows them) are matched in order to the efforts found in the ride (models.intervals; extra
efforts and missing reps allowed: the most similar lengths win). Each matched rep counts its seconds
on target (its power, smoothed within the rep over a tenth of the shortest step, 3-30 s, inside the
step's band):

    ratio   = time on target / planned work time (capped at 1)
    avg_pct = average power of the matched reps / middle of their targets
    reps    = per block: planned seconds and target, the ride's rep (start, seconds, watts), time on
              target and a verdict (hit, partial, short, low, high, missed)

result: hit (ratio >= 0.9), partial (>= 0.5), over (the work was there, above the targets), low
(tried, fell short), other (a different hard workout), easy (rode, but easy), none (no ride).
Easy days are scored on time ridden: done from 70 % of the planned minutes, else short; other when
the ride was hard. Tests are done when the ride has the effort (classifier "test" / ramp detector).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.intervals import RideIntervals
from .intensity import FRAC, IntensityMap
from .state import PlanState
from .workouts import EASY, HARD_RIDES, LADDERS, LEVELS, LSCT_STAGES, TESTS, WORK, Workout, WorkoutLibrary

BAND_LO, BAND_HI = 0.95, 1.08     # on target: 5 % under the low end .. 8 % over the high end
HIT, PARTIAL = 0.9, 0.5
EASY_DONE = 0.7
LOOKBACK_DAYS = 21
SEED_DAYS = 42                     # rides that set your starting interval levels
SCORED = (*LADDERS, *TESTS, *EASY, "lsct")
HARD_FOR_KEY = HARD_RIDES | {"sweetspot"}
SCORE_VERSION = 4                  # sessions scored by an older method are scored again (within LOOKBACK_DAYS)
SPLIT_SHARE = 0.85                 # short reps: an effort longer than SPLIT_LONGER x the rep is split into the stretches
SPLIT_LONGER = 1.6                 # at >= this x the rep's low target (2026-11-06: two 30/15 reps read as one 75-s
                                   # effort, every later rep matched one off and the last two "missed")
SHORT_REP_S = 120                  # sessions with steps this short: efforts from the raw power, even outdoors
MATCH_LENGTH = 0.5                 # a ride effort can stand for a planned rep from half to twice its length ...
MATCH_POWER = 0.8                  # ... at 80 % of the rep's low target or more


class SessionScorer:
    """Scores planned sessions from the rides' 1 Hz power."""

    @staticmethod
    def planned_blocks(workout: Workout, knots) -> list[np.ndarray]:
        """the session's work: runs of consecutive work steps, each (steps, 3) of seconds and the low / high
        target in watts (textbook %FTP through the day's map `knots`)."""
        st = workout.expanded(split_ramps=True)
        if not st:
            return []
        rows = np.array([[s.minutes * 60, s.lo, s.hi] for s in st], float)
        work = np.array([s.kind == "active" for s in st]) & (rows[:, 1] >= WORK)
        rows[:, 1:] = np.interp(rows[:, 1:], FRAC, knots)
        runs = np.flatnonzero(np.diff(np.r_[0, work.astype(int), 0])).reshape(-1, 2)
        return [rows[a:b] for a, b in runs]

    @staticmethod
    def timeline(workout: Workout, knots, reps: list[dict], seconds: int, ride_start: int = 0) -> np.ndarray:
        """The session's steps on the ride's clock: (steps, 5) of start / end second, low / high watts and
        what it is (0 easy, 1 work, 2 missed rep). Each work block starts where the ride's rep it was scored
        against starts (`reps` from score) and keeps its planned length (a short rep shows as short); the rest
        stretches between those (late starts, stops); before the first / after the last matched rep the plan's
        own clock runs. Without a matched rep that clock starts at `ride_start` (the ride's first pedal stroke: a
        recording can begin before it). A rep that starts before the last one's planned end cuts the recovery
        between them."""
        st = workout.expanded(split_ramps=True)
        if not st:
            return np.zeros((0, 5))
        rows = np.array([[s.minutes * 60, s.lo, s.hi] for s in st], float)
        work = np.array([s.kind == "active" for s in st]) & (rows[:, 1] >= WORK)
        start = np.r_[0.0, np.cumsum(rows[:, 0])[:-1]]
        runs = np.flatnonzero(np.diff(np.r_[0, work.astype(int), 0])).reshape(-1, 2)
        kind = work.astype(float)
        P, R = [], []
        for (a, b), r in zip(runs, reps):                  # blocks: tens
            if "start" in r:
                P += [start[a], start[b - 1] + rows[b - 1, 0]]
                R += [r["start"], r["start"] + P[-1] - P[-2]]
            else:
                kind[a:b] = 2
        P, R = np.array(P), np.array(R, float)
        t = np.c_[start, start + rows[:, 0]]
        if len(P):
            t = np.where(t < P[0], t - P[0] + R[0], np.where(t > P[-1], t - P[-1] + R[-1], np.interp(t, P, R)))
        else:
            t = t + ride_start
        t = np.clip(t, 0, seconds)
        out = np.c_[t, np.interp(rows[:, 1:], FRAC, knots), kind]
        return out[out[:, 1] > out[:, 0]]

    @staticmethod
    def match(sim: np.ndarray) -> list[tuple[int, int]]:
        """in-order pairs (planned, ride) with the most total similarity `sim` (planned x ride, 0 = no match);
        each side used once, either may be left over."""
        P, R = sim.shape
        V = np.zeros((P + 1, R + 1))
        for i in range(1, P + 1):                          # planned blocks: tens; each row at once
            V[i, 1:] = np.maximum.accumulate(np.maximum(V[i - 1, 1:], V[i - 1, :-1] + sim[i - 1]))
        pairs, i, j = [], P, R
        while i and j:                                     # back along the best path
            if V[i, j] == V[i - 1, j]:
                i -= 1
            elif V[i, j] == V[i, j - 1]:
                j -= 1
            else:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
        return pairs[::-1]

    @staticmethod
    def split_merged(p: np.ndarray, efforts: list[dict], threshold: float, rep_s: float) -> list[dict]:
        """efforts longer than SPLIT_LONGER x rep_s cut into their stretches at >= threshold W (3-s mean; each at least
        half a rep), so a merged pair of short reps becomes two; the others as they are."""
        out = []
        for x in efforts:                                  # tens of efforts
            if x["seconds"] <= SPLIT_LONGER * rep_s:
                out.append(x)
                continue
            seg = RideIntervals.rolling_mean(p[x["start"]:x["start"] + x["seconds"]], 3) >= threshold
            edges = np.flatnonzero(np.diff(np.r_[0, seg.astype(int), 0])).reshape(-1, 2)
            runs = [(a, b) for a, b in edges if b - a >= 0.5 * rep_s]
            if len(runs) < 2:
                out.append(x)
                continue
            out += [x | {"start": int(x["start"] + a), "seconds": int(b - a),
                         "watts": float(np.mean(p[x["start"] + a:x["start"] + b]))} for a, b in runs]
        return out

    @classmethod
    def score(cls, workout: Workout, power, knots) -> dict | None:
        """Rep by rep time on target (see the module doc); None when the session has no work steps.
        knots: the day's watts at planning.intensity.FRAC."""
        blocks = cls.planned_blocks(workout, knots)
        if not blocks or not knots[2] > 0:
            return None
        p = np.nan_to_num(np.asarray(power, float))
        shortest = min(float(b[:, 0].min()) for b in blocks)
        iv = RideIntervals(p, knots, bursty=False if shortest < SHORT_REP_S else None).efforts()
        plan_s = np.array([b[:, 0].sum() for b in blocks])
        plan_lo = np.array([b[:, 0] @ b[:, 1] for b in blocks]) / plan_s       # time-weighted over the steps
        plan_hi = np.array([b[:, 0] @ b[:, 2] for b in blocks]) / plan_s
        if shortest < SHORT_REP_S:                         # merged reps: by the block's length (an over-under block's
            iv = cls.split_merged(p, iv, SPLIT_SHARE * float(plan_lo.min()), float(plan_s.min()))   # steps are not reps)
        ride_s = np.array([x["seconds"] for x in iv], float)
        ride_w = np.array([x["watts"] for x in iv], float)
        length = np.minimum.outer(plan_s, ride_s) / np.maximum.outer(plan_s, ride_s)
        sim = np.where((length >= MATCH_LENGTH) & (ride_w[None, :] >= MATCH_POWER * plan_lo[:, None]), length, 0.0)
        pairs = dict(cls.match(sim))
        window = int(np.clip(shortest / 10, 3, 30))
        reps = []
        for i, b in enumerate(blocks):                     # planned blocks: tens
            rep = {"planned_s": int(plan_s[i]), "lo": round(float(plan_lo[i])), "hi": round(float(plan_hi[i]))}
            j = pairs.get(i)
            if j is None:
                reps.append({**rep, "on_s": 0, "result": "missed"})
                continue
            x, sec = iv[j], np.round(b[:, 0]).astype(int)
            n = int(min(x["seconds"], sec.sum()))
            seg = RideIntervals.rolling_mean(p[x["start"]:x["start"] + n], window)   # within the rep: edges count
            on = int(((seg >= BAND_LO * np.repeat(b[:, 1], sec)[:n]) & (seg <= BAND_HI * np.repeat(b[:, 2], sec)[:n])).sum())
            res = ("hit" if on >= HIT * plan_s[i] else "low" if x["watts"] < BAND_LO * plan_lo[i]
                   else "high" if x["watts"] > BAND_HI * plan_hi[i] else "short" if x["seconds"] < HIT * plan_s[i]
                   else "partial")
            reps.append({**rep, "start": x["start"], "ride_s": x["seconds"], "watts": x["watts"], "on_s": on, "result": res})
        done = [r for r in reps if "watts" in r]
        planned = float(plan_s.sum())
        mid = sum((r["lo"] + r["hi"]) / 2 * r["ride_s"] for r in done)
        return {"ratio": round(min(1.0, sum(r["on_s"] for r in reps) / planned), 2),
                "on_s": int(sum(r["on_s"] for r in reps)), "planned_s": int(planned),
                "above_s": int(sum(r["ride_s"] for r in done if r["watts"] >= BAND_LO * r["lo"])),
                "avg_pct": round(sum(r["watts"] * r["ride_s"] for r in done) / mid, 3) if mid else None,
                "reps": reps}

    @staticmethod
    def result(kind: str, ride: pd.Series, sc: dict | None, planned_min: float | None) -> dict:
        """The verdict for one session and its ride (sc: the power score, if any)."""
        wk = str(ride.get("workout") or "")
        if kind in EASY:
            mins = float(ride.get("duration_s") or 0) / 60
            q = min(1.0, mins / planned_min) if planned_min else 1.0
            res = "other" if wk in HARD_RIDES else ("done" if q >= EASY_DONE else "short")
            return {"ratio": round(q, 2), "minutes": round(mins), "result": res}
        if kind == "lsct":                                # long enough for its stages (models.lsct reads them)
            mins = float(ride.get("duration_s") or 0) / 60
            return {"minutes": round(mins), "result": "done" if mins >= sum(s[1] for s in LSCT_STAGES) else "short"}
        if kind in TESTS:
            done = wk == "test" or (kind == "ramp" and pd.notna(ride.get("ramp_top_w", np.nan)))
            return {"result": "done" if done else ("other" if wk in HARD_FOR_KEY else "easy")}
        if sc is None:                                    # no power: go by the workout type
            return {"result": "hit" if wk == kind else ("other" if wk in HARD_FOR_KEY else "easy")}
        q = sc["ratio"]
        harder = sc["above_s"] >= 0.5 * sc["planned_s"] and (sc["avg_pct"] or 0) > BAND_HI
        res = ("hit" if q >= HIT else "partial" if q >= PARTIAL else "over" if harder
               else "other" if (wk in HARD_FOR_KEY and wk != kind and q < 0.3)
               else "low" if q >= 0.15 or wk == kind or sc["above_s"] >= 0.15 * sc["planned_s"] else "easy")
        return {**sc, "result": res}

    @classmethod
    def update(cls, state: PlanState, acts: pd.DataFrame, processed, today: pd.Timestamp) -> int:
        """Score past sessions of the last 3 weeks (and today's once linked) that have no score yet or
        whose linked ride changed. Returns how many were scored."""
        from ..io import streamstore
        planned = state["planned"]
        lo, td = str((today - pd.Timedelta(days=LOOKBACK_DAYS)).date()), str(today.date())
        todo = {k: p for k, p in planned.items()
                if lo <= k[:10] and (k[:10] < td or (k[:10] == td and p.get("ride"))) and p.get("type") in SCORED
                and p.get("status") != "moved"
                and (not p.get("compliance") or p["compliance"].get("ride") != p.get("ride", p["compliance"].get("ride"))
                     or p["compliance"].get("v") != SCORE_VERSION)}
        if not todo:
            return 0
        rides = acts[acts["sport"].eq("ride")].copy() if "sport" in acts else acts.iloc[0:0]
        rides["sid"] = rides["id"].astype(str).str.replace(r"\.0$", "", regex=True) if len(rides) else []
        rides["day"] = pd.to_datetime(rides["date"]).dt.strftime("%Y-%m-%d") if len(rides) else []
        used = {str(p.get("ride") or (p.get("compliance") or {}).get("ride")) for k, p in planned.items()
                if k not in todo and (p.get("ride") or (p.get("compliance") or {}).get("ride"))}
        used |= {str(p["ride"]) for p in todo.values() if p.get("ride")}
        pick = {}
        for k in sorted(todo):                            # the day's main session chooses first
            r = cls.ride_for(todo[k], k[:10], rides, used)
            pick[k] = r
            if r is not None:
                used.add(r["sid"])
        ids = [r["sid"] for r in pick.values() if r is not None]
        streams = streamstore.read_many(processed, ids, "ride", ["power"]) if ids else {}
        hist = IntensityMap.history(processed, 0.0)               # your thresholds on each day
        for k, p in todo.items():
            r = pick[k]
            if r is None:
                p["compliance"] = {"ride": None, "result": "none", "v": SCORE_VERSION}
                continue
            z = streams.get(r["sid"])
            sc = None
            kind = PlanState.ladder_of(p)                       # a long ride's durability finish: its reps
            if kind in LADDERS and z is not None:
                knots = hist[0].row(k[:10]) if hist else IntensityMap.ftp_knots(float(r.get("ftp") or 0))
                sc = cls.score(state.workout(k), z["power"], knots)
            p["compliance"] = {"ride": r["sid"], "ride_name": str(r.get("name") or ""), "v": SCORE_VERSION,
                               "ride_workout": str(r.get("workout") or ""),
                               **cls.result(kind, r, sc, p.get("minutes"))}
        return len(todo)

    @classmethod
    def seed_levels(cls, state: PlanState, acts: pd.DataFrame, processed, today: pd.Timestamp, ftp: float) -> dict:
        """Interval levels from your rides of the last SEED_DAYS: per type the highest level of its steady
        format that one of your rides of that type would have hit at today's FTP (an old, lower FTP would
        flatter them; micro formats would credit long efforts). Only raises levels; runs once per plan state."""
        from ..io import streamstore
        state["levels_seeded"] = str(today.date())
        if "workout" not in acts or not len(acts):
            return {}
        rides = acts[acts["workout"].isin(list(LADDERS)) & (acts["date"] >= today - pd.Timedelta(days=SEED_DAYS))
                     & (acts["sport"].eq("ride") if "sport" in acts else True)]
        ids = list(rides["id"].astype(str).str.replace(r"\.0$", "", regex=True)) if len(rides) else []
        streams = streamstore.read_many(processed, ids, "ride", ["power"]) if ids else {}
        knots = IntensityMap.for_day(processed, today, ftp)[0]    # today's thresholds: an old, lower one would flatter
        found = {}
        for sid, (_, r) in zip(ids, rides.iterrows()):   # a handful of rides; the scoring is vectorised
            z, kind = streams.get(sid), r["workout"]
            if z is None or not ftp:
                continue
            steady = next(iter(LADDERS[kind]))
            for lv in range(LEVELS[kind] - 1, found.get(kind, -1), -1):
                if (cls.score(WorkoutLibrary.intervals(r["date"], kind, lv, steady, variety=False), z["power"], knots) or {})                         .get("ratio", 0) >= HIT:
                    found[kind] = lv
                    break
        changed = {k: lv for k, lv in found.items() if lv > state["levels"].get(k, 0)}
        state["levels"].update(changed)
        return changed

    @staticmethod
    def ride_for(p: dict, day: str, rides: pd.DataFrame, used: set):
        """The linked ride, else that day's ride with the most work not taken by another session and not
        one you took off this session (`not_rides`)."""
        if p.get("ride"):
            r = rides[rides["sid"].eq(str(p["ride"]))]
            if len(r):
                return r.iloc[0]
        r = rides[rides["day"].eq(day) & ~rides["sid"].isin(used) & ~rides["sid"].isin(p.get("not_rides", []))]
        if not len(r):
            return None
        return r.loc[r["work_kj"].fillna(0).idxmax()] if "work_kj" in r else r.iloc[0]

    @staticmethod
    def ftp_suggestion(state: PlanState, ftp: float, today: pd.Timestamp) -> dict | None:
        """The last 3 threshold / sweet-spot sessions on target and >= 3 % above the band middle ->
        +3 %; the last 2 threshold sessions tried but well short (<= 95 %) -> -3 %."""
        lo = str((today - pd.Timedelta(days=42)).date())
        sc = [(p["type"], p["compliance"]) for ds, p in sorted(state["planned"].items())
              if ds >= lo and p.get("type") in ("threshold", "sweetspot") and (p.get("compliance") or {}).get("avg_pct")]
        up = [c for _, c in sc[-3:]]
        if len(up) == 3 and all(c["ratio"] >= HIT and c["avg_pct"] >= 1.03 for c in up):
            pct = round((np.mean([c["avg_pct"] for c in up]) - 1) * 100)
            return {"from": round(ftp), "to": int(round(ftp * 1.03)),
                    "why": f"your last 3 threshold / sweet-spot sessions were on target and {pct} % above "
                           "the middle of the range"}
        thr = [c for t, c in sc if t == "threshold"][-2:]
        if len(thr) == 2 and all(c["result"] == "low" and c["avg_pct"] <= 0.95 for c in thr):
            return {"from": round(ftp), "to": int(round(ftp * 0.97)),
                    "why": "your last 2 threshold sessions fell well short of the targets"}
        return None
