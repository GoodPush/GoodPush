"""The plan's memory, data/processed/plan_state.json: interval levels, what was planned each day and
how it went, the accepted plan, your own changes, open proposals.

Keys of a day: "2026-10-23" for its main session, "2026-10-23#2", "#3"... for sessions you added.
    planned[key]      what was planned (type, title, minutes, level) + status / ride / compliance
    accepted[key]     the full session you have (kept 30 days) - the plan only changes it via a proposal
    manual_plan[key]  sessions you moved / added / resized - kept exactly as set
"""
from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .decay import DECAYS, LevelDecay
from .steps import flat, items_from_dicts
from .workouts import EASY, LABELS, LADDERS, LEVELS, TESTS, Z2, Adjuster, Workout, WorkoutLibrary

REST = WorkoutLibrary.rest(pd.Timestamp("2000-01-01")).to_dict()
PROPOSE_DAYS = 14          # changes in the next 2 weeks wait for you; further out the plan is a draft
KEEP_DAYS = 30
# score result -> status, and how far it moves the ladder
STATUS_OF = {"hit": "done", "done": "done", "partial": "partial", "over": "partial", "low": "low",
             "short": "short", "other": "substituted", "easy": "missed", "none": "missed"}
LADDER_STEP = {"done": 1, "low": -1}
RANK = {"recovery": 0, "endurance": 1, "long_endurance": 1, "tempo": 2, "sweetspot": 3,
        "threshold": 4, "vo2max": 5, "anaerobic": 5, "race": 5, "test": 5}


class PlanState(dict):
    """plan_state.json as a dict, with the operations on it. Use `PlanState.open(path)` to read,
    change and write it as one step (the app may tick a session off while the plan rebuilds)."""

    LOCK = threading.RLock()
    # "ladders": the version of LADDERS the levels count in. 2 (2026-11-17): two easier first levels for tempo,
    # sweet spot and threshold, so a file of version 1 moves those levels up by 2 (the same sessions)
    LADDERS_VERSION = 2
    SHIFTED = {2: {"tempo": 2, "sweetspot": 2, "threshold": 2}}      # version: {type: levels added below}

    def __init__(self, data: dict | None = None, path: Path | None = None):
        super().__init__({"levels": {k: 0 for k in LADDERS}, "ladders": self.LADDERS_VERSION, "planned": {},
                          **(data or {})})
        self.upgrade_ladders()
        self.path = path

    def upgrade_ladders(self) -> None:
        """Levels of an older LADDERS in today's: the levels, and the level of every session kept with one."""
        for v in range(int(self["ladders"]) + 1, self.LADDERS_VERSION + 1):
            for kind, n in self.SHIFTED[v].items():
                self["levels"][kind] = self["levels"].get(kind, 0) + n
            for part in ("planned", "accepted", "manual_plan"):
                for p in (self.get(part) or {}).values():
                    if p.get("type") in self.SHIFTED[v] and p.get("level") is not None:
                        p["level"] += self.SHIFTED[v][p["type"]]
        self["ladders"] = self.LADDERS_VERSION

    @classmethod
    def load(cls, path: Path) -> PlanState:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls({"ladders": 1, **data} if "levels" in data else data, path)   # a file before "ladders"
        except (OSError, ValueError):
            return cls(None, path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # LF line ends on every computer (Windows would write CRLF): synced, the same plan state is the same bytes
        self.path.write_text(json.dumps(self, indent=1, default=str), encoding="utf-8", newline="\n")

    @classmethod
    @contextmanager
    def open(cls, path: Path):
        with cls.LOCK:
            st = cls.load(path)
            yield st
            st.save()

    # --- sessions ------------------------------------------------------------------------
    def current(self, key: str) -> dict | None:
        """The session under a key as you have it: your change, else the accepted one, else rebuilt
        from its planned record."""
        d = self.get("manual_plan", {}).get(key) or self.get("accepted", {}).get(key)
        p = self["planned"].get(key)
        if d is None and p and p.get("type") not in (None, "rest"):
            d = WorkoutLibrary.rebuild(pd.Timestamp(key[:10]), p).to_dict()
        return d

    def workout(self, key: str) -> Workout:
        """The session as it was planned (for scoring a ride against it)."""
        d = self.current(key)
        if d and d["type"] != "rest":
            return Workout.from_dict(key, d)
        return WorkoutLibrary.rebuild(pd.Timestamp(key[:10]), self["planned"].get(key, {}))

    def record(self, workouts: list[Workout], today: pd.Timestamp | None = None) -> None:
        """Remember what is planned (keeping what you ticked off or linked). With `today` the workouts are the
        whole plan: days from today on that it no longer has (a deleted phase, a shorter season) are forgotten,
        unless you did, linked or changed them."""
        if today is not None:
            keep = {w.key or str(w.date.date()) for w in workouts}
            td = str(today.date())
            self["planned"] = {k: p for k, p in self["planned"].items()
                               if k[:10] < td or k in keep or p.get("status") or p.get("ride") or p.get("manual")}
        for w in workouts:
            key = w.key or str(w.date.date())
            old = self["planned"].get(key, {})
            new = {"type": w.type, "title": w.title, "trimmed": bool(w.adjusted and "trimmed" in w.title),
                   "level": w.level, "minutes": w.minutes, "tss": round(w.tss()),
                   **({"offset": w.offset} if w.offset else {}),      # your easier / harder: a rebuild keeps it
                   **({"format": w.format} if w.format else {})}
            if old.get("manual") and old.get("type") == w.type:
                new.update({k: old[k] for k in ("status", "manual", "ride", "compliance", "step") if k in old})
            self["planned"][key] = new

    # --- interval levels -----------------------------------------------------------------
    @staticmethod
    def ladder_of(p: dict) -> str | None:
        """the ladder a planned session moves: its type, or "durability" for a long ride with that finish."""
        return "durability" if p.get("type") == "long" and p.get("level") is not None else p.get("type")

    def move_ladder(self, p: dict, step: int, day: str | None = None) -> None:
        """Move the session's ladder and remember by how much (`step`), so it can be undone exactly. A session
        on `day` in a steady format behind its type ("sustained_lag") on target makes up a step of that instead."""
        kind = self.ladder_of(p)
        if not step or kind not in LADDERS or p.get("trimmed"):
            return
        if step > 0 and day and self.lags().get(kind) and \
                WorkoutLibrary.format_of(pd.Timestamp(day[:10]), kind) == LevelDecay.steady(kind):
            self.set_lag(kind, self.lags()[kind] - 1)
            p["lag_step"] = p.get("lag_step", 0) + 1
            return
        lv = self["levels"].get(kind, 0)
        new = int(min(max(lv + step, 0), LEVELS[kind] - 1))
        self["levels"][kind] = new
        p["step"] = p.get("step", 0) + new - lv

    def set_level(self, kind: str, level: int) -> None:
        """Your own interval level for a type (0-based)."""
        if kind not in LADDERS:
            raise ValueError(f"no interval type '{kind}'")
        self["levels"][kind] = int(min(max(int(level), 0), LEVELS[kind] - 1))

    # --- easier / harder per format (workouts.Adjuster) ------------------------------------------
    # "format_offsets" {type: {format: steps}} (> 0 harder): your choice, kept until you change it, on top of the
    # type's level (which keeps its meaning). Only formats you moved are in it; a file without it is all zero.
    # Peer sync merges it entry by entry like any section.
    def offsets(self) -> dict:
        return self.get("format_offsets") or {}

    def set_offset(self, kind: str, fmt: str, steps: int) -> None:
        if fmt not in LADDERS.get(kind, {}):
            raise ValueError(f"no format '{fmt}' of '{kind}'")
        per = self.setdefault("format_offsets", {}).setdefault(kind, {})
        per[fmt] = int(steps)
        if not per[fmt]:
            per.pop(fmt)
        if not per:
            self["format_offsets"].pop(kind)

    def adjustment(self, key: str) -> dict | None:
        """For an interval session of the plan's (also one you moved or copied, not one you built): its format,
        your offset as it applies to this session, and whether it can go a step easier / harder. None for any
        other session."""
        r = self.reach(key)
        return None if r is None else self.position(r, r[4])

    def reach(self, key: str) -> tuple | None:
        """(session, format, steps easier, steps harder, your offset) of an interval session easier / harder
        works on (`adjustment`); None for any other."""
        d = self.current(key) or {}
        kind, level = d.get("type"), d.get("level")
        if kind not in LADDERS or level is None:
            return None
        date = pd.Timestamp(key[:10])
        fmt = WorkoutLibrary.format_in(date, d)
        easier, harder = Adjuster.of(kind, fmt, level).reach(WorkoutLibrary.intervals(date, kind, level, fmt))
        return d, fmt, easier, harder, self.offsets().get(kind, {}).get(fmt, 0)

    @staticmethod
    def position(r: tuple, offset: int) -> dict:
        """`offset` within the reach `r` (as far as it goes): the format, the offset and which way is still open."""
        _, fmt, easier, harder, _ = r
        now = int(min(max(offset, -easier), harder))
        return {"format": fmt, "offset": now, "easier": now > -easier, "harder": now < harder}

    def adjust(self, key: str, step: int) -> dict:
        """`step` steps easier (< 0) or harder (> 0) for the format of the session under `key`, from now on (every
        session in that format; one you moved or copied is rebuilt in place, `readjust`). Raises ValueError when
        it is as easy / hard as it gets."""
        r = self.reach(key)
        if r is None:
            raise ValueError("only the plan's interval sessions can be made easier or harder here")
        a, step = self.adjustment(key), int(step)
        new = self.position(r, a["offset"] + step)
        if not step or new["offset"] == a["offset"]:
            raise ValueError(f"this is as {'hard' if step > 0 else 'easy'} as {a['format']} gets")
        self.set_offset(r[0]["type"], a["format"], new["offset"])
        if key in self.get("manual_plan", {}):
            self.readjust(key)
        return self.adjustment(key)

    def preview(self, key: str, step: int) -> tuple[Workout, dict]:
        """The session under `key` as `step` more steps easier / harder would make it, and its `adjustment` then -
        nothing saved (the session window shows it; closing the window saves it with `adjust`)."""
        r = self.reach(key)
        if r is None:
            raise ValueError("only the plan's interval sessions can be made easier or harder here")
        a = self.position(r, self.position(r, r[4])["offset"] + int(step))
        return self.adjusted(key, a["offset"]), a

    def adjusted(self, key: str, offset: int) -> Workout:
        """The interval session under `key` rebuilt at `offset` steps of its format: its length stays."""
        d = self.current(key)
        fmt = WorkoutLibrary.format_in(pd.Timestamp(key[:10]), d)
        w = WorkoutLibrary.intervals(pd.Timestamp(key[:10]), d["type"], d["level"], fmt,
                                     offsets={d["type"]: {fmt: offset}})
        try:
            return w.resized(d["minutes"]) if abs(w.minutes - d["minutes"]) >= 5 else w
        except ValueError:
            return w

    def readjust(self, key: str) -> None:
        """Your copy of an interval session (moved / copied) rebuilt at its format's offset: its length, day, lock
        and why stay; the plan's version (accepted) follows."""
        mine = self["manual_plan"][key]
        w = self.adjusted(key, self.adjustment(key)["offset"])
        new = {**mine, **{k: v for k, v in w.to_dict().items() if k in ("title", "minutes", "steps", "note")},
               "offset": w.offset, "format": w.format}
        self["manual_plan"][key] = new
        if key in self.get("accepted", {}):
            self["accepted"][key] = dict(new)

    def undo_ladder(self, p: dict) -> None:
        s, lag = p.pop("step", 0), p.pop("lag_step", 0)
        k = self.ladder_of(p)
        if s and k in LADDERS:
            self["levels"][k] = int(min(max(self["levels"].get(k, 0) - s, 0), LEVELS[k] - 1))
        if lag and k in LADDERS:
            self.set_lag(k, self.lags().get(k, 0) + lag)

    # --- levels fade without the work (planning.decay) ------------------------------------------
    # "level_clock" / "sustained_clock" {type: day}: the last day of the type's (steady) work, moved on by each
    # step it took; "sustained_lag" {type: steps}: how far the type's steady format is behind the type's level.
    def lags(self) -> dict:
        return self.get("sustained_lag") or {}

    def set_lag(self, kind: str, steps: int) -> None:
        lag = self.setdefault("sustained_lag", {})
        lag[kind] = int(min(max(steps, 0), self["levels"].get(kind, 0)))
        if not lag[kind]:
            lag.pop(kind)

    def worked(self, today: pd.Timestamp) -> tuple[dict, dict]:
        """({type: last day}, {type: last day in its steady format}) of the plan's sessions you did."""
        work, steady = {}, {}
        for ds, p in sorted(self["planned"].items()):
            kind = self.ladder_of(p)
            if kind in DECAYS and ds[:10] <= str(today.date()) and p.get("status") in ("done", "partial", "low"):
                d = pd.Timestamp(ds[:10])
                work[kind] = d
                if WorkoutLibrary.format_of(d, kind) == LevelDecay.steady(kind):
                    steady[kind] = d
        return work, steady

    def decay_levels(self, intervals: pd.DataFrame | None, today: pd.Timestamp) -> dict:
        """Levels down for the work you haven't done lately (planning.decay.LevelDecay; `intervals`: the
        IntervalLog table). Returns {type: (levels lost, steps its steady format fell behind)} where either moved."""
        rides, ride_steady = LevelDecay.last_work(intervals, today)
        plan, plan_steady = self.worked(today)
        clock, held = self.setdefault("level_clock", {}), self.setdefault("sustained_clock", {})
        out = {}
        for kind in DECAYS:
            n = LevelDecay.tick(clock, kind, (rides.get(kind), plan.get(kind)), today)
            lv = self["levels"].get(kind, 0)
            self["levels"][kind] = max(lv - n, 0)
            behind = 0
            if LevelDecay.steady(kind):
                m = LevelDecay.tick(held, kind, (ride_steady.get(kind), plan_steady.get(kind)), today)
                lag = self.lags().get(kind, 0)
                self.set_lag(kind, lag + max(m - n, 0))
                behind = self.lags().get(kind, 0) - lag
            if lv != self["levels"][kind] or behind:
                out[kind] = (lv - self["levels"][kind], behind)
        return out

    def note_test(self, kind: str, key: str) -> None:
        """the day of your last FTP / ramp test (a session's key may be "day#n")."""
        if kind in TESTS:
            self["last_test" if kind == "test" else "last_ramp"] = key[:10]

    def update_progress(self, acts: pd.DataFrame, today: pd.Timestamp) -> list:
        """Statuses of past sessions from their scores, and the ladders: on target -> next step; partly
        or above target -> same step; fell short -> one back; something else or missed -> unchanged.
        Without a score (no power) the classified workout type decides done / missed."""
        log = []
        rides = acts[acts["sport"].eq("ride")] if "sport" in acts else acts
        for ds, p in sorted(self["planned"].items()):
            d, kind = pd.Timestamp(ds[:10]), p["type"]
            comp = p.get("compliance") or {}
            if p.get("manual") and p.get("ride") and p.get("status") == "done" and "step" not in p:
                p["status"] = "linked"           # linked before rides were scored: it had moved +1
                p["step"] = 1 if self.ladder_of(p) in LADDERS and not p.get("trimmed") else 0
            if p.get("status") == "linked":      # a ride you linked: its score decides (today too)
                if comp.get("result") and d <= today:
                    self.undo_ladder(p)
                    p["status"] = STATUS_OF[comp["result"]]
                    self.move_ladder(p, LADDER_STEP.get(p["status"], 0), ds)
                    if p["status"] == "done":
                        self.note_test(kind, ds)
                    log.append((ds, kind, p["status"]))
                continue
            recheck = p.get("status") == "n/a" and kind in EASY and comp.get("result")
            if d >= today or (p.get("status") and not recheck):
                continue
            if kind in EASY and comp.get("result"):
                p["status"] = STATUS_OF[comp["result"]]
            elif kind not in LADDERS and kind not in TESTS and not (kind == "lsct" and comp.get("result")):
                p["status"] = "n/a"
                continue
            elif comp.get("result"):
                p["status"] = STATUS_OF[comp["result"]]
            else:
                near = rides[(rides["date"] >= d - pd.Timedelta(days=1)) & (rides["date"] <= d + pd.Timedelta(days=1))]
                if kind == "ramp":               # found by the ramp detector (alpha.py)
                    done = any(near.get(c, pd.Series(dtype=float)).notna().any() for c in ("ramp_lt1_w", "ramp_top_w"))
                else:
                    wk = near.get("workout", pd.Series(dtype=str)).fillna("")
                    done = any(RANK.get(w, 0) >= RANK.get(kind, 0) - (kind == "vo2max") for w in wk)
                p["status"] = "done" if done else "missed"
            if kind not in EASY or self.ladder_of(p) == "durability":
                if p["status"] == "done":
                    self.note_test(kind, ds)
                self.move_ladder(p, LADDER_STEP.get(p["status"], 0), ds)
            log.append((ds, kind, p["status"]))
        return log

    def level_reason(self, kind: str) -> str:
        """Why a session of this type moved up or down: the last scored one."""
        sc = [(ds, p) for ds, p in sorted(self["planned"].items())
              if p.get("type") == kind and p.get("status") in ("done", "partial", "low")]
        if not sc:
            return "interval progression"
        ds, p = sc[-1]
        c, d, lab = p.get("compliance") or {}, pd.Timestamp(ds[:10]), LABELS.get(kind, kind)
        pct = f" ({round(c['ratio'] * 100)} % on target)" if c.get("ratio") is not None else ""
        if c.get("result") == "over":
            return f"{d:%a}'s {lab} was ridden above the targets: same step again, held steadier"
        return {"done": f"{d:%a}'s {lab} was on target{pct}: next step",
                "partial": f"{d:%a}'s {lab} was partly on target{pct}: same step again",
                "low": f"{d:%a}'s {lab} fell short{pct}: one step back"}[p["status"]]

    # --- what you do on the Calendar -----------------------------------------------------
    def mark(self, key: str, status: str, ride: str | None = None) -> dict:
        """done + ride (dragged onto it): that ride did it; its score sets the status and ladder.
        done without a ride: counts as done. skipped: missed. "": back to automatic."""
        if status not in ("done", "skipped", ""):
            raise ValueError("status must be done, skipped or empty")
        p = self["planned"].get(key)
        if p is None:
            raise KeyError(f"nothing planned on {key}")
        self.undo_ladder(p)
        for k in ("status", "manual", "ride", "compliance"):
            p.pop(k, None)
        if status == "":
            return p
        p["manual"] = True
        if ride and status == "done":
            p["ride"], p["status"] = str(ride), "linked"
            if str(ride) in p.get("not_rides", []):        # you linked it after all
                p["not_rides"] = [r for r in p["not_rides"] if r != str(ride)]
            return p
        p["status"] = status
        if status == "done":
            self.move_ladder(p, 1, key)
            self.note_test(p.get("type"), key)
        return p

    def unlink(self, key: str) -> dict:
        """Take the ride that counts for this session off it - one you linked or one matched automatically -
        and never match that ride to it again (`not_rides`): the session is then scored like one without it
        (another ride that day, else missed). Unlike mark(key, ""), which goes back to automatic matching."""
        p = self["planned"].get(key)
        if p is None:
            raise KeyError(f"nothing planned on {key}")
        rid = p.get("ride") or (p.get("compliance") or {}).get("ride")
        self.undo_ladder(p)
        for k in ("status", "manual", "ride", "compliance"):
            p.pop(k, None)
        if rid:
            p["not_rides"] = sorted(set(p.get("not_rides", [])) | {str(rid)})
        return p

    def copy(self, src: str, dst_day: str, today: pd.Timestamp, why: str = "") -> str:
        """Add a copy of session `src` to `dst_day` (any day), next to what's there (the same day too: a
        second session). Returns the new key."""
        a = self.current(src)
        if not a or a["type"] == "rest":
            raise KeyError(f"no session on {src}")
        return self.place(self.free_key(dst_day), {**a, "why_change": why or f"copied from {pd.Timestamp(src[:10]):%a %d %b}"},
                          today)

    def add(self, d: dict, day: str, today: pd.Timestamp, key: str | None = None) -> str:
        """Your own workout `d` (type, title, steps; planning.builder) on `day` next to what's there, or in place of
        session `key`. Returns its key."""
        items = items_from_dicts(d.get("steps") or [])
        if not items:
            raise ValueError("an empty workout")
        if key is not None and key[:10] != day:
            raise ValueError("a session is replaced on its own day")
        w = {"type": d["type"], "title": d.get("title") or "My workout", "level": None,
             "minutes": round(sum(s.minutes for s in flat(items)), 1), "steps": [x.to_dict() for x in items],
             "note": d.get("note", ""), "why_change": "your workout"}
        return self.place(key or self.free_key(day), w, today)

    def place(self, key: str, d: dict, today: pd.Timestamp) -> str:
        """session `d` (as saved) under `key`: yours, not locked (the plan starts from it; `set_lock` keeps it as set).
        A future main session is also the accepted one, so the plan keeping it is no change to propose."""
        locked = (self.get("manual_plan", {}).get(key) or {}).get("locked")      # the target's lock, not the copy's
        d = {k: v for k, v in d.items() if k not in ("locked", "pinned")}
        self.setdefault("manual_plan", {})[key] = {**d, **({"locked": True} if locked else {})}
        if key[:10] < str(today.date()):          # the plan only writes today on: record it, scored like any past day
            self["planned"].pop(key, None)
            self.record([Workout.from_dict(key, d)])
        elif "#" not in key:
            self.setdefault("accepted", {})[key] = dict(self["manual_plan"][key])
        self.forget_proposals({key})
        return key

    def landing(self, day: str, today: pd.Timestamp) -> str:
        """Where a session moved to `day` goes: in place of the plan's own session there (today on, nothing of yours
        and no ride on it: the planner re-plans what it had), else next to what's there (`free_key`)."""
        p = self["planned"].get(day, {})
        if (day >= str(today.date()) and day not in self.get("manual_plan", {})
                and not (p.get("ride") or p.get("status") or (p.get("compliance") or {}).get("ride"))):
            return day
        return self.free_key(day)

    def move(self, src: str, dst_day: str, today: pd.Timestamp, ready: dict | None = None) -> str:
        """Session `src` to `dst_day` (any day; `landing`). A future source day goes back to the planner (no rest
        day of yours is left there); a past one is marked "moved" (not missed, no makeup). A session with a ride
        stays put. An unlocked hard session doesn't go onto a day a red morning (`ready`) keeps easy: the plan
        would replace it there, so it stays where it is. Returns the new key."""
        if src[:10] == dst_day:
            raise ValueError("that's the same day")
        p = self["planned"].get(src, {})
        if p.get("ride") or (p.get("compliance") or {}).get("ride"):
            raise ValueError("unlink its ride first")
        a = self.current(src)
        if not a or a["type"] == "rest":
            raise KeyError(f"no session on {src}")
        locked = (self.get("manual_plan", {}).get(src) or {}).get("locked")
        dst = self.landing(dst_day, today)
        red_days = (ready or {}).get("easy_days", 0) if (ready or {}).get("status") == "red" else 0
        if not locked and "#" not in dst and a["type"] not in EASY \
                and 0 <= (pd.Timestamp(dst_day) - today).days < red_days:
            raise ValueError(f"readiness is red, so {pd.Timestamp(dst_day):%a %d %b} stays easy and the plan would "
                             "replace this session there - lock it first to move it anyway")
        key = self.place(dst, {**a, "why_change": f"moved from {pd.Timestamp(src[:10]):%a %d %b}"}, today)
        if key[:10] >= str(today.date()):
            self.set_lock(key, bool(locked), today)     # the lock moves with it
        if src[:10] >= str(today.date()):
            self.forget_own([src])
            self.get("accepted", {}).pop(src, None)     # else the plan would keep the session there too
        else:
            a = self["manual_plan"][key]
            p = self["planned"].setdefault(src, {"type": a["type"], "title": a.get("title", "")})
            p.update(status="moved", moved_to=dst_day)
            p.pop("compliance", None)
        self.forget_proposals({src, key})
        return key

    MERGEABLE = tuple(EASY)          # easy rides merge (recovery, endurance, long); an interval session doesn't

    def merge(self, src: str, dst: str, today: pd.Timestamp) -> str:
        """Session `src` and session `dst` (two easy rides on different days) as one longer ride on `dst`'s day: their
        time together, an endurance ride (long if either was). `src`'s day is off: a future one becomes a rest day of
        yours, a past one is marked "moved". Both days from today on are locked (unlock to let the plan change them).
        Returns the merged session's key."""
        if src[:10] == dst[:10]:
            raise ValueError("that's the same day")
        a, b = self.current(src), self.current(dst)
        if not a or not b or a["type"] not in self.MERGEABLE or b["type"] not in self.MERGEABLE:
            raise ValueError("only two easy rides merge")
        for k in (src, dst):
            p = self["planned"].get(k, {})
            if p.get("ride") or (p.get("compliance") or {}).get("ride"):
                raise ValueError("unlink its ride first")
        kind = "long" if "long" in (a["type"], b["type"]) else "endurance"
        w = WorkoutLibrary.easy(pd.Timestamp(dst[:10]), kind, float(a["minutes"]) + float(b["minutes"])).to_dict()
        key = self.place(dst, {**w, "why_change": f"merged with {pd.Timestamp(src[:10]):%a %d %b}'s {a['title']}"},
                         today)
        if key[:10] >= str(today.date()):     # locked: a re-plan would size an unlocked easy ride back
            self.set_lock(key, True, today)
        if src[:10] >= str(today.date()):
            self.remove(src, f"merged into {pd.Timestamp(dst[:10]):%a %d %b}'s ride")
            self.set_lock(src, True, today)
        else:
            p = self["planned"].setdefault(src, {"type": a["type"], "title": a.get("title", "")})
            p.update(status="moved", moved_to=dst[:10])
            p.pop("compliance", None)
        self.forget_proposals({src, key})
        return key

    def delete(self, key: str, today: pd.Timestamp) -> None:
        if key[:10] < str(today.date()):
            raise ValueError("past sessions can't be deleted - mark them skipped")
        if not self.current(key) and "#" not in key:
            raise KeyError(f"no session on {key}")
        self.remove(key, "removed by you")
        self.forget_proposals({key})

    def edit(self, key: str, minutes: float | None = None, reset: bool = False) -> dict | None:
        """Change a session's length, or (reset) go back to the plan's version."""
        mp = self.setdefault("manual_plan", {})
        self.forget_proposals({key})
        if reset:
            self.get("accepted", {}).pop(key, None)
            return mp.pop(key, None) if "#" not in key else None
        d = self.current(key)
        if not d or d["type"] == "rest":
            raise KeyError(f"no session on {key}")
        mp[key] = {**Workout.from_dict(key, d).resized(minutes).to_dict(),
                   **({"locked": True} if (mp.get(key) or {}).get("locked") else {})}
        if "#" not in key:
            self.setdefault("accepted", {})[key] = dict(mp[key])
        return mp[key]

    def set_lock(self, key: str, locked: bool, today: pd.Timestamp) -> None:
        """Lock a day's session (the plan keeps it exactly as it is) or unlock it (the plan starts from it but may
        change it). Locking a session you never changed pins the plan's version; unlocking that lets it go again."""
        if key[:10] < str(today.date()):
            raise ValueError("past sessions are kept anyway")
        mp = self.setdefault("manual_plan", {})
        if locked:
            d = mp.get(key) or {**(self.current(key) or REST), "pinned": True}
            mp[key] = {**d, "locked": True}
        elif key in mp:
            if mp[key].get("pinned"):
                mp.pop(key)
            else:
                mp[key].pop("locked", None)
        self.forget_proposals({key})

    def clear_own(self, today: pd.Timestamp) -> list[str]:
        """Every change of yours from today on that isn't locked goes back to the planner. Returns their keys."""
        td = str(today.date())
        keys = [k for k, d in self.get("manual_plan", {}).items() if k[:10] >= td and not d.get("locked")]
        self.forget_own(keys)
        return sorted(keys)

    def free_key(self, day: str) -> str:
        """The day itself when nothing (or a rest day) is on it, else the next "#n"."""
        if (self.current(day) or REST)["type"] == "rest":
            return day
        taken = set(self.get("manual_plan", {})) | set(self["planned"])
        n = 2
        while f"{day}#{n}" in taken:
            n += 1
        return f"{day}#{n}"

    def remove(self, key: str, why: str) -> None:
        """Take a session off its day: an added one goes, a main one becomes a rest day of yours (locked if the
        day was)."""
        mp = self.setdefault("manual_plan", {})
        if "#" in key:
            mp.pop(key, None)
            self["planned"].pop(key, None)
        else:
            mp[key] = {**REST, "why_change": why, **({"locked": True} if (mp.get(key) or {}).get("locked") else {})}
            self.setdefault("accepted", {})[key] = dict(mp[key])

    def own_on(self, days) -> list[str]:
        """The keys of your own changes on these days that aren't locked (moved / added / resized sessions)."""
        ds = set(days)
        return sorted(k for k, d in self.get("manual_plan", {}).items() if k[:10] in ds and not d.get("locked"))

    def forget_own(self, keys) -> None:
        """The planner takes these days back: your own sessions there go (and the accepted copy of them, so the
        plan's version goes straight in), an added one with its planned record."""
        mp, acc = self.get("manual_plan", {}), self.get("accepted", {})
        for k in keys:
            if mp.pop(k, None) is not None:
                acc.pop(k, None)
            if "#" in k:
                self["planned"].pop(k, None)

    # --- proposals -----------------------------------------------------------------------
    def set_away(self, start: str, end: str, note: str = "", away: bool = True) -> list[str]:
        """Days you can't ride (travel, work...): start..end, with a note; away=False takes them off again.
        Re-plans from today (no proposals for what moves around them)."""
        days = [str(d.date()) for d in pd.date_range(start, end or start, freq="D")]
        if not days or len(days) > 366:
            raise ValueError("pick a start and end date (at most a year)")
        aw = self.setdefault("away", {})
        for ds in days:
            aw.pop(ds, None) if not away else aw.__setitem__(ds, str(note)[:80])
        return days

    def replan(self, today: pd.Timestamp) -> None:
        """You changed your settings or season: from today on the new plan applies directly (no
        proposals); your own changes stay."""
        td = str(today.date())
        self["accepted"] = {k: v for k, v in self.get("accepted", {}).items() if k < td}
        self["rejected"] = {k: v for k, v in self.get("rejected", {}).items() if k == "ftp"}
        self["proposal"] = None

    def forget_proposals(self, keys) -> None:
        prop = self.get("proposal") or {}
        keep = [c for c in prop.get("changes", []) if c.get("date") not in keys]
        self["proposal"] = {**prop, "changes": keep} if keep else None
        for k in keys:
            self.get("rejected", {}).pop(k, None)

    def decide(self, decision: str, keys=None) -> list:
        """Accept or reject proposed changes (all, or those for `keys`; "ftp" for the FTP one)."""
        if decision not in ("accept", "reject"):
            raise ValueError("decision must be accept or reject")
        prop = self.get("proposal") or {}
        done, keep = [], []
        for c in prop.get("changes", []):
            session = c.get("kind", "session") == "session"
            if keys is not None and (c["date"] if session else "ftp") not in keys:
                keep.append(c)
                continue
            if session and decision == "accept":
                self.setdefault("manual_plan" if c.get("own") else "accepted", {})[c["date"]] = c["session"]
                if not c.get("own") and not (self.get("manual_plan", {}).get(c["date"]) or {}).get("locked"):
                    self.get("manual_plan", {}).pop(c["date"], None)    # your unlocked version: the plan's now
            elif session:                    # the version too: a re-plan close to it isn't asked about again
                self.setdefault("rejected", {})[c["date"]] = {"sig": c["sig"], "session": c["session"]}
            elif decision == "reject":
                self.setdefault("rejected", {})["ftp"] = c["to"]
            done.append(c)
        self["proposal"] = {**prop, "changes": keep} if keep else None
        return done

    def propose_respec(self, today: pd.Timestamp) -> list:
        """Sessions you set yourself (from today on) whose Z2 stretches are above today's endurance band: a
        proposal each to bring them to it (Workout.respec). Accepting changes your session; rejecting keeps it."""
        changes = []
        for key, d in sorted(self.get("manual_plan", {}).items()):
            new = Workout.from_dict(key, d).respec() if key[:10] >= str(today.date()) and d["type"] != "rest" else None
            if new is None:
                continue
            new = {**d, "steps": [x.to_dict() for x in new.steps]}
            sig = hashlib.sha1(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:12]
            if not self.was_rejected(key, sig, new):
                changes.append({"kind": "session", "own": True, "date": key, "sig": sig, "session": new,
                                "why": f"your own session: its Z2 parts at the plan's endurance band now "
                                       f"({Z2[0]:.0%} - {Z2[1]:.0%} of FTP), so easy riding stays easy",
                                "from": {**{k: d.get(k) for k in ("type", "title", "minutes")},
                                         "tss": round(Workout.from_dict(key, d).tss())}})
        return changes

    def was_rejected(self, day: str, sig: str, new: dict) -> bool:
        """You said no to this change on `day`: to this very version, or to one it doesn't differ from materially
        (Workout.differs) - so a re-plan that only shifts a little (a new day's data) isn't asked again."""
        r = self.get("rejected", {}).get(day)
        if isinstance(r, dict):
            return r.get("sig") == sig or not Workout.differs(r["session"], new)
        return r == sig                      # an older entry: its signature only

    def accept_or_propose(self, workouts: list[Workout], today: pd.Timestamp) -> tuple[list[Workout], list]:
        """What you accepted stays; a material difference in the next PROPOSE_DAYS becomes a proposal.
        New days, days further out, small tweaks, readiness changes, locked sessions and a test that
        is no longer due (it follows your test history) go straight in.
        A rejected version isn't proposed again unless the plan's version moves materially from it."""
        acc, rej = self.setdefault("accepted", {}), self.setdefault("rejected", {})
        near = today + pd.Timedelta(days=PROPOSE_DAYS - 1)
        final, changes = [], []
        mine = self.get("manual_plan", {})
        for w in workouts:
            ds, new = str(w.date.date()), w.to_dict()
            old = acc.get(ds)
            own = ds in mine and not Workout.differs(mine[ds], new)     # your own session, kept: never proposed
            if (old is None or own or w.locked or w.adjusted or old.get("adjusted") or w.date > near
                    or w.offset != (old.get("offset") or 0)             # your easier / harder: no need to ask
                    or not Workout.differs(old, new) or (old["type"] in TESTS and w.type != old["type"])):
                acc[ds] = new
                final.append(w)
                continue
            sig = hashlib.sha1(json.dumps(new, sort_keys=True, default=str).encode()).hexdigest()[:12]
            if not self.was_rejected(ds, sig, new):
                same_kind = w.type == old["type"] and w.type in LADDERS and w.level != old.get("level")
                why = w.why_change or (self.level_reason(w.type) if same_kind else "re-planned from your recent rides")
                changes.append({"kind": "session", "date": ds, "sig": sig, "session": new, "why": why,
                                "from": {**{k: old.get(k) for k in ("type", "title", "minutes")},
                                         "tss": round(Workout.from_dict(ds, old).tss())}})
            final.append(Workout.from_dict(ds, old))
        keep = str((today - pd.Timedelta(days=KEEP_DAYS)).date())
        self["accepted"] = {k: v for k, v in acc.items() if k >= keep}
        self["manual_plan"] = {k: v for k, v in self.get("manual_plan", {}).items() if k >= keep}
        self["rejected"] = {k: v for k, v in rej.items() if k == "ftp" or k >= str(today.date())}
        return final, changes
