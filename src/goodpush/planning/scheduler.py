"""Which session goes on which day.

For every day inside a phase of your season (from today to the end of the last phase):
* races on their day, openers the day before, an easy spin the day after
* key days: the phase's quality sessions (build: threshold + VO2max / sweet spot by block; base:
  sweet spot; specialty: threshold + VO2max; maintenance: one a week), climbing their
  interval progression; a 20-min test and a ramp test every few weeks (never in a race week); a recovery
  week's first key day the LSCT (submaximal test: heart rate held at set levels, then heart-rate recovery)
* the longest available day is the long ride; days with no time are rest
* each week's easy volume is sized so fitness (CTL) moves as the phase asks (recovery weeks and
  race tapers less), within your hours and each day's time
* adaptations: a missed key session is made up later that week, an unplanned hard ride moves the
  next key session, and today's readiness trims or swaps the next day or two
Your own changes and sessions you ticked off stay as they are.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .load import MAX_TSS_PER_HOUR, RECOVERY_SHARE, LoadModel, VolumeRamp
from .optimizer import DAY_MINUTES, HARDNESS, SessionOptimizer
from .readiness import EASY_DAYS
from .season import Season
from .state import PlanState
from .workouts import (AT_THRESHOLD_UP, DAYS, EASY_LIMITS, EASY_LIMITS_RECOVERY, HARD_RIDES, KEY_TYPES, LABELS, LADDERS,
                       LEVELS, MAX_EFFORT_EVERY_WEEKS, MAX_EFFORT_OF, MAX_EFFORTS, TESTS, Z2, Step, TestEfforts,
                       Workout, WorkoutLibrary)

DEFAULT_PLAN = {
    "enabled": True,
    "hours_per_week": 4.0,
    "plan_starting_hours": 0.0,       # the build-up starts from at least this week's hours (else your last 4 weeks)
    "availability": [],               # hours per weekday Mon..Sun; [] = use the day settings below
    "rest_days": ["mon"],
    "key_days": ["tue", "thu"],       # quality on these (the first gets threshold / the test)
    "long_day": "sat",                # used without availability
    "recovery_days": ["fri"],
    "min_ramp": 1.5, "max_ramp": 2.5,
    "test_every_weeks": 6,
    "ramp_every_weeks": 6,            # ramp test with alphaHRV (LT1 / LT2); 0 = never
    "test_efforts": False,            # all-out efforts (1, 3, 5 min) opening key sessions in loading weeks too
    "ftp": 0,                        # 0 = use the analysis' FTP
    "load_model": "ctl",              # weekly load: "ctl" (ramp / recovery / taper rules) or "physio" (the model)
    "intensity": "model",             # "model" (the physio model picks each key day's session), "pyramidal"
                                      # (tempo / sweet spot / threshold / VO2max) or "polarized"
    "mix_draws": 10,                  # "model": posterior draws the choice is averaged over; 0 = the best fit alone
}
RACE_WEEK_QUALITY = 1        # a week with a race: at most this many quality sessions besides it
MIX_TYPES = ("tempo", "sweetspot", "threshold", "vo2max", "anaerobic")   # what the model chooses from on a key day
BASE_HARD = 1                # a base loading week: at most this many hard sessions (time on the bike builds base)
WL = WorkoutLibrary
HOURS_LIMITED_H = 0.75      # a loading week this close to its hours (cap or your days' time) is "at your hours"
FTP_TEST_LEVEL = 2           # a 20-min test once threshold or VO2max work has reached this level (from 0: the 3rd
                             # step) - never for a new rider or after a break (levels fade); the ramp test may come first
# without availability, a day's riding is at most a fixed default (e.g. 2 h, the long-ride day 4 h) - raised to what
# your own longest recent day allows (VolumeRamp.ride_cap): all of it on the long-ride day, this share on the others.
# The defaults suit someone riding a few hours a day at most; a rider who already rides 5-6 h days (a pro) gets days
# that long, else a 20-h week couldn't fit (2026-11-21: a synthetic pro's weeks stopped at ~13 h, ~500 TSS)
OTHER_DAY_SHARE = 0.75
RECOVERY_SPIN_MAX = 90.0     # a recovery spin's most minutes (the physio load model's easy hours)
RIDE_LIMIT_STEPS = 2         # ... and those days at most this many VolumeRamp growth steps past your longest recent day
MAX_RIDE_MIN = 480.0         # ... and never past the longest ride the workout library builds (Workout.resized)


class Scheduler:
    """Builds the plan's sessions from your season, settings and what you have done."""

    def __init__(self, cfg: dict, season: Season, load: LoadModel | None = None, physio=None):
        self.cfg = {**DEFAULT_PLAN, **(cfg or {})}
        av = [float(x) for x in self.cfg.get("availability") or []]
        self.avail = av if len(av) == 7 and sum(av) > 0 else None
        self.season = season
        self.events = season.events_on()
        self.load = load or LoadModel()
        self.physio = physio                     # PhysioLoad: weekly load from the physio model, else the CTL rules
        self.physio_error = ""
        self.ramp = None                         # VolumeRamp from your recent weeks, set by `plan`
        self.away = {}                           # days you can't ride (PlanState "away": day -> note), set by `plan`
        self.offsets = {}                        # your easier / harder per format (PlanState.offsets), set by `plan`
        self.kept = {}                           # ... and the days its sessions keep (`kept_days`), set by `plan`
        self.lags = {}                           # steady formats behind their type (PlanState.lags), set by `plan`
        self.chosen = None                       # intensity "model": {day: session type} on key days, set by `plan`
        self.mix_info = None                     # ... and how the choice went (for the plan's log)

    # --- the whole plan ---------------------------------------------------------------------
    def plan(self, state: PlanState, daily: pd.DataFrame, acts: pd.DataFrame, ftp: float,
             today: pd.Timestamp, ready: dict, days: pd.DatetimeIndex | None = None) -> dict:
        """Sessions from today to the end of your season (or `days`), adapted, sized and checked
        against what you accepted. Updates `state`."""
        log = state.update_progress(acts, today)
        self.away = dict(state.get("away") or {})
        self.offsets = state.offsets()
        self.lags = state.lags()
        self.kept = self.kept_days(state, today)
        past = daily.loc[:today - pd.Timedelta(days=1)]
        ctl = float(past["ctl"].iloc[-1]) if len(past) else 0.0
        atl = float(past["atl"].iloc[-1]) if len(past) else 0.0
        whole = days is None                     # the whole season (not a window): it replaces the old plan
        days = self.horizon(today, days)
        dt = self.season.day_table(days)
        days = days[dt["phase"].to_numpy() >= 0]
        dt = dt.loc[days]
        self.ramp = VolumeRamp(daily, today, dt, float(self.cfg.get("plan_starting_hours") or 0))
        levels = dict(state["levels"])
        extras = self.extras(state, today, days)
        self.chosen = None
        need_test, need_ramp, workouts = self.template(days, dt, state, acts, today, levels, ready)
        if self.model_mix(workouts, extras, dt, levels):          # the model's sessions on the key days
            need_test, need_ramp, workouts = self.template(days, dt, state, acts, today, levels, ready)
        meta = self.week_meta(workouts, extras, dt, daily, today)
        before = daily["ctl"].loc[:meta[0]["start"] - pd.Timedelta(days=1)] if meta else daily["ctl"].iloc[:0]
        c_week0 = float(before.iloc[-1]) if len(before) else ctl
        lp = self.physio_targets(meta, workouts, extras, ctl, atl) or self.load.week_targets(meta, ctl, atl, c_week0)
        solved = bool(meta) and SessionOptimizer(self).solve(workouts, meta, lp, state, acts, today, ctl, atl, ready,
                                                             extras)
        if not solved:                           # fallback: the weekly template's own adaptations
            self.adapt(workouts, state, acts, today, levels)
        weeks, done = self.fit_weeks(workouts, extras, meta, lp, ctl, atl, c_week0, resize=not solved)
        if not solved:
            self.apply_readiness(workouts, ready)
        self.easy_before_hard(workouts)
        self.vary(workouts, meta, today, ready, levels)
        self.max_efforts(workouts, meta, today, ready, acts)
        final, changes = state.accept_or_propose(workouts, today)
        changes += state.propose_respec(today)
        self.refresh_weeks(weeks, final, extras, done, ctl, atl)
        state["proposal"] = {"created": str(today.date()), "changes": changes} if changes else None
        state.record(final + extras, today if whole else None)
        sessions = sorted(final + extras, key=lambda w: (w.date, w.key))
        proj = self.projection(sessions, ctl, atl)
        return {"sessions": sessions, "readiness": ready, "projection": proj, "levels": levels, "log": log,
                "ramp_per_week": (proj[-1][1] - ctl) / max(len(proj) / 7, 1) if proj else 0.0,
                "need_test": need_test, "need_ramp": need_ramp, "ftp": ftp, "ctl_now": ctl,
                "weeks": [w for w in weeks if w["hours_planned"] > 0 or w["tss_planned"] > 0],
                "proposal": changes, "session_mix": self.mix_info}

    def choosable(self, w: Workout) -> bool:
        """a key day's interval session the model may choose for: the template's, not yours nor kept."""
        return w.type in MIX_TYPES and not (w.locked or w.own or str(w.date.date()) in self.kept)

    def base_hard_free(self, workouts, extras, dt) -> dict:
        """{Monday: hard sessions still free} for the base loading weeks: BASE_HARD less the hard ones already set
        there (a test, a session of yours or kept, one you added)."""
        load = dt.index[(dt["type"] == "base").to_numpy() & ~dt["recovery"].to_numpy(bool)]
        free = {d - pd.Timedelta(days=d.weekday()): BASE_HARD for d in load}
        fixed = [w for w in workouts if not self.choosable(w)] + list(extras)
        for w in fixed:
            d = pd.Timestamp(w.date).normalize()
            if d in load and HARDNESS.get(w.type, 0) == 2:
                free[d - pd.Timedelta(days=d.weekday())] -= 1
        return free

    def model_mix(self, workouts, extras, dt, levels) -> bool:
        """intensity "model": each key day's interval session (the template's, not one you set, nor one in a format
        you made easier / harder: that one stays, `kept`) becomes the
        MIX_TYPES session or an endurance ride as long as the template's session (no intensity at all) - whichever
        gives the physio model's best forecast (PhysioLoad.choose_sessions); a recovery week's at the first step, else
        at your level, each fitting the day. No type is held for a phase: the model decides. A base loading week has
        BASE_HARD hard sessions (optimizer.HARDNESS 2: tempo too) at most, those already set counted (a test, yours):
        the key days after that choose only between the easier MIX_TYPES (none since tempo counts as hard) and the
        easy ride. Sets `chosen` {day: type,
        None for the easy ride}; False when not in use or the model isn't there (then the pyramidal mix)."""
        if self.cfg["intensity"] != "model" or self.physio is None:
            return False
        recovery = dt["recovery"].to_numpy(bool)
        free = self.base_hard_free(workouts, extras, dt)
        slots = {}
        for i, w in enumerate(workouts):
            if not self.choosable(w):
                continue
            monday = w.date - pd.Timedelta(days=w.date.weekday())
            kinds = MIX_TYPES
            if monday in free:                               # a base loading week: one hard session
                kinds = MIX_TYPES if free[monday] > 0 else tuple(k for k in MIX_TYPES if HARDNESS[k] < 2)
                free[monday] -= 1
            cap = self.day_max(w.date)
            cands = [WL.intervals(w.date, k, 0, offsets=self.offsets, lags=self.lags) if recovery[i]
                     else self.fitting_intervals(w.date, k, levels.get(k, 0), cap)
                     for k in sorted(kinds, key=w.type.__ne__)]          # the template's first
            slots[i] = [c for c in cands if not cap or c.minutes <= cap or c.type == w.type] + \
                [WL.easy(w.date, "endurance", round(w.minutes))]
        try:
            picked, r = self.physio.choose_sessions(workouts, extras, slots, self.events)
        except Exception as e:  # noqa: BLE001 - the pyramidal mix still makes a plan
            self.mix_info = {"error": str(e)}
            return False
        self.chosen = {str(workouts[i].date.date()): k if k in MIX_TYPES else None for i, k in picked.items()}
        counts = pd.Series(list(picked.values()), dtype=object).value_counts()
        self.mix_info = {"draws": 0 if self.physio.draws is None else len(self.physio.draws),
                         "sessions": {k: int(v) for k, v in counts.items()},
                         "gain": 0.0 if r is None else float(r.start - r.objective)}
        return True

    def week_mix(self, row, start: pd.Timestamp, fixed_hard: int) -> list[str]:
        """the quality sessions a week wants. How many is one rule, whatever picks the types: the phase's number
        (`quality`) less one per fixed hard day (test, race), a race week at most RACE_WEEK_QUALITY. Which: the
        phase's, or those the model chose for the week's key days (intensity "model")."""
        mix = SessionOptimizer.phase_mix(row, fixed_hard, self.cfg["intensity"])
        monday = start - pd.Timedelta(days=start.weekday())
        n = min(len(mix), RACE_WEEK_QUALITY) if monday in self.race_weeks() else len(mix)
        if self.chosen is not None:
            end = start + pd.Timedelta(days=7)
            mix = [k for d, k in sorted(self.chosen.items()) if k and start <= pd.Timestamp(d) < end]
        return mix[:n]

    def race_weeks(self) -> set:
        """the Mondays of weeks (Mon-Sun) with a race."""
        return {pd.Timestamp(d).normalize() - pd.Timedelta(days=pd.Timestamp(d).weekday()) for d in self.events}

    @staticmethod
    def test_extra(w: Workout, day_max: float | None, instead: Workout) -> Workout:
        """A test in a loading week, with less TSS than the session it took the place of (`instead`): Z2 after the
        effort to make up the difference, within the day's time - a last resort so the week keeps its load. A test
        with as much TSS rides as it is; the test itself (warm-up, effort) never changes."""
        short = instead.tss() - w.tss()
        if short <= 0:
            return w
        per_min = Step("Z2", 1, *Z2).mean_sq() * 100 / 60 / w.ftp_ratio ** 2
        target = w.minutes + np.ceil(short / per_min / 5) * 5
        if day_max:
            target = min(target, day_max)
        return w.resized(target) if target >= w.minutes + 10 else w

    def vary(self, workouts, meta, today, ready, levels=None) -> None:
        """Plain endurance / long rides become one of their variants (spin-ups, sprints, low cadence,
        surges; tempo finish, late sweet spot). Tempo-like ones only in loading weeks of base / build /
        specialty and never when training polarized; none on days readiness may still change or that you
        set yourself. The long ride of a loading week gets the durability finish instead (a small addition,
        not a hard day), a level up each week as if the last was done."""
        loading = {i for W in meta if W["kind"] == "load" and not W["taper"]
                   and W["row"].type in ("base", "build", "specialty") for i in W["idx"]}
        hold = (ready or {}).get("easy_days", EASY_DAYS.get((ready or {}).get("status"), 0))   # days left plain
        level = (levels or {}).get("durability", 0)
        for i, w in enumerate(workouts):
            if w.locked or w.own or w.adjusted or (w.date - today).days < hold:
                continue
            if w.type == "long" and i in loading and len(w.steps) <= 3:
                new = WL.durability(w, level, self.cfg["intensity"] == "polarized")
                if new is not w:
                    workouts[i] = new
                    level = min(level + 1, LEVELS["durability"] - 1)
                    continue
            workouts[i] = WL.varied(w, i in loading and self.cfg["intensity"] != "polarized")

    def max_efforts(self, workouts, meta, today, ready, acts) -> None:
        """Timed all-out efforts to measure, not infer (2026-11-04): at the end of build (its push week) and in
        specialty's loading weeks without a race (a race is an all-out effort itself), the week's first VO2max session
        opens with 5 minutes all-out, else its anaerobic one with 1 minute (WorkoutLibrary.max_effort) - not within
        MAX_EFFORT_EVERY_WEEKS of the last such effort your heart rate confirmed or planned (that keeps them rare),
        nor on days readiness may still change or that you set yourself. With test_efforts: every loading week
        without a race, its first session one of the lengths is due for (TestEfforts) - 1, 3 and 5 min."""
        hold = (ready or {}).get("easy_days", EASY_DAYS.get((ready or {}).get("status"), 0))
        last = self.last_max_efforts(acts)
        rotation = TestEfforts(last) if self.cfg.get("test_efforts") else None
        for W in meta:
            row = W["row"]
            due = W["push"] if row.type == "build" else (row.type == "specialty" and W["kind"] == "load"
                                                          and not W["taper"])
            if rotation is not None:
                due = W["kind"] == "load" and not W["taper"]
            if not due:
                continue
            for i in sorted(W["idx"], key=lambda i: (workouts[i].type != "vo2max", workouts[i].date)):
                w = workouts[i]
                sec = rotation.pick(w.type, w.date) if rotation is not None else MAX_EFFORT_OF.get(w.type)
                if not sec or w.locked or w.own or w.adjusted or (w.date - today).days < hold:
                    continue
                if rotation is None and last.get(sec) is not None and \
                        (w.date - last[sec]).days < 7 * MAX_EFFORT_EVERY_WEEKS:
                    continue
                new = WL.max_effort(w, sec)
                if new.minutes > min(w.minutes + 5, self.day_max(w.date) or np.inf):   # doesn't fit your time
                    continue
                workouts[i], last[sec] = new, w.date
                if rotation is not None:
                    rotation.placed(sec, w.date)
                break

    @staticmethod
    def last_max_efforts(acts: pd.DataFrame) -> dict:
        """{seconds: the day of your last effort that long your heart rate confirmed as all-out, or that you rode
        as a planned all-out step (models.all_out: all_out_s)} (MAX_EFFORTS)."""
        from ..metrics.effort_hr import EffortHeartRate
        if acts is None or not len(acts):
            return {}
        rides = acts[acts["sport"].eq("ride")] if "sport" in acts else acts
        day = pd.to_datetime(rides["date"])
        out = {s: None for s in MAX_EFFORTS}
        if "max_hr" in rides:
            hit = EffortHeartRate.maxed(rides, float(rides["max_hr"].quantile(0.995) or 190))
            out |= {s: day[hit[s]].max() for s in MAX_EFFORTS if s in hit and hit[s].any()}
        if "all_out_s" in rides:
            stepped = day[rides["all_out_s"].notna()].groupby(rides["all_out_s"].dropna().astype(int)).max()
            out |= {s: max(d for d in (out[s], stepped[s]) if d is not None) for s in MAX_EFFORTS if s in stepped}
        return out

    def physio_targets(self, meta, workouts, extras, ctl, atl) -> dict | None:
        """weekly targets from the physio model, or None (not in use, nothing to plan, or it failed)."""
        if self.physio is None or not meta or self.cfg["load_model"] != "physio":
            return None
        try:
            return self.physio.week_targets(meta, workouts, extras, self.events, self.load, ctl, atl)
        except Exception as e:  # noqa: BLE001 - the CTL rules still make a plan
            self.physio_error = str(e)
            return None

    def horizon(self, today: pd.Timestamp, days: pd.DatetimeIndex | None) -> pd.DatetimeIndex:
        if days is not None:
            return pd.DatetimeIndex(days)
        end = self.season.end
        return pd.date_range(today, end, freq="D") if end is not None and end >= today else pd.DatetimeIndex([])

    # --- day by day ---------------------------------------------------------------------------
    def template(self, days, dt, state, acts, today, levels, ready=None) -> tuple[bool, bool, list[Workout]]:
        """A session for every day from the phase's pattern; tests every few weeks; your locked days."""
        cfg = self.cfg
        last_test, last_ramp = self.last_tests(state, acts)
        need_test = last_test is None or (today - last_test).days > 7 * cfg["test_every_weeks"]
        need_ramp = bool(cfg["ramp_every_weeks"]) and (last_ramp is None or (today - last_ramp).days > 7 * cfg["ramp_every_weeks"])
        test_due = today if need_test else last_test + pd.Timedelta(weeks=cfg["test_every_weeks"], days=1)
        ramp_due = None if not cfg["ramp_every_weeks"] else (
            today if need_ramp else last_ramp + pd.Timedelta(weeks=cfg["ramp_every_weeks"], days=1))
        # once you've ridden today, a re-plan mustn't turn today's session into a test
        was_today = state["planned"].get(str(today.date()), {}).get("type")
        rode_today = bool(len(acts)) and "sport" in acts and bool((acts["date"].eq(today) & acts["sport"].eq("ride")).any())
        long_dn = DAYS[int(np.argmax(self.avail))] if self.avail else cfg["long_day"]
        ahead = dict(levels)        # each later session of a type a step up, as if the earlier were done
        test_week, out = None, []
        race_weeks = self.race_weeks()
        quality_in = {}                 # race weeks: quality sessions placed so far (at most RACE_WEEK_QUALITY)
        unready = (ready or {}).get("easy_days", EASY_DAYS.get((ready or {}).get("status"), 0))   # days a test must wait
        # tests you placed yourself (moved, copied, added) are the plan's: none of that kind in the weeks before
        # one of yours, and the next is due the usual interval after it
        own_tests = {t: sorted({pd.Timestamp(k[:10]) for k, v in state.get("manual_plan", {}).items()
                           if v.get("type") == t and k[:10] >= str(today.date())}) for t in ("test", "ramp")}
        for d, row in zip(days, dt.itertuples()):
            ds, dn = str(d.date()), DAYS[d.weekday()]
            monday = d - pd.Timedelta(days=d.weekday())
            if d in own_tests["test"]:
                test_due, test_week = d + pd.Timedelta(weeks=cfg["test_every_weeks"]), monday
            if d in own_tests["ramp"] and ramp_due is not None:
                ramp_due = d + pd.Timedelta(weeks=cfg["ramp_every_weeks"])
            test_before = any(0 <= (t - d).days < 7 * cfg["test_every_weeks"] for t in own_tests["test"])
            ramp_before = any(0 <= (t - d).days < 7 * cfg["ramp_every_weeks"] for t in own_tests["ramp"])
            day_max = self.day_max(d)
            ev = self.events.get(ds)
            race_soon = any(0 <= (pd.Timestamp(e) - d).days <= 6 for e in self.events)
            a_race_close = any(e["priority"] == "A" and 1 <= (pd.Timestamp(k) - d).days <= 2 for k, e in self.events.items())
            free = d != today or not rode_today
            ready_ok = (d - today).days >= unready
            instead = None                       # a test's day: the session it took the place of
            if ds in self.away:
                w = replace(WL.rest(d), title="Away", note=self.away[ds], locked=True)
            elif ev:
                w = WL.race(d, ev)
            elif str((d + pd.Timedelta(days=1)).date()) in self.events:
                w = WL.openers(d)
            elif str((d - pd.Timedelta(days=1)).date()) in self.events:
                w = WL.easy(d, "recovery", 45)
            elif dn in cfg["rest_days"] or day_max == 0:
                w = WL.rest(d)
            elif dn in cfg["key_days"] and row.type != "recovery" and not a_race_close:
                first = dn == cfg["key_days"][0]
                mix = self.quality(row.type, row.recovery, row.block, row.week, cfg["intensity"])
                kind = mix[0] if first and mix else (mix[1] if not first and len(mix) > 1 else None)
                kind = self.kept.get(ds) or (self.chosen or {}).get(ds, kind)   # yours kept, else the model's choice
                if monday in race_weeks and kind:                 # a race week: the race + at most one more
                    kind = kind if quality_in.get(monday, 0) < RACE_WEEK_QUALITY else None
                    quality_in[monday] = quality_in.get(monday, 0) + (kind is not None)
                if (first and d >= test_due and not race_soon and ready_ok and not test_before
                        and self.test_ready(ahead) and (free or was_today in (None, "test"))):
                    w, test_week = WL.ftp_test(d), monday
                    instead = self.test_instead(d, kind, row, ahead, day_max)
                    test_due = d + pd.Timedelta(weeks=cfg["test_every_weeks"])
                elif (ramp_due is not None and d >= ramp_due and not first and not row.recovery and not race_soon
                      and not ramp_before
                      and test_week != monday and ready_ok and (free or was_today in (None, "ramp"))):
                    w = WL.ramp_test(d)
                    instead = self.test_instead(d, kind, row, ahead, day_max)
                    ramp_due = d + pd.Timedelta(weeks=cfg["ramp_every_weeks"])
                elif row.recovery:                       # a recovery week: the LSCT on its first key day
                    w = WL.lsct(d) if first and ready_ok else WL.easy(d, "endurance", 60)
                elif kind is None:
                    w = WL.easy(d, "endurance", 60)
                else:
                    w = self.fitting_intervals(d, kind, ahead.get(kind, 0), day_max)
                    ahead[kind] = min(ahead.get(kind, 0) + 1, LEVELS[kind] - 1)
            elif ds in self.kept and not a_race_close and row.type != "recovery":   # one kept on another day
                kind = self.kept[ds]
                w = WL.intervals(d, kind, 0, offsets=self.offsets) if row.recovery else \
                    self.fitting_intervals(d, kind, ahead.get(kind, 0), day_max)
            elif dn == long_dn:
                hours = self.hours_cap(row, d)
                long_h = max(1.5, 0.3 * hours) if row.recovery else max(2.0, min(4.0, 0.33 * hours))
                w = WL.easy(d, "long", long_h * 60)
            elif dn in cfg["recovery_days"]:
                w = WL.easy(d, "recovery", 45)
            else:
                w = WL.easy(d, "endurance", 60)
            if instead and not row.recovery and not race_soon and row.type != "recovery":
                own = self.avail[d.weekday()] * 60 if self.avail else None     # (the ramp may trim it below)
                w = self.test_extra(w, own, instead)         # a test inside a training block: its load made up
            if day_max and w.minutes > day_max and w.type not in ("rest", "race"):   # the time you have
                try:
                    w = w.resized(day_max)
                except ValueError:
                    pass
            p_old = state["planned"].get(ds, {})
            if p_old.get("manual"):              # marked / linked by you: the session as you had it, kept
                had = state.current(ds)          # (the full copy: the record alone loses e.g. durability efforts)
                w = Workout.from_dict(ds, had) if had and had.get("type") == p_old.get("type") else WL.rebuild(d, p_old)
                w.locked = True
            mine = state.get("manual_plan", {}).get(ds)
            if mine:                             # you moved / changed this day: the plan starts from it, locked = kept
                w = Workout.from_dict(ds, mine)
                w.locked = bool(mine.get("locked"))
                w.own = not w.locked
            out.append(w)
        need_test = need_test and any(w.type == "test" for w in out)      # due and planned (not waiting for levels)
        return need_test, need_ramp, out

    def kept_days(self, state: PlanState, today: pd.Timestamp) -> dict[str, str]:
        """{day: type} of the plan's interval sessions from today on in a format you made easier / harder
        (PlanState "format_offsets"): each keeps its day and type, so harder can't move the format away (the
        model's choice, the optimizer and the template leave them; a red morning can still make them easy)."""
        return {k: p["type"] for k, p in state["planned"].items()
                if "#" not in k and k >= str(today.date()) and p.get("type") in LADDERS
                and (self.offsets.get(p["type"]) or {}).get(WL.format_in(pd.Timestamp(k), p))}

    def test_instead(self, d, kind: str | None, row, ahead: dict, day_max: float | None) -> Workout:
        """The session a test takes the place of on a key day: its interval session, else an hour's endurance."""
        if kind is None or row.recovery:
            return WL.easy(d, "endurance", 60)
        return self.fitting_intervals(d, kind, ahead.get(kind, 0), day_max)

    def fitting_intervals(self, d, kind: str, level: int, day_max: float | None) -> Workout:
        """The interval session at `level` (with your easier / harder for its format, and its steady format's lag),
        or the highest lower step that fits the day's time."""
        w = WL.intervals(d, kind, level, offsets=self.offsets, lags=self.lags)
        while day_max and w.minutes > day_max and w.level > 0:
            w = WL.intervals(d, kind, w.level - 1, offsets=self.offsets)
        return w

    @staticmethod
    def quality_count(phase_type: str, recovery: bool) -> int:
        """how many quality sessions a week has - one rule whatever picks their types (pyramidal, polarized, the
        model): build / specialty loading weeks 2, base and maintenance 1, a recovery week or phase none (a
        recovery week has its LSCT instead: Scheduler.template)."""
        if phase_type == "recovery" or recovery:
            return 0
        return 2 if phase_type in ("build", "specialty") else 1

    @staticmethod
    def quality(phase_type: str, recovery: bool, block: int, week: int, style: str = "pyramidal") -> list[str]:
        """A week's quality sessions, in key-day order; as many as `quality_count`.
        pyramidal: base sweet spot (one middle-zone day: tempo comes in the long ride's finish, Scheduler.vary),
        build threshold + VO2max / sweet spot (by block), specialty threshold + VO2max / anaerobic (by week),
        maintenance threshold / sweet spot.
        polarized: only above LT2 - VO2max + VO2max / anaerobic (by block; specialty: anaerobic), VO2max in
        maintenance. The rest is easy riding. ("model" starts from pyramidal.)"""
        n = Scheduler.quality_count(phase_type, recovery)
        if style == "polarized":
            mix = ["vo2max", "anaerobic" if phase_type == "specialty" or block % 2 else "vo2max"]
        elif phase_type == "build":
            mix = ["threshold", "vo2max" if block % 2 == 0 else "sweetspot"]
        elif phase_type == "base":
            mix = ["sweetspot"]
        elif phase_type == "specialty":
            mix = ["threshold", "vo2max" if week % 2 == 0 else "anaerobic"]   # race-specific punch
        else:
            mix = ["threshold" if week % 2 == 0 else "sweetspot"]
        return mix[:n]

    @staticmethod
    def test_ready(levels: dict) -> bool:
        """threshold-type work (threshold or VO2max) at FTP_TEST_LEVEL: a 20-min test is worth riding (`levels`
        as planned up to that day)."""
        return max(levels.get("threshold", 0), levels.get("vo2max", 0)) >= FTP_TEST_LEVEL

    @staticmethod
    def last_tests(state: PlanState, acts: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Latest 20-min test and ramp test: what you marked and what the rides show."""
        lt = pd.Timestamp(state["last_test"]) if state.get("last_test") else None
        lr = pd.Timestamp(state["last_ramp"]) if state.get("last_ramp") else None
        if "workout" in acts and acts["workout"].eq("test").any():
            t = acts.loc[acts["workout"].eq("test"), "date"].max()
            lt = max(lt, t) if lt is not None else t
        if "ramp_top_w" in acts and acts["ramp_top_w"].notna().any():
            r = acts.loc[acts["ramp_top_w"].notna(), "date"].max()
            lr = max(lr, r) if lr is not None else r
        return lt, lr

    def easy_before_hard(self, workouts) -> None:
        """At or above threshold (threshold, VO2max, anaerobic, a test): rest or a recovery spin the day before - a spin you may
        swap for a day off. An endurance / long ride there becomes the spin; a day you set yourself or another
        quality session stays (the optimizer doesn't plan those; this catches the fallback and moved sessions)."""
        for j in range(1, len(workouts)):
            w, prev = workouts[j], workouts[j - 1]
            if w.type not in AT_THRESHOLD_UP or (w.date - prev.date).days != 1 or prev.locked or prev.own:
                continue
            if prev.type in ("endurance", "long"):
                prev = workouts[j - 1] = replace(WL.easy(prev.date, "recovery", 45), why_change=prev.why_change)
            if prev.type == "recovery":
                prev.note = f"Optional: spin easy or take the day off before tomorrow's {LABELS.get(w.type, w.type)}."

    def ride_limit(self, d: pd.Timestamp, default: float, long: bool) -> float:
        """the most minutes of riding on `d` without availability: `default` (the setting a few-hours-a-day rider
        gets), raised to your own longest recent day (VolumeRamp.ride_cap; on other days than the long ride
        OTHER_DAY_SHARE of it); never past day_max (your time, the ramp)."""
        cap = self.ramp.ride_cap(d) if self.ramp else np.inf
        if np.isfinite(cap):          # ride_cap grows a step every loading week ahead: months out it ran to 8-h days
            cap = min(cap, 60 * self.ramp.ride_h * (1 + VolumeRamp.GROWTH) ** RIDE_LIMIT_STEPS, MAX_RIDE_MIN)
        out = max(default, cap * (1.0 if long else OTHER_DAY_SHARE)) if np.isfinite(cap) else default
        dm = self.day_max(d)
        return out if dm is None else min(out, dm)

    def day_max(self, d: pd.Timestamp) -> float | None:
        """minutes you can ride on `d`: your availability, and no more than the ramp allows (None: no limit);
        0 on a day you're away."""
        if str(pd.Timestamp(d).date()) in self.away:
            return 0.0
        caps = ([self.avail[d.weekday()] * 60] if self.avail else []) + ([self.ramp.ride_cap(d)] if self.ramp else [])
        return min(caps) if caps and min(caps) < np.inf else None

    def week_hours(self, row) -> float:
        """The week's hours: the phase's, else yours; half in a recovery phase, RECOVERY_SHARE in a recovery
        week; never more than your weekly availability."""
        h = row.hours if row.hours == row.hours else float(self.cfg["hours_per_week"]) * (0.5 if row.type == "recovery" else 1)
        h *= RECOVERY_SHARE if row.recovery else 1.0
        return min(h, sum(self.avail)) if self.avail else h

    def steady_hours(self, row, d: pd.Timestamp, kind: str, taper: bool) -> float:
        """The hours a loading week should reach (VolumeRamp.week_steady, within the week's hours); 0 for a
        recovery week or phase and a taper (the block shape and the races decide those)."""
        if kind != "load" or taper or not self.ramp:
            return 0.0
        return float(min(self.week_hours(row), self.ramp.week_steady(d)))

    def hours_cap(self, row, d: pd.Timestamp) -> float:
        """The week's hours within the volume ramp (VolumeRamp: +GROWTH a loading week on what you rode). A push
        week too (2026-11-18: safe increases first): its overload is the load target, not hours past the ramp."""
        if not self.ramp:
            return self.week_hours(row)
        return min(self.week_hours(row), self.ramp.week_cap(d))

    @staticmethod
    def extras(state: PlanState, today: pd.Timestamp, days: pd.DatetimeIndex) -> list[Workout]:
        """Sessions you added next to a day's main one (kept as you set them; locked when you locked them)."""
        if not len(days):
            return []
        lo, hi = str(today.date()), str(days[-1].date())
        out = [Workout.from_dict(k, v) for k, v in sorted(state.get("manual_plan", {}).items())
               if "#" in k and lo <= k[:10] <= hi and v["type"] != "rest"]
        for w in out:
            w.locked = bool(state["manual_plan"][w.key].get("locked"))
        return out

    # --- adapting to what you did ---------------------------------------------------------------
    @staticmethod
    def free_easy_day(workouts, first: int, last_day: pd.Timestamp, yesterday_hard: bool = False) -> int | None:
        """Index of an endurance day (not next to a key session) from `first` up to `last_day`."""
        for j in range(first, len(workouts)):
            w = workouts[j]
            if w.date > last_day:
                break
            if w.type != "endurance" or w.locked or w.own or (j == 0 and yesterday_hard):
                continue
            if not any(workouts[k].type in KEY_TYPES for k in (j - 1, j + 1) if 0 <= k < len(workouts)):
                return j
        return None

    def adapt(self, workouts, state, acts, today, levels) -> None:
        """An unplanned hard ride yesterday -> today's key session moves to the next free easy day;
        a key session missed earlier this week -> made up on a free easy day before Sunday."""
        if not workouts:
            return
        planned = state["planned"]
        rides = acts[acts["sport"].eq("ride")] if "sport" in acts and len(acts) else acts.iloc[0:0]
        rode_today = bool(len(rides)) and bool((rides["date"] == today).any())
        y = today - pd.Timedelta(days=1)
        y_hard = sorted(set(rides.loc[rides["date"] == y, "workout"].fillna("")) & HARD_RIDES) if len(rides) else []
        week_end = today + pd.Timedelta(days=6 - today.weekday())
        w0 = workouts[0]
        if (y_hard and planned.get(str(y.date()), {}).get("type") not in KEY_TYPES and w0.date == today
                and w0.type in KEY_TYPES - {"race"} and not w0.locked and not rode_today):
            j = self.free_easy_day(workouts, 2, today + pd.Timedelta(days=5))
            if j is not None:
                what, dj = LABELS.get(y_hard[0], y_hard[0]), workouts[j].date
                moved = WL.test(w0.type, dj) if w0.type in TESTS else \
                    WL.intervals(dj, w0.type, w0.level or 0, offsets=self.offsets)
                moved.why_change = f"moved from {today:%a}: yesterday's {what} wasn't in the plan"
                easy = WL.easy(today, "endurance", 60)
                easy.why_change = f"easy after yesterday's unplanned {what}; {w0.title} moved to {dj:%a}"
                workouts[0], workouts[j] = easy, moved
        week0 = today - pd.Timedelta(days=today.weekday())
        missed = [(k, p) for k, p in sorted(planned.items(), reverse=True)
                  if week0 <= pd.Timestamp(k[:10]) < today and p.get("type") in LADDERS and p.get("status") == "missed"]
        if not missed:
            return
        k, p = missed[0]                                 # one makeup a week: the latest missed
        kind = p["type"]
        later = [q for dq, q in planned.items() if k < dq < str(today.date()) and q.get("type") == kind]
        if any(w.type == kind for w in workouts if w.date <= week_end) or \
                any(q.get("status") in ("done", "partial", "low") for q in later):
            return                                       # one is coming / already made up
        j = self.free_easy_day(workouts, 1 if rode_today else 0, week_end, bool(y_hard))
        if j is not None:
            w = WL.intervals(workouts[j].date, kind, levels.get(kind, 0), offsets=self.offsets, lags=self.lags)
            w.why_change = f"makeup for {pd.Timestamp(k[:10]):%a}'s missed {LABELS.get(kind, kind)}"
            workouts[j] = w

    def apply_readiness(self, workouts, ready: dict) -> None:
        """Red: today and tomorrow easy, the next key day only if you've bounced back. Amber:
        threshold / VO2max trimmed, a test moved to the next free easy day."""
        if ready["status"] == "green" or not workouts:
            return
        red, why, pending = ready["status"] == "red", "; ".join(ready["reasons"]), None
        for j in range(min(ready.get("easy_days", EASY_DAYS[ready["status"]]), len(workouts))):
            w = workouts[j]
            if w.locked or w.type in ("rest", "race"):
                continue
            if w.own and not red:                # amber: your session is your call (SessionOptimizer.own_on_amber)
                w.adjusted = why + " - your session, kept: back off if your HR runs high"
                continue
            if red:
                if w.type in LADDERS or w.type in TESTS:
                    pending = w
                workouts[j] = WL.easy(w.date, "recovery", 30 if j == 0 else 45)
                workouts[j].adjusted = why
            elif w.type in TESTS:
                workouts[j] = WL.easy(w.date, "endurance", 60)
                workouts[j].adjusted = why + " - test moved"
                pending = w
            elif w.type in ("threshold", "vo2max", "anaerobic"):
                workouts[j] = w.trimmed(why)
            elif w.type in ("sweetspot", "tempo", "long", "endurance"):
                w.adjusted = why + " - keep it in Z2, cut short if HR drifts"
        if pending is None:
            return
        hard = set(LADDERS) | set(TESTS)
        for j in range(2, len(workouts)):
            w = workouts[j]
            if red and w.type in hard:
                if pending.type in TESTS:
                    w = workouts[j] = WL.test(pending.type, w.date)
                    w.adjusted = f"test moved from {pending.date.date()} (readiness red)"
                w.adjusted = (w.adjusted + "; " if w.adjusted else "") + \
                    "only if resting HR is back to normal and you feel fine - otherwise ride easy"
                return
            nb = [workouts[k].type for k in (j - 1, j + 1) if 0 <= k < len(workouts)]
            if not red and w.type == "endurance" and not set(nb) & hard and not w.locked:
                workouts[j] = WL.test(pending.type, w.date)
                workouts[j].adjusted = f"moved from {pending.date.date()} (readiness amber)"
                return

    # --- weekly volume --------------------------------------------------------------------------
    def week_meta(self, workouts, extras, dt, daily, today) -> list[dict]:
        """Per planned week: its days, phase row, load already done, your added sessions, hours, races,
        and the bounds the weekly load program needs."""
        wk = pd.DatetimeIndex([w.date for w in workouts]).to_period("W-SUN").start_time
        x_tss = pd.Series([x.tss() for x in extras], [x.date for x in extras], dtype=float).groupby(level=0).sum()
        x_min = pd.Series([x.minutes for x in extras], [x.date for x in extras], dtype=float).groupby(level=0).sum()
        a_weeks = {pd.Timestamp(d).to_period("W-SUN").start_time for d, e in self.events.items() if e["priority"] == "A"}
        weeks = []
        for w in wk.unique():
            idx = np.flatnonzero(wk == w)
            row = dt.loc[workouts[idx[0]].date]
            done = daily.loc[w: today - pd.Timedelta(days=1)]
            dates = [workouts[i].date for i in idx]
            ext_t, ext_m = float(x_tss.reindex(dates).sum()), float(x_min.reindex(dates).sum())
            done_t, done_h = float(done["tss"].sum()), float(done["hours"].sum())
            races = [((pd.Timestamp(d) - dates[0]).days, e["priority"]) for d, e in self.events.items()
                     if dates[0] <= pd.Timestamp(d) <= dates[-1]]
            kind = "recovery_phase" if row.type == "recovery" else "recovery" if row.recovery else "load"
            build_weeks = self.season.phases[row.phase].build_weeks
            taper = bool(races) or (w + pd.Timedelta(weeks=1)) in a_weeks
            push = kind == "load" and row.type in ("base", "build") and not taper and bool(row.push)
            cap_h = self.hours_cap(row, w)
            day_max = [self.day_max(d) for d in dates]
            ride_h = sum(day_max) / 60 if None not in day_max else cap_h       # the days' time
            weeks.append({"start": w, "idx": idx, "row": row, "done": (done_t, done_h), "extra": (ext_t, ext_m),
                          "cap_h": cap_h, "steady_h": self.steady_hours(row, w, kind, taper),
                          "day_max": day_max, "days": len(idx), "kind": kind, "races": races,
                          "easy_max": self.easy_max([workouts[i] for i in idx], kind == "load"),
                          "taper": taper, "ramp": self.load.ramp_bounds(self.cfg, row.type, row.week_in_block, build_weeks),
                          "push": push,
                          "cap": MAX_TSS_PER_HOUR * max(0.0, min(cap_h - done_h, ride_h))})
        return weeks

    def fit_weeks(self, workouts, extras, weeks, lp, ctl, atl, c_week0, resize: bool) -> tuple[list, dict]:
        """Week rows for the plan's summary; with `resize` (no optimiser) each week's easy volume is first
        sized to its target. Returns the rows and each week's done load (rides so far + added sessions)."""
        rows, done_wk, c_run, a_run, c_prev = [], {}, ctl, atl, c_week0
        for wi, W in enumerate(weeks):
            idx, row = W["idx"], W["row"]
            (done_t, done_h), (ext_t, ext_m) = W["done"], W["extra"]
            d_tss, d_h = done_t + ext_t, done_h + ext_m / 60
            done_wk[W["start"]] = (d_tss, d_h)
            planned = float(lp["tss"][wi])                    # the days still to come, extras included
            target = done_t + planned
            sub = [workouts[i] for i in idx]
            if resize:
                caps = self.easy_caps(sub, W["kind"] == "load")
                sub = self.fit_volume(sub, planned - ext_t, max(0.0, (W["cap_h"] - d_h) * 60), W["kind"] != "load", caps)
            capped = sum(s.tss() for s in sub) + d_tss < 0.95 * target
            if (resize and capped and W["kind"] == "load" and row.type in ("base", "build") and not W["taper"]
                    and self.cfg["intensity"] != "polarized"):
                blk = 15 + 5 * min(int(row.week_in_block), 2)    # out of hours: tempo in the long ride
                sub = [WL.long_tempo(s.date, s.minutes, blk)
                       if s.type == "long" and s.minutes >= 90 and not (s.locked or s.own)
                       else s for s in sub]
            for i, s in zip(idx, sub):
                workouts[i] = s
            c, a = self.load.project(self.daily_tss(sub, extras), c_run, a_run)
            c_run, a_run = c[-1], a[-1]
            plan_tss = d_tss + sum(s.tss() for s in sub)
            plan_h = d_h + sum(s.minutes for s in sub) / 60
            ramp, c_prev = float(lp["ctl"][wi]) - c_prev, float(lp["ctl"][wi])
            time_h = W["cap_h"] if None in W["day_max"] else min(W["cap_h"], sum(W["day_max"]) / 60)
            rows.append({"week": str(W["start"].date()), "phase": row.phase_name, "type": row.type,
                         "block": int(row.block) + 1, "week_in_block": int(row.week_in_block) + 1,
                         "recovery": bool(row.recovery), "taper": W["taper"], "push": W["push"],
                         "ramp_target": round(ramp, 1),
                         "tss_target": round(target), "tss_planned": round(plan_tss), "hours_planned": round(plan_h, 1),
                         "hours_cap": round(W["cap_h"], 1), "ctl_end": round(float(c_run), 1),
                         "capped": bool(plan_tss < 0.95 * target and plan_h >= W["cap_h"] - 0.25),
                         # a loading week using (nearly) all the hours it may: more load would need more time
                         "hours_limited": bool(W["kind"] == "load" and not W["taper"] and plan_h >= time_h - HOURS_LIMITED_H)})
        return rows, done_wk

    def easy_caps(self, sub: list[Workout], load_week: bool) -> list[float | None]:
        """each session's most minutes for the volume fit: an endurance / long ride within EASY_LIMITS raised to
        your own longest recent day (`ride_limit`) in a loading week, within EASY_LIMITS_RECOVERY otherwise; the
        day's time for the rest."""
        if load_week:
            return [self.ride_limit(s.date, EASY_LIMITS[s.type][1], s.type == "long") if s.type in EASY_LIMITS
                    else self.day_max(s.date) for s in sub]
        return [min(EASY_LIMITS_RECOVERY[s.type][1], self.day_max(s.date) or np.inf) if s.type in EASY_LIMITS_RECOVERY
                else self.day_max(s.date) for s in sub]

    def easy_max(self, sub: list[Workout], load_week: bool) -> list[float]:
        """each day's most easy minutes for the physio load model: an endurance / long ride's limit (`ride_limit`), a
        recovery spin's RECOVERY_SPIN_MAX; on an interval day of a loading week the Z2 that fits around its intervals
        within the day (as pros ride them; SessionOptimizer allows the same); 0 otherwise."""
        out = []
        for s in sub:                                    # a week's days
            if s.type in EASY_LIMITS:
                out.append(self.ride_limit(s.date, EASY_LIMITS[s.type][1], s.type == "long"))
            elif s.type == "recovery":
                out.append(min(RECOVERY_SPIN_MAX, self.day_max(s.date) or np.inf))
            elif load_week and s.type in MIX_TYPES:
                out.append(max(0.0, self.ride_limit(s.date, DAY_MINUTES["other"], False) - s.minutes))
            else:
                out.append(0.0)
        return out

    def refresh_weeks(self, rows, workouts, extras, done_wk, ctl, atl) -> None:
        """Week totals as they stand after keeping what you accepted."""
        c_run, a_run = ctl, atl
        for r in rows:
            w = pd.Timestamp(r["week"])
            sub = [s for s in workouts if w <= s.date < w + pd.Timedelta(days=7)]
            d_tss, d_h = done_wk.get(w, (0.0, 0.0))
            r["tss_planned"] = round(d_tss + sum(s.tss() for s in sub))
            r["hours_planned"] = round(d_h + sum(s.minutes for s in sub) / 60, 1)
            if sub:
                c, a = self.load.project(self.daily_tss(sub, extras), c_run, a_run)
                c_run, a_run = c[-1], a[-1]
                r["ctl_end"] = round(float(c_run), 1)

    @staticmethod
    def fit_volume(workouts, need_tss: float, max_min: float, recovery_week: bool, caps=None) -> list[Workout]:
        """Scale the endurance / long rides (within limits and each day's time) to reach need_tss
        without going over max_min: one factor for all, found by bisection."""
        caps = caps or [None] * len(workouts)
        flex = [i for i, w in enumerate(workouts) if w.type in EASY_LIMITS and not (w.locked or w.own)]
        if not flex:
            return workouts
        fixed_tss = sum(w.tss() for i, w in enumerate(workouts) if i not in flex)
        fixed_min = sum(w.minutes for i, w in enumerate(workouts) if i not in flex)
        base = np.array([workouts[i].minutes for i in flex], float)
        rate = np.array([workouts[i].tss() for i in flex]) / np.maximum(base, 1)
        limits = EASY_LIMITS_RECOVERY if recovery_week else EASY_LIMITS
        lo = np.array([limits[workouts[i].type][0] for i in flex], float)
        hi = np.array([limits[workouts[i].type][1] if caps[i] is None else caps[i] for i in flex], float)   # (easy_caps)
        lo = np.minimum(lo, hi)
        f = np.linspace(0.1, 6.0, 600)[:, None]           # every factor at once, no bisection loop
        mins = np.clip(base * f, lo, hi)
        ok = (fixed_tss + mins @ rate < need_tss) & (fixed_min + mins.sum(1) < max_min)
        k = max(int(np.flatnonzero(ok)[-1]) if ok.any() else 0, 0)
        out = list(workouts)
        for i, m in zip(flex, mins[k]):
            out[i] = WL.easy(workouts[i].date, workouts[i].type, m)
            out[i].why_change = workouts[i].why_change
        return out

    @staticmethod
    def daily_tss(workouts, extras=()) -> np.ndarray:
        """Planned TSS per day of `workouts` (with your added sessions on those days)."""
        days = {w.date for w in workouts}
        ws = list(workouts) + [x for x in extras if x.date in days]
        return pd.Series([w.tss() for w in ws], [w.date for w in ws], dtype=float).groupby(level=0).sum().to_numpy()

    def projection(self, sessions, ctl: float, atl: float) -> list[tuple]:
        """(date, CTL, ATL, TSB) for every planned day."""
        if not sessions:
            return []
        dates = sorted({w.date for w in sessions})
        c, a = self.load.project(self.daily_tss(sessions), ctl, atl)
        return list(zip(dates, c, a, c - a))
