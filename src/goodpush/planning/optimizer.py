"""Which session on which free day: one mixed-integer program over the whole plan (scipy milp / HiGHS).

Fixed days (tests, races, openers, the day after a race, your rest days, days with no time, your own
changes and ticked-off sessions) come from the scheduler's template. For every other day the program picks one
option - rest, a recovery spin, endurance of any length, or a quality session the phase uses - so that

hard rules
* neighbouring days' hardness adds up to at most 3 (VO2max / threshold / sweet spot / tempo / test / race 2,
  openers / LSCT 1, easy 0): VO2max -> openers -> endurance is fine, VO2max -> tempo is not; yesterday's
  actual ride counts. Tempo counts as hard: the morning after a ride that reaches LT1 costs about as much
  whatever the session (the cost is per day, not per minute above LT1)
* at or above threshold (threshold, VO2max, anaerobic, a test): rest or a recovery spin the day before
* a week whose load is under SPIN_FROM_HOURS of endurance riding rides endurance (from 30 min) on its easy days:
  a recovery spin only the day after a hard session or the day before one at threshold
* at least one rest day in a full week; each session fits its day's time; the week fits your hours
* never three days in a row without a ride (the days before today count) - soft, heavily weighted: it gives
  way only when the hours or the days' time leave no room
* no day above the TSS spike cap (3 x CTL, at least 150)
* per week at most the phase's quality sessions
* sessions you added next to a day's own ("#2" ...) count with their hardness: a hard one is one of the week's
  quality sessions, counts in the neighbouring days, wants an easy day before and a fresh morning
soft goals (weighted)
* each week's TSS on the weekly target from LoadModel.week_targets (in a loading week an interval day's
  spare time can carry Z2 after the intervals)
* the phase's quality sessions all placed (a missed one this week is still owed -> makeups)
* the same type of session as you already have on that day (no reshuffling from run to run)
* your key / recovery days from the settings; the long ride on your longest-available day
health, fatigue and form
* readiness red (resting HR, HRV, sleep, your illness / overreaching markers, how you feel ...):
  today and tomorrow easy (no hard session, at most an easy spin's load); amber: today at most
  sweet spot / tempo. These apply straight away (no proposal) with the reasons attached.
* race mornings: form (TSB) >= +10 for an A race, >= -5 for a B race
* every planned morning: form (TSB) >= -30, a threshold / VO2max day only from TSB >= -20, and
  fatigue (ATL) <= 1.3 x fitness (CTL) - all from today's real CTL / ATL, soft but heavily weighted
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .load import TSB_A, TSB_B
from .readiness import EASY_DAYS
from .workouts import (AT_THRESHOLD_UP, DAYS, EASY_BEFORE, LABELS, LADDERS, LEVELS, SPIN_FROM_HOURS, Workout,
                       WorkoutLibrary as WL)

HARDNESS = {"rest": 0, "recovery": 0, "endurance": 0, "long": 0, "openers": 1, "lsct": 1, "tempo": 2,
            "sweetspot": 2, "threshold": 2, "vo2max": 2, "test": 2, "ramp": 2, "race": 2, "anaerobic": 2}
MAX_PAIR_HARDNESS = 3
TSB_FLOOR, TSB_HARD_DAY, ACWR_MAX = -30.0, -20.0, 1.3
RED_TSS = 40.0                   # most a red day may carry (an easy spin)
BIG_M = 100.0
FIXED_TYPES = ("test", "ramp", "race", "openers", "lsct")
DONE_STATUSES = ("done", "partial", "low", "linked", "substituted")
LONG_MIN = 150                    # an endurance ride this long is "the long ride"
DAY_MINUTES = {"long": 240.0, "other": 120.0}   # a day's time without availability, at least (SessionOptimizer.day_minutes)
EASY_WEEK_MINUTES = 120.0         # an endurance ride in a recovery week / phase, at most
SPIKE_CTL, SPIKE_MIN = 3.0, 150.0
# objective weights (per TSS point / per session / per day)
W_TSS, W_MIX, W_KEEP, W_KEY, W_RECOVERY = 1.0, 200.0, 20.0, 25.0, 10.0
W_MINE = 80.0                    # your own (unlocked) session on its day: kept unless the plan has a real reason
W_FORM = 20.0                    # per TSB / ATL point over a limit
W_LONG = 0.2                     # per endurance minute on the long-ride day (moves volume, never adds it)
W_GAP = 150.0                    # per three days in a row without a ride
W_STAY = 0.3                     # per minute a day moves from the version you last saw (re-plans don't reshuffle)
MAX_DAYS_OFF = 2                 # ... so at most this many in a row
TIME_LIMIT = 20.0


class ConstraintRows:
    """Linear constraint rows lo <= coefs . x <= hi, collected one at a time."""

    def __init__(self, n: int):
        self.n, self.rows, self.lo, self.hi = n, [], [], []

    def add(self, coefs: dict, lo: float, hi: float) -> None:
        r = np.zeros(self.n)
        for k, v in coefs.items():
            r[k] += v
        self.rows.append(r)
        self.lo.append(lo)
        self.hi.append(hi)

    def add_rows(self, matrix: np.ndarray, lo, hi) -> None:
        """Several rows at once (dense, one per row of `matrix`)."""
        self.rows += list(matrix)
        self.lo += list(np.broadcast_to(lo, len(matrix)))
        self.hi += list(np.broadcast_to(hi, len(matrix)))

    def constraint(self) -> LinearConstraint:
        return LinearConstraint(np.array(self.rows), self.lo, self.hi)


class SessionOptimizer:
    """Chooses the sessions of the free days (see the module doc). `solve` changes the workouts in place."""

    def __init__(self, scheduler):
        self.cfg = scheduler.cfg
        self.avail = scheduler.avail
        self.day_max = scheduler.day_max
        self.ride_limit = scheduler.ride_limit
        self.fitting_intervals = scheduler.fitting_intervals
        self.events = scheduler.events
        self.week_mix = scheduler.week_mix
        self.kept = scheduler.kept                             # sessions in a format you moved: they keep their day
        self.model_mix = scheduler.chosen is not None          # the model chose the key days' sessions
        self.long_dn = DAYS[int(np.argmax(self.avail))] if self.avail else self.cfg["long_day"]

    @staticmethod
    def phase_mix(row, fixed_hard: int, style: str = "pyramidal") -> list[str]:
        """The quality sessions a week of this phase wants (Scheduler.quality), minus one per fixed hard
        day (test, race)."""
        from .scheduler import Scheduler
        mix = Scheduler.quality(row.type, bool(row.recovery), int(row.block), int(row.week), style)
        return mix[:max(0, len(mix) - fixed_hard)]

    def day_minutes(self, d: pd.Timestamp) -> float:
        """the day's time: your availability, else DAY_MINUTES raised to your own longest recent day
        (Scheduler.ride_limit) - a rider who rides 5-h days gets 5-h days."""
        if self.avail:
            return self.day_max(d)
        long = DAYS[d.weekday()] == self.long_dn
        return self.ride_limit(d, DAY_MINUTES["long" if long else "other"], long)

    def options(self, d: pd.Timestamp, kinds: list[str], level: dict, easy_week: bool) -> list[tuple]:
        """(name, minutes, tss, hardness) per option of a free day; endurance has variable minutes (from 30 in an
        easy or short week, else 45)."""
        cap = self.day_minutes(d)
        out = [("rest", 0.0, 0.0, 0)]
        if cap >= 45:
            w = WL.easy(d, "recovery", 45)
            out.append(("recovery", w.minutes, w.tss(), 0))
        if cap >= (30 if easy_week else 45):
            out.append(("endurance", None, None, 0))
        for k in kinds:
            w = self.fitting_intervals(d, k, level.get(k, 0), cap)
            if w.minutes <= cap:
                out.append((k, w.minutes, w.tss(), HARDNESS[k]))
        return out

    @staticmethod
    def mine(state, d: pd.Timestamp) -> dict | None:
        """Your own session on this day when it isn't locked (a locked one is a fixed day)."""
        m = state.get("manual_plan", {}).get(str(d.date()))
        return m if m and not m.get("locked") else None

    @staticmethod
    def option_of(kind: str) -> str:
        """The day option a session type is (a long ride is endurance minutes)."""
        return "endurance" if kind == "long" else kind

    def owed(self, state, today: pd.Timestamp, mix: list[str]) -> list[str]:
        """This week's quality sessions not yet done (a missed one is still owed)."""
        week0 = today - pd.Timedelta(days=today.weekday())
        done = [p.get("type") for k, p in state["planned"].items()
                if week0 <= pd.Timestamp(k[:10]) < today and p.get("status") in DONE_STATUSES]
        left = list(mix)
        for t in done:
            if t in left:
                left.remove(t)
        return left

    def solve(self, workouts: list[Workout], weeks: list[dict], lp: dict, state, acts: pd.DataFrame,
              today: pd.Timestamp, ctl: float, atl: float, ready: dict, extras: list[Workout]) -> bool:
        """Pick the free days' sessions; False (workouts untouched) when the program can't be solved.
        `extras`: the sessions you added next to a day's own (fixed; their load and hardness count)."""
        n_days = len(workouts)
        week_of = np.empty(n_days, int)
        for wi, W in enumerate(weeks):
            week_of[W["idx"]] = wi
        extra_tss, x_types = {}, {}
        for x in extras:
            extra_tss[x.date] = extra_tss.get(x.date, 0.0) + x.tss()
            x_types.setdefault(x.date, set()).add(x.type)
        x_hard = np.array([max([HARDNESS.get(t, 0) for t in x_types.get(w.date, ())], default=0) for w in workouts])
        x_above = np.array([bool(x_types.get(w.date, set()) & AT_THRESHOLD_UP) for w in workouts])
        x_busy = np.array([bool(x_types.get(w.date, set()) - set(EASY_BEFORE)) for w in workouts])   # not an easy day
        after_event = {i for i, w in enumerate(workouts) if i and workouts[i - 1].type == "race"}
        rode_today = "sport" in acts and bool((acts["date"].eq(today) & acts["sport"].eq("ride")).any())
        fixed = np.array([w.locked or w.type in FIXED_TYPES or i in after_event or self.day_minutes(w.date) == 0
                          or (w.type == "rest" and DAYS[w.date.weekday()] in self.cfg["rest_days"])
                          or (w.date == today and rode_today)              # ridden already: keep it
                          or self.own_on_amber(w, ready, today)            # amber: your session is your call
                          or (self.kept.get(str(w.date.date())) == w.type      # your harder / easier format
                              and not (ready.get("status") == "red" and (w.date - today).days <= 1))
                          for i, w in enumerate(workouts)])
        level = dict(state["levels"])
        rate = WL.easy(today, "endurance", 60).tss() / 60   # TSS per endurance minute
        low = self.low_volume(weeks, lp, rate)                 # weeks of little riding: endurance, not spins
        mixes, opts, var = [], [], []                          # var: (day, option index)
        for wi, W in enumerate(weeks):
            hard_fixed = sum(1 for i in W["idx"] if fixed[i] and HARDNESS.get(workouts[i].type, 0) == 2) \
                + int(np.sum(x_hard[W["idx"]] == 2))              # your added hard sessions are the week's too
            mix = self.week_mix(W["row"], W["start"], hard_fixed)
            if wi == 0 and W["idx"].size and workouts[W["idx"][0]].date == today and not self.model_mix:
                mix = self.owed(state, today, mix)     # (the model chose for the days still to come)
            mixes.append(mix)
        # each week's interval steps: the start step plus one per earlier week that has that session
        week_levels = [{k: min(level.get(k, 0) + sum(k in m for m in mixes[:wi]), LEVELS[k] - 1)
                        for k in LADDERS} for wi in range(len(weeks))]
        for i, w in enumerate(workouts):
            if fixed[i]:
                opts.append([])
                continue
            mix = mixes[week_of[i]]
            day_opts = self.options(w.date, list(dict.fromkeys(mix)), week_levels[week_of[i]],
                                    weeks[week_of[i]]["kind"] != "load" or low[week_of[i]])
            mine = self.mine(state, w.date)
            if mine and self.option_of(mine["type"]) not in [o[0] for o in day_opts]:     # yours is always an option
                own = Workout.from_dict(str(w.date.date()), mine)
                day_opts.append((mine["type"], own.minutes, own.tss(), HARDNESS.get(mine["type"], 1)))
            if ready.get("status") == "red" and (w.date - today).days <= 1:     # red: rest or a spin
                day_opts = [o for o in day_opts if o[0] in ("rest", "recovery")]
            opts.append(day_opts)
            var += [(i, o) for o in range(len(opts[i]))]
        if not var:
            return True
        # variables: y (one per day option) | m (endurance minutes per free day) | under, over per week | short per week
        ny, W = len(var), len(weeks)
        end_vars = {i: k for k, (i, o) in enumerate(var) if opts[i][o][0] == "endurance"}
        free = sorted({i for i, _ in var})
        m_idx = {i: ny + j for j, i in enumerate(free)}
        U, O, SH = ny + len(free), ny + len(free) + W, ny + len(free) + 2 * W
        F = SH + W                                             # 3 slacks a day: form, hard-day freshness, ACWR
        windows = self.gap_windows(workouts, acts, today)
        G = F + 3 * n_days                                     # a slack per window of days that needs a ride
        seen = self.seen_minutes(state, workouts, free)
        S = G + len(windows)                                   # per day you saw: minutes it moves from that version
        n = S + len(seen)
        rows = ConstraintRows(n)
        add = rows.add
        by_day: dict[int, list[int]] = {}
        for k, (i, o) in enumerate(var):
            by_day.setdefault(i, []).append(k)
        for i in free:          # one option a day; easy minutes: an endurance ride, or Z2 after the intervals
            add({k: 1 for k in by_day[i]}, 1, 1)
            easy = weeks[week_of[i]]["kind"] != "load"
            day_min = self.day_minutes(workouts[i].date)
            room = {k: -((min(day_min, EASY_WEEK_MINUTES) if easy else day_min) if opts[i][var[k][1]][0] == "endurance"
                         else max(0.0, day_min - opts[i][var[k][1]][1]) if opts[i][var[k][1]][3] and not easy else 0.0)
                    for k in by_day[i]}
            add({m_idx[i]: 1, **room}, -np.inf, 0)
            k = end_vars.get(i)
            if k is not None:
                add({m_idx[i]: 1, k: -(30 if easy or low[week_of[i]] else 45)}, 0, np.inf)
        hard = {}                                              # day -> hardness expression
        for i in range(n_days):                                # (+ the day's added sessions)
            hard[i] = ({}, HARDNESS.get(workouts[i].type, 0) + x_hard[i]) if fixed[i] else \
                ({k: opts[i][var[k][1]][3] for k in by_day.get(i, [])}, x_hard[i])
        y_kind = self.yesterday_hard(state, acts, today)
        h_yday = HARDNESS.get(y_kind, 0)
        for i in range(n_days):                                # neighbouring hardness (yesterday too)
            if i == 0:
                if workouts[0].date == today and not fixed[0]:
                    add(hard[0][0], -np.inf, max(0, MAX_PAIR_HARDNESS - h_yday - hard[0][1]))
                continue
            if (workouts[i].date - workouts[i - 1].date).days != 1:
                continue
            (ca, ka), (cb, kb) = hard[i - 1], hard[i]
            coefs = dict(ca)
            for k, v in cb.items():
                coefs[k] = coefs.get(k, 0) + v
            if coefs:                                          # (fixed / added load past the limit: no more on top)
                add(coefs, -np.inf, max(0, MAX_PAIR_HARDNESS - ka - kb))
        for i in range(1, n_days):                             # at / above threshold: rest / a spin the day before
            if (workouts[i].date - workouts[i - 1].date).days != 1:
                continue
            above = {k: 1 for k in by_day.get(i, []) if opts[i][var[k][1]][0] in AT_THRESHOLD_UP}
            easy = {} if x_busy[i - 1] else \
                {k: -1 for k in by_day.get(i - 1, []) if opts[i - 1][var[k][1]][0] in EASY_BEFORE}
            a0 = int((fixed[i] and workouts[i].type in AT_THRESHOLD_UP) or x_above[i])
            e0 = int(fixed[i - 1] and workouts[i - 1].type in EASY_BEFORE and not x_busy[i - 1])
            if above or easy:                                  # two fixed days: nothing to choose
                add({**above, **easy}, -np.inf, e0 - a0 if easy else max(e0 - a0, 0))   # (nothing to fix there)
        self.add_spin_rule(rows, workouts, low, week_of, fixed, var, opts, by_day, x_hard, x_above, h_yday, today)
        spike = max(SPIKE_MIN, SPIKE_CTL * ctl)
        for i in free:
            add({**{k: opts[i][var[k][1]][2] or 0 for k in by_day[i]}, m_idx[i]: rate}, -np.inf, spike)
        self.add_health(rows, workouts, fixed, var, opts, by_day, m_idx, rate, F, today, ctl, atl, ready, extra_tss,
                        x_hard)
        for g, days in enumerate(windows):                     # no long breaks: a ride in every window
            rides = {k: 1 for i in days if not fixed[i] for k in by_day[i] if opts[i][var[k][1]][0] != "rest"}
            ridden = sum(1 for i in days if (fixed[i] and workouts[i].minutes > 0 and workouts[i].type != "rest")
                         or workouts[i].date in x_types)                # a session you added is a ride
            if not ridden:
                add({**rides, G + g: 1}, 1, np.inf)
        for j, (i, prev) in enumerate(seen.items()):           # |the day's minutes - the version you saw|
            total = {**{k: opts[i][var[k][1]][1] or 0.0 for k in by_day[i]}, m_idx[i]: 1.0}
            add({**total, S + j: -1}, -np.inf, prev)
            add({**total, S + j: 1}, prev, np.inf)
        for wi, Wk in enumerate(weeks):                        # weekly TSS, hours, rest day, quality mix
            idx = Wk["idx"]
            fixed_tss = sum(workouts[i].tss() for i in idx if fixed[i])
            fixed_min = sum(workouts[i].minutes for i in idx if fixed[i])
            tss, mins = {}, {}
            for i in idx:
                if fixed[i]:
                    continue
                for k in by_day[i]:
                    name, mi, t, _ = opts[i][var[k][1]]
                    tss[k], mins[k] = t or 0.0, mi or 0.0
                tss[m_idx[i]], mins[m_idx[i]] = rate, 1.0
            target = float(lp["tss"][wi]) - Wk["extra"][0] - fixed_tss
            add({**tss, U + wi: 1, O + wi: -1}, target, target)
            (_, done_h), (_, ext_m) = Wk["done"], Wk["extra"]
            add(mins, -np.inf, max(0.0, (Wk["cap_h"] - done_h) * 60 - ext_m - fixed_min))
            if idx.size == 7 and not any(fixed[i] and workouts[i].type == "rest" for i in idx):
                add({k: 1 for i in idx if not fixed[i] for k in by_day[i] if opts[i][var[k][1]][0] == "rest"}, 1, np.inf)
            mix = mixes[wi]
            for kind in set(mix):
                ks = {k: 1 for i in idx if not fixed[i] for k in by_day[i] if opts[i][var[k][1]][0] == kind}
                need = mix.count(kind)
                if ks:
                    add({**ks, SH + wi: 1} if need else ks, need if need else -np.inf, need)
                elif need:
                    add({SH + wi: 1}, need, np.inf)
        cost = np.zeros(n)
        cost[U:U + W] = cost[O:O + W] = W_TSS
        cost[SH:SH + W] = W_MIX
        cost[F:G] = W_FORM
        cost[G:S] = W_GAP
        cost[S:] = W_STAY
        for i in free:
            if DAYS[workouts[i].date.weekday()] == self.long_dn:
                cost[m_idx[i]] -= W_LONG
        accepted = state.get("accepted", {})
        for k, (i, o) in enumerate(var):
            name, d = opts[i][o][0], workouts[i].date
            dn, had = DAYS[d.weekday()], (accepted.get(str(d.date())) or {}).get("type")
            had = "endurance" if had == "long" else had
            if had and had == name:
                cost[k] -= W_KEEP
            mine = self.mine(state, d)
            if mine and self.option_of(mine["type"]) == name:
                cost[k] -= W_MINE
            if HARDNESS.get(name, 0) and dn in self.cfg["key_days"]:
                cost[k] -= W_KEY
            if name == "recovery" and dn in self.cfg["recovery_days"]:
                cost[k] -= W_RECOVERY
            cost[k] += 1e-3 * i                                # ties: earlier days first
        integrality = np.zeros(n)
        integrality[:ny] = 1
        bounds = Bounds(np.zeros(n), np.r_[np.ones(ny), np.full(n - ny, np.inf)])
        res = milp(cost, constraints=rows.constraint(), integrality=integrality,
                   bounds=bounds, options={"time_limit": TIME_LIMIT, "mip_rel_gap": 0.01})
        if res.x is None:
            return False
        self.apply(workouts, res.x, var, opts, m_idx, [week_levels[week_of[i]] for i in range(n_days)],
                   state, today, y_kind)
        self.note_readiness(workouts, ready, today)
        return True

    @staticmethod
    def low_volume(weeks: list[dict], lp: dict, rate: float) -> np.ndarray:
        """per week: its load (done + target) less than SPIN_FROM_HOURS of endurance riding - little time on the
        bike, so its easy days are endurance rides, not recovery spins. `rate`: TSS per endurance minute."""
        load = np.array([W["done"][0] for W in weeks]) + np.asarray(lp["tss"][:len(weeks)], float)
        return load < SPIN_FROM_HOURS * 60 * rate

    @staticmethod
    def add_spin_rule(rows: ConstraintRows, workouts, low, week_of, fixed, var, opts, by_day, x_hard, x_above,
                      h_yday: int, today: pd.Timestamp) -> None:
        """In a low-volume week (`low`, per week) a recovery spin only the day after a hard session (yesterday's ride
        counts) or the day before one at threshold: spin <= hard yesterday + at threshold tomorrow."""
        n_days = len(workouts)
        for i in sorted(by_day):
            if fixed[i] or not low[week_of[i]]:
                continue
            spin = {k: 1 for k in by_day[i] if opts[i][var[k][1]][0] == "recovery"}
            if not spin:
                continue
            coefs, const = dict(spin), 0
            if i == 0:
                const += int(workouts[0].date == today and h_yday == 2)
            elif (workouts[i].date - workouts[i - 1].date).days == 1:
                if fixed[i - 1]:
                    const += int(HARDNESS.get(workouts[i - 1].type, 0) == 2)
                coefs.update({k: -1 for k in by_day.get(i - 1, []) if opts[i - 1][var[k][1]][3] == 2})
                const += int(x_hard[i - 1] == 2)
            if i + 1 < n_days and (workouts[i + 1].date - workouts[i].date).days == 1:
                if fixed[i + 1]:
                    const += int(workouts[i + 1].type in AT_THRESHOLD_UP)
                coefs.update({k: -1 for k in by_day.get(i + 1, []) if opts[i + 1][var[k][1]][0] in AT_THRESHOLD_UP})
                const += int(x_above[i + 1])
            rows.add(coefs, -np.inf, const)

    @staticmethod
    def gap_windows(workouts, acts: pd.DataFrame, today: pd.Timestamp) -> list[list[int]]:
        """the runs of MAX_DAYS_OFF + 1 consecutive planned days (indices) that need a ride; the first ones shorter
        by the days you've already gone without riding before today."""
        dates = pd.DatetimeIndex([w.date for w in workouts])
        rides = (acts.loc[acts["sport"].eq("ride") & acts["date"].lt(today), "date"]
                 if len(acts) and "sport" in acts else pd.Series(dtype="datetime64[ns]"))
        last = pd.Timestamp(rides.max()) if len(rides) else today - pd.Timedelta(days=MAX_DAYS_OFF + 1)
        off = int(min(max((today - last).days - 1, 0), MAX_DAYS_OFF))     # days off just before today
        span = MAX_DAYS_OFF + 1
        out = []
        for i in range(len(workouts)):
            start = dates[i] - pd.Timedelta(days=span - 1)
            first = max(start, today)
            if start < today and (today - start).days > off:              # reaches back past your last ride
                continue
            j = np.flatnonzero((dates >= first) & (dates <= dates[i]))
            if len(j) == (dates[i] - first).days + 1:                    # consecutive planned days only
                out.append(j.tolist())
        return out

    def add_health(self, rows: ConstraintRows, workouts, fixed, var, opts, by_day, m_idx, rate, F, today,
                   ctl: float, atl: float, ready: dict, extra_tss: dict, x_hard: np.ndarray) -> None:
        """Readiness caps on today / tomorrow and the daily form / fatigue limits (see the module doc).
        Daily TSS is T @ x + const; morning CTL / ATL are lower-triangular decay weights @ that.
        x_hard (days,): the hardest session you added on each day (a hard one wants a fresh morning too)."""
        n_days, n = len(workouts), rows.n
        pos = np.array([(w.date - today).days for w in workouts])
        T, H, H2 = np.zeros((n_days, n)), np.zeros((n_days, n)), np.zeros((n_days, n))
        k_all = np.array([k for k, _ in enumerate(var)], int)
        if k_all.size:
            day = np.array([i for i, _ in var])
            info = np.array([opts[i][o][2] or 0.0 for i, o in var]), np.array([opts[i][o][3] for i, o in var])
            T[day, k_all], H[day, k_all] = info
            H2[day, k_all] = info[1] == 2
        free = np.array(sorted(m_idx), int)
        if free.size:
            T[free, [m_idx[i] for i in free]] = rate
        const = np.array([extra_tss.get(w.date, 0.0) + (w.tss() if fixed[i] else 0.0) for i, w in enumerate(workouts)])
        status = ready.get("status", "green")
        if status != "green":
            capped = np.flatnonzero((pos < ready.get("easy_days", EASY_DAYS[status])) & ~np.asarray(fixed))
            rows.add_rows(H[capped], np.full(capped.size, -np.inf), np.full(capped.size, 0 if status == "red" else 1))
            if status == "red":
                rows.add_rows(T[capped], np.full(capped.size, -np.inf), RED_TSS - const[capped])
        lag = pos[:, None] - pos[None, :] - 1                  # days between an earlier day and this morning
        load = {}
        for name, days_tc, y0 in (("ctl", 42, ctl), ("atl", 7, atl)):
            k, a = 1 - np.exp(-1 / days_tc), np.exp(-1 / days_tc)
            Wm = np.where(lag >= 0, k * a ** np.maximum(lag, 0), 0.0)
            load[name] = (Wm @ T, a ** pos * y0 + Wm @ const)
        (Mc, c0), (Ma, a0) = load["ctl"], load["atl"]
        later = np.arange(1, n_days)                           # this morning is a fact
        S = np.zeros((n_days, n))
        S[np.arange(n_days), F + 3 * np.arange(n_days)] = 1
        rows.add_rows((Mc - Ma + S)[later], (TSB_FLOOR - (c0 - a0))[later], np.full(later.size, np.inf))
        hard_days = later[H2[later].any(1)]
        G = np.zeros((n_days, n))
        G[np.arange(n_days), F + 3 * np.arange(n_days) + 1] = 1
        rows.add_rows((Mc - Ma + G - BIG_M * H2)[hard_days], (TSB_HARD_DAY - BIG_M - (c0 - a0))[hard_days],
                      np.full(hard_days.size, np.inf))
        added = later[x_hard[later] == 2]                      # your added hard sessions: always hard days
        rows.add_rows((Mc - Ma + G)[added], (TSB_HARD_DAY - (c0 - a0))[added], np.full(added.size, np.inf))
        # race mornings, day by day: the taper's form (A >= TSB_A low end, B >= TSB_B); a race day has no
        # session to choose, so its hard-day slack is free for this
        prio = np.array([(self.events.get(str(w.date.date())) or {}).get("priority", "") if w.type == "race" else ""
                         for w in workouts])
        race = later[np.isin(prio[later], ("A", "B"))]
        if race.size:
            lo = np.where(prio[race] == "A", TSB_A[0], TSB_B)
            rows.add_rows((Mc - Ma + G)[race], lo - (c0 - a0)[race], np.full(race.size, np.inf))
        R = np.zeros((n_days, n))
        R[np.arange(n_days), F + 3 * np.arange(n_days) + 2] = -1
        rows.add_rows((Ma - ACWR_MAX * Mc + R)[later], np.full(later.size, -np.inf), (ACWR_MAX * c0 - a0)[later])

    @staticmethod
    def own_on_amber(w: Workout, ready: dict, today: pd.Timestamp) -> bool:
        """Your own (moved / changed) session on a day an amber morning keeps easy: kept as you set it - amber is
        advice, the choice is yours (red still makes the day easy)."""
        return bool(w.own and ready.get("status") == "amber"
                    and 0 <= (w.date - today).days < ready.get("easy_days", EASY_DAYS["amber"]))

    @staticmethod
    def note_readiness(workouts, ready: dict, today: pd.Timestamp) -> None:
        """Today (and tomorrow when red) carry the readiness reasons, so the change applies straight away;
        after a red day the next hard session is conditional."""
        if ready.get("status", "green") == "green":
            return
        why = "; ".join(ready.get("reasons", [])) or f"readiness {ready['status']}"
        days = ready.get("easy_days", EASY_DAYS[ready["status"]])
        red = ready["status"] == "red"
        for w in workouts:
            if SessionOptimizer.own_on_amber(w, ready, today):
                w.adjusted = f"{why} - your session, kept: back off if your HR runs high"
            elif (w.date - today).days < days and not w.locked:
                w.adjusted = why
            elif red and HARDNESS.get(w.type, 0) == 2 and not w.locked:
                w.adjusted = "only if resting HR is back to normal and you feel fine - otherwise ride easy"
                return

    def apply(self, workouts, x, var, opts, m_idx, day_levels, state, today, y_kind: str) -> None:
        """Workouts from the chosen options (interval steps as priced); why sessions moved."""
        chosen = {i: opts[i][o][0] for k, (i, o) in enumerate(var) if x[k] > 0.5}
        before = {i: workouts[i].type for i in chosen}
        week_end = today + pd.Timedelta(days=6 - today.weekday())
        missed = {p["type"]: pd.Timestamp(k[:10]) for k, p in sorted(state["planned"].items())
                  if today - pd.Timedelta(days=today.weekday()) <= pd.Timestamp(k[:10]) < today
                  and p.get("status") in ("missed", "skipped") and p.get("type") in LADDERS}
        for i in sorted(chosen):
            d, name = workouts[i].date, chosen[i]
            mine = self.mine(state, d)
            if mine and self.option_of(mine["type"]) == name:        # your session kept: exactly as you set it
                new = Workout.from_dict(str(d.date()), mine)
                new.own = True
            elif name == "rest":
                new = WL.rest(d)
            elif name == "recovery":
                new = WL.easy(d, "recovery", 45)
            elif name == "endurance":
                mins = max(30, round(x[m_idx[i]] / 5) * 5)
                new = WL.easy(d, "long" if mins >= LONG_MIN else "endurance", mins)
            else:
                new = self.fitting_intervals(d, name, day_levels[i].get(name, 0), self.day_minutes(d))
                pad = int(x[m_idx[i]] / 5) * 5                 # the week needs more: Z2 after the intervals
                if pad >= 10:
                    new = new.resized(new.minutes + pad)
                if name in missed and d <= week_end:
                    new.why_change = f"makeup for {missed.pop(name):%a}'s missed {LABELS[name]}"
            workouts[i] = new
        # an unplanned hard ride yesterday: today's key session went easy and later
        had = (state.get("accepted", {}).get(str(today.date())) or {}).get("type") or before.get(0)
        if 0 in chosen and workouts[0].date == today and y_kind and HARDNESS.get(had, 0) == 2 \
                and HARDNESS.get(chosen[0], 0) < 2:
            what = LABELS.get(y_kind, y_kind)
            later = next((j for j in sorted(chosen) if j > 0 and chosen[j] == had), None)
            workouts[0].why_change = f"easier after yesterday's unplanned {what}" + \
                (f"; {LABELS.get(had, had)} moved to {workouts[later].date:%a}" if later is not None else "")
            if later is not None:
                workouts[later].why_change = f"moved from {today:%a}: yesterday's {what} wasn't in the plan"

    @staticmethod
    def seen_minutes(state, workouts: list[Workout], free: list[int]) -> dict[int, float]:
        """{free day: its minutes in the version you last saw (PlanState "accepted")} - days the plan has shown."""
        acc = state.get("accepted", {})
        seen = {i: (acc.get(str(workouts[i].date.date())) or {}).get("minutes") for i in free}
        return {i: float(m) for i, m in seen.items() if m is not None}

    @staticmethod
    def yesterday_hard(state, acts: pd.DataFrame, today: pd.Timestamp) -> str:
        """The hardest ride type yesterday ("" if none hard). Planned hard days count as planned."""
        y = today - pd.Timedelta(days=1)
        if not len(acts) or "workout" not in acts:
            return ""
        kinds = [str(t) for t in acts.loc[acts["date"] == y, "workout"].fillna("")]
        hard = sorted((k for k in kinds if HARDNESS.get(k, 0) == 2), key=lambda k: k != "race")
        return hard[0] if hard else ""
