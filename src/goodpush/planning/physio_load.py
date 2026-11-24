"""Weekly load targets from the physiological model (`[plan] load_model = "physio"`): model-predictive.

Same output as LoadModel.week_targets (TSS per week for the days still to come, end-of-week CTL), but
the amounts come from physio.plan.DosePlan: the template's sessions stay (phases, blocks, recovery weeks,
key days, tests, races: the block structure), the model picks each week's easy hours and how much of its
key work to keep, for the best forecast on your A / B race mornings (all-out, durations by race type) and
your fitness at the end of the plan, without more fatigue than you have handled before (95th
percentile of the fitted fatigue over the last two years). Replaces the ramp, push, recovery-share and
taper rules; the daily optimiser then places the sessions as before.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .intensity import IntensityMap
from .load import LoadModel
from .workouts import EASY_LIMITS, Workout

RACE_DURATIONS = {"road": (300, 1200, 3600), "crit": (60, 300, 1200), "tt": (1200, 3600), "mtb": (300, 1200, 3600),
                  "gravel": (1200, 3600, 7200), "cx": (60, 300, 1200), "hillclimb": (600, 1200, 2400),
                  "fondo": (3600, 7200), "other": (300, 1200, 3600)}
END_DURATIONS = (300, 1200, 3600)
RACE_WEIGHT = {"A": 1.0, "B": 0.3}
END_WEIGHT, END_WEIGHT_NO_RACE = 0.3, 1.0
EASY_SHARE = {"long": 2.0, "endurance": 1.0, "recovery": 0.5}
KEY_EASY_SHARE = 0.5            # an interval day's Z2 around the intervals: its share of the week's easy hours
SCALED = {"threshold", "vo2max", "sweetspot", "tempo", "anaerobic"}
MIN_RIDE_DAYS = 120             # history needed before "your highest fatigue" is a backstop
BACKSTOP_GROWTH = 1.10          # the backstop: 10 % above the highest so far (it rises as you reach new highs)



class PhysioLoad:
    def __init__(self, model, z, ftp: float, easy_ftp: float = 0.635, draws=None):
        """The cost of fatigue is learned (adaptation and performance, as fitted to your data). The only
        limit is a backstop: BACKSTOP_GROWTH above the highest fatigue in your
        history, so each block may go a little beyond anything you've done and, once you have, the ceiling
        moves up with you; only with MIN_RIDE_DAYS of rides behind it (little history: no backstop).
        `draws` (n, P): posterior draws the session mix is chosen over (`choose_sessions`); none = the mode."""
        self.model, self.z, self.ftp = model, np.asarray(z), float(ftp)
        self.draws = np.asarray(draws) if draws is not None and len(draws) else None
        self.easy_ftp = float(easy_ftp)             # easy riding, fraction of FTP (the middle of your endurance band)
        self.power_w = np.asarray(model.data.power_w)
        self.last_day = pd.Timestamp(model.data.days[-1])
        ride_days = int((np.asarray(model.data.hist_h).sum(1) > 0).sum())
        _, F, end = model.simulate(self.z)                   # your history: once for all of the below
        self.fatigue_cap = BACKSTOP_GROWTH * float(np.max(F)) if ride_days >= MIN_RIDE_DAYS else float("inf")
        self.result = None
        # today's thresholds turn the sessions' %FTP into the watts you'll ride (planning/intensity.py)
        self.now = IntensityMap.from_states(model, [self.last_day + pd.Timedelta(days=1)],
                                            np.asarray(end[0])[None], ftp)

    def intensity_map(self) -> IntensityMap:
        """thresholds on each planned day: the last solve's forecast, else today's."""
        r = self.result
        if r is None or not len(r.states):
            return self.now
        days = pd.date_range(self.last_day + pd.Timedelta(days=1), periods=len(r.states))
        return IntensityMap.from_states(self.model, days, r.states, self.ftp)

    def hist(self, w: Workout) -> np.ndarray:
        """hours per power bin of a session in your watts (each step at its middle)."""
        st = w.expanded()
        out = np.zeros(len(self.power_w))
        if not st:
            return out
        m = np.array([s.minutes for s in st], float)
        p = self.now.watts(w.date, [(s.lo + s.hi) / 2 for s in st])
        width = self.power_w[1] - self.power_w[0]
        b = np.clip((p / width).astype(int), 0, len(out) - 1)
        return np.bincount(b, m / 60, minlength=len(out))

    def week_targets(self, weeks: list[dict], workouts: list[Workout], extras: list[Workout], events: dict,
                     load: LoadModel, ctl: float, atl: float) -> dict | None:
        """{"tss", "ctl", "ok", "physio"} like LoadModel.week_targets, or None when nothing lies past the fit."""
        from ..physio.plan import DosePlan
        W = len(weeks)
        dates = pd.DatetimeIndex([w.date for w in workouts])
        j = (dates - self.last_day).days.to_numpy() - 1               # forecast day of each session
        D = int(j.max()) + 1 if W and len(j) else 0
        if D <= 0:
            return None
        week_of = np.zeros(len(workouts), int)
        for wi, wk in enumerate(weeks):
            week_of[wk["idx"]] = wi
        day_week = np.zeros(D, int)
        ok = j >= 0
        day_week[j[ok]] = week_of[ok]
        B = len(self.power_w)
        H = np.stack([self.hist(w) for w in workouts]) if workouts else np.zeros((0, B))
        types = np.array([w.type for w in workouts])
        locked = np.array([w.locked for w in workouts], bool)
        easy = np.isin(types, list(EASY_SHARE)) & ~locked
        key = np.isin(types, list(SCALED)) & ~locked
        fixed = ~easy & ~key
        fixed_hist, key_hist = np.zeros((D, B)), np.zeros((D, B))
        np.add.at(fixed_hist, j[ok & fixed], H[ok & fixed])
        np.add.at(key_hist, j[ok & key], H[ok & key])
        for x in extras:                                              # your added sessions stay
            jx = (pd.Timestamp(x.date) - self.last_day).days - 1
            if 0 <= jx < D:
                fixed_hist[jx] += self.hist(x)
        # each day's most easy minutes (Scheduler.easy_max: grows with your own longest days; on an interval day the
        # Z2 that fits around its intervals); weeks without it (older callers): the fixed limits within your time
        day_max = np.full(len(workouts), np.inf)                        # your time / the ramp, per day
        limit = np.full(len(workouts), np.nan)
        for wk in weeks:                                                # the plan's weeks
            day_max[wk["idx"]] = [np.inf if m is None else m for m in wk.get("day_max", [None] * len(wk["idx"]))]
            if "easy_max" in wk:
                limit[wk["idx"]] = wk["easy_max"]
        fixed_limit = np.minimum([EASY_LIMITS.get(t, (0, 90))[1] for t in types], day_max)
        limit = np.where(np.isnan(limit), np.where(easy, fixed_limit, 0.0), limit)
        pad = key & (limit > 0)                                          # Z2 around the intervals
        weight = np.where(easy, pd.Series(types).map(EASY_SHARE).fillna(0).to_numpy(float), 0.0) + KEY_EASY_SHARE * pad
        wsum = np.bincount(week_of, weight, minlength=W)
        share = np.zeros(D)
        np.add.at(share, j[ok], (weight / np.maximum(wsum[week_of], 1e-9))[ok])
        long_share = np.zeros(D)
        long_share[j[ok & easy & (types == "long")]] = 1.0
        # easy hours: within the week's hours (after its fixed and key sessions) and the easy days' limits
        mins = np.array([w.minutes for w in workouts], float)
        fixed_h = np.bincount(week_of, np.where(fixed | key, mins, 0) / 60, minlength=W)
        limit_h = np.bincount(week_of, np.where(easy | pad, limit, 0) / 60, minlength=W)
        room = np.array([wk["cap_h"] - wk["done"][1] - wk["extra"][1] / 60 for wk in weeks]) - fixed_h
        easy_max = np.clip(np.minimum(room, limit_h), 0, None)
        targets = self.targets(events, D)
        easy_bin = np.zeros(B)
        easy_w = float(self.now.watts(self.last_day, self.easy_ftp))
        easy_bin[min(int(easy_w / (self.power_w[1] - self.power_w[0])), B - 1)] = 1.0
        ramp, recover = self.block_pairs(weeks)
        steady = np.array([max(0.0, wk.get("steady_h", 0.0) - wk["done"][1] - wk["extra"][1] / 60) for wk in weeks])
        plan = DosePlan(self.model, self.z, fixed_hist, key_hist, share, day_week, easy_bin, easy_max,
                        targets, self.fatigue_cap, long_share, ramp, recover, steady)
        r = plan.solve()
        tss = self.weekly_tss(weeks, workouts, week_of, fixed, key, r)
        self.result = r
        return {"tss": tss, "ctl": self.week_ctl(weeks, tss, load, ctl, atl), "ok": r.converged,
                "physio": {"easy_h": r.easy_h, "key_scale": r.key_scale, "fatigue_cap": self.fatigue_cap,
                           "objective": r.objective}}

    def targets(self, events: dict, D: int) -> list:
        """your A / B race mornings (all-out at the race kind's durations) and your fitness at the end of the D days."""
        from ..physio.plan import Target
        out = [Target(int((pd.Timestamp(d) - self.last_day).days - 1), RACE_WEIGHT[e["priority"]],
                      RACE_DURATIONS.get(e.get("kind"), RACE_DURATIONS["other"]))
               for d, e in events.items() if e["priority"] in RACE_WEIGHT
               and 0 <= (pd.Timestamp(d) - self.last_day).days - 1 < D]
        out.append(Target(D - 1, END_WEIGHT if out else END_WEIGHT_NO_RACE, END_DURATIONS, all_out=False))
        return out

    def choose_sessions(self, workouts: list[Workout], extras: list[Workout], slots: dict, events: dict):
        """{workout index: session type} for the key days `slots` {index: [candidate sessions, the first the
        template's]}: the sessions that give the best forecast on your race mornings and at the end (physio.plan.
        MixPlan), averaged over the posterior draws if there are any; and the MixResult (None: nothing to choose)."""
        from ..physio.plan import LONG_AFTER_H, MixPlan
        dates = pd.DatetimeIndex([w.date for w in workouts])
        j = (dates - self.last_day).days.to_numpy() - 1               # forecast day of each session
        D = int(j.max()) + 1 if len(j) else 0
        slots = {i: c for i, c in slots.items() if j[i] >= 0 and c}
        if D <= 0 or not slots:
            return {}, None
        B = len(self.power_w)
        base = np.zeros((D, B))
        others = [i for i in range(len(workouts)) if i not in slots and j[i] >= 0]
        if others:
            np.add.at(base, j[others], np.stack([self.hist(workouts[i]) for i in others]))
        for x in extras:                                              # your added sessions
            jx = (pd.Timestamp(x.date) - self.last_day).days - 1
            if 0 <= jx < D:
                base[jx] += self.hist(x)
        long_h = np.zeros(D)
        is_long = np.array([w.type == "long" for w in workouts]) & (j >= 0)
        long_h[j[is_long]] = np.clip(np.array([w.minutes for w in workouts])[is_long] / 60 - LONG_AFTER_H, 0, None)
        index = list(slots)
        K = max(len(c) for c in slots.values())
        cand, ok = np.zeros((len(index), K, B)), np.zeros((len(index), K), bool)
        for s, i in enumerate(index):                                 # a few candidates a key day
            cand[s, :len(slots[i])] = np.stack([self.hist(w) for w in slots[i]])
            ok[s, :len(slots[i])] = True
        Z = self.draws if self.draws is not None else self.z[None]
        plan = MixPlan(self.model, Z, base, long_h, j[index], cand, ok, np.zeros(len(index), int),
                       self.targets(events, D), self.fatigue_cap)
        r = plan.solve()
        return {i: slots[i][c].type for i, c in zip(index, r.choice)}, r

    @staticmethod
    def block_pairs(weeks) -> tuple[np.ndarray, np.ndarray]:
        """for each pair of consecutive whole weeks in one block: (loading -> loading, loading -> recovery).
        Not across blocks, phases, tapers, or the week already under way."""
        pairs = list(zip(weeks[:-1], weeks[1:]))
        ok = np.array([(a["row"].phase, a["row"].block) == (b["row"].phase, b["row"].block) and a["days"] == 7 and b["days"] == 7 and a["kind"] == "load"
                       and not a["taper"] and not b["taper"] for a, b in pairs], bool)
        nxt = np.array([b["kind"] for _, b in pairs])
        return (ok & (nxt == "load")).astype(float), (ok & (nxt == "recovery")).astype(float)

    def forecast_frame(self) -> pd.DataFrame | None:
        """the last solve's forecast: daily capacities (W) and fatigue if you ride the plan."""
        r = self.result
        if r is None:
            return None
        Q = np.asarray(self.model.quantities(r.states))                 # lt1, cp, map, 1-h power
        days = pd.date_range(self.last_day + pd.Timedelta(days=1), periods=len(r.fatigue), name="date")
        out = pd.DataFrame({"lt1": Q[0], "cp": Q[1], "map": Q[2], "ftp": Q[3], "fatigue": r.fatigue}, index=days)
        out.attrs["fatigue_cap"] = self.fatigue_cap if np.isfinite(self.fatigue_cap) else np.nan
        return out

    def weekly_tss(self, weeks, workouts, week_of, fixed, key, r) -> np.ndarray:
        """the days still to come: fixed sessions + scaled key sessions + easy hours + your added sessions."""
        W = len(weeks)
        t = np.array([w.tss() for w in workouts], float)
        out = np.bincount(week_of, np.where(fixed, t, 0), minlength=W)
        out += r.key_scale * np.bincount(week_of, np.where(key, t, 0), minlength=W)
        out += r.easy_h * self.easy_ftp ** 2 * 100
        return out + np.array([wk["extra"][0] for wk in weeks], float)

    @staticmethod
    def week_ctl(weeks, tss, load: LoadModel, ctl: float, atl: float) -> np.ndarray:
        """end-of-week CTL with each week's TSS spread evenly over its days."""
        days = np.array([wk["days"] for wk in weeks], int)
        daily = np.repeat(np.asarray(tss) / np.maximum(days, 1), days)
        c, _ = load.project(daily, ctl, atl)
        return c[np.cumsum(days) - 1]
