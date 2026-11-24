"""Training load for planning: fitness (CTL) / fatigue (ATL) projections and the weekly load targets.

Weekly targets are one linear program over every week of the plan (scipy HiGHS, milliseconds).
With a week's TSS spread evenly over its days, CTL and ATL on every day are linear in the weekly
totals, so the constraints are exact for that spread:

* loading weeks (base / build): CTL rises at least min_ramp a week (soft) and at most the week's
  step of the min -> max staircase (hard); specialty holds 0..+1, maintenance -0.5..+0.5
* the last loading week of a block is the push: no ramp cap, instead form (TSB) at its end comes down
  to TSB_PUSH - tired enough to want the recovery week, short of overreaching (TSB < -30)
* weekly load grows at most 15 % on the last loading week
* recovery weeks: at most 70 % of a steady week's load (7 x CTL; about 40 % under the loading weeks); a recovery phase 50 %
* each week within your hours: the planned days' availability and the weekly cap, at
  MAX_TSS_PER_HOUR (43 TSS an hour)
* races: TSB on race morning +10..+25 for an A race, >= -5 for a B race (soft) - the taper falls out
Objective: the highest CTL on your A races' days (weighted 10) and at the end of the plan. Taper weeks
(a race in the week, or an A race next week) have no minimum ramp, so fitness may dip to get fresh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.signal import lfilter

MAX_TSS_PER_HOUR = 43.0          # a mostly-easy week with two interval days (what the planned weeks reach)
RECOVERY_SHARE = 0.7          # of a steady week (7 x CTL) in a recovery week
RECOVERY_PHASE_SHARE = 0.5
TSS_GROWTH = 1.15             # weekly load (per day) vs the last loading week
TSB_A = (10.0, 25.0)          # race-morning form for an A race
TSB_B = -5.0                  # B race: a mini-taper (2-3 easier days), not a fresh week
TSB_PUSH = -20.0              # form at the end of a block's last loading week
PENALTY_RAMP, PENALTY_TSB = 10.0, 100.0
WEIGHT_RACE_CTL, WEIGHT_END_CTL = 10.0, 1.0


class VolumeRamp:
    """How much more than you have been riding the plan may ask, from your recent weeks: a week at most GROWTH
    above the average week of the last LOOKBACK_WEEKS (before this one), one more GROWTH after each loading week
    ahead (a recovery week or phase adds none: the week after it is at most GROWTH above the last loading week);
    one day's riding the same way from the longest day of those weeks. Each re-plan starts again from what you
    rode. Without any history: no limit. `start_hours`: the week to start from when it's more than that average
    (the plan setting plan_starting_hours: what you know you can ride already)."""
    GROWTH = 0.10
    LOOKBACK_WEEKS = 4
    MIN_WEEK_H = 3.0         # after a break: never below this a week ...
    MIN_RIDE_H = 1.5         # ... or a day

    def __init__(self, daily: pd.DataFrame, today: pd.Timestamp, days: pd.DataFrame | None = None,
                 start_hours: float = 0.0):
        """days: the plan's day table (Season.day_table: type, recovery) - which weeks ahead are loading weeks;
        without it every week is."""
        self.start = (today - pd.Timedelta(days=today.weekday())).normalize()       # this week's Monday
        self.growth_weeks = None
        if days is not None and len(days):
            monday = days.index - pd.to_timedelta(days.index.day_of_week, unit="D")
            load = (~days["recovery"].astype(bool) & days["type"].ne("recovery")).groupby(monday).first()
            self.growth_weeks = 1 + load.cumsum().shift(fill_value=0).astype(int)   # 1 + loading weeks before
        past = daily.loc[self.start - pd.Timedelta(weeks=self.LOOKBACK_WEEKS): self.start - pd.Timedelta(days=1)]
        if not len(past):
            self.week_h = self.ride_h = float("inf")
            if start_hours > 0:
                self.week_h = start_hours
            return
        weeks = min(self.LOOKBACK_WEEKS, max(1.0, (past.index[-1] - past.index[0]).days / 7 + 1 / 7))
        ride = past["ride_hours"] if "ride_hours" in past else past["hours"]
        self.week_h = max(self.MIN_WEEK_H, start_hours, float(past["hours"].sum()) / weeks)
        self.ride_h = max(self.MIN_RIDE_H, float(ride.max()))

    def steps(self, d: pd.Timestamp) -> int:
        """steps of growth by the week of `d`: 1 this week, one more after each loading week."""
        d = pd.Timestamp(d).normalize()
        monday = d - pd.Timedelta(days=d.weekday())
        if self.growth_weeks is not None and monday in self.growth_weeks.index:
            return int(self.growth_weeks[monday])
        return max(0, (monday - self.start).days // 7) + 1

    def week_cap(self, d: pd.Timestamp) -> float:
        """most hours for the week of `d`."""
        return self.week_h * (1 + self.GROWTH) ** self.steps(d)

    def week_steady(self, d: pd.Timestamp) -> float:
        """the hours the week of `d` should reach: what you have been riding, +GROWTH after each loading week
        (one step below `week_cap`) - steady growth of time on the bike."""
        return self.week_h * (1 + self.GROWTH) ** (self.steps(d) - 1)

    def ride_cap(self, d: pd.Timestamp) -> float:
        """most minutes of riding on day `d`."""
        return 60 * self.ride_h * (1 + self.GROWTH) ** self.steps(d)


class RampRange:
    """How fast your fitness may build: the 4-week CTL ramp (points a week, as in the profile's "How fast can you
    ramp?" table) below which you gained little and above which you got ill more often - from your history where
    it has enough weeks (and, for the top, enough illnesses), else the usual guidance. `flag` marks the plan's
    weeks outside it ("slow" / "fast"); n = 1 and observational: a hint, not a rule."""
    USUAL = (1.5, 5.0)       # below ~1.5 a week fitness barely moves; ~5+ for weeks on end is an aggressive build
    MIN_WEEKS = 20           # a ramp band counts with at least this many weeks ...
    MIN_ILLNESSES = 3        # ... and the top only with this many illnesses on record
    GAIN_PCT = 2.0           # bottom: the slowest band whose weeks were followed by at least this power gain
    ILL_FACTOR = 1.5         # top: the fastest band (from the bottom up) whose illness rate stayed under this x usual

    def __init__(self, buildup: dict | None = None):
        """buildup: the analysis' results["buildup"] (ramp_table, illness_base_rate, illness_episodes)."""
        from ..models.buildup import RAMP_BINS, RAMP_LABELS
        b = buildup or {}
        edges = dict(zip(RAMP_LABELS, zip(RAMP_BINS[:-1], RAMP_BINS[1:])))
        rows = [(edges[r["ctl_ramp_per_week"]], r) for r in b.get("ramp_table") or []
                if edges.get(r.get("ctl_ramp_per_week"), (-1, 0))[0] >= 0 and (r.get("weeks") or 0) >= self.MIN_WEEKS]
        self.lo, self.hi = self.USUAL
        self.lo_learned = self.hi_learned = False
        gains = [lo for (lo, _), r in rows if (r.get("power_change_pct") or -99) >= self.GAIN_PCT]
        if gains:
            self.lo, self.lo_learned = float(gains[0]), True
        base = b.get("illness_base_rate")
        if (b.get("illness_episodes") or 0) >= self.MIN_ILLNESSES and base and base == base and rows:
            hi = None
            for (lo, top), r in rows:
                ill = r.get("illness_next_6wk")
                if ill is None or ill != ill or ill > self.ILL_FACTOR * base:
                    break
                hi = top
            if hi is not None and hi > self.lo:
                self.hi, self.hi_learned = float(hi), True

    def to_dict(self) -> dict:
        return {"lo": self.lo, "hi": self.hi, "lo_learned": self.lo_learned, "hi_learned": self.hi_learned}

    def flag(self, weeks: list[dict], daily: pd.DataFrame) -> None:
        """Each plan week row gets ramp_4wk (CTL at its end vs 4 weeks before, per week; earlier weeks from
        `daily`) and ramp_flag: "fast" above the range, "slow" below it in a base / build loading week (specialty and
        maintenance hold fitness), else None."""
        if not weeks:
            return
        start = pd.Timestamp(weeks[0]["week"])
        past = daily["ctl"].loc[:start - pd.Timedelta(days=1)].resample("W-SUN").last().iloc[-4:]
        ctl = pd.concat([past.reset_index(drop=True), pd.Series([w["ctl_end"] for w in weeks], dtype=float)],
                        ignore_index=True)
        ramp = ((ctl - ctl.shift(4)) / 4).to_numpy()[len(past):]
        for w, r in zip(weeks, ramp):
            ok = r == r
            w["ramp_4wk"] = round(float(r), 1) if ok else None
            load = w.get("type") in ("base", "build") and not (w.get("recovery") or w.get("taper"))
            w["ramp_flag"] = ("fast" if ok and r > self.hi else "slow" if ok and load and r < self.lo else None)


class LoadModel:
    """Exponentially weighted load: CTL (42 d) and ATL (7 d), as in metrics/load.py."""

    def __init__(self, ctl_days: int = 42, atl_days: int = 7):
        self.k_ctl = 1 - np.exp(-1 / ctl_days)
        self.k_atl = 1 - np.exp(-1 / atl_days)

    def project(self, tss, ctl0: float, atl0: float) -> tuple[np.ndarray, np.ndarray]:
        """CTL and ATL after each day of `tss`, starting from ctl0 / atl0 (one lfilter call each)."""
        t = np.asarray(tss, float)
        out = []
        for k, y0 in ((self.k_ctl, ctl0), (self.k_atl, atl0)):
            y, _ = lfilter([k], [1, k - 1], t, zi=[(1 - k) * y0])
            out.append(y)
        return out[0], out[1]

    @staticmethod
    def ramp_bounds(cfg: dict, phase_type: str, week_in_block: int, build_weeks: int) -> tuple[float, float]:
        """(lowest wanted, highest allowed) CTL change in a loading week of this phase."""
        if phase_type == "specialty":
            return 0.0, 1.0
        if phase_type == "maintenance":
            return -0.5, 0.5
        k = min(week_in_block / max(1, build_weeks - 1), 1.0)
        return cfg["min_ramp"], cfg["min_ramp"] + (cfg["max_ramp"] - cfg["min_ramp"]) * k

    def week_targets(self, weeks: list[dict], ctl0: float, atl0: float, ctl_week0: float) -> dict:
        """Planned TSS per week (the days still to come).

        weeks: per week {"days": planned days, "kind": "load" | "recovery" | "recovery_phase",
        "ramp": (lo, hi), "cap": max TSS, "races": [(planned days before the race, "A" | "B")],
        "taper": bool (optional: no minimum ramp), "push": bool (optional: TSB_PUSH instead of the ramp cap)}.
        ctl0 / atl0: yesterday's values; ctl_week0: CTL at the start of the first week (its ramp
        counts from there). Returns {"tss", "ctl" (end of each week), "ok"}."""
        W = len(weeks)
        if not W:                                       # nothing planned (no phases)
            return {"tss": np.zeros(0), "ctl": np.zeros(0), "ok": True}
        races = [(w, d, p) for w, wk in enumerate(weeks) for d, p in wk["races"] if p in ("A", "B")]
        E = len(races)
        n = 4 * W + 2 * E                              # x, C, A, ramp slack per week; 2 TSB slacks per race
        X, C, A, S, T = 0, W, 2 * W, 3 * W, 4 * W
        days = np.array([wk["days"] for wk in weeks], float)
        ac, aa = 1 - self.k_ctl, 1 - self.k_atl
        # equalities: end-of-week CTL / ATL from the previous week's and this week's load
        eq = np.zeros((2 * W, n))
        beq = np.zeros(2 * W)
        for j, (off, a, y0) in enumerate(((C, ac, ctl0), (A, aa, atl0))):
            an = a ** days
            r = j * W + np.arange(W)
            eq[r, off + np.arange(W)] = 1
            eq[r[1:], off + np.arange(W - 1)] = -an[1:]
            eq[r, X + np.arange(W)] = -(1 - an) / days
            beq[r[0]] = an[0] * y0
        ub, bub = [], []
        last_load = None
        for w, wk in enumerate(weeks):
            prev, const = (C + w - 1, 0.0) if w else (None, ctl_week0)
            if wk["kind"] == "load":
                lo, hi = wk["ramp"]
                r = np.zeros(n)
                if wk.get("push"):                        # C_w - A_w >= TSB_PUSH (the objective pushes to it)
                    r[A + w], r[C + w] = 1, -1
                    ub.append(r)
                    bub.append(-TSB_PUSH)
                else:                                     # C_w - C_prev <= hi
                    r[C + w] = 1
                    if prev is not None:
                        r[prev] = -1
                    ub.append(r)
                    bub.append(hi + const)
                r = np.zeros(n)                           # C_w - C_prev + s_w >= lo (not while tapering)
                if wk.get("taper"):
                    lo = -1e3
                r[C + w], r[S + w] = -1, -1
                if prev is not None:
                    r[prev] = 1
                ub.append(r)
                bub.append(-lo - const)
                if last_load is not None:                 # per-day load grows at most 15 %
                    r = np.zeros(n)
                    r[X + w], r[X + last_load] = 1 / days[w], -TSS_GROWTH / days[last_load]
                    ub.append(r)
                    bub.append(0.0)
                last_load = w
            else:                                         # recovery: a share of a steady week
                share = RECOVERY_SHARE if wk["kind"] == "recovery" else RECOVERY_PHASE_SHARE
                r = np.zeros(n)
                r[X + w] = 1
                if prev is not None:
                    r[prev] = -share * days[w]
                ub.append(r)
                bub.append(share * days[w] * (ctl0 if prev is None else 0.0))
        for e, (w, d, p) in enumerate(races):            # race-morning TSB
            tsb = np.zeros(n)
            const = 0.0
            for off, a, y0 in ((C, ac, ctl0), (A, aa, atl0)):
                sign = 1 if off == C else -1
                ad = a ** d
                if w:
                    tsb[off + w - 1] += sign * ad
                else:
                    const += sign * ad * y0
                tsb[X + w] += sign * (1 - ad) / days[w]
            lo, hi = TSB_A if p == "A" else (TSB_B, 1e3)
            r_lo, r_hi = -tsb, tsb.copy()                 # tsb + s_lo >= lo ; tsb - s_hi <= hi
            r_lo[T + 2 * e], r_hi[T + 2 * e + 1] = -1, -1
            ub += [r_lo, r_hi]
            bub += [const - lo, hi - const]
        cost = np.zeros(n)
        a_races = [(w, d) for w, d, p in races if p == "A"]
        cost[C + W - 1] -= WEIGHT_END_CTL                   # fitness at the end of the plan ...
        for w, d in a_races:                             # ... and, above all, on A-race mornings
            ad = ac ** d
            cost[X + w] -= WEIGHT_RACE_CTL * (1 - ad) / days[w]
            if w:
                cost[C + w - 1] -= WEIGHT_RACE_CTL * ad
        cost[S:S + W] = PENALTY_RAMP
        cost[T:] = PENALTY_TSB
        bounds = [(0, wk["cap"]) for wk in weeks] + [(None, None)] * (2 * W) + [(0, None)] * (W + 2 * E)
        res = linprog(cost, A_ub=np.array(ub) if ub else None, b_ub=np.array(bub) if ub else None,
                      A_eq=eq, b_eq=beq, bounds=bounds, method="highs")
        if res.status != 0:                               # shouldn't happen (all soft); keep a steady load
            tss = np.minimum([wk["cap"] for wk in weeks], ctl0 * days)
            return {"tss": tss, "ctl": np.full(W, ctl0), "ok": False}
        return {"tss": res.x[X:X + W], "ctl": res.x[C:C + W], "ok": True}
