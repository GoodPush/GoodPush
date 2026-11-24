"""A ramp test ridden from the plan (WorkoutLibrary.ramp_test): what it says without alpha1, from the ride's laps
(one per workout step, exact) and its 1-s power / heart rate.

* top: the last step held to the end (LENIENCY_S short still counts: the step change lags the clock) plus the
  share of the next one you rode (Kuipers et al. 1985, Int J Sports Med 6:197);
* heart rate at the end of each step (its last HR_END_S: HR lags the power), and so at your thresholds' watts;
* the highest heart rate, and how far it fell RECOVERY_S after you stopped (heart-rate recovery: Cole et al.
  1999, N Engl J Med 341:1351; as a training-status marker, Daanen et al. 2012, Int J Sports Physiol Perform
  7:251);
* thresholds without alpha1: the power where the ramp reached your LT1 / LT2 heart rate (from alpha1 on your
  rides before the test; LTHR for LT2 without it) - goodpush's own rule, not a published one. Heart rate lags a
  ramp, so these read high - the physio model fits that bias (sources ramp_hr_lt1 / ramp_hr_lt2). Where alpha1
  was recorded its ramp thresholds lead. The app's Sources page lists all of these.

`RampTest.add_to` puts them in the activities table at analysis (ramp_* columns), where the model and the
Rider page take them from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.fit import FIT_EPOCH_OFFSET

LENIENCY_S = 10           # a step this close to its full length counts as held
HR_END_S = 30             # a step's heart rate: its last 30 s
RECOVERY_S = (60, 120)    # heart-rate recovery: this long after you stopped
STOP_FRAC = 0.5           # stopped: power under half the top
LAP_STEP = 71             # FIT lap field: the workout step it was (wkt_step_index)
LT2_MAX_SHARE = 0.92      # a ramp LT2 at or above this share of the top is no threshold (CP ~ 75-90 % of MAP)
LT2_HR_MARGIN = 3         # ... nor one within this many bpm of the ride's highest heart rate
HR_COLUMNS = ("ramp_hrr_60", "ramp_hr_lt1_w", "ramp_hr_lt1_hr", "ramp_hr_lt2_w", "ramp_hr_lt2_hr")


def window_means(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """nan-aware means of x over [a, b) for many windows at once (prefix sums); NaN where a window has no data."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    s, c = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))], np.r_[0, np.cumsum(ok)]
    a, b = np.clip(a, 0, x.size).astype(int), np.clip(b, 0, x.size).astype(int)
    n = c[b] - c[a]
    with np.errstate(all="ignore"):
        return np.where(n > 0, (s[b] - s[a]) / n, np.nan)


class RampTest:
    """steps: one row per ramp step you rode (target_w, planned_s, start, seconds ridden, power, hr_end, hr_max,
    cadence), in order; power / hr: the ride's 1-s streams."""

    @staticmethod
    def lt2_check(acts: pd.DataFrame) -> pd.Series:
        """Why a ramp's LT2 can't be a threshold (per row; "" = fine), so it isn't taken in: within LT2_MAX_SHARE
        of the ramp's top (LT2 sits well below the top - CP is ~75-90 % of MAP), or with heart rate within
        LT2_HR_MARGIN of the ride's highest (alpha1 crossed 0.5 only at exhaustion, catching up, not at a threshold).
        Checked against everything else, a reading that passes is weighed by the physio model (its own bias)."""
        col = RampTest.column
        top, mx = col(acts, "ramp_top_w"), col(acts, "ramp_max_hr")
        why = pd.Series("", index=acts.index)
        for w, hr in (("ramp_lt2_w", "ramp_lt2_hr"), ("ramp_hr_lt2_w", "ramp_hr_lt2_hr")):
            has = col(acts, w).notna()
            why = why.mask(why.eq("") & has & (col(acts, w) >= LT2_MAX_SHARE * top),
                           f"at {LT2_MAX_SHARE:.0%}+ of the ramp's top: LT2 sits well below it")
            why = why.mask(why.eq("") & has & (col(acts, hr) >= mx - LT2_HR_MARGIN),
                           "alpha1 crossed 0.5 only at your highest heart rate (catching up at exhaustion)")
        return why

    @staticmethod
    def column(acts: pd.DataFrame, c: str) -> pd.Series:
        """acts[c] as numbers (NaN where missing, or all NaN without the column)"""
        return pd.to_numeric(acts[c], errors="coerce") if c in acts else pd.Series(np.nan, index=acts.index)

    def __init__(self, steps: pd.DataFrame, power: np.ndarray, hr: np.ndarray):
        held = (steps["seconds"] >= steps["planned_s"] - LENIENCY_S).to_numpy()
        end = int(np.argmin(held)) + 1 if not held.all() else held.size      # the ramp ends at the step you didn't hold
        self.steps = steps.iloc[:end].assign(held=held[:end])                  # (what follows: easy riding, skipped steps)
        self.power, self.hr = np.asarray(power, float), np.asarray(hr, float)

    @classmethod
    def from_ride(cls, laps: list[dict], start: float, workout, ftp: float, z: dict) -> RampTest | None:
        """from the ride's FIT laps, its unix `start`, the planned workout (its ramp steps' %FTP x `ftp`) and its
        streams `z`. Laps without the step they were (rides before LAP_STEP) are taken in step order. None when
        no lap is a ramp step."""
        from ..planning.workouts import RAMP_NAME
        plan = workout.expanded()
        ramp = np.array([s.name.startswith(RAMP_NAME) for s in plan])
        if not laps or not ramp.any():
            return None
        lap = pd.DataFrame({"start": [x.get(2, 0) + FIT_EPOCH_OFFSET - start for x in laps],
                            "seconds": [x.get(8, 0) / 1000 for x in laps],
                            "step": [i if x.get(LAP_STEP) is None else x[LAP_STEP] for i, x in enumerate(laps)]})
        lap = lap[(lap["step"] < len(plan)) & ramp[np.minimum(lap["step"], len(plan) - 1)]]
        if lap.empty:
            return None
        mid = np.array([s.mid for s in plan])
        planned = np.array([s.minutes * 60 for s in plan])
        a = lap["start"].to_numpy(float)
        b = a + lap["seconds"].to_numpy(float)
        p, hr, cad = (np.asarray(z[k], float) for k in ("power", "heart_rate", "cadence"))
        hr_max = np.array([np.nanmax(hr[int(i):max(int(j), int(i) + 1)], initial=np.nan) for i, j in zip(a, b)])
        steps = pd.DataFrame({"name": [plan[i].name for i in lap["step"]], "target_w": mid[lap["step"]] * ftp,
                              "planned_s": planned[lap["step"]], "start": a, "seconds": lap["seconds"].to_numpy(float),
                              "power": window_means(p, a, b), "hr_end": window_means(hr, np.maximum(b - HR_END_S, a), b),
                              "hr_max": hr_max, "cadence": window_means(cad, a, b)}).reset_index(drop=True)
        return cls(steps, p, hr)

    @classmethod
    def of_ride(cls, proj, x: pd.Series, workout, z: dict | None = None) -> RampTest | None:
        """the ramp test ridden as `workout` on ride `x` (an activities row): from its FIT laps and streams (`z`, else
        read); None without them."""
        from ..io import streamstore
        from ..io.fit import FitError, MESG_LAP, read_messages
        f = proj.path("strava") / str(x.get("filename") or "")
        z = z or streamstore.read(proj.processed, str(x["id"]).removesuffix(".0"), "ride")
        if not z or not f.is_file():
            return None
        try:
            laps = read_messages(f, [MESG_LAP]).get(MESG_LAP, [])
        except (FitError, OSError):
            return None
        return cls.from_ride(laps, float(z["start"]), workout, float(x.get("ftp") or 0) or 200.0, z)

    @classmethod
    def add_to(cls, proj, acts: pd.DataFrame, lthr: float | None, progress=print) -> pd.DataFrame:
        """the ramp tests you rode from the plan into `acts` (after the alpha1 columns): ramp_start_min, ramp_min,
        ramp_top_w, ramp_max_hr (as the alpha1 detector fills them), ramp_hrr_60, and ramp_hr_lt1_w / ramp_hr_lt2_w
        with the heart rates they're at (ramp_hr_lt1_hr / ramp_hr_lt2_hr) where alpha1 gave no ramp threshold."""
        from ..io import streamstore
        from ..planning import PlanState
        st = PlanState.load(proj.processed / "plan_state.json")
        linked = cls.planned_rides(st)
        sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        rows = np.flatnonzero(sid.isin(linked).to_numpy())
        for c in ("ramp_start_min", "ramp_min", "ramp_top_w", "ramp_max_hr", "ramp_lt1_w", "ramp_lt2_w", *HR_COLUMNS):
            if c not in acts:
                acts[c] = np.nan
        if not rows.size:
            return acts
        anchors = cls.hr_anchors(acts, lthr)
        streams = streamstore.read_many(proj.processed, list(sid.iloc[rows]), "ride")
        done = 0
        for i in rows:                                 # your ramp tests: a handful
            x = acts.iloc[i]
            r = cls.of_ride(proj, x, st.workout(linked[sid.iloc[i]]), streams.get(sid.iloc[i]))
            if r is None or r.last_held is None:
                continue
            a = int(r.steps["start"].iloc[0])
            hr1, hr2 = anchors.iloc[i]
            vals = {"ramp_start_min": a / 60, "ramp_min": (r.stop() - a) / 60, "ramp_top_w": r.top(),
                    "ramp_max_hr": float(np.nanmax(r.hr[a:r.stop() + max(RECOVERY_S)], initial=np.nan)),
                    "ramp_hrr_60": r.recovery()[60]}
            for col, alpha, hr in (("ramp_hr_lt1", "ramp_lt1_w", hr1), ("ramp_hr_lt2", "ramp_lt2_w", hr2)):
                if not np.isfinite(x.get(alpha, np.nan)):         # alpha1's own ramp threshold leads
                    vals[f"{col}_w"], vals[f"{col}_hr"] = r.power_at_hr(hr), hr
            acts.loc[acts.index[i], list(vals)] = [np.nan if v is None else v for v in vals.values()]
            done += 1
        progress(f"ramp tests: {done} from the plan's ramp sessions")
        return acts

    @staticmethod
    def planned_rides(st) -> dict[str, str]:
        """ride id -> plan day of every ride linked to a planned ramp session (PlanState `st`): ramp tests for
        sure, whatever FTP says (FTP only set the workout's watts; the test is what tells whether it moved)."""
        return {str(p["ride"]): k for k, p in st["planned"].items() if p.get("type") == "ramp" and p.get("ride")}

    @staticmethod
    def hr_anchors(acts: pd.DataFrame, lthr: float | None) -> pd.DataFrame:
        """each ride's LT1 / LT2 heart rate as known before it: the latest alpha1 estimate from an earlier ride
        (hrvt1_hr_28d / hrvt2_hr_42d); LTHR for LT2 without one."""
        d = pd.to_datetime(acts["date"])
        out = pd.DataFrame(index=acts.index)
        for name, col in (("lt1", "hrvt1_hr_28d"), ("lt2", "hrvt2_hr_42d")):
            v = acts[col] if col in acts else pd.Series(np.nan, index=acts.index)
            known = v.groupby(d).last().sort_index().ffill().shift(1)          # by the day before
            out[name] = known.reindex(d).to_numpy()
        out["lt2"] = out["lt2"].fillna(lthr if lthr else np.nan)
        return out

    def power_at_hr(self, hr: float) -> float | None:
        """the ramp's power where its end-of-step heart rate reached `hr` (held steps); None outside them."""
        s = self.steps[self.steps["held"] & self.steps["hr_end"].notna()].sort_values("target_w")
        y = np.maximum.accumulate(s["hr_end"].to_numpy(float))
        if len(y) < 2 or not np.isfinite(hr) or not y[0] <= hr <= y[-1]:
            return None
        return float(np.interp(hr, y, s["target_w"].to_numpy(float)))

    @property
    def last_held(self) -> int | None:
        """the row of the last step held to the end (only the ramp's last step can be one you didn't hold)."""
        n = int(self.steps["held"].sum())
        return n - 1 if n else None

    def top(self) -> float | None:
        """the last held step's target plus the share of the next one you rode."""
        k = self.last_held
        if k is None:
            return None
        s = self.steps
        if k + 1 >= len(s):
            return float(s["target_w"].iloc[k])
        nxt = s.iloc[k + 1]
        share = min(nxt["seconds"] / nxt["planned_s"], 1.0)
        return float(s["target_w"].iloc[k] + share * (nxt["target_w"] - s["target_w"].iloc[k]))

    def stop(self) -> int:
        """the second you stopped: the first after the ramp's last step began where power fell under STOP_FRAC x
        the top (else the last step's end)."""
        last = self.steps.iloc[-1]
        a, b = int(last["start"]), int(last["start"] + last["seconds"])
        top = self.top() or float(last["target_w"])
        low = np.flatnonzero(np.nan_to_num(self.power[a:], nan=0.0) < STOP_FRAC * top)
        return a + int(low[0]) if low.size else min(b, self.power.size - 1)

    def recovery(self) -> dict[int, float | None]:
        """how far heart rate fell RECOVERY_S after you stopped (from its highest in the 10 s around the stop)."""
        t = self.stop()
        peak = np.nanmax(self.hr[max(t - 10, 0):t + 10], initial=np.nan)
        later = window_means(self.hr, np.array(RECOVERY_S) - 5 + t, np.array(RECOVERY_S) + 5 + t)
        return {s: (None if np.isnan(d) else round(float(d))) for s, d in zip(RECOVERY_S, peak - later)}

    def hr_at(self, watts: dict[str, float]) -> dict[str, dict]:
        """heart rate at each threshold's watts: between the held steps' (end-of-step HR), none outside them."""
        s = self.steps[self.steps["held"] & self.steps["hr_end"].notna()].sort_values("target_w")
        x, y = s["target_w"].to_numpy(float), s["hr_end"].to_numpy(float)
        out = {}
        for name, w in watts.items():             # a handful of thresholds
            inside = len(x) > 1 and w and x[0] <= w <= x[-1]
            out[name] = {"w": round(w) if w else None, "hr": round(float(np.interp(w, x, y))) if inside else None}
        return out

    def best(self, seconds: int = 60) -> float | None:
        """the best `seconds` of power from the first ramp step to the stop."""
        a, t = int(self.steps["start"].iloc[0]), self.stop()
        starts = np.arange(a, max(t - seconds + 1, a + 1))
        m = window_means(self.power, starts, starts + seconds)
        return round(float(np.nanmax(m))) if np.isfinite(m).any() else None

    def summary(self, watts: dict[str, float], ftp: float) -> dict:
        """for the ride page: the steps, the top (W, x FTP), best minute, max HR, recovery, HR at `watts`."""
        k, top = self.last_held, self.top()
        t = self.stop()
        mx = np.nanmax(self.hr[int(self.steps["start"].iloc[0]):t + max(RECOVERY_S)], initial=np.nan)
        rnd = lambda v, d=0: None if v is None or v != v else round(float(v), d)  # noqa: E731
        return {"top_w": rnd(top), "top_ftp": rnd(top / ftp, 2) if top and ftp else None,
                "last_held": None if k is None else self.steps["name"].iloc[k],
                "best_1min": self.best(60), "max_hr": rnd(mx), "recovery": self.recovery(), "hr_at": self.hr_at(watts),
                "steps": [{"name": r.name, "target_w": rnd(r.target_w), "seconds": rnd(r.seconds), "planned_s": rnd(r.planned_s),
                           "power": rnd(r.power), "hr_end": rnd(r.hr_end), "hr_max": rnd(r.hr_max),
                           "cadence": rnd(r.cadence), "held": bool(r.held)} for r in self.steps.itertuples()]}
