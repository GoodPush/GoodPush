"""All-out steps ridden from the plan (planning.workouts.MAX_EFFORTS: 1, 3 or 5 min opening a key
session, WorkoutLibrary.max_effort): where the ride's lap for that step is, and its best power over the step's
length. `AllOutStep.add_to` puts them in the activities table at analysis (all_out_s, all_out_w), where the physio
model's efforts (models.physio.PhysioInputs.efforts) count that best as a test - the curve's own point - instead of
a hard ride's best that may be far below it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.fit import FIT_EPOCH_OFFSET
from .ramp import LAP_STEP

COLUMNS = ("all_out_s", "all_out_w")


class AllOutStep:
    """seconds: the step's planned length; watts: the best mean power over that length inside its lap."""
    SLACK_S = 5          # the lap's edges: a second or two of the effort may fall into the laps around it

    def __init__(self, seconds: int, watts: float):
        self.seconds, self.watts = int(seconds), float(watts)

    @classmethod
    def from_ride(cls, laps: list[dict], start: float, workout, power: np.ndarray) -> AllOutStep | None:
        """from the ride's FIT laps, its unix `start`, the planned workout and its 1-s power; laps without the step
        they were (rides before LAP_STEP) are taken in step order. None without an all-out step ridden."""
        from ..planning.workouts import ALL_OUT_NAME
        plan = workout.expanded() if workout is not None else []
        mine = [i for i, s in enumerate(plan) if s.name.startswith(ALL_OUT_NAME)]
        if not laps or not mine:
            return None
        step = [i if x.get(LAP_STEP) is None else x[LAP_STEP] for i, x in enumerate(laps)]
        at = [j for j, s in enumerate(step) if s == mine[0]]
        if not at:
            return None
        x = laps[at[0]]
        seconds = int(round(plan[mine[0]].minutes * 60))
        lap = int(x.get(2, 0) + FIT_EPOCH_OFFSET - start)
        a = max(lap - cls.SLACK_S, 0)
        b = min(lap + int(x.get(8, 0) / 1000) + cls.SLACK_S, len(power))
        p = np.nan_to_num(np.asarray(power, float)[a:b])
        if p.size < seconds:
            return None
        c = np.r_[0.0, np.cumsum(p)]
        return cls(seconds, float(((c[seconds:] - c[:-seconds]) / seconds).max()))

    @classmethod
    def of_ride(cls, proj, x: pd.Series, workout, z: dict | None = None) -> AllOutStep | None:
        """the all-out step ridden as `workout` on ride `x` (an activities row): from its FIT laps and power
        stream (`z`, else read); None without them."""
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
        return cls.from_ride(laps, float(z["start"]), workout, z["power"])

    @staticmethod
    def planned_rides(st) -> dict[str, str]:
        """ride id -> plan day of every ride linked to a planned session that opens with an all-out step."""
        from ..planning.workouts import ALL_OUT_NAME
        out = {}
        for k, p in st["planned"].items():
            rid = p.get("ride") or (p.get("compliance") or {}).get("ride")
            if rid and str(p.get("title") or "").startswith(ALL_OUT_NAME):
                out[str(rid)] = k
        return out

    @classmethod
    def add_to(cls, proj, acts: pd.DataFrame, progress=print) -> pd.DataFrame:
        """the all-out steps you rode from the plan into `acts` (COLUMNS)."""
        from ..io import streamstore
        from ..planning import PlanState
        st = PlanState.load(proj.processed / "plan_state.json")
        linked = cls.planned_rides(st)
        for c in COLUMNS:
            if c not in acts:
                acts[c] = np.nan
        sid = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True)
        rows = np.flatnonzero(sid.isin(linked).to_numpy())
        if not rows.size:
            return acts
        streams = streamstore.read_many(proj.processed, list(sid.iloc[rows]), "ride")
        done = 0
        for i in rows:                                 # one every week or two
            r = cls.of_ride(proj, acts.iloc[i], st.workout(linked[sid.iloc[i]]), streams.get(sid.iloc[i]))
            if r is None:
                continue
            acts.loc[acts.index[i], list(COLUMNS)] = [r.seconds, r.watts]
            done += 1
        progress(f"all-out steps: {done} from the plan's sessions")
        return acts
