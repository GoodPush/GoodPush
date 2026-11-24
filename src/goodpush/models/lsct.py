"""An LSCT ridden from the plan (WorkoutLibrary.lsct; Lamberts et al. 2011, Br J Sports Med 45:797): what each
stage took and how fast heart rate fell, from the ride's laps (one per workout step, exact) and 1-s streams.

* per stage: mean power and heart rate over its second half (heart rate lags the power, and you settle onto the
  stage's heart rate in its first minutes; ridden in resistance mode);
* heart-rate recovery: the heart rate over the last END_S of the last stage less the heart rate RECOVERY_S into
  sitting still (Lamberts: the fall in the first minute).

Fitter: more power at the same heart rates, and a faster fall. Heavy fatigue can also speed the fall (heart rate
doesn't rise as far), so read the recovery together with the stage powers and how the legs felt.
`LSCT.add_to` puts them in the activities table at analysis (lsct_* columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io.fit import FIT_EPOCH_OFFSET
from .ramp import LAP_STEP, window_means

END_S = 5                  # heart rate at the end of the last stage: its last 5 s
RECOVERY_S = 60            # ... less the heart rate this long into sitting still (mean over +/- WINDOW_S)
WINDOW_S = 3
COLUMNS = ("lsct_w_1", "lsct_w_2", "lsct_w_3", "lsct_hr_1", "lsct_hr_2", "lsct_hr_3", "lsct_hrr_60")


class LSCT:
    """stages: one row per LSCT stage ridden (name, share of max HR, start, seconds, power, hr), in order; stop: the
    second the sitting still began (None: not ridden); hr: the ride's 1-s heart rate."""

    def __init__(self, stages: pd.DataFrame, stop: float | None, hr: np.ndarray):
        self.stages, self.stop, self.hr = stages, stop, np.asarray(hr, float)

    @classmethod
    def from_ride(cls, laps: list[dict], start: float, workout, z: dict) -> LSCT | None:
        """from the ride's FIT laps, its unix `start`, the planned workout and its streams `z`; laps without the step
        they were (rides before LAP_STEP) are taken in step order. None when no lap is an LSCT stage."""
        from ..planning.workouts import LSCT_STOP
        plan = workout.expanded()
        if not laps or not any(s.hr for s in plan):
            return None
        lap = pd.DataFrame({"start": [x.get(2, 0) + FIT_EPOCH_OFFSET - start for x in laps],
                            "seconds": [x.get(8, 0) / 1000 for x in laps],
                            "step": [i if x.get(LAP_STEP) is None else x[LAP_STEP] for i, x in enumerate(laps)]})
        lap = lap[lap["step"] < len(plan)]
        share = np.array([s.hr or 0.0 for s in plan])[lap["step"]]
        st = lap[share > 0]
        if st.empty:
            return None
        p, hr = (np.asarray(z[k], float) for k in ("power", "heart_rate"))
        a = st["start"].to_numpy(float)
        b = a + st["seconds"].to_numpy(float)
        half = (a + b) / 2
        stages = pd.DataFrame({"name": [plan[i].name for i in st["step"]], "share": share[share > 0],
                               "start": a, "seconds": st["seconds"].to_numpy(float),
                               "power": window_means(p, half, b), "hr": window_means(hr, half, b)})
        still = lap[[plan[i].name == LSCT_STOP for i in lap["step"]]]
        return cls(stages.reset_index(drop=True), float(still["start"].iloc[0]) if len(still) else None, hr)

    @classmethod
    def of_ride(cls, proj, x: pd.Series, workout, z: dict | None = None) -> LSCT | None:
        """the LSCT ridden as `workout` on ride `x` (an activities row): from its FIT laps and streams (`z`, else
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
        return cls.from_ride(laps, float(z["start"]), workout, z)

    def recovery(self) -> float | None:
        """how far heart rate fell RECOVERY_S into sitting still (bpm); None without the sitting still."""
        if self.stop is None:
            return None
        t = self.stop
        end, later = window_means(self.hr, np.array([t - END_S, t + RECOVERY_S - WINDOW_S]),
                                  np.array([t, t + RECOVERY_S + WINDOW_S]))
        d = end - later
        return None if np.isnan(d) else round(float(d))

    def values(self) -> dict[str, float | None]:
        """the lsct_* COLUMNS of this ride (stages not ridden: None)."""
        out = {c: None for c in COLUMNS}
        for i, r in enumerate(self.stages.itertuples()):         # three stages
            if i < 3:
                out[f"lsct_w_{i + 1}"] = None if r.power != r.power else round(float(r.power), 1)
                out[f"lsct_hr_{i + 1}"] = None if r.hr != r.hr else round(float(r.hr), 1)
        out["lsct_hrr_60"] = self.recovery()
        return out

    @staticmethod
    def planned_rides(st) -> dict[str, str]:
        """ride id -> plan day of every ride linked to (or scored for) a planned LSCT (PlanState `st`)."""
        out = {}
        for k, p in st["planned"].items():
            rid = p.get("ride") or (p.get("compliance") or {}).get("ride")
            if p.get("type") == "lsct" and rid:
                out[str(rid)] = k
        return out

    @classmethod
    def add_to(cls, proj, acts: pd.DataFrame, progress=print) -> pd.DataFrame:
        """the LSCTs you rode from the plan into `acts` (COLUMNS)."""
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
        for i in rows:                                 # one every recovery week: a handful
            r = cls.of_ride(proj, acts.iloc[i], st.workout(linked[sid.iloc[i]]), streams.get(sid.iloc[i]))
            if r is None:
                continue
            vals = r.values()
            acts.loc[acts.index[i], list(vals)] = [np.nan if v is None else v for v in vals.values()]
            done += 1
        progress(f"LSCTs: {done} from the plan's LSCT sessions")
        return acts
