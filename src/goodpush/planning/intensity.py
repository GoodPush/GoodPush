"""Your watts for a session's targets: textbook %FTP mapped through your own thresholds.

The workout library writes targets as fractions of FTP (endurance 0.55-0.65, VO2max 1.08-1.15, ...),
which assumes everyone's LT1 is 75 % of FTP, MAP 120 % and so on. Here each fraction becomes watts by
straight lines between your thresholds on the session's day (the physio model's forecast for that day,
or today's values):

    %FTP    0     0.75   1.00   1.20         1.50                 2.50
    watts   0     LT1    CP     MAP (5 min)  REPS_1MIN x 1-min    SPRINT x Pmax

So endurance stays below your LT1 wherever it is, threshold sits at your CP, VO2max between CP and
your MAP, repeated 1-min efforts at 80 % of what you could do all-out for a minute. Sessions are stored
in textbook %FTP (accepted versions don't churn as the thresholds move); this map turns them into
watts wherever watts are shown or exported. Forecasts change as the model updates with new rides.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .steps import map_steps

FRAC = np.array([0.0, 0.75, 1.0, 1.2, 1.5, 2.5])
REPS_1MIN = 0.80          # repeated 1-min efforts: share of an all-out minute
SPRINT = 0.90             # sprints: share of Pmax
FILE = "intensity_map.json"
TESTS = ("test", "ramp")


class IntensityMap:
    def __init__(self, days: pd.DatetimeIndex, knots: np.ndarray, ftp: float):
        """knots (n_days, len(FRAC)): watts at each FRAC point per day; days sorted."""
        self.days = pd.DatetimeIndex(days)
        self.knots = np.maximum.accumulate(np.asarray(knots, float), axis=1)    # never decreasing
        self.ftp = float(ftp)

    @classmethod
    def from_states(cls, model, days, states, ftp: float) -> IntensityMap:
        """from physio states (n_days, 5) on `days` (model.capacity / PowerCurve give the thresholds)."""
        return cls(days, cls.knots_of(model.capacity(np.asarray(states, float))), ftp)

    @staticmethod
    def knots_of(c) -> np.ndarray:
        """watts at the FRAC points (..., 6) from capacities: lt1, cp, map, w_prime, pmax, fade (arrays or
        numbers; a dict or a physio_states row)."""
        from ..physio import PowerCurve
        cp, lt1, mp, pmax = (np.asarray(c[k], float) for k in ("cp", "lt1", "map", "pmax"))
        p60 = np.asarray(PowerCurve.power(60.0, cp, np.asarray(c["w_prime"], float), pmax, np.asarray(c["fade"], float)))
        return np.stack([np.zeros_like(cp), lt1, cp, mp, REPS_1MIN * p60, SPRINT * pmax], axis=-1)

    @classmethod
    def history(cls, processed: Path, ftp: float) -> tuple[IntensityMap, np.ndarray] | None:
        """the map for every fitted past day (processed/physio_states.parquet) and each day's W' (J);
        None without a fit."""
        from ..models.physio import PhysioRun
        if processed is None:
            return None
        try:
            st = pd.read_parquet(Path(processed) / PhysioRun.STATES, columns=["cp", "lt1", "map", "w_prime", "pmax", "fade"])
        except (OSError, ValueError, KeyError):
            return None
        return (cls(st.index, cls.knots_of(st), ftp), st["w_prime"].to_numpy(float)) if len(st) else None

    @classmethod
    def for_day(cls, processed: Path, date, ftp: float) -> tuple[np.ndarray, float | None]:
        """(knots, W' in J) on a past day from the fitted history (the nearest day it has); without a
        fit the textbook map and no W'."""
        h = cls.history(processed, ftp)
        if h is None:
            return cls.ftp_knots(ftp), None
        m, w_prime = h
        i = m.index(date)
        return m.knots[i], float(w_prime[i])

    @staticmethod
    def ftp_knots(ftp: float) -> np.ndarray:
        """the textbook map (FRAC x FTP): without a physio fit."""
        return FRAC * float(ftp)

    def index(self, date) -> int:
        """the row of a day (the nearest the map covers)."""
        return int(np.clip(self.days.searchsorted(pd.Timestamp(date).normalize()), 0, len(self.days) - 1))

    def row(self, date) -> np.ndarray:
        """that day's knots."""
        return self.knots[self.index(date)]

    def watts(self, date, frac) -> np.ndarray:
        return np.interp(np.asarray(frac, float), FRAC, self.row(date))

    def apply(self, w):
        """the session with its targets as your watts, written back as fractions of FTP (for exports).
        Tests keep their protocol (the ramp's even steps, the 20-min test's %FTP pacing)."""
        if not w.steps or w.type in TESTS:
            return w
        k = self.row(w.date)
        steps = map_steps(w.steps, lambda s: s.mapped(lambda x: round(float(np.interp(x, FRAC, k)) / self.ftp, 3)))
        return replace(w, steps=steps, ftp_ratio=self.ftp_on(w.date) / self.ftp)

    def ftp_on(self, date) -> float:
        """the FTP you'll have that day: today's, moved with the forecast CP (TSS on a future day is
        relative to it, as your past rides are to theirs)."""
        return self.ftp * self.row(date)[2] / self.knots[0][2]

    def today(self) -> dict:
        """today's anchors in watts (for the app)."""
        k = self.knots[0]
        return {"lt1": round(k[1]), "cp": round(k[2]), "map": round(k[3]), "reps_1min": round(k[4]),
                "sprint": round(k[5]), "from": str(self.days[0].date())}

    def save(self, processed: Path) -> None:
        Path(processed, FILE).write_text(json.dumps({"days": [str(d.date()) for d in self.days],
                                                     "knots": np.round(self.knots, 1).tolist(), "ftp": self.ftp}))

    @classmethod
    def load(cls, processed: Path) -> IntensityMap | None:
        try:
            js = json.loads(Path(processed, FILE).read_text())
            return cls(pd.DatetimeIndex(js["days"]), np.array(js["knots"]), js["ftp"]) if js["days"] else None
        except (OSError, ValueError, KeyError):
            return None
