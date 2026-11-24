"""Variety for interval sessions: the same work as a ladder level, in another shape.

A level of a format with one set of reps of 2 min or more (threshold 3 x 10 min, VO2max 5 x 4 min, ...)
can be ridden as a pyramid, descending or ascending reps, broken reps, reps that ramp in, reps that open
with a surge, or reps that alternate low and high cadence. Every pattern keeps the level's time on
target and its band (rests keep the rep's work : rest ratio), so the session's load, your level and its
scoring stay as they were. Warm-ups vary too: steady, a ramp, or with spin-ups.

Choices are seeded by the date and the type, so rebuilding the plan keeps a day's session. The patterns of
a type take turns in a shuffled order (each once per cycle), so the same shape doesn't come round too soon.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .steps import EASY_OFF, WORK, Repeat, Step

# per interval type: the patterns it may use, and the shortest rep (min) each needs
PATTERNS = {
    "threshold": ("straight", "pyramid", "descending", "broken", "ramp-in", "cadence", "hard starts"),
    "vo2max": ("straight", "pyramid", "descending", "ascending", "hard starts"),
    "sweetspot": ("straight", "descending", "broken", "ramp-in", "cadence"),
    "tempo": ("straight", "ramp-in", "cadence", "broken"),
}
SHORTEST = {"straight": 0, "pyramid": 2, "descending": 2, "ascending": 2, "broken": 8, "ramp-in": 5,
            "cadence": 3, "hard starts": 3}
FEWEST = {"pyramid": 3, "descending": 2, "ascending": 2, "cadence": 2}     # reps a pattern needs
ABOUT = {
    "pyramid": "Reps build up and come back down: the same time on target, the longest in the middle.",
    "descending": "Longest rep first, while you're fresh; each one shorter, so you can hold the power to the end.",
    "ascending": "Reps get longer: start sharp, finish with the longest effort on tired legs.",
    "broken": "Each rep split by 30 s easy: the same time at threshold, more of it at good quality.",
    "ramp-in": "Each rep ramps up over its first minute: settle in instead of jumping on the pedals.",
    "cadence": "Reps alternate at 70 and 100 rpm: force, then leg speed, at the same power.",
    "hard starts": "Each rep opens with 30 s well above target, then settles: holding power after a surge.",
}
DURATION_STEP = 0.5          # rep lengths of pyramids and ladders round to this many minutes
SURGE = 1.12                 # hard starts: the opening 30 s at this x the band
# step-down: each rep in three steps (share of the rep, x the band); the band is the middle step. Hard first to
# raise oxygen uptake fast, then lower to hold it there (Lisbôa et al. 2015, J Strength Cond Res 29:2434)
STEP_DOWN = ((.3, 1.10), (.3, 1.0), (.4, .93))
CADENCES = (70, 100)


class IntervalVariety:
    """Shapes for a session's main set; WorkoutLibrary.intervals asks for one."""

    @staticmethod
    def seed(date, *keys) -> int:
        """a stable seed for this day and these keys (the same on every rebuild)."""
        return pd.Timestamp(date).toordinal() * 1009 + zlib.crc32("|".join(map(str, keys)).encode())

    @staticmethod
    def turn(date, n: int) -> int:
        """which of `n` options a session on `date` takes: one more each week, and from Thursday half a cycle
        on, so two sessions of a type in a week differ."""
        d = pd.Timestamp(date)
        return (d.toordinal() // 7 + (d.weekday() >= 3) * ((n + 1) // 2)) % n

    @classmethod
    def pattern(cls, date, kind: str, on: float, reps: int) -> str:
        """this week's pattern for `kind`: the next in its shuffled cycle, if it fits `reps` reps of `on` min."""
        options = PATTERNS.get(kind, ("straight",))
        n = len(options)
        order = np.random.default_rng(zlib.crc32(kind.encode()) + pd.Timestamp(date).toordinal() // 7 // n).permutation(n)
        name = options[order[cls.turn(date, n)]]
        return name if on >= SHORTEST[name] and reps >= FEWEST.get(name, 1) else "straight"

    @staticmethod
    def lengths(total: float, reps: int, shape: str) -> np.ndarray:
        """rep lengths (min) for `reps` reps adding up to about `total`: pyramid, descending or ascending."""
        i = np.arange(reps) / max(reps - 1, 1)
        w = {"pyramid": 1 + .5 * (1 - np.abs(2 * i - 1)), "descending": 1.25 - .5 * i, "ascending": .75 + .5 * i}[shape]
        m = np.round(total * w / w.sum() / DURATION_STEP) * DURATION_STEP
        return np.maximum(m, 1.0)

    @staticmethod
    def with_rest(st: Step, on: float, off: float, off_name: str, off_band: tuple[float, float]) -> list[Step]:
        """a rep and its rest, the rest in the level's work : rest ratio (none when the level has no rest)."""
        if not off:
            return [st]
        return [st, Step(off_name, off * st.minutes / on, *off_band, "rest" if off_band[0] < WORK else "active")]

    @classmethod
    def main_set(cls, pattern: str, on_name: str, off_name: str, reps: int, on: float, off: float,
                 band: tuple[float, float], off_band: tuple[float, float]) -> tuple[list, str]:
        """the main set in `pattern`, and the title's description of it ("" = the format's own)."""
        lo, hi = band
        rest = (on, off, off_name, off_band)
        if pattern in ("pyramid", "descending", "ascending"):
            m = cls.lengths(reps * on, reps, pattern)
            steps = [x for k in m for x in cls.with_rest(Step(on_name, float(k), lo, hi), *rest)]
            return steps, f"{pattern} " + "-".join(f"{k:g}" for k in m) + " min"
        if pattern == "cadence":
            low, high = (cls.with_rest(Step(on_name, on, lo, hi, cadence=c, cue=f"{c} rpm"), *rest) for c in CADENCES)
            return (low + high) * (reps // 2) + (low if reps % 2 else []), "cadence 70 / 100"
        if pattern == "broken":
            rep = [Step(on_name, on / 2, lo, hi), Step("Easy", .5, *EASY_OFF, "rest"), Step(on_name, on / 2, lo, hi)]
        elif pattern == "ramp-in":
            rep = [Step(f"{on_name} (ramp in)", 1, lo - .15, hi - .15, lo_end=lo, hi_end=hi), Step(on_name, on - 1, lo, hi)]
        elif pattern == "hard starts":
            rep = [Step("Surge", .5, lo * SURGE, hi * SURGE, cue="Hard start, then settle"), Step(on_name, on - .5, lo, hi)]
        elif pattern == "step-down":
            rep = [Step(on_name, on * f, lo * x, hi * x, cue="Step down" if k else "Hard start")
                   for k, (f, x) in enumerate(STEP_DOWN)]
        else:
            rep = [Step(on_name, on, lo, hi)]
        last = cls.with_rest(Step(on_name, on, lo, hi), *rest)[1:]          # the rest after each rep
        return [Repeat(reps, rep + last)], "" if pattern == "straight" else pattern

    @staticmethod
    def warm_up(date, kind: str) -> list[Step]:
        """15 min: steady, a ramp, or with three spin-ups (by date)."""
        pick = np.random.default_rng(IntervalVariety.seed(date, kind, "warm up")).integers(3)
        if pick == 1:
            return [Step("Warm up", 12, .45, .50, "warmup", lo_end=.68, hi_end=.72), Step("Z2", 3, .55, .65)]
        if pick == 2:
            return [Step("Warm up", 10, .50, .65, "warmup"),
                    Repeat(3, [Step("Spin-up", .5, .70, .75, cadence=110, cue="110+ rpm, smooth"),
                               Step("Easy", .5, .50, .55, "rest")]),
                    Step("Z2", 2, .55, .65)]
        return [Step("Warm up", 15, .50, .65, "warmup")]
