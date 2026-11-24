"""A workout's parts: `Step` (power as a fraction of FTP; a ramp, cadence, cue, a mode it suits) and
`Repeat` (reps of Steps and Repeats, so sets of intervals nest), with the walks over them."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

EASY_OFF = (.34, .36)          # recovery between efforts and sets: zone 1 (45-55 % left no recovery in a 30/15's
                               # 15 s: the efforts could not be finished)
WORK = .76                     # steps at or above this (fraction of FTP) are "work"


@dataclass
class Step:
    name: str
    minutes: float
    lo: float              # fraction of FTP (a ramp: at its start)
    hi: float
    kind: str = "active"   # warmup | active | rest | cooldown
    lo_end: float | None = None   # a ramp: the band at its end, moving evenly from lo / hi
    hi_end: float | None = None
    cadence: int | None = None    # target rpm
    cue: str = ""                 # shown when the step starts
    mode: str = ""                # a trainer mode it asks for (resistance | sim): the trainer leaves ERG for it and
                                  # goes back after (ride.session.WorkoutRun.follow_step); other modes stay
    hr: float | None = None       # a heart-rate target (fraction of max HR) you hold (resistance mode);
                                  # lo / hi are only a guide

    @property
    def ramp(self) -> bool:
        return self.lo_end is not None

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2

    @property
    def mid_end(self) -> float:
        return (self.lo_end + self.hi_end) / 2 if self.ramp else self.mid

    def mean_sq(self) -> float:
        """mean of the squared target over the step (TSS): (a² + ab + b²) / 3 along a ramp."""
        a, b = self.mid, self.mid_end
        return (a * a + a * b + b * b) / 3

    def pieces(self, minutes: float = 1.0) -> list[Step]:
        """a ramp as even steps about `minutes` long, for what can't ramp (FIT workouts, scoring); else itself."""
        if not self.ramp:
            return [self]
        n = max(int(round(self.minutes / minutes)), 2)
        f = (np.arange(n) + .5) / n
        lo, hi = self.lo + (self.lo_end - self.lo) * f, self.hi + (self.hi_end - self.hi) * f
        return [Step(self.name, self.minutes / n, float(l), float(h), self.kind, cadence=self.cadence,
                     cue=self.cue if i == 0 else "", mode=self.mode, hr=self.hr) for i, (l, h) in enumerate(zip(lo, hi))]

    def mapped(self, fn) -> Step:
        """its targets through fn (scaled, or as your watts)."""
        return replace(self, lo=fn(self.lo), hi=fn(self.hi), lo_end=None if self.lo_end is None else fn(self.lo_end),
                       hi_end=None if self.hi_end is None else fn(self.hi_end))

    def to_dict(self) -> dict:
        d = {"name": self.name, "minutes": self.minutes, "lo": self.lo, "hi": self.hi, "kind": self.kind}
        d.update({k: getattr(self, k) for k in STEP_EXTRAS if getattr(self, k) not in (None, "")})
        return d


STEP_EXTRAS = ("lo_end", "hi_end", "cadence", "cue", "mode", "hr")


@dataclass
class Repeat:
    """`reps` times through `steps`: Steps and Repeats (a set of micro intervals nests in the sets)."""
    reps: int
    steps: list

    def to_dict(self) -> dict:
        return {"reps": self.reps, "steps": [x.to_dict() for x in self.steps]}


def flat(items: list, split_ramps: bool = False) -> list[Step]:
    """Steps and Repeats written out in riding order (ramps as 1-min pieces with split_ramps)."""
    out = []
    for x in items:                      # a workout's few dozen items
        out += flat(x.steps, split_ramps) * x.reps if isinstance(x, Repeat) else \
            (x.pieces() if split_ramps else [x])
    return out


def map_steps(items: list, fn) -> list:
    """the same structure with every Step through fn."""
    return [replace(x, steps=map_steps(x.steps, fn)) if isinstance(x, Repeat) else fn(x) for x in items]


def items_from_dicts(rows: list[dict]) -> list:
    return [Repeat(int(d["reps"]), items_from_dicts(d["steps"])) if "reps" in d else Step(**d) for d in rows]


def flat_dicts(rows: list[dict]) -> list[dict]:
    """a saved workout's steps written out (its JSON: nested {reps, steps})."""
    out = []
    for d in rows:
        out += flat_dicts(d["steps"]) * int(d["reps"]) if "reps" in d else [d]
    return out


def nest_repeat(steps: list, repeat) -> list:
    """an older saved workout's one repeated block (first, last, count) as a Repeat."""
    if not repeat:
        return steps
    a, b, n = repeat
    return steps[:a] + [Repeat(int(n), steps[a:b + 1])] + steps[b + 1:]
