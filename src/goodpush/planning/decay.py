"""Interval levels fade without the work (detraining): `LevelDecay`.

The rule, the same for every rider (codebase constants; a rider's history only tests them, `replay`):
after GRACE_DAYS without a type's work its level drops one step, then one more every STEP_DAYS, down to the
first level. Aerobic and threshold adaptations fade noticeably within 2-4 weeks without the stimulus (Coyle et
al. 1984; Mujika & Padilla 2000); a step is about 10-20 % more work, so a step per 10 days is gentle.

A type's work: its intervals in your rides (models.intervals.IntervalLog, zone = the type) or a planned session
of it you did. Tempo, sweet spot and threshold also have a steady format (the first in LADDERS) that needs long
steady efforts: over-unders are threshold work but not 20 minutes at threshold. Its own clock counts only
SUSTAINED efforts (at least SUSTAINED_S long, power changing at most once every PIECE_S, even within STEADY_CV)
in its zone or a higher one of these three; without them the steady format falls behind the type's level
(PlanState "sustained_lag"), and each steady session on target makes up a step on top of the usual one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .workouts import LADDERS

GRACE_DAYS = 14                 # without the work this long: the first step down
STEP_DAYS = 10                  # then one step every this many days
SUSTAINED_S = 300               # a steady effort: at least 5 min ...
PIECE_S = 120                   # ... with at most one change of power every 2 min (over-unders change every 1-2)
STEADY_CV = 0.10                # ... and its power this even (sd / mean): a ramp test's climbing steps are not
SUSTAINED_ORDER = ("tempo", "sweetspot", "threshold")      # higher steady work keeps the lower ones too
DECAYS = tuple(k for k in LADDERS if k != "durability")    # the long ride's finish: a small add-on, no clock


class LevelDecay:
    """When each type's work was last done, and how many steps its level has lost since."""

    @staticmethod
    def steps(gap_days: int) -> int:
        """steps down after `gap_days` without the work."""
        return 0 if gap_days < GRACE_DAYS else (int(gap_days) - GRACE_DAYS) // STEP_DAYS + 1

    @classmethod
    def tick(cls, clock: dict, kind: str, days, today) -> int:
        """The steps due `today` for `kind` on `clock` ({type: day}, kept in the plan state), its work last done
        on the latest of `days` (None: not seen) and the clock's day; the clock moves on by the steps taken, so
        asking again the same day takes none. No work ever seen: none."""
        seen = [pd.Timestamp(d) for d in (*days, clock.get(kind)) if d is not None]
        if not seen:
            return 0
        last = max(seen)
        n = cls.steps((pd.Timestamp(today) - last).days)
        clock[kind] = str((last + pd.Timedelta(days=n * STEP_DAYS)).date())
        return n

    @staticmethod
    def steady(kind: str) -> str | None:
        """the type's steady format (needs sustained efforts), None for the others."""
        return next(iter(LADDERS[kind])) if kind in SUSTAINED_ORDER else None

    @staticmethod
    def sustained(iv: pd.DataFrame) -> np.ndarray:
        """which intervals are steady efforts (long, power changing rarely)."""
        secs = iv["seconds"].to_numpy(float)
        return (secs >= SUSTAINED_S) & (secs / np.maximum(iv["steps"].to_numpy(float), 1) >= PIECE_S) \
            & (iv["cv"].to_numpy(float) <= STEADY_CV)

    @classmethod
    def last_work(cls, iv: pd.DataFrame | None, today) -> tuple[dict, dict]:
        """({type: last day of its work}, {type: last day of its sustained work}) from the intervals table,
        up to `today`. Tempo, sweet spot and threshold: work in a higher one of the three counts too."""
        if iv is None or not len(iv):
            return {}, {}
        iv = iv[pd.to_datetime(iv["date"]).dt.normalize() <= pd.Timestamp(today)]
        day = pd.to_datetime(iv["date"]).dt.normalize()
        last = day.groupby(iv["zone"]).max()
        rank = iv["zone"].map({z: i for i, z in enumerate(SUSTAINED_ORDER)})
        held = cls.sustained(iv)
        work = {k: last[k] for k in DECAYS if k in last.index and k not in SUSTAINED_ORDER}
        steady = {}
        for i, k in enumerate(SUSTAINED_ORDER):
            up = (rank >= i).to_numpy()
            if up.any():
                work[k] = day[up].max()
            if (up & held).any():
                steady[k] = day[up & held].max()
        return work, steady

    @classmethod
    def replay(cls, iv: pd.DataFrame) -> pd.DataFrame:
        """Each return to sustained work after GRACE_DAYS or more without it, per steady type: the gap, the steps
        the rule takes off, and the sustained minutes in zone of the session back vs the median of the last
        three sessions before the gap (`ratio`). What a rider then rode is their choice, not their limit: a test
        of the rule's direction and size, not a fit."""
        held = iv[cls.sustained(iv)].assign(day=lambda x: pd.to_datetime(x["date"]).dt.normalize())
        rank = held["zone"].map({z: i for i, z in enumerate(SUSTAINED_ORDER)})
        rows = []
        for i, k in enumerate(SUSTAINED_ORDER):                    # three types; the sessions are vectorised
            mine = held[rank.to_numpy() == i]
            per = mine.groupby("day")["seconds"].sum() / 60
            kept = pd.Series(held["day"][rank.to_numpy() >= i].unique()).sort_values().reset_index(drop=True)
            if len(per) < 4:
                continue
            gaps = kept.diff().dt.days
            back = kept[gaps >= GRACE_DAYS]
            back = back[back.isin(per.index)]
            before = per.rolling(3).median().shift(1)
            g = gaps[back.index].to_numpy()
            rows.append(pd.DataFrame({"type": k, "day": back.to_numpy(), "gap_days": g,
                                      "steps": [cls.steps(d) for d in g],
                                      "minutes": per[back].to_numpy(), "before": before[back].to_numpy()}))
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["type", "day", "gap_days", "steps", "minutes", "before"])
        return out.dropna().assign(ratio=lambda x: x["minutes"] / x["before"])
