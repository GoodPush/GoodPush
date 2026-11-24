"""What a planned session is: a `Workout` of `Step`s (power as a fraction of FTP), and the
`WorkoutLibrary` that builds every kind of session the plan uses."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from functools import partial

import numpy as np
import pandas as pd

from .variety import ABOUT, IntervalVariety as IV
from .steps import EASY_OFF, WORK, Repeat, Step, flat, flat_dicts, items_from_dicts, map_steps, nest_repeat

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# interval sessions: per type its formats (they take turns week by week), each
# (on-step name, off-step name, off power, what it trains, progression with one step per level).
# A step: (sets, reps per set, minutes on, minutes off, on low, on high (fraction of FTP), minutes between sets).
# The off power is easy recovery, or the "under" of over-unders / the sweet spot between bursts.
# All formats of a type have the same number of levels: your level carries over between them. Tempo, sweet spot
# and threshold start at ~20-25 min of work (2x12 / 3x8 / 4x5): where a rider comes back after a break. Levels fade
# without the work (planning.decay.LevelDecay); a type's first (steady) format fades on its own without long
# steady efforts.
# Progressive overload without crushing you: each format adds reps only up to its proven size (Seiler's
# 4 x 8, Rønnestad's 3 x 13 x 30/15, ~25 min of VO2max reps, ~50 min at threshold); from there its levels
# shorten the rests, and only the last adds ~1 % power. Getting stronger raises every target with your
# thresholds, so the same session keeps getting harder in watts.
# Easier / harder (`Adjuster`, your choice per format, PlanState "format_offsets"): the same rule as an offset on
# top of the level, below the first level and above the last too. Structure first, power last, so the power
# target holds as long as the structure can give way. Reps and sets are one ladder: harder = a rep, a rep, then a
# set more with a rep fewer (3x3, 3x4, 3x5, 4x4, 4x5, ...: every step more work) up to one set more than the
# format's most, then shorter rests - never more power (the level's targets stay); easier = the same ladder down
# first (3x5, 3x4, 3x3, 2x4: two reps fewer, then a set fewer; 2026-11-18, the rider: longer recovery first only
# stretched the rests), then longer recovery (micro intervals' off: 30/15, 30/30, 30/45), shorter reps, power.
LADDERS = {
    "threshold": {
        "Threshold": ("Threshold", "Recover", EASY_OFF, "Long steady efforts just under FTP.",
                      [(1, 4, 5, 3, .95, 1.0, 0), (1, 3, 8, 4, .95, 1.0, 0),
                       (1, 3, 10, 5, .95, 1.0, 0), (1, 3, 12, 5, .95, 1.0, 0), (1, 2, 20, 8, .94, .99, 0),
                       (1, 3, 15, 5, .95, 1.0, 0), (1, 2, 25, 8, .94, .99, 0), (1, 2, 25, 6, .94, .99, 0),
                       (1, 2, 25, 5, .95, 1.0, 0)]),
        "Over-unders": ("Over", "Under", (.90, .95),
                        "1 min over FTP, then hold just under it while you clear the lactate: threshold work "
                        "with surges, as in a race.",
                        # never more than 5 overs a set: progress by sets and shorter rests, not longer sets
                        [(2, 3, 1, 2, 1.05, 1.10, 5), (2, 4, 1, 2, 1.05, 1.10, 5),
                         (3, 3, 1, 2, 1.05, 1.10, 5), (3, 4, 1, 2, 1.05, 1.10, 5), (3, 4, 1, 2, 1.05, 1.10, 4),
                         (3, 5, 1, 2, 1.05, 1.10, 5), (3, 5, 1, 2, 1.05, 1.10, 4), (3, 5, 1, 2, 1.05, 1.10, 3),
                         (3, 5, 1, 2, 1.06, 1.11, 3)]),
    },
    "vo2max": {
        "VO2max": ("VO2max", "Recover", EASY_OFF, "3-5 min efforts at the top of your aerobic power.",
                   [(1, 5, 3, 3, 1.08, 1.15, 0), (1, 5, 4, 4, 1.08, 1.14, 0), (1, 6, 4, 4, 1.08, 1.14, 0),
                    (1, 5, 5, 5, 1.06, 1.12, 0), (1, 5, 5, 4, 1.07, 1.13, 0)]),
        # supra-threshold reps of at most 6 min with a long rest (coaching practice: 5 x 6 min with 4 min easy);
        # 8-min reps above FTP with 2-3 min rests are hard to finish at a power you can hold. Always straight reps
        # (FIXED_PATTERNS): a pyramid or descending set would stretch a rep past 6 min
        "6-min": ("Hard", "Recover", EASY_OFF,
                  "6-min efforts just above FTP with a long rest: hard but even to the last one.",
                  [(1, 2, 6, 4, 1.02, 1.05, 0), (1, 3, 6, 4, 1.02, 1.05, 0), (1, 4, 6, 4, 1.02, 1.05, 0),
                   (1, 5, 6, 4, 1.02, 1.05, 0), (1, 5, 6, 3.5, 1.03, 1.06, 0)]),
        # VO2max over-unders: short overs above threshold, then unders well below it, at the bottom of tempo, so
        # each over starts with some lactate cleared. Overs high with unders near threshold make every set one
        # long effort above FTP - too hard to repeat. Never more than 5 overs a set
        "Over-unders 1:1": ("Over", "Under", (.76, .80),
                            "1 min over FTP, 1 min at upper tempo, no rest between: repeated surges above "
                            "threshold, clearing lactate in the unders without stopping.",
                            [(3, 4, 1, 1, 1.05, 1.10, 5), (3, 5, 1, 1, 1.05, 1.10, 5), (4, 5, 1, 1, 1.05, 1.10, 5),
                             (4, 5, 1, 1, 1.05, 1.10, 4), (4, 5, 1, 1, 1.06, 1.11, 4)]),
        # each rep opens with a surge (FIXED_PATTERNS), then holds VO2max power; rest as long as the rep
        "Hard-start VO2max": ("VO2max", "Recover", EASY_OFF,
                              "3.5-min efforts that open with 30 s hard, then settle at VO2max power: oxygen "
                              "uptake rises fast and stays high.",
                              [(1, 4, 3.5, 3.5, 1.10, 1.15, 0), (1, 5, 3.5, 3.5, 1.10, 1.15, 0),
                               (1, 6, 3.5, 3.5, 1.10, 1.15, 0), (1, 6, 3.5, 3, 1.10, 1.15, 0),
                               (1, 6, 3.5, 3, 1.11, 1.16, 0)]),
        # 5-min reps in three steps down (FIXED_PATTERNS; variety.STEP_DOWN): about 115 / 105 / 98 % at the
        # first level. Longer near VO2max for the same cost than a rep at constant power; never more than 5 reps
        "Step-down VO2max": ("VO2max", "Recover", EASY_OFF,
                             "5-min efforts that start hard and step down twice: oxygen uptake rises fast, "
                             "then the lower steps hold it high for longer than a steady effort could.",
                             [(1, 3, 5, 3, 1.03, 1.07, 0), (1, 4, 5, 3, 1.03, 1.07, 0),
                              (1, 5, 5, 3, 1.03, 1.07, 0), (1, 5, 5, 2.5, 1.05, 1.09, 0),
                              (1, 5, 5, 2.5, 1.07, 1.11, 0)]),
        "30/15s": ("On", "Off", EASY_OFF,
                   "Rønnestad micro intervals: the short rests keep oxygen uptake high for longer than long "
                   "intervals do. Ride each 30 s hard but even, not a sprint.",
                   [(3, 10, .5, .25, 1.18, 1.28, 3), (3, 11, .5, .25, 1.18, 1.28, 3), (3, 12, .5, .25, 1.18, 1.28, 3),
                    (3, 13, .5, .25, 1.18, 1.28, 3), (3, 13, .5, .25, 1.19, 1.29, 3)]),
        "40/20s": ("On", "Off", EASY_OFF, "Micro intervals, a bit longer on: aerobic power and repeated surges.",
                   [(2, 9, 2 / 3, 1 / 3, 1.12, 1.20, 4), (3, 8, 2 / 3, 1 / 3, 1.12, 1.20, 4),
                    (3, 9, 2 / 3, 1 / 3, 1.12, 1.20, 4), (3, 10, 2 / 3, 1 / 3, 1.12, 1.20, 4),
                    (3, 10, 2 / 3, 1 / 3, 1.13, 1.21, 3)]),
    },
    "sweetspot": {
        "Sweet spot": ("Sweet spot", "Recover", EASY_OFF, "A big aerobic stimulus at manageable fatigue.",
                       [(1, 3, 8, 3, .88, .93, 0), (1, 3, 10, 4, .88, .93, 0),
                        (1, 3, 12, 4, .88, .93, 0), (1, 2, 20, 5, .88, .93, 0), (1, 3, 15, 5, .88, .93, 0),
                        (1, 3, 20, 5, .88, .93, 0), (1, 2, 30, 6, .88, .92, 0)]),
        "Sweet spot bursts": ("Burst", "Sweet spot", (.88, .93),
                              "Sweet spot with a 15-s burst every 5 min: holding the pace after a surge, "
                              "as in a group ride.",
                              [(2, 2, .25, 4.75, 1.40, 1.60, 4), (1, 5, .25, 4.75, 1.40, 1.60, 0),
                               (3, 2, .25, 4.75, 1.40, 1.60, 4), (2, 4, .25, 4.75, 1.40, 1.60, 5),
                               (3, 3, .25, 4.75, 1.40, 1.60, 5), (3, 4, .25, 4.75, 1.40, 1.60, 5),
                               (2, 6, .25, 4.75, 1.40, 1.60, 6)]),
    },
    "tempo": {
        "Tempo": ("Tempo", "Recover", EASY_OFF, "Steady, comfortably hard aerobic work.",
                  [(1, 2, 12, 4, .76, .85, 0), (1, 3, 10, 4, .76, .85, 0), (1, 2, 20, 5, .76, .85, 0),
                   (1, 2, 30, 5, .76, .85, 0), (1, 1, 60, 0, .76, .85, 0)]),
        "Tempo bursts": ("Burst", "Tempo", (.76, .85),
                         "Tempo with a 15-s kick every 5 min: steady aerobic work that also keeps the legs "
                         "used to surges.",
                         [(2, 2, .25, 4.75, 1.20, 1.40, 4), (2, 3, .25, 4.75, 1.20, 1.40, 5),
                          (2, 4, .25, 4.75, 1.20, 1.40, 5), (2, 6, .25, 4.75, 1.20, 1.40, 5),
                          (1, 12, .25, 4.75, 1.20, 1.40, 0)]),
        "Tempo-plus": ("Upper tempo", "Tempo", (.76, .81),
                       "3 min at the top of tempo, 3 min lower, without a break: long aerobic work just "
                       "under sweet spot.",
                       [(2, 2, 3, 3, .86, .90, 5), (2, 3, 3, 3, .86, .90, 5),
                       (2, 4, 3, 3, .86, .90, 5), (2, 5, 3, 3, .86, .90, 5), (1, 10, 3, 3, .86, .90, 0)]),
    },
    "anaerobic": {
        "1-min efforts": ("Hard", "Easy", EASY_OFF, "Anaerobic capacity: hard 1-min efforts for attacks and "
                          "short climbs (crits, cyclocross, road races).",
                          [(1, 6, 1, 3, 1.30, 1.50, 0), (1, 8, 1, 3, 1.30, 1.50, 0), (2, 5, 1, 2, 1.30, 1.50, 5),
                           (2, 6, 1, 2, 1.30, 1.50, 5), (2, 6, 1, 2, 1.31, 1.51, 4)]),
        "30-s efforts": ("Hard", "Easy", EASY_OFF, "Short, very hard efforts: the punch to close a gap or attack.",
                         [(1, 8, .5, 2.5, 1.50, 1.70, 0), (1, 10, .5, 2.5, 1.50, 1.70, 0),
                          (2, 6, .5, 2, 1.50, 1.70, 5), (2, 7, .5, 2, 1.50, 1.70, 5), (2, 7, .5, 2, 1.51, 1.71, 4)]),
    },
    # not a session type of its own: the finish of a long ride in a loading week (WorkoutLibrary.durability;
    # the long ride keeps type "long" and carries this ladder's level). A small addition on top of the week's key
    # sessions, never a hard day of its own: 2 x 2.5 min up to 4 x 2.5 min. Holding power on tired legs
    # (Maunder et al. 2021); no trial has tested these exact sessions: coaching practice
    "durability": {
        "Durability": ("Durability", "Z2", (.55, .65),
                       "Efforts just under FTP at the end of the long ride, on tired legs: holding your power "
                       "late in a ride (durability). Eat and drink well before them.",
                       [(1, 2, 2.5, 2.5, .95, 1.0, 0), (1, 3, 2.5, 2.5, .95, 1.0, 0), (1, 4, 2.5, 2.5, .95, 1.0, 0)]),
    },
}
LEVELS = {k: len(next(iter(f.values()))[4]) for k, f in LADDERS.items()}   # levels per type
DURABILITY_AFTER_MIN = 90       # riding before the durability finish
DURABILITY_POLARIZED = (1.06, 1.12)   # training polarized: the efforts above CP (nothing between LT1 and LT2)
# formats whose title is their on / off in seconds: "30/15 3x13", "30/30 3x13" when easier (not "... × 30 s / 15 s")
FORMAT_TITLES = {"30/15s": "30/15", "40/20s": "40/20"}
# formats that always take one shape (planning/variety.py), with or without variety
FIXED_PATTERNS = {"6-min": "straight", "Hard-start VO2max": "hard starts", "Step-down VO2max": "step-down"}
# formats that narrow the Adjuster's limits beyond their ladder's: micro intervals' off never gets shorter (the
# point of the format), up to 3x as long when easier; on time, reps and sets come from the ladder (Adjuster.of)
ADJUST_LIMITS = {"30/15s": {"rest": (1.0, 3.0)}, "40/20s": {"rest": (1.0, 3.0)}}
# a format's own step for a knob (Adjuster.STEP): micro intervals' off one off-length at a time (30/15, 30/30,
# 30/45), never shorter; the rest between their sets keeps the usual step
ADJUST_STEPS = {"30/15s": {"rest": 1.0}, "40/20s": {"rest": 1.0}}
EASY = {"recovery": (.45, .55), "endurance": (.55, .65), "long": (.55, .65)}
# endurance = the lower to middle part of the zone, well below LT1 (0.75): easy rides stay easy, so the hard
# days can be hard. Every Z2 stretch (easy rides, between and after intervals) uses this band.
Z2 = EASY["endurance"]
# Variety for easy days (Scheduler.vary picks one by date): (name, shortest ride (min), tempo-like (only in
# loading weeks, never polarized), ([one repeat: (step name, minutes, lo, hi, kind)], times), about).
# Long rides put their repeat at the end. Same purpose and about the same load as the steady ride.
EASY_VARIANTS = {
    "endurance": [
        ("steady", 0, False, None),
        ("spin-ups", 45, False, ([("High cadence 110+ rpm", 1, .55, .65, "active"), ("Z2", 2, .55, .65, "active")], 6),
         "Six 1-min spin-ups at 110+ rpm, same easy power: smooth pedalling."),
        ("sprints", 60, False, ([("Sprint, all-out", 1 / 6, 1.80, 2.50, "active"), ("Z2", 5, .55, .65, "active")], 6),
         "Six 10-s sprints, 5 min easy between: top-end speed without fatigue. Resistance / slope mode, not ERG."),
        ("low cadence", 75, True, ([("Low cadence 60 rpm", 5, .76, .82, "active"), ("Z2", 5, .55, .65, "active")], 4),
         "Four 5-min efforts at 60 rpm, seated, upper tempo: on-bike strength."),
        ("surges", 60, False, ([("Surge", .5, 1.20, 1.30, "active"), ("Z2", 6, .55, .65, "active")], 5),
         "Five 30-s surges, then settle back into Z2: like a group ride."),
    ],
    "long": [
        ("steady", 0, False, None),
        ("tempo finish", 150, True, ([("Tempo finish", 30, .76, .85, "active")], 1),
         "The last half hour at tempo, on tired legs: durability."),
        ("late sweet spot", 150, True, ([("Sweet spot", 10, .88, .93, "active"), ("Z2", 5, .55, .65, "active")], 2),
         "Two 10-min sweet-spot efforts in the last hour: holding power late in a long ride."),
    ],
}
EASY_LIMITS = {"endurance": (45, 150), "long": (90, 240)}      # minutes the volume fit may use
EASY_LIMITS_RECOVERY = {"endurance": (30, 90), "long": (60, 120)}   # ... in a recovery week / phase
TESTS = ("test", "ramp")
KEY_TYPES = {"threshold", "vo2max", "sweetspot", "anaerobic", "test", "ramp", "race"}   # need fresh legs
HARD_RIDES = {"threshold", "vo2max", "anaerobic", "race", "test"}         # classifier labels
AT_THRESHOLD_UP = {"threshold", "vo2max", "anaerobic", "test", "ramp"}   # rest or a spin the day before (races: openers)
EASY_BEFORE = ("rest", "recovery")
# a week whose load is less than this many hours of endurance riding rides endurance on its easy days: a recovery
# spin only the day after a hard session or the day before one at threshold (SessionOptimizer.add_spin_rule); with
# more time on the bike spins fill in
SPIN_FROM_HOURS = 6.0
LABELS = {"threshold": "threshold", "vo2max": "VO2max", "sweetspot": "sweet spot", "tempo": "tempo",
          "anaerobic": "anaerobic ride", "race": "race", "test": "test", "ramp": "ramp test", "durability": "durability finish",
          "lsct": "LSCT"}
# the Lamberts and Lambert submaximal cycle test (LSCT; Lamberts et al. 2011, Br J Sports Med 45:797): 6 min at 60 %,
# 6 min at 80 % and 3 min at 90 % of max heart rate, then 90 s sitting still: the power each stage takes and how far
# heart rate falls in the first minute track fitness and fatigue. Ridden in resistance mode: you hold the heart rate.
# Per stage: (name, minutes, share of max HR, guide band (fraction of FTP): a rough guide, not a target)
LSCT_STAGES = (("LSCT 60 % HRmax", 6, .60, (.45, .55)), ("LSCT 80 % HRmax", 6, .80, (.70, .76)),
               ("LSCT 90 % HRmax", 3, .90, (.86, .92)))
LSCT_STOP_MIN = 1.5                    # sitting still after the last stage (a step at 0 W: its clock runs regardless)
LSCT_STOP = "Stop, sit still"          # that step's name (models.lsct finds it so)
RAMP_STEPS = np.round(.50 + .06 * np.arange(16), 2)               # 50 % -> 140 % FTP, 2 min each
RAMP_NAME = "Ramp"                     # the ramp test's steps are named "Ramp 50 %" ... (models.ramp finds them so)
# timed all-out efforts opening a session (WorkoutLibrary.max_effort; Scheduler.max_efforts): by length, the step's
# name (ALL_OUT_NAME first: models.all_out finds it so), a guide band (fraction of FTP; ride by feel), minutes easy
# after it, and what it is for
ALL_OUT_NAME = "All-out"
PRIMER_NAME = "Primer"                 # before it, ERG off (resistance mode): find your gear at about half effort
MAX_EFFORTS = {
    300: (f"{ALL_OUT_NAME} 5 min", 1.15, 1.35, 10,
          "Starts with a 5-minute all-out effort: pace it so the last minute is your hardest; your heart rate should "
          "be at its maximum by the end. That measures your 5-minute power (MAP) instead of inferring it."),
    180: (f"{ALL_OUT_NAME} 3 min", 1.25, 1.50, 8,
          "Starts with a 3-minute all-out effort: hard from the start, hold on to the end. That measures how much you "
          "have above critical power (W') instead of inferring it."),
    60: (f"{ALL_OUT_NAME} 1 min", 1.60, 2.00, 6,
         "Starts with a 1-minute all-out effort: go from the start and hold on; your heart rate peaks just after. "
         "That measures your 1-minute power instead of inferring it."),
}
MAX_EFFORT_OF = {"vo2max": 300, "anaerobic": 60}   # which max effort opens which session type (push weeks)
MAX_EFFORT_EVERY_WEEKS = 4             # not again sooner than this after the last one your heart rate confirmed
# [plan] test_efforts: all-out efforts in ordinary loading weeks too (TestEfforts), each length at most this often
# (weeks) and before these session types (the long ones tire you for the set after them). No sprint: Pmax stays
# inferred - a sprint varies too much with the day to be worth its own effort (2026-11-22)
TEST_EFFORT_WEEKS = {60: 4, 180: 6, 300: 8}
TEST_EFFORT_BEFORE = {60: {"vo2max", "anaerobic"}, 180: {"vo2max", "threshold"}, 300: {"vo2max"}}
# race intensity (fraction of FTP) and typical length by kind
RACE = {"road": (.75, .85, 180), "crit": (.85, .95, 60), "tt": (.95, 1.0, 60), "mtb": (.80, .90, 120),
        "gravel": (.70, .80, 240), "cx": (.90, 1.0, 60), "hillclimb": (.95, 1.05, 45),
        "fondo": (.65, .75, 300), "other": (.75, .85, 120)}


@dataclass
class Workout:
    date: pd.Timestamp
    type: str              # rest|recovery|endurance|long|threshold|vo2max|sweetspot|tempo|test|ramp|race|openers|lsct
    minutes: float = 0
    title: str = ""
    steps: list = field(default_factory=list)     # Steps and Repeats
    level: int | None = None
    note: str = ""
    adjusted: str = ""     # why readiness changed it
    locked: bool = False   # you ticked it off or locked it: kept as it is
    own: bool = False      # your own change, not locked: the plan may replace it, but never tweaks it
    why_change: str = ""   # why the plan proposes it (makeup, moved, ...)
    key: str = ""          # "day#n" for a session you added to a day that already had one
    ftp_ratio: float = 1.0 # steps are fractions of today's FTP; the day's FTP is this x today's (TSS uses it)
    offset: int = 0        # an interval session's steps easier (< 0) / harder than its level (Adjuster), as built
    format: str = ""       # an interval session's format (a LADDERS key): moved to another day it keeps it

    def expanded(self, split_ramps: bool = False) -> list[Step]:
        """Steps with the repeats written out."""
        return flat(self.steps, split_ramps)

    def total_minutes(self) -> float:
        """the session length; math.fsum: exactly rounded, the same on every Python (sum() differs by version)"""
        return math.fsum(s.minutes for s in self.expanded())

    def tss(self) -> float:
        st = self.expanded()
        if not st:
            return 0.0
        m = np.array([s.minutes for s in st])
        f2 = np.array([s.mean_sq() for s in st]) / self.ftp_ratio ** 2
        return float((m / 60 * f2).sum() * 100)

    def to_dict(self) -> dict:
        return {"type": self.type, "title": self.title, "minutes": self.minutes, "level": self.level,
                "note": self.note, "adjusted": self.adjusted, "why_change": self.why_change,
                "steps": [x.to_dict() for x in self.steps], **({"offset": self.offset} if self.offset else {}),
                **({"format": self.format} if self.format else {})}

    @classmethod
    def from_dict(cls, key: str, d: dict) -> Workout:
        """From its JSON; `key` is the day, or "day#n" for an added session. Older copies kept one repeated
        block as "repeat": (first, last, count)."""
        steps = nest_repeat(items_from_dicts(d.get("steps", [])), d.get("repeat"))
        return cls(pd.Timestamp(key[:10]), d["type"], d.get("minutes", 0), d.get("title", ""), steps,
                   d.get("level"), d.get("note", ""), d.get("adjusted", ""), why_change=d.get("why_change", ""),
                   key=key if "#" in key else "", offset=int(d.get("offset") or 0), format=d.get("format") or "")

    @staticmethod
    def shape(d: dict) -> tuple:
        """What makes two versions of a day different: type, level and the hard parts, rep by rep."""
        st = flat_dicts(d.get("steps", []))
        if d.get("repeat"):                                   # an older copy: its repeat written out
            a, b, n = d["repeat"]
            st = st[:a] + st[a:b + 1] * n + st[b + 1:]
        hard = tuple((s["name"], round(s["minutes"]), round(s["lo"], 2)) for s in st
                     if s["kind"] == "active" and s["lo"] >= WORK)
        return d["type"], d.get("level"), hard

    @staticmethod
    def differs(old: dict, new: dict) -> bool:
        """A change worth asking about: another session, or easy volume off by > 15 min / 20 %."""
        if Workout.shape(old) != Workout.shape(new):
            return True
        return abs(new.get("minutes", 0) - old.get("minutes", 0)) > max(15, 0.2 * old.get("minutes", 0))

    def z2_band(self) -> tuple[float, float] | None:
        """(lo, hi) of its first main step: a plain easy ride's band."""
        s = next((s for s in self.expanded() if s.kind == "active"), None)
        return (s.lo, s.hi) if s else None

    def respec(self) -> Workout | None:
        """With its Z2 stretches at today's band (Z2), when any sits above it - a session saved under an
        earlier band; None when there is nothing to change."""
        def now(s: Step) -> Step:
            old = s.name == "Z2" and s.kind == "active" and not s.ramp and s.hi > Z2[1] + 1e-9
            return replace(s, lo=Z2[0], hi=Z2[1]) if old else s
        steps = map_steps(self.steps, now)
        return replace(self, steps=steps) if steps != self.steps else None

    def resized(self, minutes: float) -> Workout:
        """At a new total length: a plain easy ride is rebuilt; otherwise the longest easy part
        (outside the intervals) stretches or shrinks and the hard parts stay."""
        minutes = int(round(float(minutes) / 5) * 5)
        if self.type in EASY and len(self.steps) <= 3:            # a plain easy ride: rebuilt
            if not 20 <= minutes <= 480:
                raise ValueError("an easy ride can be 20 min to 8 h")
            return replace(WorkoutLibrary.easy(self.date, self.type, minutes, self.z2_band()),
                           why_change=self.why_change, key=self.key)
        flex = [i for i, s in enumerate(self.steps) if isinstance(s, Step) and s.kind == "active" and s.lo < WORK
                and not s.ramp and s.hr is None]
        if not flex:
            raise ValueError("this session has no easy part to lengthen or shorten")
        i = max(flex, key=lambda k: self.steps[k].minutes)
        total = self.total_minutes()
        new = self.steps[i].minutes + (minutes - total)
        if new < 5:
            raise ValueError(f"it can't be shorter than {int(total - self.steps[i].minutes + 5)} min")
        steps = list(self.steps)
        steps[i] = replace(steps[i], minutes=new)
        title = re.sub(r"\d+:\d\d h|\d+ min", WorkoutLibrary.duration(minutes), self.title, count=1) if self.type == "long" else self.title
        return replace(self, steps=steps, minutes=minutes, title=title)

    def trimmed(self, why: str) -> Workout:
        """An amber day: one rep fewer per set of the main set (while it keeps two or more) and 2 % lower work
        targets (Adjuster.TRIM; not counted as progress)."""
        w = Adjuster().move(self, Adjuster.TRIM)
        return replace(w, title=self.title + " (trimmed)", adjusted=why)


class Adjuster:
    """Moves an interval workout easier or harder by whole steps, on any step tree (Steps and Repeats, sets of
    reps): one mechanism for your easier / harder choice per format, the readiness trim and your own workouts.

    Its knobs work on the main set (the top-level Repeat with the most work; its inner Repeat when it has sets):
    volume (reps and sets as one ladder, `volume`), rests (rest steps that close a rep or a set, or stand between the
    main set's efforts),
    minutes on (a rep's steps before its rest: a shaped rep scales as a whole, its shape kept) and the power of
    every work step from the first effort to the last. Easier walks EASIER, harder HARDER (the rule above LADDERS):
    each knob gives the steps its limits allow, then the next takes over; when none is left the workout stays as
    it is (`reach` says how far it goes). An offset counts from the workout as built, so harder then easier is the
    same workout again and n steps at once are n single steps.

    Limits (`limits`, knob -> (lo, hi)): reps and sets as counts, minutes on as minutes, rests and power as factors
    on the workout as built. A knob without one takes the workout's own: reps down to half (at least 2) and never
    more, sets down to one and up one, reps down to half their length, FACTORS for rests and power."""
    EASIER = ("volume", "rest", "on", "power")       # fewer reps and sets first
    HARDER = ("volume", "rest")                       # never power: harder is more work at the same targets
    KNOBS = ("reps", "sets", "volume", "rest", "on", "power")
    TOWARD_HARDER = {"reps": 1, "sets": 1, "volume": 1, "rest": -1, "on": 1, "power": 1}   # its direction when harder
    VOLUME_BACK = 1                                   # a set more, this many reps fewer (when the reps allow)
    STEP = {"rest": .25, "on": .1, "power": .01}      # a factor knob's step: x (1 + STEP k)
    FACTORS = {"rest": (.75, 2.0), "power": (.90, 1.03)}
    TRIM = {"reps": -1, "power": -2}                  # an amber day (Workout.trimmed)
    ROUND_S = 5                                       # a changed step's length, to this many seconds

    def __init__(self, limits: dict | None = None, steps: dict | None = None):
        self.limits = dict(limits or {})
        self.steps = {**self.STEP, **(steps or {})}     # a format's own steps (ADJUST_STEPS)

    @classmethod
    def of(cls, kind: str, fmt: str, level: int) -> Adjuster:
        """the limits of a LADDERS format at `level`, from its levels of the same build (one set or several): reps
        up to its proven size (its most reps, and no more work in all than its biggest level) and down to two fewer
        than its fewest (at least 2), minutes on (a rep's work, the off too when it is work) within its range, sets
        down to one and up to one more than its most; ADJUST_LIMITS narrow them, ADJUST_STEPS its steps."""
        ladder, work_off = LADDERS[kind][fmt][4], LADDERS[kind][fmt][2][0] >= WORK
        row = ladder[int(min(max(level, 0), len(ladder) - 1))]
        rows = np.array([r for r in ladder if (r[0] > 1) == (row[0] > 1)], float)
        on = rows[:, 2] + rows[:, 3] * work_off
        few = int(rows[:, 1].min())
        most = min(int(rows[:, 1].max()),
                   int(np.floor((rows[:, 0] * rows[:, 1] * on).max() / (row[0] * (row[2] + row[3] * work_off)) + 1e-9)))
        return cls({"reps": (min(few, max(2, few - 2)), most), "sets": (1, int(rows[:, 0].max()) + 1),
                    "on": (float(on.min()), float(on.max())), **ADJUST_LIMITS.get(fmt, {})}, ADJUST_STEPS.get(fmt))

    # --- the main set ----------------------------------------------------------------------------
    @classmethod
    def work(cls, x) -> float:
        """minutes of work (active steps at or above WORK) in a Step or Repeat."""
        if isinstance(x, Repeat):
            return x.reps * sum(cls.work(y) for y in x.steps)
        return x.minutes if x.kind == "active" and x.lo >= WORK else 0.0

    @classmethod
    def parts(cls, items: list) -> tuple[int, int, int | None, int | None]:
        """(first, last): the top-level items from the first effort to the last; the main Repeat's index among
        them (None: no Repeat there) and, when it holds sets, the index of its reps' Repeat inside it."""
        w = np.array([cls.work(x) for x in items])
        hit = np.flatnonzero(w > 0)
        if not hit.size:
            return 0, -1, None, None
        a, b = int(hit[0]), int(hit[-1])
        reps = [i for i in range(a, b + 1) if isinstance(items[i], Repeat)]
        if not reps:
            return a, b, None, None
        m = reps[int(np.argmax(w[reps]))]
        inner = [j for j, y in enumerate(items[m].steps) if isinstance(y, Repeat)]
        k = inner[int(np.argmax([cls.work(items[m].steps[j]) for j in inner]))] if inner else None
        return a, b, m, k

    @staticmethod
    def closing(body: list) -> int | None:
        """index of the rest step that closes a rep or set (its last step), if it has one."""
        return len(body) - 1 if body and isinstance(body[-1], Step) and body[-1].kind == "rest" else None

    def room(self, w: Workout) -> dict[str, tuple[int, int]]:
        """steps each knob can take (down, up) in its own direction: reps / sets / volume / on / rest / power up =
        more."""
        out = {k: (0, 0) for k in self.KNOBS}
        a, b, m, k = self.parts(w.steps)
        if b < a:
            return out
        items = w.steps
        rests = any(isinstance(x, Step) and x.kind == "rest" for x in items[a:b + 1])
        out["power"] = self.factor_room("power")
        if m is not None:
            reps_rep = items[m] if k is None else items[m].steps[k]
            body = reps_rep.steps
            end = self.closing(body)
            r0, m0 = reps_rep.reps, sum(s.minutes for s in flat(body[:end]))
            rests = rests or end is not None or (k is not None and self.closing(items[m].steps) is not None)
            lo, hi = self.limits.get("reps", (min(r0, max(2, -(-r0 // 2))), r0))
            out["reps"] = (max(r0 - lo, 0), max(hi - r0, 0))
            if k is not None:
                lo, hi = self.limits.get("sets", (1, items[m].reps + 1))
                out["sets"] = (max(items[m].reps - lo, 0), max(hi - items[m].reps, 0))
            v = self.volume(w)
            out["volume"] = (v["p0"], len(v["states"]) - 1 - v["p0"])
            lo, hi = self.limits.get("on", (m0 / 2, m0))
            out["on"] = self.steps_within(lo / m0, hi / m0, self.steps["on"]) if m0 > 0 else (0, 0)
        if rests:
            out["rest"] = self.factor_room("rest")
        return out

    def volume(self, w: Workout) -> dict | None:
        """the main set's reps and sets as one ladder: {"states": [(sets, reps), ...] easiest first, "p0": the
        workout's place in it, "sets", "reps": as built}. In each set the reps go from VOLUME_BACK below the most
        to the most, then a set more starts at the bottom again; the set it starts in also reaches down to its own
        reps, one set down to the fewest; below the workout as built only states with fewer reps in all than every
        step above them (3x3 easier: 2x4, not 2x5). None without a Repeat."""
        _, _, m, k = self.parts(w.steps)
        if m is None:
            return None
        r0 = (w.steps[m] if k is None else w.steps[m].steps[k]).reps
        s0 = 1 if k is None else w.steps[m].reps
        few, most = self.limits.get("reps", (min(r0, max(2, -(-r0 // 2))), r0))
        hi = max(most, r0)
        lo = max(few, hi - self.VOLUME_BACK)
        sets_hi = 1 if k is None else max(self.limits.get("sets", (1, s0 + 1))[1], s0)
        states = [(n, r) for n in range(1, sets_hi + 1)
                  for r in range(min(few, r0) if n == 1 else min(lo, r0) if n == s0 else lo, hi + 1)]
        p0 = states.index((s0, r0))
        total = np.array([n * r for n, r in states[:p0 + 1]])
        later = np.minimum.accumulate(total[::-1])[::-1]       # the least reps in all from each state to as built
        easier = [st for st, t, m in zip(states[:p0], total[:-1], later[1:]) if t < m]    # less work every step
        return {"states": easier + states[p0:], "p0": len(easier), "sets": s0, "reps": r0}

    def factor_room(self, knob: str) -> tuple[int, int]:
        lo, hi = self.limits.get(knob, self.FACTORS[knob])
        return self.steps_within(lo, hi, self.steps[knob])

    @staticmethod
    def steps_within(lo: float, hi: float, step: float) -> tuple[int, int]:
        """whole steps of x (1 + step k) from 1 that stay within (lo, hi): (down, up)."""
        return max(int(np.floor((1 - lo) / step + 1e-9)), 0), max(int(np.floor((hi - 1) / step + 1e-9)), 0)

    def reach(self, w: Workout) -> tuple[int, int]:
        """how many steps easier and harder `w` can go."""
        room = self.room(w)
        return tuple(sum(room[k][(d * self.TOWARD_HARDER[k] > 0)] for k in walk)
                     for d, walk in ((-1, self.EASIER), (1, self.HARDER)))

    def moves(self, w: Workout, offset: int) -> dict[str, int]:
        """the knob moves (in each knob's own direction) for `offset` steps (> 0 harder), as far as the limits go."""
        d, left, out = int(np.sign(offset)), abs(int(offset)), {}
        room = self.room(w)
        for knob in (self.HARDER if d > 0 else self.EASIER):
            s = d * self.TOWARD_HARDER[knob]
            n = min(left, room[knob][s > 0])
            if n:
                out[knob], left = s * n, left - n
        if "volume" in out:                             # the ladder's position -> reps and sets
            v = self.volume(w)
            sets, reps = v["states"][v["p0"] + out.pop("volume")]
            out.update({k: x for k, x in (("reps", reps - v["reps"]), ("sets", sets - v["sets"])) if x})
        return out

    def apply(self, w: Workout, offset: int) -> Workout:
        """`w` moved `offset` steps (> 0 harder, < 0 easier); `w` itself at 0 or when nothing is left to move."""
        return self.move(w, self.moves(w, offset)) if offset else w

    # --- changing the steps ----------------------------------------------------------------------
    def move(self, w: Workout, moves: dict[str, int]) -> Workout:
        """`w` with these knob moves (each clipped to its room); `w` itself when none is left."""
        room = self.room(w)
        mv = {k: int(np.clip(v, -room[k][0], room[k][1])) for k, v in moves.items()}
        if not any(mv.values()):
            return w
        f = {k: 1 + self.steps[k] * mv.get(k, 0) for k in self.steps}
        a, b, m, k = self.parts(w.steps)
        items = list(w.steps)
        for i in range(a, b + 1):
            x = self.powered(items[i], f["power"])
            items[i] = self.timed(x, f["rest"]) if isinstance(x, Step) and x.kind == "rest" else x
        if m is not None:
            main = items[m]
            if k is None:
                items[m] = self.rep_moved(main, main.reps + mv.get("reps", 0), f["on"], f["rest"])
            else:
                inner = list(main.steps)
                inner[k] = self.rep_moved(inner[k], inner[k].reps + mv.get("reps", 0), f["on"], f["rest"])
                end = self.closing(inner)
                if end is not None:
                    inner[end] = self.timed(inner[end], 1 + self.STEP["rest"] * mv.get("rest", 0))   # between sets
                items[m] = Repeat(main.reps + mv.get("sets", 0), inner)
        out = replace(w, steps=items)
        out.minutes = out.total_minutes()
        return out

    def rep_moved(self, rep: Repeat, reps: int, on: float, rest: float) -> Repeat:
        """a reps' Repeat with `reps` reps, each rep's steps x `on` long and its closing rest x `rest`."""
        body, end = list(rep.steps), self.closing(rep.steps)
        body = map_steps(body[:end], partial(self.timed, factor=on)) + \
            ([self.timed(body[end], rest)] if end is not None else [])
        return Repeat(reps, body)

    def powered(self, x, factor: float):
        """a Step or Repeat with its work steps' targets x `factor`."""
        if factor == 1:
            return x
        if isinstance(x, Repeat):
            return replace(x, steps=[self.powered(y, factor) for y in x.steps])
        return x.mapped(factor.__mul__) if x.kind == "active" and x.lo >= WORK else x

    @classmethod
    def timed(cls, s: Step, factor: float) -> Step:
        """a step `factor` x as long."""
        return s if factor == 1 else replace(s, minutes=cls.minutes(s.minutes, factor))

    @classmethod
    def minutes(cls, minutes: float, factor: float) -> float:
        """minutes x factor to whole ROUND_S seconds (at least one); unchanged at factor 1."""
        if factor == 1:
            return minutes
        return max(round(minutes * factor * 60 / cls.ROUND_S), 1) * cls.ROUND_S / 60


class WorkoutLibrary:
    """Builds every kind of session the plan uses."""

    @staticmethod
    def rest(date) -> Workout:
        return Workout(date, "rest", 0, "Rest day")

    @staticmethod
    def easy(date, kind: str, minutes: float, band: tuple[float, float] | None = None) -> Workout:
        """a plain easy ride; band: (lo, hi) of its Z2 part (default EASY's)."""
        lo, hi = band or EASY[kind]
        minutes = int(round(minutes / 5) * 5)
        label = {"recovery": "Recovery spin", "endurance": "Endurance", "long": "Long endurance"}[kind]
        return Workout(date, kind, minutes, f"{label} {WorkoutLibrary.duration(minutes)}",
                       [Step("Warm up", 10, .45, .60, "warmup"),
                        Step("Z2" if kind != "recovery" else "Easy", max(minutes - 15, 10), lo, hi),
                        Step("Cool down", 5, .40, .50, "cooldown")])

    @staticmethod
    def varied(w: Workout, middle_ok: bool) -> Workout:
        """A plain endurance / long ride as one of its EASY_VARIANTS (by date, so a day keeps its variant),
        same length; the steady ride when the variant doesn't fit (too short, or tempo-like when
        `middle_ok` is False: not a loading week, or training polarized)."""
        options = EASY_VARIANTS.get(w.type)
        if not options or len(w.steps) > 3:
            return w
        name, shortest, middle, spec, *about = options[pd.Timestamp(w.date).toordinal() % len(options)]
        if spec is None or w.minutes < shortest or (middle and not middle_ok):
            return w
        block, times = spec
        block = [Step(n, m, lo, hi, k) for n, m, lo, hi, k in block]
        work = sum(s.minutes for s in block) * times
        z2 = w.minutes - 15 - work
        if z2 < 10:
            return w
        lo, hi = w.z2_band() or EASY[w.type]
        late = w.type == "long"                                    # long rides: the work comes at the end
        head = [Step("Warm up", 10, .45, .60, "warmup"), Step("Z2", z2 if late else z2 / 2, lo, hi)]
        tail = ([] if late else [Step("Z2", z2 / 2, lo, hi)]) + [Step("Cool down", 5, .40, .50, "cooldown")]
        steps = head + ([Repeat(times, block)] if times > 1 else block) + tail
        label = {"endurance": "Endurance", "long": "Long endurance"}[w.type]
        return replace(w, title=f"{label} {WorkoutLibrary.duration(w.minutes)}, {name}", steps=steps,
                       note=(about[0] if about else ""))

    @staticmethod
    def duration(minutes: float) -> str:
        """30 s, 2 min, 4:45, and from an hour h:mm: 1:30 h."""
        if minutes < 1:
            return f"{round(minutes * 60)} s"
        m, s = divmod(round(minutes * 60), 60)
        if m >= 60:
            return f"{m // 60}:{m % 60:02d} h"
        return f"{m} min" if not s else f"{m}:{s:02d}"

    @staticmethod
    def format_of(date, kind: str) -> str:
        """the format a `kind` session on `date` takes when none is asked for (the formats take turns)."""
        formats = list(LADDERS[kind])
        return formats[IV.turn(date, len(formats))]

    @staticmethod
    def format_in(date, d: dict) -> str:
        """the format of interval session `d` (saved, or a planned record) on `date`: as saved, else the longest
        format its title starts with (saved before formats were), else the day's turn (`format_of`)."""
        kind, title = d["type"], d.get("title") or ""
        named = [f for f in LADDERS[kind] if title.startswith(f)]
        return d.get("format") or max(named, key=len, default="") or WorkoutLibrary.format_of(date, kind)

    @staticmethod
    def intervals(date, kind: str, level: int, fmt: str | None = None, variety: bool = True,
                  offsets: dict | None = None, lags: dict | None = None) -> Workout:
        """A `kind` session at `level`, in `fmt` or this week's format (the formats take turns). With
        `variety` a set of reps of 2 min or more takes this week's shape (pyramid, broken, ...; the same
        work) and the warm-up varies (planning/variety.py). `offsets`: your easier / harder choice per format
        (PlanState "format_offsets", {kind: {format: steps}}), applied by the format's Adjuster. `lags`: how
        many levels the type's steady format (its first) is behind the type's `level` (PlanState
        "sustained_lag", {kind: steps}); the session's level is then the steady format's own."""
        fmt = fmt or WorkoutLibrary.format_of(date, kind)
        on_name, off_name, off_band, about, ladder = LADDERS[kind][fmt]
        level -= (lags or {}).get(kind, 0) if fmt == next(iter(LADDERS[kind])) else 0
        lvl = int(min(max(level, 0), len(ladder) - 1))
        sets, reps, on, off, lo, hi, rest = ladder[lvl]
        pattern, shape = "", ""
        if sets == 1:                                    # one set of reps: in this week's shape
            fixed = fmt in FIXED_PATTERNS
            pattern = FIXED_PATTERNS[fmt] if fixed else IV.pattern(date, kind, on, reps) if variety else "straight"
            main, shape = IV.main_set(pattern, on_name, off_name, reps, on, off, (lo, hi), off_band)
            shape = "" if fixed else shape                   # the format's name says it already
            about = about + (f" {ABOUT[pattern]}" if pattern in ABOUT and not fixed else "")
        else:                                            # sets of on / off pairs with a rest between
            pair = [Step(on_name, on, lo, hi)] + \
                ([Step(off_name, off, *off_band, "rest" if off_band[0] < WORK else "active")] if off else [])
            main = [Repeat(sets, [Repeat(reps, pair), Step("Recover", rest, *EASY_OFF, "rest")])]
        warm = IV.warm_up(date, kind) if variety else [Step("Warm up", 15, .50, .65, "warmup")]
        steps = [*warm, *main, Step("Z2", 10, .55, .65), Step("Cool down", 10, .40, .50, "cooldown")]
        w = Workout(date, kind, 0, "", steps, lvl, note=about, format=fmt)
        offset =((offsets or {}).get(kind) or {}).get(fmt, 0)
        if offset:
            adj = Adjuster.of(kind, fmt, lvl)
            mv = adj.moves(w, offset)
            easier, harder = adj.reach(w)
            w = adj.move(w, mv)
            w.offset = int(min(max(offset, -easier), harder))                          # as far as it goes
            sets, reps = sets + mv.get("sets", 0), reps + mv.get("reps", 0)
            on = Adjuster.minutes(on, 1 + adj.steps["on"] * mv.get("on", 0))
            off = Adjuster.minutes(off, 1 + adj.steps["rest"] * mv.get("rest", 0)) if off_band[0] < WORK else off
        w.title = WorkoutLibrary.interval_title(fmt, pattern, shape, sets, reps, on, off)
        w.minutes = w.total_minutes()
        return w

    @staticmethod
    def interval_title(fmt: str, pattern: str, shape: str, sets: int, reps: int, on: float, off: float) -> str:
        """"VO2max 5x4 min", "Threshold pyramid 8-10-12-10-8 min", "Over-unders 3x4 × 1 min / 2 min", "30/15 3x13";
        `pattern` "" for sets of reps."""
        dur = WorkoutLibrary.duration
        if pattern:
            return f"{fmt} {shape}" if pattern in ("pyramid", "descending", "ascending") else \
                f"{fmt} {reps}x{dur(on)}" + (f", {shape}" if shape else "")
        return f"{round(on * 60)}/{round(off * 60)} {sets}x{reps}" if fmt in FORMAT_TITLES else \
            f"{fmt} {sets}x{reps} × {dur(on)} / {dur(off)}"

    @staticmethod
    def level_titles(offsets: dict | None = None) -> dict[str, list[str]]:
        """Per type and level: the sessions of each format at that level, with your easier / harder `offsets`
        (the Plan page's level picker)."""
        day = pd.Timestamp("2026-01-05")
        return {k: [" · ".join(WorkoutLibrary.intervals(day, k, lv, f, variety=False, offsets=offsets).title
                               for f in v) for lv in range(LEVELS[k])] for k, v in LADDERS.items()}

    @staticmethod
    def durability(w: Workout, level: int, polarized: bool = False) -> Workout:
        """A plain long ride `w` with the durability finish (LADDERS["durability"]) at the highest level up to
        `level` that fits its length after DURABILITY_AFTER_MIN of riding; `w` itself when none fits.
        `polarized`: the efforts at DURABILITY_POLARIZED instead of just under FTP."""
        on_name, off_name, off_band, about, ladder = LADDERS["durability"]["Durability"]
        lo, hi = w.z2_band() or EASY[w.type]
        for lv in range(int(min(max(level, 0), len(ladder) - 1)), -1, -1):
            _, reps, on, off, plo, phi, _ = ladder[lv]
            plo, phi = DURABILITY_POLARIZED if polarized else (plo, phi)
            finish = reps * on + (reps - 1) * off
            z2 = w.minutes - 15 - finish
            if z2 + 10 >= DURABILITY_AFTER_MIN:
                pair = [Step(on_name, on, plo, phi), Step(off_name, off, *off_band, "active")]
                main = [Repeat(reps - 1, pair), Step(on_name, on, plo, phi)] if reps > 1 else [Step(on_name, on, plo, phi)]
                steps = [Step("Warm up", 10, .45, .60, "warmup"), Step("Z2", z2, lo, hi), *main,
                         Step("Cool down", 5, .40, .50, "cooldown")]
                dur = WorkoutLibrary.duration
                return replace(w, steps=steps, level=lv, note=about,
                               title=f"Long endurance {dur(w.minutes)}, durability {reps}x{dur(on)}")
        return w

    @staticmethod
    def long_tempo(date, minutes: float, block_min: int) -> Workout:
        """Long ride with two tempo blocks (when your hours cap the load)."""
        minutes = int(round(minutes / 5) * 5)
        z2 = max(minutes - 15 - 2 * block_min - 10, 20)
        steps = [Step("Warm up", 10, .45, .60, "warmup"), Step("Z2", z2 / 2, .55, .65),
                 Step("Tempo", block_min, .76, .85), Step("Z2", 10, .55, .65),
                 Step("Tempo", block_min, .76, .85), Step("Z2", z2 / 2, .55, .65),
                 Step("Cool down", 5, .40, .50, "cooldown")]
        return Workout(date, "long", sum(s.minutes for s in steps),
                       f"Long ride {WorkoutLibrary.duration(minutes)} with 2x{block_min} min tempo", steps)

    @staticmethod
    def ftp_test(date) -> Workout:
        steps = [Step("Warm up", 15, .50, .65, "warmup"),
                 Repeat(3, [Step("Opener", 1, 1.0, 1.05), Step("Easy", 1, .50, .55, "rest")]),
                 Step("Easy", 5, .50, .55, "rest"), Step("All-out, even pace", 20, 1.0, 1.10),
                 Step("Easy", 5, .40, .50, "rest"), Step("Z2 after the test", 5, .55, .65),
                 Step("Cool down", 5, .40, .50, "cooldown")]
        w = Workout(date, "test", 0, "20-min FTP test", steps,
                    note="After the openers: 5 min easy, then 20 min as hard as you can hold evenly. "
                         "FTP is about 95 % of the 20-min average power.")
        w.minutes = sum(s.minutes for s in w.expanded())
        return w

    @staticmethod
    def ramp_test(date) -> Workout:
        """Ramp to failure for the alpha1 thresholds: 2-min steps so alpha1's 2-min window keeps up."""
        steps = [Step("Warm up", 10, .45, .55, "warmup")]
        steps += [Step(f"{RAMP_NAME} {round(f * 100)} %", 2, f - .01, f + .01) for f in RAMP_STEPS]
        steps += [Step("Easy", 5, .40, .50, "rest"), Step("Z2 after the test", 5, .55, .65),
                  Step("Cool down", 5, .40, .50, "cooldown")]
        return Workout(date, "ramp", sum(s.minutes for s in steps), "Ramp test (alpha1 thresholds)", steps,
                       note="Chest strap on and alphaHRV recording. Every 2 minutes the target goes up about 6 % "
                            "of FTP; keep cadence steady and keep going until you can't hold it (still going at the "
                            "last step: hold it as long as you can), then spin easy. goodpush reads LT1 and LT2 "
                            "(alpha1 0.75 / 0.5) and your heart rate at each from this ride. Ride it rested.")

    @staticmethod
    def lsct(date, minutes: float = 60) -> Workout:
        """The LSCT (LSCT_STAGES) as the start of an easy ride `minutes` long: the stages hold heart rates, ridden in
        resistance mode (you set the effort; the watts are only a guide), then LSCT_STOP_MIN sitting still for
        heart-rate recovery, then Z2 and a cool-down."""
        stages = [Step(n, m, lo, hi, cue=f"Hold {round(f * 100)} % of max heart rate", mode="resistance", hr=f)
                  for n, m, f, (lo, hi) in LSCT_STAGES]
        stop = Step(LSCT_STOP, LSCT_STOP_MIN, 0.0, 0.0, "rest",
                    cue="Stop pedalling and sit still: heart-rate recovery is being recorded")
        test = sum(s.minutes for s in stages) + LSCT_STOP_MIN
        z2 = max(int(round(minutes / 5) * 5) - test - 5, 5)
        steps = [*stages, stop, Step("Z2", z2, *Z2), Step("Cool down", 5, .40, .50, "cooldown")]
        return Workout(date, "lsct", test + z2 + 5, "LSCT (submaximal test) + easy ride", steps,
                       note="Heart-rate strap on; ride it rested, at about the same time of day and the same "
                            "conditions each time. No warm-up: the first stage is the warm-up. The trainer starts "
                            "in resistance mode: ride each stage at 60, 80 and 90 % of your max heart rate "
                            "(the bar shows the bpm; the watts are only a guide). After the third "
                            "stage stop pedalling and sit still for 90 s - the ride keeps recording - then ride "
                            "easy. Note how hard each stage felt (RPE).")

    @staticmethod
    def race(date, event: dict) -> Workout:
        lo, hi, minutes = RACE.get(event.get("kind"), RACE["other"])
        steps = [Step("Warm up", 20, .50, .70, "warmup"), Step("Race", minutes, lo, hi)]
        return Workout(date, "race", minutes + 20, f"{event.get('name') or 'Race'} ({event.get('priority', 'B')})",
                       steps, note="Race day.")

    @staticmethod
    def max_effort(w: Workout, seconds: int) -> Workout:
        """Interval session `w` opening with a timed all-out effort (MAX_EFFORTS): a good warm-up, a primer, then as
        hard as you can for that long - goodpush counts it as a measurement (models.all_out: the step's lap; for
        the long ones also your heart rate at its maximum, metrics.effort_hr) - and the session's own set after a
        recovery. Primer to all-out ride in resistance mode: the trainer leaves ERG for them and goes back after
        (ride.session.WorkoutRun)."""
        name, lo, hi, recover, about = MAX_EFFORTS[seconds]
        warm = IV.warm_up(w.date, w.type)
        rest = w.steps[len(warm):] if w.steps[:len(warm)] == warm else \
            [s for s in w.steps if not (isinstance(s, Step) and s.kind == "warmup")]
        res = "resistance"
        head = [Step("Warm up", 15, .50, .65, "warmup"),
                Repeat(2, [Step("Opener", .5, 1.10, 1.20), Step("Easy", 1.5, .50, .55, "rest")]),
                Step(PRIMER_NAME, 1, .80, .95, mode=res, cue="ERG off: find your gear, about half effort"),
                Step("Easy", 3, .50, .60, "rest", mode=res, cue="Easy, the gear ready"),
                Step(name, seconds / 60, lo, hi, mode=res, cue="As hard as you can, the whole way"),
                Step("Recover", recover, .40, .55, "rest")]
        # it stands in for the first interval: the session keeps its length (your time that day) - the easy parts
        # after the set and the cool-down shrink first (to 5 min), then reps go (never below 2)
        over = sum(s.minutes for s in flat(head)) + sum(s.minutes for s in flat(rest)) - w.total_minutes()
        rest = list(rest)
        for i in sorted((i for i, s in enumerate(rest) if isinstance(s, Step) and s.lo < WORK and s.kind in ("active", "cooldown")),
                        key=lambda i: -rest[i].minutes):
            cut = min(max(over, 0), rest[i].minutes - 5)
            if cut > 0:
                rest[i], over = replace(rest[i], minutes=rest[i].minutes - cut), over - cut
        r = next((i for i, s in enumerate(rest) if isinstance(s, Repeat) and s.reps > 2), None)
        if r is not None and over > 0:
            per = sum(s.minutes for s in flat(rest[r].steps))
            drop = int(min(rest[r].reps - 2, np.ceil(over / per)))
            rest[r] = replace(rest[r], reps=rest[r].reps - drop)
        new = replace(w, steps=head + rest, title=f"{name} + {w.title}", note=f"{about} {w.note}".strip())
        new.minutes = new.total_minutes()
        return new

    @staticmethod
    def openers(date) -> Workout:
        """The day before a race: easy with a few short efforts to wake the legs up."""
        steps = [Step("Warm up", 15, .50, .65, "warmup"),
                 Repeat(3, [Step("Opener", 1, 1.05, 1.15), Step("Easy", 2, .50, .55, "rest")]),
                 Step("Z2", 15, .55, .65), Step("Cool down", 5, .40, .50, "cooldown")]
        w = Workout(date, "openers", 0, "Openers 3x1 min", steps, note="Short and sharp; stay fresh.")
        w.minutes = w.total_minutes()
        return w

    @staticmethod
    def test(kind: str, date) -> Workout:
        return WorkoutLibrary.ramp_test(date) if kind == "ramp" else WorkoutLibrary.ftp_test(date)

    @staticmethod
    def rebuild(date, record: dict) -> Workout:
        """A session from its planned record (type, level, offset, minutes, title) when no full copy was kept."""
        kind = record.get("type")
        if kind in TESTS:
            w = WorkoutLibrary.test(kind, date)
        elif kind in LADDERS:
            fmt = WorkoutLibrary.format_in(date, record)
            w = WorkoutLibrary.intervals(date, kind, record.get("level") or 0, fmt,
                                         offsets={kind: {fmt: record.get("offset") or 0}})
            opening = str(record.get("title") or "").split(" + ")[0]       # "All-out 3 min + ...": its opening too
            sec = next((s for s, v in MAX_EFFORTS.items() if v[0] == opening), None)
            if sec:
                w = WorkoutLibrary.max_effort(w, sec)
        elif kind == "rest":
            w = WorkoutLibrary.rest(date)
        elif kind == "lsct":
            w = WorkoutLibrary.lsct(date, record.get("minutes") or 60)
        else:
            w = WorkoutLibrary.easy(date, kind if kind in EASY else "endurance", record.get("minutes") or 60)
        if record.get("title"):
            w.title = record["title"]
        w.locked = True
        return w


class TestEfforts:
    """[plan] test_efforts: which all-out effort opens a session (WorkoutLibrary.max_effort), so the power curve is
    measured at every length now and then - 1, 3 and 5 min (MAX_EFFORTS) - and not only inferred.
    `last`: {seconds: the day of the last such effort, done or planned}; it moves as efforts are placed."""
    __test__ = False                    # not a pytest class

    def __init__(self, last: dict | None = None):
        self.last = {s: None for s in TEST_EFFORT_WEEKS} | {s: v for s, v in (last or {}).items() if s in TEST_EFFORT_WEEKS}

    def pick(self, kind: str, date) -> int | None:
        """the effort (seconds) to open a `kind` session on `date`: of the lengths due (none within its
        TEST_EFFORT_WEEKS) and suited to it (TEST_EFFORT_BEFORE), the longest overdue - None if none is."""
        date = pd.Timestamp(date)
        due = [s for s in TEST_EFFORT_WEEKS if kind in TEST_EFFORT_BEFORE[s]
               and (self.last[s] is None or (date - pd.Timestamp(self.last[s])).days >= 7 * TEST_EFFORT_WEEKS[s])]
        if not due:
            return None
        over = {s: np.inf if self.last[s] is None else (date - pd.Timestamp(self.last[s])).days / TEST_EFFORT_WEEKS[s]
                for s in due}
        return max(due, key=lambda s: (over[s], s))

    def placed(self, seconds: int, date) -> None:
        self.last[seconds] = pd.Timestamp(date)
