"""A synthetic rider's description: who they are, their physiology at the start and how they train (a TOML file,
`goodpush synth new --profile`). Every number has a default, so a profile only needs what differs."""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import tomlkit

from ..app.tomledit import TomlEdit

DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class RiderProfile:
    name: str = "Synthetic rider"
    seed: int = 1
    truth: str = "model"            # where the true physiology comes from: "model" (the physio model run forward:
    #                                 can its parameters be recovered?) or "simple" (an independent generator: how far
    #                                 off is the model on a rider it doesn't describe exactly?)
    draw_dynamics: bool = True      # "model": the training response drawn from the model's prior (else its centre)
    days: int = 540                 # how long the history is
    end: str = ""                   # its last day (default: yesterday)
    timezone: str = "UTC"
    sex: str = "female"
    birth_year: int = 1988
    weight_kg: float = 64.0
    # physiology at the start, fresh (W)
    cp_w: float = 215.0
    map_w: float = 270.0            # best 5 minutes
    lt1_w: float = 155.0
    pmax_w: float = 820.0
    fade: float = 0.08              # how fast power falls past ~30 min (the physio model's durability = 1 / fade)
    max_hr: float = 186.0
    lthr: float = 166.0             # heart rate held at CP
    resting_hr: float = 52.0
    enter_hr: bool = False          # write max / threshold / resting heart rate into goodpush.toml (else estimated)
    # training
    hours_start: float = 5.0        # hours a week at the start ...
    hours_end: float = 8.0          # ... and at the end (rising evenly)
    ride_days: list = field(default_factory=lambda: ["tue", "thu", "sat", "sun"])
    key_days: list = field(default_factory=lambda: ["tue", "thu"])
    long_day: str = "sun"
    key_mix: dict = field(default_factory=lambda: {"sweetspot": 2.0, "threshold": 2.0, "vo2max": 1.0, "tempo": 1.0})
    recovery_every: int = 4         # every n-th week is a recovery week (0 = never)
    recovery_share: float = 0.6     # ... with this share of the hours
    outdoor_share: float = 0.4      # easy and long rides outside (terrain, coasting, surges)
    surge_share: float = 0.0        # easy and long rides indoors with the outdoor surges (short efforts near the curve)
    ftp_test_weeks: int = 8         # a 20-min FTP test this often (0 = never)
    ramp_weeks: int = 8             # a ramp test (0 = never)
    lsct_weeks: int = 4             # an LSCT (0 = never)
    test_efforts: bool = False      # key sessions open with all-out efforts now and then (as [plan] test_efforts)
    all_out_per_month: float = 2.0  # all-out efforts (5 s to 20 min) in ordinary rides
    strap_share: float = 1.0        # rides with a chest strap's beat intervals (alpha1)
    missed_beats: float = 0.002     # share of beats the strap misses
    plan: bool = True               # turn the plan on in goodpush.toml
    # wellness (a watch's nights: sleep, HRV, resting heart rate)
    wellness: bool = True
    sleep_h: float = 7.4
    hrv_ms: float = 58.0
    # illness and breaks
    illness_per_year: float = 3.0
    breaks_per_year: float = 1.0    # a week or two away without the bike

    @classmethod
    def load(cls, path: str | Path | None) -> RiderProfile:
        """the profile in a TOML file (unknown keys refused); None: all defaults."""
        if path is None:
            return cls()
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        bad = sorted(set(data) - known)
        if bad:
            raise ValueError(f"{path}: unknown profile keys {', '.join(bad)}")
        return cls(**data).checked()

    def checked(self) -> RiderProfile:
        if self.truth not in ("model", "simple"):
            raise ValueError('truth must be "model" or "simple"')
        if not self.lt1_w < self.cp_w < self.map_w < self.pmax_w:
            raise ValueError("the physiology needs LT1 < CP < MAP < Pmax")
        if not self.resting_hr < self.lthr < self.max_hr:
            raise ValueError("the heart rates need resting < threshold < maximum")
        days = set(self.ride_days) | set(self.key_days) | {self.long_day}
        if not days <= set(DAY_NAMES):
            raise ValueError(f"days must be among {', '.join(DAY_NAMES)}")
        return self

    def text(self) -> str:
        """the profile as TOML (kept with the rider, so it can be made again)."""
        return tomlkit.dumps({k: TomlEdit.value(v) for k, v in asdict(self).items()})
