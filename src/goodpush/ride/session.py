"""Riding a session: `WorkoutRun` = the plan's steps in watts on a clock that runs while you pedal,
`RideSession` = devices + workout + trainer control + recording, moved on by `tick(now)`."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics import dfa
from .recorder import RideRecorder
from .sensors import SensorHub

RIDING_W = 30              # power above this: you're riding and the clocks run; at or below it they stop
GAP_S = 10.0              # s: power readings dropping out (missed ANT+ messages, not a low reading) this long
                          # leave the clocks as they were, or cadence says (pedalling: they run)
RESISTANCE = 30.0          # % resistance for free rides and resistance mode (trainer's own scale)
# how the trainer is steered; only you switch (the mode button on the bar):
#   erg: it holds the step's target power whatever your cadence (a free ride, or after the workout: free_w)
#   resistance: a fixed brake level, power = your effort x gear
#   sim: a road of this gradient, with your weight, so sprints and gears feel like outside
MODES = ("erg", "resistance", "sim")
GRADE_LIMITS = (-10.0, 20.0)   # % gradient in sim mode
FREE_W, FREE_STEP_W = 150.0, 5.0   # ERG with no step to ride: this power (cmd_ride: 65 % of FTP), +/- moves it
FREE_FTP = 0.65


class WorkoutRun:
    """steps: plan.json session rows [{name, min, lo, hi (W), kind, and lo_end / hi_end for a ramp, cadence,
    cue, mode, hr}]; none = free ride.
    `mode` is one of MODES; `scale` makes ERG targets harder / easier, `resistance` is the level in
    resistance mode, `grade` the road's gradient (%) in sim mode, `free_w` what ERG holds with no step (a free
    ride, or once the workout is done). `max_hr` turns heart-rate steps (the LSCT's, ridden in resistance mode:
    you hold the heart rate) into the bpm the bar shows (0: unknown). A step at 0 W (`still`, the LSCT's sitting
    still) runs on the clock whether you pedal or not."""

    def __init__(self, title: str, steps: list[dict], key: str = "", resistance: float = RESISTANCE,
                 max_hr: float = 0.0):
        self.title, self.key, self.steps = title, key, list(steps)
        self.ends = np.cumsum([s["min"] * 60 for s in self.steps]) if self.steps else np.zeros(0)
        self.elapsed = 0.0
        self.paused = False
        self.scale = 1.0
        first = self.steps[0].get("mode") if self.steps else None
        self.mode = first if first in MODES else "erg"         # a workout that opens in another mode (the LSCT)
        self.mode_before = None    # the ERG left for steps that ask for another mode (follow_step): back after them
        self.at = self.index       # the step follow_step last saw
        self.resistance = resistance
        self.grade = 0.0
        self.free_w = FREE_W
        self.max_hr = float(max_hr or 0)

    @classmethod
    def from_plan(cls, plan_path: Path, key: str | None = None, today: pd.Timestamp | None = None,
                  resistance: float = RESISTANCE, max_hr: float = 0.0) -> WorkoutRun | None:
        """The session `key` ("2026-10-27" or "2026-10-27#2"), else today's first one with steps."""
        sessions = json.loads(Path(plan_path).read_text(encoding="utf-8")).get("sessions") or []
        day = str((today or pd.Timestamp.today()).date())
        s = next((s for s in sessions if (s["key"] == key if key else s["date"] == day and s["steps"])), None)
        return cls(s["title"], s["steps"], s["key"], resistance, max_hr) if s and s["steps"] else None

    def ends_at(self, now: float) -> float | None:
        """when the workout ends (unix time) if you ride on without stopping; None for a free ride or once done."""
        return now + self.total - self.elapsed if self.total and not self.done else None

    @property
    def still(self) -> bool:
        """the step now is one to sit still in (0 W): its clock runs without pedalling."""
        s = self.step
        return s is not None and s["hi"] == 0 and s.get("kind") == "rest"

    def hr_target(self, step: dict | None = None) -> float | None:
        """the heart rate (bpm) a step (default: the one now) holds; None without one or without max_hr."""
        s = self.step if step is None else step
        return round(s["hr"] * self.max_hr) if s and s.get("hr") and self.max_hr else None

    @property
    def total(self) -> float:
        return float(self.ends[-1]) if len(self.ends) else 0.0

    @property
    def done(self) -> bool:
        return bool(len(self.ends)) and self.elapsed >= self.total

    @property
    def index(self) -> int:
        """the current step, -1 for a free ride or once the workout is over."""
        i = int(np.searchsorted(self.ends, self.elapsed, "right"))
        return i if i < len(self.steps) else -1

    @property
    def step(self) -> dict | None:
        i = self.index
        return self.steps[i] if i >= 0 else None

    @property
    def next_step(self) -> dict | None:
        i = self.index
        return self.steps[i + 1] if 0 <= i < len(self.steps) - 1 else None

    @property
    def step_left(self) -> float:
        i = self.index
        return float(self.ends[i] - self.elapsed) if i >= 0 else 0.0

    def advance(self, dt: float, riding: bool) -> None:
        if (riding or self.still) and not self.paused and not self.done:
            end = self.total if riding else float(self.ends[self.index])    # sitting still: to the step's end
            self.elapsed = min(self.elapsed + dt, end)
            self.follow_step()

    def follow_step(self) -> None:
        """On a new step: one that asks for another mode (an all-out's primer and effort: resistance) leaves ERG
        for it - in resistance or sim already, it stays as it is - and the first step after them that doesn't ask
        goes back to ERG. A mode you pick yourself (choose_mode) stays."""
        i = self.index
        if i == self.at:
            return
        self.at = i
        want = (self.step or {}).get("mode")
        if want in MODES and want != "erg":
            if self.mode == "erg":
                self.mode_before, self.mode = "erg", want
        elif self.mode_before is not None:
            self.mode, self.mode_before = self.mode_before, None

    def choose_mode(self, mode: str) -> None:
        """the mode you picked: kept, no going back after a step's own."""
        self.mode, self.mode_before = mode, None

    def watts(self, step: dict | None, frac: float = 0.0) -> float | None:
        """ERG target of a step `frac` of the way through it (a ramp moves evenly from its start band to its
        end band); no step: free_w. None when the trainer isn't holding a power (not ERG; steps that ask for
        resistance leave ERG for them, follow_step)."""
        if self.mode != "erg":
            return None
        if step is None:
            return self.free_w
        a = (step["lo"] + step["hi"]) / 2
        b = (step["lo_end"] + step["hi_end"]) / 2 if "lo_end" in step else a
        return (a + (b - a) * frac) * self.scale

    def target(self) -> float | None:
        step = self.step
        return self.watts(step, 1 - self.step_left / (step["min"] * 60) if step else 0.0)

    def command(self) -> tuple[str, float]:
        """what the trainer should do now: ("power", W), ("resistance", %) or ("grade", %)."""
        t = self.target()
        if t is not None:
            return "power", round(t)
        if self.mode == "sim":
            return "grade", round(self.grade, 1)
        return "resistance", round(self.resistance)

    def next_mode(self) -> None:
        self.choose_mode(MODES[(MODES.index(self.mode) + 1) % len(MODES)])

    def nudge(self, percent: float) -> None:
        """Harder (+) / easier (-): ERG targets by percent (with no step, free_w by FREE_STEP_W), the resistance
        level by percent points, the sim gradient by percent points."""
        kind = self.command()[0]
        if kind == "power" and self.step is None:
            self.free_w = float(np.clip(self.free_w + FREE_STEP_W * percent, 50, 1000))
        elif kind == "power":
            self.scale = float(np.clip(self.scale + percent / 100, 0.5, 1.5))
        elif kind == "grade":
            self.grade = float(np.clip(self.grade + percent, *GRADE_LIMITS))
        else:
            self.resistance = float(np.clip(self.resistance + percent, 0, 100))

    def skip(self) -> None:
        i = self.index
        if i >= 0:
            self.seek(self.ends[i])

    def seek(self, seconds: float) -> None:
        """Move the workout clock to `seconds` into the workout (back to redo a part, ahead to skip it)."""
        self.elapsed = float(np.clip(seconds, 0, self.total))
        self.follow_step()


class RideSession:
    """One ride. `link` is an AntLink (or SimLink); `folder` = where rides are saved (data/raw/goodpush)."""

    RESEND = 30.0          # s: the trainer command again even if it didn't change (a trainer that restarted)
    AUTOSAVE = 60          # s: partial copy of the recording
    LIVE_BEATS = 600       # alpha1 from the last beats: dfa.WINDOW_S of them (up to 400 at 200 bpm) and the ones
                           # before, for the artifact filter's neighbourhood

    def __init__(self, link, hub: SensorHub, run: WorkoutRun, folder: Path, now: float | None = None):
        self.link, self.hub, self.run, self.folder = link, hub, run, Path(folder)
        self.start = time.time() if now is None else now
        self.recorder = RideRecorder(self.start, run.title)
        self.moving = 0.0          # s ridden: the ride clock, stopped while there's no power
        self.second = int(self.start)
        self.last = None
        self.sent = (None, -np.inf)
        self.known = (-np.inf, False)     # (time, pedalling) the readings last told
        self.alpha1 = (float("nan"), float("nan"), 0.0)     # live DFA alpha1, artifact share, window filled
        self.alpha_at = -np.inf                                  # ... worked out every dfa.STEP_S

    def pedalling(self, now: float) -> bool | None:
        """from the readings: power above RIDING_W, or while power readings drop out cadence above 0; None
        with neither."""
        p = self.hub.get("power", now)
        if p == p:
            return p > RIDING_W
        c = self.hub.get("cadence", now)
        return c > 0 if c == c else None

    def riding(self, now: float) -> bool:
        """the clocks run: pedalling, or with no readings at all, what they last told for up to GAP_S."""
        on = self.pedalling(now)
        return on if on is not None else self.known[1] and now - self.known[0] <= GAP_S

    def tick(self, now: float) -> None:
        on = self.pedalling(now)
        if on is not None:
            self.known = (now, on)
        if self.last is None:                  # the ride starts at the first pedal stroke: standing by the
            if not on:                         # trainer isn't recorded (the trainer is told its target already)
                self.hub.take_beats()
                self.control(now)
                return
            self.start, self.second = now, int(now)
            self.recorder = RideRecorder(now, self.run.title)
        dt = 0.0 if self.last is None else min(now - self.last, 2.0)
        self.last = now
        riding = self.riding(now)
        self.recorder.beats += self.hub.take_beats()         # every beat, riding or not (alpha1 needs them all)
        if now - self.alpha_at >= dfa.STEP_S:
            self.alpha1, self.alpha_at = self.live_alpha1(), now
        self.moving += dt * (riding and not self.run.paused)
        self.run.advance(dt, riding)
        self.control(now)
        while self.second + 1 <= now:         # each whole second since the last tick (normally one)
            g = self.hub.get
            self.recorder.add(self.second, g("power", now), g("cadence", now), g("hr", now), g("speed", now),
                              self.run.index, riding and not self.run.paused)
            self.second += 1
            if self.recorder.seconds % self.AUTOSAVE == 0:
                self.recorder.save_partial(self.folder)

    def live_alpha1(self) -> tuple[float, float, float]:
        """(alpha1, artifact share, share of the window filled) from the beats of the last dfa.WINDOW_S
        (metrics.dfa: corrected, detrended). alpha1 is NaN until the beats fill the window; shown whatever its
        artifact share (the bar shows that next to it)."""
        b = np.array([x[:2] for x in self.recorder.beats[-self.LIVE_BEATS:]], float).reshape(-1, 2)
        filled = min(b[:, 1].sum() / dfa.WINDOW_S, 1.0)
        if filled < 1:
            return float("nan"), float("nan"), filled
        rr, end = b[:, 1], b[-1, 0]
        a, share = dfa.series(rr, end - rr.sum(), np.array([end]), max_artifacts=1.0)
        return float(a[0]), float(share[0]), filled

    def control(self, now: float) -> None:
        want = self.run.command()
        if want != self.sent[0] or now - self.sent[1] >= self.RESEND:
            getattr(self.link, f"set_{want[0]}")(want[1])          # set_power / set_resistance / set_grade
            self.sent = (want, now)

    def power3(self) -> float:
        """3-s average power (what the overlay shows)."""
        p = np.array([r[1] for r in self.recorder.rows[-3:]], float)
        return float(np.nanmean(p)) if np.isfinite(p).any() else float("nan")

    def reconnect(self, choice: dict[str, str]) -> None:
        self.link.close()
        self.hub.choose(choice)
        self.link.connect(self.hub.devices())
        self.sent = (None, -np.inf)

    def finish(self, min_seconds: int = 60) -> Path | None:
        """Saves the ride (None if it was shorter than min_seconds: nothing kept)."""
        self.link.close()
        if self.recorder.seconds < min_seconds:
            self.recorder.discard(self.folder)
            return None
        return self.recorder.save(self.folder)

    def discard(self) -> None:
        self.link.close()
        self.recorder.discard(self.folder)
