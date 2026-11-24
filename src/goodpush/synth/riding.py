"""One synthetic ride, 1 Hz: power from the session's steps and what the rider can do today (Capacity, all-out
curve), heart rate from power, and the beat-to-beat intervals of a chest strap with a set DFA alpha1."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

from ..models.physio import PhysioInputs
from ..models.stamina import RideStamina
from ..planning.steps import flat
from ..planning.workouts import ALL_OUT_NAME, LSCT_STOP, RAMP_NAME
from .truth import Capacity

FTP_TEST_STEP = "All-out, even pace"          # WorkoutLibrary.ftp_test's 20 minutes


class HeartModel:
    """Steady heart rate from power: resting + (LTHR - resting) (P / CP)^EXPONENT up to CP, then on to maximum at
    MAP; reached with a lag (TAU_S), drifting up over the first hour (DRIFT_BPM at CP by DRIFT_S, then level), held
    down by fatigue (FATIGUE_BPM per unit of 1 - form), with a day's offset (heat, hydration) and a little noise."""
    EXPONENT = 0.8
    TAU_S = 30.0
    DRIFT_BPM = 4.0
    DRIFT_S = 3600.0
    FATIGUE_BPM = 40.0
    DAY_SD = 2.0
    SECOND_SD = 1.0

    def __init__(self, profile, rng: np.random.Generator):
        self.rest, self.lthr, self.max = profile.resting_hr, profile.lthr, profile.max_hr
        self.rng = rng

    def steady(self, power: np.ndarray, cap: Capacity) -> np.ndarray:
        x = np.maximum(power, 0.0) / cap.cp
        below = self.rest + (self.lthr - self.rest) * np.minimum(x, 1.0) ** self.EXPONENT
        above = (self.max - self.lthr) * np.clip((x - 1.0) / (cap.map / cap.cp - 1.0), 0.0, 1.2)
        return below + above

    def offset(self, cap: Capacity) -> float:
        """the ride's own shift (bpm): the day's offset less what fatigue holds down."""
        return float(self.DAY_SD * self.rng.standard_normal() - self.FATIGUE_BPM * (1.0 - cap.form))

    def hr(self, power: np.ndarray, cap: Capacity, offset: float) -> np.ndarray:
        """the heart's own rate, smooth (the beats come from it; the recorded stream adds `noisy`'s wobble)."""
        a = np.exp(-1.0 / self.TAU_S)
        ss = self.steady(power, cap)
        lagged = lfilter([1.0 - a], [1.0, -a], ss, zi=[a * (self.rest + 15.0)])[0]
        rise = np.minimum(np.arange(len(power)) / self.DRIFT_S, 1.0)
        drift = self.DRIFT_BPM * rise * np.clip(ss / self.lthr, 0.0, 1.0)
        return np.clip(lagged + drift + offset, self.rest + 3.0, self.max)

    def noisy(self, hr: np.ndarray) -> np.ndarray:
        """the 1-Hz heart rate a watch records."""
        return np.clip(hr + self.SECOND_SD * self.rng.standard_normal(hr.size), self.rest + 3.0, self.max)

    def power_at(self, hr: np.ndarray, cap: Capacity, offset: float) -> np.ndarray:
        """the steady power that gives heart rate `hr` (below threshold) on a ride with this offset."""
        share = np.clip((np.asarray(hr, float) - offset - self.rest) / (self.lthr - self.rest), 0.05, 1.0)
        return cap.cp * share ** (1.0 / self.EXPONENT)


class BeatMaker:
    """Beat-to-beat intervals around the heart rate: fractional Gaussian noise in blocks of BLOCK beats whose
    exponent follows the intensity, so DFA alpha1 crosses 0.75 at LT1 and 0.5 at CP (ALPHA_AT); its size falls
    with intensity (SD_MS at rest-ish, x exp(-SD_DROP P / CP)). A share of beats the strap misses (two intervals
    as one)."""
    BLOCK = 256
    ALPHA_AT = (np.array([0.6, 1.0]), np.array([1.05, 0.75]))      # (power / LT1, alpha1) below LT1
    ALPHA_ABOVE = (np.array([1.0, 1.15]), np.array([0.5, 0.4]))    # (power / CP, alpha1) above CP
    # the alpha1 metrics.dfa measures on this noise for each exponent (beats at ~130 bpm, detrended)
    CALIBRATION = (np.array([0.435, 0.51, 0.57, 0.67, 0.73, 0.83, 0.94, 1.02, 1.11, 1.20]),
                   np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]))
    SD_MS = 40.0
    SD_DROP = 2.5
    SMOOTH_S = 60

    def __init__(self, profile, rng: np.random.Generator):
        self.missed = profile.missed_beats
        self.rng = rng

    def alpha(self, power: np.ndarray, cap: Capacity) -> np.ndarray:
        """the alpha1 the rider's heart shows at each second's (smoothed) power."""
        p = uniform_filter1d(np.asarray(power, float), self.SMOOTH_S)
        lo = np.interp(p / cap.lt1, *self.ALPHA_AT)
        mid = np.interp(p, [cap.lt1, cap.cp], [0.75, 0.5])
        hi = np.interp(p / cap.cp, *self.ALPHA_ABOVE)
        return np.where(p < cap.lt1, lo, np.where(p < cap.cp, mid, hi))

    def noise(self, hurst: np.ndarray) -> np.ndarray:
        """unit-sd fractional Gaussian noise, one row of BLOCK beats per exponent (spectral synthesis)."""
        n, rows = self.BLOCK, len(hurst)
        f = np.fft.rfftfreq(2 * n)[1:]
        amp = f[None, :] ** ((1.0 - 2.0 * hurst[:, None]) / 2.0)
        c = (self.rng.standard_normal((rows, f.size)) + 1j * self.rng.standard_normal((rows, f.size))) * amp
        x = np.fft.irfft(np.c_[np.zeros(rows), c], n=2 * n)[:, :n]
        return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)

    def beats(self, hr: np.ndarray, power: np.ndarray, cap: Capacity, start: float) -> list:
        """(time heard, interval s, beats it spans) for RideRecorder."""
        count = np.r_[0.0, np.cumsum(hr / 60.0)]
        k = np.arange(1, int(count[-1]))
        t = np.interp(k, count, np.arange(len(count), dtype=float))
        rr = np.diff(np.r_[0.0, t])
        sec = np.minimum(t.astype(int), len(power) - 1)
        x = np.maximum(power[sec], 0.0) / cap.cp
        rows = -(-rr.size // self.BLOCK)
        a = np.r_[self.alpha(power, cap)[sec], np.zeros(rows * self.BLOCK - rr.size)]
        h = np.interp(a.reshape(rows, self.BLOCK).mean(1), *self.CALIBRATION)
        z = self.noise(h).ravel()[:rr.size]
        rr = np.maximum(rr + self.SD_MS / 1000.0 * np.exp(-self.SD_DROP * x) * z, 0.25)
        spans = np.ones(rr.size)
        miss = np.flatnonzero(self.rng.random(rr.size - 1) < self.missed)
        miss = miss[np.r_[True, np.diff(miss) > 1]] if miss.size else miss     # never two in a row
        rr[miss + 1] += rr[miss]
        spans[miss + 1] = 2
        keep = np.ones(rr.size, bool)
        keep[miss] = False
        rr, spans = rr[keep], spans[keep]
        heard = start + np.cumsum(rr)
        return list(zip(heard.tolist(), rr.tolist(), spans.tolist()))


class RideMaker:
    """A session (planning Workout) ridden: each step at its share of the rider's FTP (what they believe it is),
    held only as far as today's all-out curve allows (REP_HOLD x all-out for the step's length) and W' lasts
    (`paced`); tests ridden all-out; LSCT stages ridden to heart rate. Outside: terrain (a slow wander), coasting and
    short surges."""
    EXECUTION_SD = 0.02          # a step ridden this far off its target
    SECOND_SD = 0.03             # second-to-second wobble
    # a training rep at most this share of all-out for its length: reps ridden at 97 %, fresh each time, put a
    # synthetic rider's best powers next to the curve all session long, and the fit raised CP 13 % (2026-11-21)
    REP_HOLD = 0.92
    EASE = 0.97                  # W' run out: the rest of the effort just under CP
    PACE_PASSES = 40             # `paced`: at most this many efforts cut short in a ride
    ALL_OUT_SD = 0.03            # an all-out effort's scatter around the curve (as the physio model's)
    TERRAIN_SD, TERRAIN_TAU_S = 0.25, 40.0
    COAST_EVERY_S, COAST_S = 150.0, (5, 40)
    SURGE_EVERY_S, SURGE_S = 900.0, (8, 20)
    ALL_OUT_S = (5, 60, 300, 1200)
    EASY_AFTER = 0.45            # share of FTP spun after a failed ramp / an all-out effort

    def __init__(self, profile, rng: np.random.Generator):
        self.rng = rng
        self.heart = HeartModel(profile, rng)
        self.beats = BeatMaker(profile, rng)

    def plan(self, workout) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        """the expanded steps, and per second: the step index, the target (share of FTP; ramps move evenly) and
        each step's first second."""
        steps = flat(workout.steps)
        n = np.array([max(int(round(s.minutes * 60)), 1) for s in steps])
        first = np.r_[0, np.cumsum(n)[:-1]]
        idx = np.repeat(np.arange(len(steps)), n)
        pos = (np.arange(n.sum()) - first[idx]) / n[idx]
        a = np.array([s.mid for s in steps])
        b = np.array([(s.lo_end + s.hi_end) / 2 if s.ramp else s.mid for s in steps])
        return steps, idx, a[idx] + (b - a)[idx] * pos, first

    def ride(self, workout, ftp: float, cap: Capacity, all_out, outdoor: bool, effort_s: int | None = None,
             surges: bool = False) -> dict:
        """power, step, heart rate (and the ride's heart-rate offset) for `workout`; all_out(t): today's all-out
        power; effort_s: an all-out effort this long in the ride's middle; surges: the short surges of a ride
        outside (always there outdoors)."""
        steps, idx, frac, first = self.plan(workout)
        n = np.bincount(idx, minlength=len(steps))
        names = np.array([s.name for s in steps])
        hr_share = np.array([s.hr or 0.0 for s in steps])[idx]
        best = all_out(n.astype(float))                        # each step's length all-out
        execution = np.exp(self.EXECUTION_SD * self.rng.standard_normal(len(steps)))
        tested = best * np.exp(self.ALL_OUT_SD * self.rng.standard_normal(len(steps)))
        offset = self.heart.offset(cap)
        ramp = np.flatnonzero(np.char.startswith(names, RAMP_NAME))
        planned = np.minimum(frac * ftp * execution[idx], self.REP_HOLD * best[idx])
        flat_out = ((names == FTP_TEST_STEP) | np.char.startswith(names, ALL_OUT_NAME))[idx]   # tests, all-out steps
        planned = np.where(flat_out, tested[idx], planned)
        # a ramp runs W' out by design (`ramp`); tests and all-out steps are ridden flat out, and the reps after
        # them feel what they took (W' balance)
        p = np.where(np.isin(idx, ramp) | flat_out, planned, self.paced(planned, cap))
        p = np.where(hr_share > 0, self.heart.power_at(hr_share * self.heart.max, cap, offset), p)
        p = np.where((names == LSCT_STOP)[idx], 0.0, p)
        if ramp.size:
            self.ramp(p, int(first[ramp[0]]), int(first[ramp[-1]] + n[ramp[-1]]), ftp, all_out)
        p = p * np.exp(self.SECOND_SD * self.rng.standard_normal(p.size))
        if outdoor:
            p = self.outside(p)
        if outdoor or surges:
            p = self.surged(p, all_out)
        if effort_s:            # last: terrain or a surge on top put it up to 19 % over the curve (2026-11-22)
            self.effort(p, effort_s, ftp, float(all_out(float(effort_s))))
        return {"power": np.round(np.maximum(p, 0.0)), "step": idx, "names": names,
                "heart": self.heart.hr(p, cap, offset)}

    def paced(self, p: np.ndarray, cap: Capacity) -> np.ndarray:
        """the riding as far as W' lasts (models.stamina.RideStamina's W' balance on the day's CP and W', fatigue
        included): where it runs out, the rest of that stretch above CP is ridden just under CP. One pass per
        stretch cut short (each pass finds the first)."""
        p = np.array(p, float)
        cp, wp = cap.cp * cap.form, cap.w_prime * cap.form
        for _ in range(self.PACE_PASSES):                       # a ride's few efforts that run W' out
            out = np.flatnonzero(RideStamina(p, cp, wp, cap.lt1).w_balance() < 0)
            if not out.size:
                break
            i = int(out[0])
            below = np.flatnonzero(p[i:] <= cp)
            j = i + int(below[0]) if below.size else p.size
            p[i:j] = self.EASE * cp
        return p

    def ramp(self, p: np.ndarray, start: int, end: int, ftp: float, all_out) -> None:
        """the ramp (seconds start..end of `p`) ridden until its mean power over some effort duration
        (PhysioInputs.EFFORT_S) reaches today's all-out power for it (one draw of the day's scatter), then easy for
        the rest: no stretch of a ramp beats the rider's own curve (2026-11-22: ended at the 2-min power, its last
        5-20 min sat up to 23 % above it - CP +14 %, W' -34 % in the fit)."""
        q = p[start:end]
        c = np.r_[0.0, np.cumsum(q)]
        luck = np.exp(self.ALL_OUT_SD * self.rng.standard_normal())
        fail = q.size
        for d in (s for s in PhysioInputs.EFFORT_S if s <= q.size):     # a dozen durations
            over = np.flatnonzero((c[d:] - c[:-d]) / d >= luck * float(all_out(float(d))))
            if over.size:
                fail = min(fail, int(over[0]) + d)
        p[start + fail:end] = self.EASY_AFTER * ftp

    def effort(self, p: np.ndarray, seconds: int, ftp: float, watts: float) -> None:
        """an all-out effort `seconds` long a third into the ride, then easy for a few minutes."""
        a = len(p) // 3
        if a + seconds + 180 > len(p):
            return
        p[a:a + seconds] = watts * np.exp(self.ALL_OUT_SD * self.rng.standard_normal())
        p[a + seconds:a + seconds + 180] = self.EASY_AFTER * ftp

    def outside(self, p: np.ndarray) -> np.ndarray:
        """terrain (a slow log-normal wander) and coasting stretches."""
        a = np.exp(-1.0 / self.TERRAIN_TAU_S)
        x = lfilter([np.sqrt(1.0 - a * a)], [1.0, -a], self.rng.standard_normal(p.size))
        p = p * np.exp(self.TERRAIN_SD * x - self.TERRAIN_SD ** 2 / 2.0)
        p[self.stretches(p.size, self.COAST_EVERY_S, self.COAST_S)] = 0.0
        return p

    def surged(self, p: np.ndarray, all_out) -> np.ndarray:
        """short surges (about one every SURGE_EVERY_S, at half to most of the 5-s power), each at most today's
        all-out for its length (one draw of the day's scatter), nothing above the 5-s power (2026-11-22: surges at
        85 % of the 5-s power beat the 15-30 s curve - W' -19 %, Pmax +28 % in the fit)."""
        p = np.array(p, float)
        sprint = float(all_out(5.0))
        k = self.rng.poisson(p.size / self.SURGE_EVERY_S)
        start = self.rng.integers(0, p.size, k)
        length = self.rng.integers(self.SURGE_S[0], self.SURGE_S[1] + 1, k)
        luck = np.exp(self.ALL_OUT_SD * self.rng.standard_normal())
        level = np.minimum(sprint * self.rng.uniform(0.5, 0.85), luck * np.asarray(all_out(length.astype(float))))
        at = np.repeat(start, length) + np.arange(length.sum()) - np.repeat(np.cumsum(length) - length, length)
        keep = at < p.size
        p[at[keep]] = np.repeat(level, length)[keep]
        return np.minimum(p, sprint)

    def stretches(self, size: int, every: float, length: tuple) -> np.ndarray:
        """a mask of random stretches, about one every `every` s, each `length` (lo, hi) s long."""
        k = self.rng.poisson(size / every)
        start = self.rng.integers(0, size, k)
        return self.covered(size, start, start + self.rng.integers(length[0], length[1] + 1, k))

    @staticmethod
    def covered(size: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """a mask of `size` that is True inside any [start, end) (ends past `size` cut off)."""
        d = np.zeros(size + 1, int)
        np.add.at(d, np.minimum(start, size), 1)
        np.add.at(d, np.minimum(end, size), -1)
        return np.cumsum(d)[:size] > 0
