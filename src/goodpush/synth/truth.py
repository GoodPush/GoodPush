"""A synthetic rider's true physiology, day by day. Each morning it says what the rider can do (`morning`,
`all_out`); the day's riding then moves it on (`advance`). Two kinds:

* ModelTruth: the physio model itself run forward (PhysioModel.step) from parameters set by the profile and
  drawn from the model's prior - a fit of the generated data can be scored against the very parameters;
* SimpleTruth: an independent generator (fitness filters per kind of riding, its own power-duration curve) - the
  model meets a rider it doesn't describe exactly.

The days run one at a time: each morning's capacity sets that day's riding, which sets the next morning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..metrics.power import POWER_BINS, power_histogram
from ..models.physio import SOURCES, PhysioInputs
from ..physio.curve import PowerCurve
from ..physio.model import DYNAMICS, MAP_S, RATE_OF, RECOVERY, TRAINED, PhysioData, PhysioModel


@dataclass
class Capacity:
    """what the rider can do this morning (W, J), fresh (`form` = the share fatigue leaves of it)."""
    cp: float
    w_prime: float
    pmax: float
    fade: float
    map: float
    lt1: float
    ftp: float
    fatigue: float
    form: float

    def row(self) -> dict:
        return asdict(self)


class ModelTruth:
    KIND = "model"
    DRAWN = DYNAMICS + ("log_s_f",)    # drawn from the prior with draw_dynamics (the training response, fatigue)
    DRAW_SD = 0.5                      # ... at this share of its prior sd (full draws made riders doubling CP in
    #                                    months), and at most CLIP_SD of those from its centre
    CLIP_SD = 2.0
    ILLNESS_ETA = -1.5                 # an illness's week: its unexplained change (prior sds) on the trained states
    ILLNESS_NOISE = PhysioInputs.ILLNESS_NOISE

    def __init__(self, profile, days: pd.DatetimeIndex, illness_onsets: list, rng: np.random.Generator):
        T = len(days)
        week = (np.arange(T) + days[0].day_of_week) // 7
        ill = np.isin(np.arange(week[-1] + 1), week[days.get_indexer(pd.DatetimeIndex(illness_onsets))])
        none_i, none_f = np.zeros(0, int), np.zeros(0)
        # the model on these days without any riding yet: its parameters' layout and priors, its noise
        self.data = PhysioData(days=days.to_numpy().astype("datetime64[D]"), power_w=PhysioInputs.POWER_W,
                               hist_h=np.zeros((T, PhysioInputs.BINS)), long_h=np.zeros(T),
                               recovery_z=np.zeros((T, len(RECOVERY))), week=week, effort_day=none_i,
                               effort_s=none_f, effort_w=none_f, effort_kind=none_i, reading_day=none_i,
                               reading_w=none_f, reading_source=none_i, sources=SOURCES,
                               level_w=(profile.map_w, profile.pmax_w),
                               noise_scale=np.where(ill[week], self.ILLNESS_NOISE, 1.0))
        model = PhysioModel(self.data, PhysioInputs.EFFORT_S)
        self.spec = model.spec
        z = self.spec.mean.copy()
        if profile.draw_dynamics:
            for name in self.DRAWN:
                o, n = self.spec.offsets[name], self.spec.unpack(z)[name].size
                z[o:o + n] += self.DRAW_SD * self.spec.sd[o:o + n] * np.clip(rng.standard_normal(n), -self.CLIP_SD, self.CLIP_SD)
        p = self.spec.unpack(z)
        p["x0"][:] = self.start(profile)
        p["eta"][:] = rng.standard_normal(p["eta"].shape)
        p["eta"][np.ix_(ill, TRAINED > 0)] = self.ILLNESS_ETA        # an illness's week (Monday weeks from 0)
        self.p = p
        self.noise = np.asarray(PhysioModel.noise({k: jnp.asarray(v) for k, v in p.items()}, model.arrays))[:T]
        self.s_f = float(np.exp(p["log_s_f"][0]))
        self.ready((jnp.asarray(p["x0"]), jnp.zeros(len(TRAINED)), jnp.asarray(0.0)))

    def ready(self, carry) -> None:
        """the parameters (self.p) packed and compiled into the day step, starting from `carry`."""
        self.z = self.spec.pack(self.p)
        dyn = PhysioModel.dynamics({k: jnp.asarray(v) for k, v in self.p.items()})
        self.stepper = jax.jit(partial(PhysioModel.step, jnp.log(PhysioInputs.POWER_W), dyn, PhysioModel.gains(dyn)))
        self.carry = self.carry0 = carry
        self.day = 0

    def settle(self, usual: list[list[np.ndarray]]) -> None:
        """the start in balance with the rider's usual training (`usual`: each day's rides, ridden at the start): a
        constant daily dose d holds a trained state at floor + effect . d, so the floors go where the usual dose holds
        the profile's watts; the adaptation pool and fatigue start at their steady levels too. The training response
        itself (effects, time constants) stays as drawn. Without this, a rider whose hours are far above what the
        prior's centre sustains (a pro) gained CP by a third within months."""
        hist, long = np.mean([self.hours(r) for r in usual], axis=0), np.mean([self.long(r) for r in usual])
        p = {k: jnp.asarray(v) for k, v in self.p.items()}
        x0 = p["x0"]
        dose = PhysioModel.doses(jnp.log(PhysioInputs.POWER_W), PhysioModel.capacity(x0), jnp.asarray(hist),
                                 jnp.asarray(long))
        gain = PhysioModel.gains(p) @ dose                           # what a day adds to each state's pool
        held = np.asarray(gain * jnp.exp(p["log_tau"]))
        self.p["floor"][TRAINED > 0] = np.asarray(x0)[TRAINED > 0] - held[TRAINED > 0]
        cp = float(PhysioModel.capacity(x0)["cp"])
        fat = float(hist @ PhysioInputs.POWER_W ** 2 / cp ** 2 / (1.0 - np.exp(-1.0 / np.exp(float(p["log_tau_f"][0])))))
        rate = 1.0 - np.exp(-np.exp(np.asarray(p["log_rate"]))[RATE_OF] * np.exp(-np.exp(float(p["log_gamma"][0])) * fat))
        self.ready((x0, jnp.asarray(np.asarray(gain) / rate * TRAINED), jnp.asarray(fat)))

    @staticmethod
    def hours(rides: list[np.ndarray]) -> np.ndarray:
        """a day's rides (1-Hz power each) as the model's hours per power bin (as PhysioInputs makes them)."""
        H = np.array([power_histogram(r) for r in rides], float).reshape(len(rides), POWER_BINS)   # a day: 0-2 rides
        return PhysioInputs.power_hours(H).sum(0)

    @staticmethod
    def long(rides: list[np.ndarray]) -> float:
        return float(PhysioInputs.long_hours([len(r) for r in rides]).sum())

    @staticmethod
    def start(profile) -> np.ndarray:
        """the model's unconstrained states for the profile's watts (fade = 1 / durability)."""
        w = float(PowerCurve.w_prime_for(MAP_S, profile.map_w, profile.cp_w, profile.pmax_w))
        hour = float(PowerCurve.power(3600.0, profile.cp_w, w, profile.pmax_w, profile.fade))
        share = np.array([profile.lt1_w / hour, profile.cp_w / profile.map_w])
        if not ((share > 0.25) & (share < 0.98)).all():
            raise ValueError("LT1 / 1-h power and CP / MAP must be between 0.25 and 0.98")
        f_e, f_t = np.log(share / (1.0 - share))
        return np.array([np.log(profile.map_w), f_e, f_t, np.log(profile.pmax_w), np.log(1.0 / profile.fade)])

    def morning(self) -> Capacity:
        x, _, fat = self.carry
        c = {k: float(v) for k, v in PhysioModel.capacity(x).items()}
        ftp = float(PhysioModel.quantities(x)[3])
        return Capacity(c["cp"], c["w_prime"], c["pmax"], c["fade"], c["map"], c["lt1"], ftp, float(fat),
                        float(np.exp(-self.s_f * float(fat))))

    def all_out(self, t) -> np.ndarray:
        """what the rider can hold all-out for t seconds this morning (fatigue included)."""
        x, _, fat = self.carry
        return np.asarray(PhysioModel.curve(x, jnp.asarray(t, float))) * np.exp(-self.s_f * float(fat))

    def advance(self, rides: list[np.ndarray]) -> None:
        """the day's rides (1-Hz power each) -> the next morning."""
        hist = self.hours(rides)
        inputs = (jnp.asarray(hist), jnp.asarray(self.long(rides)), jnp.asarray(hist @ PhysioInputs.POWER_W ** 2),
                  jnp.asarray(self.noise[self.day]), jnp.asarray(True))
        self.carry = self.stepper(self.carry, inputs)[0]
        self.day += 1

    def params(self) -> dict:
        """the true parameters by name (as a fit names them), the noise terms left out (`settle` first: it sets the
        floors)."""
        return {n: float(v) for n, v in zip(self.spec.names(), self.z) if not n.startswith("eta")}


class SimpleTruth:
    """Fitness per kind of riding (hours below LT1, LT1-CP, above CP), each an exponential filter of its daily hours
    (TAU_D); CP is a detrained floor plus each fitness's part, LT1 a share of CP that endurance hours raise, W' moves a
    little with severe work, Pmax wanders. Fatigue: a 7-day filter of the daily load (TSS) costing all-out power.
    Its curve: P(t) = CP + W' / (t + W' / (Pmax - CP)), falling with log time past T_LONG."""
    KIND = "simple"
    KINDS = ("easy", "threshold", "severe")
    TAU_D = np.array([60.0, 35.0, 21.0])
    SHARE = np.array([0.4, 0.35, 0.25])     # each kind's part of the trained CP above the floor at the start
    FLOOR = 0.85                            # CP with no riding, as a share of the start's
    LT1_FLOOR = 0.9                         # LT1 / CP with no riding, as a share of the start's
    LT1_MAX = 0.92                          # LT1 / CP at most
    W_SEVERE = 0.15                         # W' change per relative change of severe fitness
    PMAX_WALK = 0.003                       # Pmax's daily random walk (log)
    FATIGUE_TAU_D = 7.0
    FATIGUE_COST = 6e-4                     # all-out power lost per TSS a day of recent load (log)
    T_LONG = 1200.0
    ILLNESS_DROP = 0.05                     # an illness takes this share of CP, W' and LT1 ...
    ILLNESS_TAU_D = 14.0                    # ... and gives it back over about this long
    MIN_HOURS = 0.01                        # a day's hours of a kind at least (a kind never ridden still has a part)

    def __init__(self, profile, days: pd.DatetimeIndex, illness_onsets: list, rng: np.random.Generator):
        self.rng = rng
        self.days = days
        self.cp0, self.lt1_0, self.fade = profile.cp_w, profile.lt1_w, profile.fade
        a, s = profile.map_w - profile.cp_w, profile.pmax_w - profile.cp_w
        self.w0 = MAP_S * a / (1.0 - a / s)
        self.pmax = profile.pmax_w
        self.onsets = pd.DatetimeIndex(illness_onsets)
        self.day = 0
        self.calibrate(np.ones(len(self.KINDS)), 0.0)        # the profile's start, until `settle`

    def calibrate(self, fit0: np.ndarray, atl: float) -> None:
        """gains such that daily hours `fit0` of each kind hold the profile's start; that load's fatigue."""
        self.fit0 = np.maximum(fit0, self.MIN_HOURS)
        self.fitness = self.fit0.copy()
        self.gain = self.SHARE * (1.0 - self.FLOOR) * self.cp0 / self.fit0
        self.lt1_gain = (1.0 - self.LT1_FLOOR) * (self.lt1_0 / self.cp0) / self.fit0[0]
        self.atl = atl

    def kinds(self, p: np.ndarray, cp: float, lt1: float) -> np.ndarray:
        """hours below LT1, LT1-CP and above CP in 1-Hz power p."""
        return np.array([(p > 0) & (p < lt1), (p >= lt1) & (p < cp), p >= cp]).sum(1) / 3600.0

    def settle(self, usual: list[list[np.ndarray]]) -> None:
        """the start in balance with the rider's usual training (each day's rides, ridden at the start)."""
        days = [np.concatenate(r) if r else np.zeros(0) for r in usual]
        hours = np.mean([self.kinds(p, self.cp0, self.lt1_0) for p in days], axis=0)
        load = float(np.mean([100.0 * np.sum((p / self.cp0) ** 2) / 3600.0 for p in days]))
        self.calibrate(hours, load)

    def illness(self) -> float:
        """the share an illness leaves today (1 = none)."""
        d = (self.days[self.day] - self.onsets).days.to_numpy(float)
        d = d[d >= 0]
        return float(np.prod(1.0 - self.ILLNESS_DROP * np.exp(-d / self.ILLNESS_TAU_D)))

    def levels(self) -> tuple[float, float, float]:
        ill = self.illness()
        cp = (self.FLOOR * self.cp0 + self.gain @ self.fitness) * ill
        rel = self.fitness[2] / self.fit0[2]
        w = self.w0 * float(np.clip(1.0 + self.W_SEVERE * (rel - 1.0), 0.8, 1.3)) * ill
        share = min(self.LT1_FLOOR * self.lt1_0 / self.cp0 + self.lt1_gain * self.fitness[0], self.LT1_MAX)
        return cp, w, share * cp

    @classmethod
    def power(cls, t, cp, w_prime, pmax, fade) -> np.ndarray:
        t = np.asarray(t, float)
        short = cp + w_prime / (t + w_prime / max(pmax - cp, 1.0))
        return short * (1.0 - fade * np.log(np.maximum(t, cls.T_LONG) / cls.T_LONG))

    def morning(self) -> Capacity:
        cp, w, lt1 = self.levels()
        curve = self.power(np.array([MAP_S, 3600.0]), cp, w, self.pmax, self.fade)
        return Capacity(cp, w, self.pmax, self.fade, float(curve[0]), lt1, float(curve[1]), self.atl,
                        float(np.exp(-self.FATIGUE_COST * self.atl)))

    def all_out(self, t) -> np.ndarray:
        cp, w, _ = self.levels()
        return self.power(t, cp, w, self.pmax, self.fade) * np.exp(-self.FATIGUE_COST * self.atl)

    def advance(self, rides: list[np.ndarray]) -> None:
        cp, _, lt1 = self.levels()
        p = np.concatenate(rides) if rides else np.zeros(0)
        k = np.exp(-1.0 / self.TAU_D)
        self.fitness = self.fitness * k + self.kinds(p, cp, lt1) * (1.0 - k)
        kf = np.exp(-1.0 / self.FATIGUE_TAU_D)
        self.atl = self.atl * kf + 100.0 * float(np.sum((p / cp) ** 2)) / 3600.0 * (1.0 - kf)
        self.pmax *= float(np.exp(self.PMAX_WALK * self.rng.standard_normal()))
        self.day += 1

    def params(self) -> dict:
        return {"fitness_tau_d": self.TAU_D.tolist(), "gain_w_per_h": self.gain.tolist(), "w_prime_start": self.w0,
                "fatigue_cost": self.FATIGUE_COST, "fatigue_tau_d": self.FATIGUE_TAU_D}


TRUTHS = {t.KIND: t for t in (ModelTruth, SimpleTruth)}
