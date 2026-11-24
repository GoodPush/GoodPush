"""Model-predictive weekly load: how much to ride each week so the model's forecast is best on the days
that matter, within your hours.

The week's sessions come from outside (the plan's template: its key sessions, tests, races, which days
are easy). Per week two numbers are chosen:

* easy hours e_w (spread over the week's easy days, the long day weighted LONG_WEIGHT),
* key scale s_w in [0, 1]: how much of the week's key sessions to keep (1 = all of them, less = an
  unloading week).

Forecast = PhysioModel.forecast from the last fitted day, no process noise. How much fatigue costs -
slower adaptation, less on the day - is learned from your data, not a limit (your sleep / HRV / resting
HR / bedtime act through readiness, day to day).
Objective (minimised):

- sum over targets of weight x mean log power over the target's durations (all-out with fatigue, or
  capacity), + TIME_COST per hour ridden, + FATIGUE_PENALTY x (fatigue above the backstop)^2 (the
  highest fatigue in your history: never planned beyond what you have been through),
  + KEEP_WEIGHT x (1 - s_w)^2 (keep the template's key sessions unless the forecast gains),
  + RAMP_WEIGHT x (block shape violated, hours)^2: within a block each loading week at least RAMP_MIN
  more hours than the week before, the recovery week at most RECOVERY_MAX of the week before (the
  block structure you train in; the model picks the levels),
  + STEADY_WEIGHT x (hours short of the week's steady hours)^2: time on the bike grows steadily (the
  volume ramp: what you have been riding, more after each loading week) unless the forecast clearly
  gains otherwise. A week far from the target days barely moves the forecast - without this its hours
  would be wherever the optimiser stops.

Smooth in (e, s), so L-BFGS-B with box bounds and JAX gradients; a few dozen variables, < 1 s.

The loss is compiled once for the class, with the problem's arrays as an argument; days, weeks and
targets are padded up to buckets (padding has no load, no weight and is masked out), so a re-plan with
a day less to go or another phase reuses the compiled loss instead of compiling a new one.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .model import DAY_BUCKET, MAP_S, STATES, ParamSpec, PhysioModel, bucket, pad

TIME_COST = 2e-4            # log-power units per hour: an hour must buy at least 0.02 % on the target
FATIGUE_PENALTY = 1.0
KEEP_WEIGHT = 2e-3          # ~ what 0.2 % on the target is worth
RAMP_WEIGHT = 0.05          # per hour^2 short of the block shape
RAMP_MIN = 0.08             # loading weeks: +8 % hours a week within a block
STEADY_WEIGHT = 1e-3        # per hour^2 short of the steady hours: ~ 0.1 % on the target; decides a flat week
RECOVERY_MAX = 0.65         # recovery week: at most 65 % of the push week's hours
LONG_AFTER_H = 2.0


@dataclass
class Target:
    day: int                 # forecast day index (0 = the morning after the last fitted day)
    weight: float
    durations: tuple         # seconds
    all_out: bool = True     # all-out with fatigue (race) or capacity (fitness at the end)


@dataclass
class DoseResult:
    easy_h: np.ndarray       # (W,)
    key_scale: np.ndarray    # (W,)
    states: np.ndarray       # (D, 5) mornings
    fatigue: np.ndarray      # (D,)
    objective: float
    converged: bool


class DosePlan:
    TARGET_BUCKET = 4          # targets padded to a multiple of this
    MAX_DURATIONS = 4          # durations per target, padded

    def __init__(self, model: PhysioModel, z, fixed_hist, key_hist, easy_share, week, easy_hist_per_h,
                 easy_max_h, targets: list[Target], fatigue_cap: float, long_share=None, ramp=None, recover=None,
                 steady=None):
        """fixed_hist / key_hist (D, B): hours per power bin of the sessions that stay / that scale with s;
        easy_share (D,): each day's share of its week's easy hours (sums to 1 per week, 0 on non-easy days);
        week (D,) week index; easy_hist_per_h (B,): one easy hour's histogram; easy_max_h (W,);
        long_share (D,) optional: share of the week's easy hours that count as one long ride on that day;
        ramp / recover (W-1,) optional: week w+1 is a loading / the recovery week after loading week w in
        the same block (both whole weeks) - the block shape applies;
        fatigue_cap: the backstop (your highest fatigue so far); steady (W,) optional: hours each week should
        reach (0: none) - the volume ramp's steady growth."""
        self.model = model
        self.easy_max = np.asarray(easy_max_h, float)
        self.D, self.W = len(week), len(self.easy_max)
        D, W = self.padded(self.D, self.W)
        fixed, key = pad(fixed_hist, D), pad(key_hist, D)
        week = pad(np.asarray(week, int), D)
        self.fixed_wh = np.bincount(week, fixed.sum(1), minlength=W)        # hours per week
        self.key_wh = np.bincount(week, key.sum(1), minlength=W)
        p = model.spec.unpack(jnp.asarray(z))
        none = np.zeros(max(self.W - 1, 0))
        self.problem = dict(
            power_w=model.power_w, dynamics=model.dynamics(p), s_f=jnp.exp(p["log_s_f"][0]),
            carry=model.simulate(jnp.asarray(z))[2],       # today's state: once, not per evaluation
            fixed=jnp.asarray(fixed), key=jnp.asarray(key), share=jnp.asarray(pad(easy_share, D)),
            week=jnp.asarray(week), easy=jnp.asarray(easy_hist_per_h, float),
            easy_max=jnp.asarray(pad(self.easy_max, W)),
            long_share=jnp.asarray(pad(np.zeros(self.D) if long_share is None else long_share, D)),
            ramp=jnp.asarray(pad(none if ramp is None else ramp, W - 1)),
            recover=jnp.asarray(pad(none if recover is None else recover, W - 1)),
            steady=jnp.asarray(pad(np.zeros(self.W) if steady is None else np.asarray(steady, float), W)),
            fixed_wh=jnp.asarray(self.fixed_wh), key_wh=jnp.asarray(self.key_wh),
            day_mask=jnp.asarray(pad(np.ones(self.D), D)), week_mask=jnp.asarray(pad(np.ones(self.W), W)),
            cap=jnp.asarray(float(fatigue_cap)), **self.target_arrays(targets))

    @classmethod
    def target_arrays(cls, targets: list[Target]) -> dict:
        """the targets as padded arrays (TARGET_BUCKET targets, MAX_DURATIONS durations; padding weighs 0)."""
        K = bucket(len(targets), cls.TARGET_BUCKET)
        durations = np.full((K, cls.MAX_DURATIONS), MAP_S)          # padding: any valid duration, masked
        n_dur = np.zeros(K)
        for i, t in enumerate(targets):           # a handful of targets
            durations[i, :len(t.durations)] = t.durations
            n_dur[i] = len(t.durations)
        return dict(target_day=jnp.asarray(pad(np.array([t.day for t in targets], int), K)),
                    target_weight=jnp.asarray(pad(np.array([t.weight for t in targets], float), K)),
                    target_all_out=jnp.asarray(pad(np.array([t.all_out for t in targets], bool), K)),
                    durations=jnp.asarray(durations),
                    duration_mask=jnp.asarray(np.arange(cls.MAX_DURATIONS)[None, :] < n_dur[:, None], float))

    @staticmethod
    def target_gain(q: dict, X, F, s_f):
        """sum over targets of weight x mean log power over its durations (all-out with fatigue, or capacity)."""
        pw = PhysioModel.curve(X[q["target_day"]][:, None, :], q["durations"])              # (K, durations)
        pw = jnp.where(q["target_all_out"][:, None], pw * jnp.exp(-s_f * F[q["target_day"]])[:, None], pw)
        mean_log = (jnp.log(pw) * q["duration_mask"]).sum(1) / jnp.maximum(q["duration_mask"].sum(1), 1.0)
        return jnp.sum(q["target_weight"] * mean_log)

    @classmethod
    def padded(cls, days: int, weeks: int) -> tuple[int, int]:
        """the bucket sizes (days, weeks) a problem of this size is solved at."""
        d = bucket(days, DAY_BUCKET)
        return d, max(d // 7 + 2, weeks)

    @staticmethod
    def inputs(q: dict, e, s):
        """daily histograms (D, B) and hours past 2 h (D,) for weekly easy hours e and key scales s."""
        easy_day = e[q["week"]] * q["share"]
        hist = q["fixed"] + s[q["week"]][:, None] * q["key"] + easy_day[:, None] * q["easy"][None, :]
        ride_h = q["fixed"].sum(1) + s[q["week"]] * q["key"].sum(1) + easy_day
        long = jnp.where(q["long_share"] > 0, jax.nn.softplus(4.0 * (ride_h - LONG_AFTER_H)) / 4.0, 0.0)
        return hist, long

    @staticmethod
    def forecast(q: dict, e, s):
        """morning states (D, 5) and fatigue (D,) for easy hours e and key scales s (padded)."""
        hist, long = DosePlan.inputs(q, e, s)
        n = len(long)
        _, (X, F) = PhysioModel.run(q["power_w"], q["dynamics"], hist, long, jnp.zeros((n, len(STATES))), q["carry"])
        return X, F

    @staticmethod
    def loss(theta, q: dict):
        W = len(q["easy_max"])
        e = theta[:W] * q["easy_max"]
        s = theta[W:]
        X, F = DosePlan.forecast(q, e, s)
        gain = DosePlan.target_gain(q, X, F, q["s_f"])
        hours = jnp.sum(q["fixed"].sum(1)) + jnp.sum(s[q["week"]] * q["key"].sum(1)) + jnp.sum(e)
        over = jax.nn.relu(F - q["cap"]) * q["day_mask"]
        return (-gain + TIME_COST * hours + FATIGUE_PENALTY * jnp.sum(over ** 2)
                + KEEP_WEIGHT * jnp.sum(q["week_mask"] * (1.0 - s) ** 2) + RAMP_WEIGHT * DosePlan.block_shape(q, e, s)
                + STEADY_WEIGHT * DosePlan.steady_short(q, e, s))

    @staticmethod
    def block_shape(q: dict, e, s):
        """squared hours by which the weeks miss the block shape (ramp up, then recover)."""
        h = q["fixed_wh"] + s * q["key_wh"] + e
        short = jax.nn.relu((1.0 + RAMP_MIN) * h[:-1] - h[1:])
        over = jax.nn.relu(h[1:] - RECOVERY_MAX * h[:-1])
        return jnp.sum(q["ramp"] * short ** 2 + q["recover"] * over ** 2)

    @staticmethod
    def steady_short(q: dict, e, s):
        """squared hours by which the weeks fall short of their steady hours."""
        h = q["fixed_wh"] + s * q["key_wh"] + e
        return jnp.sum(jax.nn.relu(q["steady"] - h) ** 2)

    value_and_grad = staticmethod(jax.jit(jax.value_and_grad(loss.__func__)))   # once for the class

    def weekly_hours(self, e, s) -> np.ndarray:
        return np.asarray(self.fixed_wh[:self.W] + s * self.key_wh[:self.W] + e)

    def objective(self, theta):
        v, g = self.value_and_grad(jnp.asarray(theta), self.problem)
        return float(v), np.asarray(g, float)

    def solve(self, e0=None, s0=None) -> DoseResult:
        W = len(self.problem["easy_max"])
        e0 = 0.5 * self.easy_max if e0 is None else np.asarray(e0, float)
        s0 = np.ones(self.W) if s0 is None else np.asarray(s0, float)
        theta0 = np.r_[pad(np.where(self.easy_max > 0, e0 / np.maximum(self.easy_max, 1e-9), 0.0), W),
                       s0, np.ones(W - self.W)]                         # padded weeks: no easy hours, s = 1
        r = minimize(self.objective, np.clip(theta0, 0, 1), jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * (2 * W), options=dict(maxiter=500))
        e, s = r.x[:W] * np.asarray(self.problem["easy_max"]), r.x[W:]
        X, F = self.forecast(self.problem, jnp.asarray(e), jnp.asarray(s))
        return DoseResult(e[:self.W], s[:self.W], np.asarray(X)[:self.D], np.asarray(F)[:self.D], float(r.fun),
                          bool(r.success))


@dataclass
class MixResult:
    choice: np.ndarray       # (S,) the chosen candidate of each key slot
    objective: float         # mean loss over the parameter sets with it
    start: float             # ... with the starting choice
    rounds: int


class MixPlan:
    """Which session on each key day: the one that makes the forecast best (DosePlan's targets, time cost and
    fatigue backstop), averaged over parameter sets - the posterior draws, so a training effect the data barely pin
    down doesn't win by fitting noise - or the posterior mode alone.

    Discrete choice by repeated linearisation: the gradient of the mean loss by the days' power histograms prices
    every candidate session at once (first-order change of the loss if it replaced the slot's current one); all slots
    switch to their best, kept when the real loss drops, else only the most promising slot switches; until nothing
    promises a gain (or MAX_ROUNDS). One forecast + gradient per parameter set and round, BATCH sets at a time."""
    MAX_ROUNDS = 20
    TOL = 1e-7                 # least predicted gain (log power) worth a switch

    def __init__(self, model: PhysioModel, Z, base_hist, long_h, slot_day, cand_hist, cand_ok, start,
                 targets: list[Target], fatigue_cap: float):
        """Z (n, P) parameter sets; base_hist (D, B) hours per power bin of the days' other sessions; long_h (D,) hours
        past LONG_AFTER_H in a ride; slot_day (S,) the forecast day of each key slot; cand_hist (S, K, B) each
        candidate session there; cand_ok (S, K) the candidates that exist; start (S,) the candidate to start from."""
        self.D = len(long_h)
        D = bucket(self.D, DAY_BUCKET)
        self.base = pad(np.asarray(base_hist, float), D)
        self.long = jnp.asarray(pad(np.asarray(long_h, float), D))
        self.slot_day = np.asarray(slot_day, int)
        self.cand = np.asarray(cand_hist, float)
        self.ok = np.asarray(cand_ok, bool)
        self.start = np.asarray(start, int)
        self.q = dict(power_w=model.power_w, cap=jnp.asarray(float(fatigue_cap)),
                      day_mask=jnp.asarray(pad(np.ones(self.D), D)), **DosePlan.target_arrays(targets))
        Z = np.asarray(Z, float)
        self.n = len(Z)
        Z = np.concatenate([Z, np.repeat(Z[-1:], -self.n % PhysioModel.BATCH, axis=0)])   # whole batches
        self.weight = np.r_[np.ones(self.n), np.zeros(len(Z) - self.n)] / self.n                 # padding weighs 0
        each = jax.vmap(partial(self.draw_inputs, layout=model.spec.layout, a=model.arrays))
        self.sets = model.batched(each, Z)                     # each set's dynamics and today's state: once

    @staticmethod
    def draw_inputs(z, layout, a) -> dict:
        """what a forecast with parameters z needs: its dynamics, the cost of fatigue on the day, today's state."""
        p = ParamSpec.split(layout, z)
        return dict(dyn=PhysioModel.dynamics(p), s_f=jnp.exp(p["log_s_f"][0]),
                    carry=PhysioModel.simulate_of(z, layout, a)[2])

    @staticmethod
    def draw_loss(hist, long, q: dict, d: dict):
        """one parameter set's loss: - target gain + TIME_COST x hours + FATIGUE_PENALTY x (fatigue over the cap)^2."""
        _, (X, F) = PhysioModel.run(q["power_w"], d["dyn"], hist, long, jnp.zeros((len(long), len(STATES))), d["carry"])
        over = jax.nn.relu(F - q["cap"]) * q["day_mask"]
        return (-DosePlan.target_gain(q, X, F, d["s_f"]) + TIME_COST * jnp.sum(hist)
                + FATIGUE_PENALTY * jnp.sum(over ** 2))

    @staticmethod
    def batch_loss(hist, long, q: dict, d: dict, weight):
        return jnp.sum(weight * jax.vmap(MixPlan.draw_loss, in_axes=(None, None, None, 0))(hist, long, q, d))

    value_and_grad = staticmethod(jax.jit(jax.value_and_grad(batch_loss.__func__)))   # once for the class

    def hist_of(self, choice) -> np.ndarray:
        h = self.base.copy()
        np.add.at(h, self.slot_day, self.cand[np.arange(len(choice)), choice])
        return h

    def evaluate(self, choice) -> tuple[float, np.ndarray]:
        """mean loss over the parameter sets and its gradient by the days' histograms (D, B)."""
        h = jnp.asarray(self.hist_of(choice))
        value, grad, B = 0.0, np.zeros(self.base.shape), PhysioModel.BATCH
        for i in range(0, len(self.weight), B):                 # BATCH parameter sets at a time
            part = jax.tree.map(partial(self.rows, i=i, n=B), self.sets)
            v, g = self.value_and_grad(h, self.long, self.q, part, jnp.asarray(self.weight[i:i + B]))
            value, grad = value + float(v), grad + np.asarray(g)
        return value, grad

    @staticmethod
    def rows(a, i: int, n: int):
        return a[i:i + n]

    def solve(self) -> MixResult:
        S = len(self.slot_day)
        choice = self.start.copy()
        v, g = self.evaluate(choice)
        v0, rounds = v, 0
        while S and rounds < self.MAX_ROUNDS:
            rounds += 1
            price = np.where(self.ok, np.einsum("skb,sb->sk", self.cand, g[self.slot_day]), np.inf)
            promise = price[np.arange(S), choice] - price.min(1)      # predicted drop of the loss by switching
            if promise.max() <= self.TOL:
                break
            best = price.argmin(1)
            trial = np.where(promise > self.TOL, best, choice)
            vt, gt = self.evaluate(trial)
            if vt >= v:                                            # the slots interact: only the most promising one
                trial = choice.copy()
                trial[promise.argmax()] = best[promise.argmax()]
                vt, gt = self.evaluate(trial)
                if vt >= v:
                    break
            choice, v, g = trial, vt, gt
        return MixResult(choice, v, v0, rounds)
