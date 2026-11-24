"""Fitting: MAP (minimum negative log posterior, L-BFGS) and uncertainty by NUTS samples (exact in the limit;
about half an hour on years of data). A Laplace approximation (a Gaussian at the mode) was tried and removed
(2026-11-10): on real data its draws were worth about 1 in 200 by importance sampling.
The potential, its gradient and the Hessian for the preconditioner come from `NumbaPosterior` (hand-written
backward pass; the JAX potential stays the reference it is tested against).

Both work on standardised parameters w = (z - prior mean) / prior sd: every coordinate on the same
scale, so the first L-BFGS steps are sane and NUTS starts with a sensible mass matrix.
"""
from __future__ import annotations

import os
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .model import EFFECT_STATE, GAIN_EFFECT, GAIN_INDEX, STATES, PhysioModel
from .numba_posterior import NumbaPosterior


class GainShares:
    """NUTS coordinates for the training effects: per state, the log total effect at the average dose mix and
    the logit shares of its dose domains, instead of one log effect per domain. The data pin down what the
    usual mix of training does, not each domain alone (z1 and z2 mostly come together), a curved ridge in
    log effects that NUTS crawled along (2026-11-01: ESS 4-8 of 400 for f_E's gains). The map is one to one
    and the potential is Jacobian-corrected: the same posterior, other coordinates.
    For the LEVEL states the total is the response after HORIZON days of the usual dose instead of the long-run one,
    r = log total + log(1 - exp(-HORIZON / tau)), and the detrained floor's entry is the level there, floor + exp(r).
    For a short tau r is the long-run effect, for a long one the first gain x HORIZON: what the data pin down in
    both cases. With the long-run total the chains froze for 25-200 draws at long time constants, where a small step
    in log tau moved the total and the floor a long way (2026-11-12). Years of rides fix that level; how it splits
    into floor and training lift they barely say, and with the lift a log coordinate that ridge is curved
    (2026-11-12: 1.8 % divergent draws, most of them where MAP's lift was large and its floor low). Both are shears
    (r moves by a function of log tau, the level by one of r): their Jacobian is 1.
    v: standardised w with the log-effect entries replaced by these coordinates, the LEVEL states' floors by levels."""
    # not LT1's share of 1-h power: its lift runs to many logits (the share saturates), so no level is pinned down
    LEVEL = ("map", "f_t")
    HORIZON = 84.0                                 # days of the usual dose the LEVEL states' response is taken at

    def __init__(self, model: PhysioModel, log_dose: np.ndarray):
        """log_dose: log of each gain's average daily dose (its domain's), from the mode's path (`average_doses`)."""
        names = model.spec.names()
        self.index = np.array([i for i, n in enumerate(names) if n.startswith("log_effect")])
        rows = EFFECT_STATE
        self.groups = [np.flatnonzero(rows == r) for r in np.unique(rows)]
        self.log_dose = np.asarray(log_dose, float)
        self.mean, self.sd = model.spec.mean[self.index], model.spec.sd[self.index]
        # per LEVEL state: its floor's entry, its total-effect coordinate (the group's first) and the floor's prior sd
        level = [g for g in self.groups if STATES[rows[g[0]]] in self.LEVEL]
        self.floor = np.array([names.index(f"floor[{rows[g[0]]}]") for g in level], int)
        self.total = self.index[[g[0] for g in level]]
        self.floor_sd = model.spec.sd[self.floor]
        # ... and its detraining time constant's entry (not changed: v holds the standardised log tau there)
        self.tau = np.array([names.index(f"log_tau[{rows[g[0]]}]") for g in level], int)
        self.tau_mean, self.tau_sd = model.spec.mean[self.tau], model.spec.sd[self.tau]

    def horizon_share(self, w_tau, xp=np):
        """log(1 - exp(-HORIZON / tau)) for the LEVEL states, from their standardised log tau: how much of the long-run
        response HORIZON days give (log, <= 0)."""
        return xp.log(-xp.expm1(-self.HORIZON * xp.exp(-(self.tau_mean + self.tau_sd * w_tau))))

    @staticmethod
    def average_doses(model: PhysioModel, z: np.ndarray) -> np.ndarray:
        """log of the average daily dose of each training effect's domains on the path at z (summed when an effect
        is shared by domains, GAIN_EFFECT)."""
        X, _, _ = model.simulate(z)
        d = model.data
        D = jax.vmap(PhysioModel.doses, in_axes=(None, 0, 0, 0, 0))(
            jnp.log(model.power_w), PhysioModel.capacity(X), jnp.asarray(d.hist_h, float), jnp.asarray(d.long_h, float),
            jnp.asarray(d.hr_hours))
        mean = np.asarray(D).mean(0)[GAIN_INDEX[1]]
        return np.log(np.maximum(np.bincount(GAIN_EFFECT, weights=mean), 1e-12))

    def log_effects(self, q):
        """coordinates q (one per training effect) -> log effects, and the log-determinant of its Jacobian (JAX)."""
        out, logdet = [], 0.0
        for g in self.groups:                      # three states with gains
            logits = jnp.concatenate([jnp.zeros(1), q[g[1:]]])
            log_share = logits - jax.scipy.special.logsumexp(logits)
            out.append(q[g[0]] + log_share - self.log_dose[g])
            J = jnp.concatenate([jnp.ones((len(g), 1)), (jnp.eye(len(g)) - jnp.exp(log_share)[None, :])[:, 1:]], axis=1)
            logdet = logdet + jnp.linalg.slogdet(J)[1]
        return jnp.concatenate(out), logdet

    def to_w(self, v):
        """v -> standardised w, and the log-determinant of dw/dv (JAX)."""
        w = v.at[self.floor].add(-jnp.exp(v[self.total]) / self.floor_sd)        # level -> floor
        w = w.at[self.total].add(-self.horizon_share(v[self.tau], jnp))           # response at HORIZON -> long run
        y, logdet = self.log_effects(w[self.index])
        return w.at[self.index].set((y - self.mean) / self.sd), logdet - jnp.sum(jnp.log(self.sd))

    def w_of(self, v):
        """v -> standardised w (JAX)."""
        return self.to_w(v)[0]

    def from_w(self, w: np.ndarray) -> np.ndarray:
        """standardised w (..., D) -> v (numpy)."""
        v = np.array(w, float)
        y = self.mean + self.sd * v[..., self.index]
        q = np.zeros_like(y)
        for g in self.groups:
            t = y[..., g] + self.log_dose[g]
            q[..., g[0]] = np.logaddexp.reduce(t, axis=-1)
            q[..., g[1:]] = t[..., 1:] - t[..., :1]
        v[..., self.index] = q
        v[..., self.total] += self.horizon_share(v[..., self.tau])                 # long run -> response at HORIZON
        v[..., self.floor] += np.exp(v[..., self.total]) / self.floor_sd         # floor -> level
        return v


class RunClock:
    """Elapsed time of a NUTS run and a rough time to go, from the pace of the latest block (warm-up steps
    and draws cost about the same per iteration)."""

    def __init__(self, iterations_left: int):
        self.start = self.last = time.time()
        self.left = iterations_left

    def after(self, n: int, more: bool = True) -> str:
        """after a block of n iterations; `more`: the count of what's left is known (not with an ESS target)."""
        now = time.time()
        pace, self.last = (now - self.last) / max(n, 1), now
        self.left = max(self.left - n, 0)
        togo = f", ~{pace * self.left / 60:.0f} min to go" if more and self.left else ""
        return f"{(now - self.start) / 60:.1f} min so far{togo}"


class ChainProgress:
    """One progress line for chains running in parallel, instead of NumPyro's bar per chain: the count, elapsed
    and remaining time of the slowest chain (a block ends when its slowest chain does). Stands in for
    `numpyro.util.progress_bar_factory` inside `active(label)`."""

    label = ""

    def __init__(self, num_samples: int, num_chains: int, progress_rate: int | None = None):
        from tqdm.auto import tqdm
        # NumPyro < 0.20 (e.g. the Intel Mac's, with jax pinned) doesn't pass the rate: its own, 20 reports a run
        self.total, self.rate = num_samples, progress_rate or max(num_samples // 20, 1)
        self.counts = [0] * num_chains
        self.next_chain, self.lock = 0, Lock()
        self.bar = tqdm(total=num_samples, desc=f"{self.label}, slowest chain", unit="it", leave=False)

    @classmethod
    @contextmanager
    def active(cls, label: str):
        import numpyro.util
        before, cls.label = numpyro.util.progress_bar_factory, label
        numpyro.util.progress_bar_factory = cls.factory
        try:
            yield
        finally:
            numpyro.util.progress_bar_factory = before

    @classmethod
    def factory(cls, num_samples: int, num_chains: int, progress_rate: int | None = None):
        return cls(num_samples, num_chains, progress_rate).decorate

    def tick(self, increment, chain):
        """host side: `chain` (-1: not numbered yet) did `increment` more iterations; the line follows the slowest."""
        increment, chain = int(increment), int(chain)
        with self.lock:
            if chain == -1:
                chain, self.next_chain = self.next_chain, self.next_chain + 1
            self.counts[chain] += increment
            slowest = min(self.counts)
            if slowest > self.bar.n:
                self.bar.update(slowest - self.bar.n)
            if slowest >= self.total:
                self.bar.close()
        return np.asarray(chain, np.int64)

    def report(self, increment: int, chain):
        from jax.experimental import io_callback
        return io_callback(self.tick, jax.ShapeDtypeStruct((), jnp.int64), increment, chain)

    def keep(self, chain):
        return chain

    def advance(self, i, chain):
        """traced: iteration i (1-based) of `chain`, reported every `rate` iterations as NumPyro's bars are."""
        from jax import lax
        chain = jnp.asarray(chain, jnp.int64)
        chain = lax.cond(i == 1, partial(self.report, 0), self.keep, chain)
        chain = lax.cond(i % self.rate == 0, partial(self.report, self.rate), self.keep, chain)
        return lax.cond(i == self.total, partial(self.report, self.total % self.rate), self.keep, chain)

    def decorate(self, body):
        return partial(self.loop_body, body)

    def loop_body(self, body, i, vals):
        sub, chain = vals
        return body(i, sub), self.advance(i + 1, chain)


class Whitening:
    """The linear map NUTS samples through: v = v_mode + L u, NUTS on u with an identity mass. L = L_1 L_2 ... L_k,
    each stage L_i = diag(s_i) (I + V_i diag(c_i) V_i^T) with V_i orthonormal columns (low rank: a leapfrog step costs
    D x rank, not D^2). The first stage is the inverse Hessian's square root at the mode (`from_hessian`); each
    warm-up block adds one learned from that block's draws and gradients (`refined`). The mode's curvature is local:
    a long or curved ridge, or a region the mode doesn't see, is wider than it says. Learning from where the chains
    go catches that in every run, whatever the model or rider (2026-11-09: ~15 directions 10-28x wider than the
    Hessian said - start level, floor and training effect trading off - put two chains at the tree limit, 1023 steps
    a draw). Draws have covariance S and gradients S^-1: too wide directions stand out in the draws, too narrow
    ones in the gradients (`refined`)."""

    CUTOFF = 2.0      # a learned direction (or coordinate) is used when its variance is this many times off 1, either way
    MIN_VAR = 1 / 16  # a stage squeezes at most to this variance (sd x0.25; 2026-11-09: x0.5 bound 10-16 directions
    #                   in every warm-up block, keeping the step size small)
    MIN_DRAWS = 10    # fewer pooled draws: no stage

    def __init__(self, v_mode: np.ndarray, stages: tuple = ()):
        self.v_mode = np.asarray(v_mode, float)
        self.stages = tuple(stages)            # (s, V, c) each, outermost first

    @classmethod
    def from_hessian(cls, v_mode: np.ndarray, lam: np.ndarray, V: np.ndarray, min_eigen: float) -> Whitening:
        """the inverse Hessian's symmetric square root (eigenvalues lam, vectors V), in the directions with
        curvature above min_eigen (identity in the rest)."""
        keep = lam > min_eigen
        return cls(v_mode, [(np.ones(lam.size), V[:, keep], lam[keep] ** -0.5 - 1.0)])

    @property
    def rank(self) -> int:
        return sum(V.shape[1] for _, V, _ in self.stages)

    @staticmethod
    def apply(stage: tuple, u):
        """L_i u for u (..., D): numpy or JAX."""
        s, V, c = stage
        return s * (u + ((u @ V) * c) @ V.T)

    @staticmethod
    def invert(stage: tuple, x):
        """L_i^-1 x."""
        s, V, c = stage
        x = x / s
        return x + ((x @ V) * (1.0 / (1.0 + c) - 1.0)) @ V.T

    def to_v(self, u):
        for stage in reversed(self.stages):
            u = self.apply(stage, u)
        return self.v_mode + u

    def to_u(self, v):
        x = np.asarray(v) - self.v_mode
        for stage in self.stages:
            x = self.invert(stage, x)
        return x

    def refined(self, U: np.ndarray, G: np.ndarray) -> tuple[Whitening, tuple]:
        """this whitening with the stages learned from draws U (n, D) of u and the potential's gradients G (n, D) at
        them (positions move over with to_u(old.to_v(u))), and the new stages: `misfit` (directions), then each
        coordinate's scale from what is left (variance of the draws)."""
        X, G = U - U.mean(0), G - G.mean(0)
        misfit = self.misfit(X, G)
        X = self.invert(misfit, X)
        var = np.mean(X ** 2, 0)
        var = np.where((var > self.CUTOFF) | (var < 1 / self.CUTOFF), np.maximum(var, self.MIN_VAR), 1.0)
        stages = (misfit, (np.sqrt(var), np.zeros((X.shape[1], 0)), np.zeros(0)))
        return Whitening(self.v_mode, self.stages + stages), stages

    MISFIT = 0.25     # a direction is a candidate when (x - g)'s mean square along it is above this (variance 2 or 1/2: 0.5)

    @classmethod
    def misfit(cls, X: np.ndarray, G: np.ndarray) -> tuple:
        """the directions where the whitening is wrong, and their variance. For a Gaussian with covariance S, each
        draw and its gradient satisfy x = S g, so x - g = (I - S^-1) x is zero wherever S is already 1 and lies only
        in the wrong directions, wide or narrow: its singular vectors find them with no few-draws noise (a principal
        direction of the draws alone is off by ~D / n: 100-200 draws in 1000-2000 dimensions left a 28x ridge at
        ~4x, 2026-11-09). In that subspace (fewer dimensions than draws) the variance is the geometric mean of the
        draws' covariance and the inverse of the gradients' (exact for a Gaussian: the subspace is invariant)."""
        n = len(X)
        _, sv, Vt = np.linalg.svd(X - G, full_matrices=False)
        Q = Vt[:n // 2][sv[:n // 2] ** 2 / n > cls.MISFIT].T
        x, g = X @ Q, G @ Q
        lam, E = np.linalg.eigh(cls.geometric_mean(x.T @ x / n, np.linalg.inv(g.T @ g / n)))
        lam = np.maximum(lam, cls.MIN_VAR)
        keep = (lam > cls.CUTOFF) | (lam < 1 / cls.CUTOFF)
        return np.ones(X.shape[1]), Q @ E[:, keep], np.sqrt(lam[keep]) - 1.0

    @staticmethod
    def geometric_mean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """A # B = A^1/2 (A^-1/2 B A^-1/2)^1/2 A^1/2 for symmetric positive definite A, B: the S with S A^-1 S = B."""
        a, P = np.linalg.eigh(A)
        root, inv_root = (P * np.sqrt(a)) @ P.T, (P / np.sqrt(a)) @ P.T
        m, R = np.linalg.eigh(inv_root @ B @ inv_root)
        out = root @ ((R * np.sqrt(np.maximum(m, 0.0))) @ R.T) @ root
        return 0.5 * (out + out.T)

    @staticmethod
    def renamed(stages: tuple, names: list[str], new_names: list[str]) -> tuple | None:
        """stages over the parameters `names`, put on a vector named `new_names` by name: parameters they didn't have
        (e.g. new weeks' eta) get none of the correction (scale 1, zero rows in V). Parameters `new_names` doesn't have
        (e.g. a backtest fold's later weeks) are dropped: each stage becomes its restriction to the kept ones,
        diag(s_k)(I + V_k C V_k^T), back in the stage form - V_k = Q R, R C R^T = U D U^T, so V = Q U (orthonormal
        again) and c = D (still > -1: R's singular values are at most 1)."""
        at = {n: i for i, n in enumerate(new_names)}
        kept = np.array([n in at for n in names], bool)
        pos = np.array([at[n] for n in np.asarray(names)[kept]], int)
        out = []
        for s, V, c in stages:
            V, c = np.asarray(V)[kept], np.asarray(c)
            if not kept.all() and V.shape[1]:
                Q, R = np.linalg.qr(V)
                d, U = np.linalg.eigh(R @ (c[:, None] * R.T))
                live = np.abs(d) > 1e-12                    # directions that lived only on dropped parameters: gone
                V, c = Q @ U[:, live], d[live]
            s2, V2 = np.ones(len(new_names)), np.zeros((len(new_names), V.shape[1]))
            s2[pos], V2[pos] = np.asarray(s)[kept], V
            out.append((s2, V2, c))
        return tuple(out)

    @staticmethod
    def describe(stages: tuple, n: int) -> str:
        (_, _, c), (s, _, _) = stages
        sd, rescaled = 1.0 + c, s != 1.0
        parts = [f"{int((sd > 1).sum())} directions wider" + (f" (up to x{sd.max():.1f})" if (sd > 1).any() else ""),
                 f"{int((sd < 1).sum())} narrower" + (f" (down to x{sd.min():.2f})" if (sd < 1).any() else ""),
                 f"{int(rescaled.sum())} coordinates rescaled"
                 + (f" (x{s[rescaled].min():.2f}-x{s[rescaled].max():.1f})" if rescaled.any() else "")]
        return f"whitening learned from {n} draws: " + ", ".join(parts)


@dataclass
class MapResult:
    z: np.ndarray
    neg_log_post: float
    iterations: int
    converged: bool
    message: str


class PhysioFit:
    NONFINITE = 1e20
    RESTARTS = 5
    RESTART_TOL = 0.1
    MIN_EIGEN = 1.0         # curvature floor, standardised: the prior alone gives 1, so never wider than the prior
    # NUTS preconditioner: the inverse Hessian's square root, kept in the directions with curvature above this
    # (identity in the rest: <= 5 % off in scale there). ~600 of 2455 directions on real data (2026-11-01):
    # a quarter of a dense mass matrix's work per leapfrog step
    LOW_RANK_EIGEN = 1.1

    def __init__(self, model: PhysioModel):
        self.model = model
        self.callback = None
        self.fast = NumbaPosterior(model)      # value + gradient for MAP and NUTS (several times JAX's speed)

    @staticmethod
    def potential_of(w, layout, a):
        """negative log posterior at standardised parameters w, on the model's padded arrays `a`."""
        return -PhysioModel.log_posterior_of(a["mean"] + a["sd"] * w, layout, a)

    # compiled once for the class per shape bucket (and kept on disk): a re-fit as days are added reuses it
    value_and_grad = staticmethod(jax.jit(jax.value_and_grad(potential_of.__func__), static_argnums=1))

    def potential(self, w):
        return self.potential_of(w, self.model.spec.layout, self.model.arrays)

    def to_w(self, z) -> np.ndarray:
        return (np.asarray(z, float) - self.model.spec.mean) / self.model.spec.sd

    def to_z(self, w) -> np.ndarray:
        return self.model.spec.mean + self.model.spec.sd * np.asarray(w, float)

    def objective(self, w: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = self.fast.value_and_grad(w)
        if not np.isfinite(v):                       # a wild line-search trial: tell it to step back
            return self.NONFINITE, np.zeros_like(w)
        return float(v), np.asarray(g, float)

    def map(self, z0: np.ndarray | None = None, maxiter: int = 5000, callback=None) -> MapResult:
        """posterior mode from z0 (default: prior means). `callback(z)` after each iteration."""
        w0 = np.zeros(self.model.spec.size) if z0 is None else self.to_w(z0)
        self.callback = callback
        best, its = None, 0
        for _ in range(self.RESTARTS):              # a restart drops L-BFGS's memory after a bad line search
            r = minimize(self.objective, w0, jac=True, method="L-BFGS-B", callback=self.report if callback else None,
                         options=dict(maxiter=maxiter - its, maxcor=30))
            its += int(r.nit)
            improved = best is None or r.fun < best.fun - self.RESTART_TOL
            best, w0 = (r, r.x) if best is None or r.fun < best.fun else (best, w0)
            if not improved or its >= maxiter:
                break
        return MapResult(self.to_z(best.x), float(best.fun), its, bool(best.success), str(best.message))

    def report(self, w: np.ndarray) -> None:
        self.callback(self.to_z(w))

    # warm-up: blocks of FIRST_BLOCK steps, doubling up to WARMUP_BLOCK, each adding a learned `Whitening` stage from
    # its second half; then TUNE_BLOCK steps (at most a quarter of the warm-up) tune only the step size for the draws.
    # A checkpoint after each block
    FIRST_BLOCK = 50
    WARMUP_BLOCK = 200
    TUNE_BLOCK = 100
    STORED_BLOCK = 100       # starting from the stored whitening (an earlier run's learned stages): one learning block
    DRAW_BLOCK = 200         # draws a chain between checkpoints / ESS checks (was 50 before the learned whitening)
    MAX_STEPS = 2 ** 10 - 1  # NUTS's tree limit (sets max_tree_depth: 10; the tests use fewer)
    SAMPLER = "shares+horizon+learned2+ta0.9"   # in a checkpoint's tag: a run made by another sampler is not gone on from
    # NUTS's target acceptance: smaller steps than numpyro's 0.8 - two runs (2026-11-04) each had a chain stick in a
    # narrow region of the weekly noise terms (funnel: up to 268 of 300 draws divergent), losing its ESS
    TARGET_ACCEPT = 0.9
    MAX_R_HAT = 1.01         # with a target ESS: the chains must also agree this well
    MAX_DRAWS_FACTOR = 4     # ... and drawing never goes past this many times the draws asked for (no endless run)
    UNJUDGED = ("pmax",)     # states whose own parameters the ESS target leaves out (still reported): sprints are
    #                          trained outside the plan, and Pmax's weekly noise mixes slowest
    PER_STATE = ("x0", "floor", "log_tau", "eta")   # blocks with one entry per state (eta: per week and state)

    def judged(self) -> np.ndarray:
        """mask of the parameters the ESS / R-hat target is judged on: all but the UNJUDGED states' own."""
        skip = [STATES.index(s) for s in self.UNJUDGED]
        out = np.ones(self.model.spec.size, bool)
        for b in self.model.spec.blocks:
            if b.name in self.PER_STATE:
                o = self.model.spec.offsets[b.name]
                out[o:o + b.mean.size] = ~np.isin(np.arange(b.mean.size) % len(STATES), skip)
        return out

    def nuts(self, z_init: np.ndarray, num_warmup: int = 500, num_samples: int = 500, seed: int = 0,
             progress: bool = True, chains: int | None = None, checkpoint: Path | None = None, tag: str = "",
             block: int = DRAW_BLOCK, say=print, target_ess: float | None = None, max_minutes: float | None = None,
             seed_draws: np.ndarray | None = None, whitening_from: Path | None = None,
             whitening_to: Path | None = None,
             structure: str = "", start: np.ndarray | None = None,
             max_r_hat: float = MAX_R_HAT) -> tuple[np.ndarray, dict]:
        """posterior samples (chains x num_samples, D) of z with NumPyro's NUTS, and diagnostics.
        Chains run in parallel (one CPU device each, physio.CHAINS), started around z_init (the MAP) with
        a spread of half the whitened scale. Coordinates: the training effects as `GainShares`; NUTS runs on u with identity mass and
        v = `Whitening`.to_v(u): first the inverse Hessian's square root at z_init (low rank), then a stage learned
        from each warm-up block's draws and gradients (`warmup_blocks`), so directions the mode's curvature gets
        wrong are caught in every run; a last block tunes only the step size. Every block reports steps a draw,
        the share at the tree limit and step size per chain (`tree_report`, with a warning for odd chains). The
        potential is numba's. info: steps per draw, divergences, step size, worst split R-hat and smallest
        effective sample size over all parameters.

        With `checkpoint` (a file) the run is saved after every warm-up block and every `block` draws, and a later
        call with the same `tag` (what the posterior is) goes on from there: stopping loses at most a block.
        Another tag starts over. A finished run's checkpoint is gone on from too (more draws, or a target_ess).

        target_ess: after num_samples draws, more blocks until the smallest effective sample size reaches it and
        the worst split R-hat is <= max_r_hat (default MAX_R_HAT; a refresh stops sooner), or max_minutes of
        drawing have passed, or MAX_DRAWS_FACTOR x num_samples draws a chain are reached; each block prints both.
        seed_draws (chains, n, D) of z: an earlier run's draws to go on from when there is no checkpoint - they are
        kept, the chains start at their last draws, and the warm-up runs again from there.
        start (chains, D) of z: positions the chains start from when there is no checkpoint (e.g. the last draws of
        another day's run) - unlike seed_draws NOT kept as draws (another posterior): the run begins with none.

        whitening_from: stored learned stages to start from (a refresh; `load_whitening`): a run without its own
        checkpoint that finds ones made by this `structure` and SAMPLER takes the Hessian stage fresh (the mode
        moved), the stored stages after it (by parameter name), and the short warm-up (`warmup_blocks(stored=True)`).
        whitening_to: where to write the stages this run learned at the end of its warm-up (a full run;
        `save_whitening`), replacing what was there. A run going on from its checkpoint keeps the checkpoint's."""
        from numpyro.infer import MCMC

        from . import CHAINS
        chains = chains or CHAINS
        method = "parallel" if jax.local_device_count() >= chains else "sequential"
        ck = self.load_checkpoint(checkpoint) if checkpoint else None
        if ck is None or ck["tag"] != tag:
            say(f"NUTS: preconditioner - the Hessian at the mode ({self.model.spec.size} parameters) ...")
            t0 = time.time()
            shares = GainShares(self.model, GainShares.average_doses(self.model, z_init))
            v_mode = shares.from_w(self.to_w(z_init))
            J = np.asarray(jax.jacfwd(shares.w_of)(jnp.asarray(v_mode)))
            lam, V = np.linalg.eigh(J.T @ self.fast.hessian(self.to_w(z_init)) @ J)   # the Hessian in v (gradient ~0)
            lam = np.maximum(lam, self.MIN_EIGEN)
            whitening = Whitening.from_hessian(v_mode, lam, V, self.LOW_RANK_EIGEN)
            say(f"NUTS: preconditioner ready in {time.time() - t0:.0f} s ({whitening.rank} informed directions)")
            learned = self.load_whitening(whitening_from, structure, say)
            if learned is not None:
                whitening = Whitening(v_mode, whitening.stages + learned)
                say(f"NUTS: starting from the stored whitening: {self.warmup_blocks(num_warmup, True)[-1][1]} "
                    f"warm-up steps ({sum(V.shape[1] for _, V, _ in learned)} learned directions)")
            eps = np.random.default_rng(seed).standard_normal((chains, lam.size))
            ck = {"tag": tag, "days": len(self.model.data.days), "warmup_done": 0, "state": None,
                  "rng": np.asarray(jax.random.PRNGKey(seed)), "log_dose": shares.log_dose,
                  "whitening": whitening, "step_size": 1.0, "stored_whitening": learned is not None,
                  "z": 0.5 * eps, "W": np.zeros((chains, 0, lam.size)),           # z: chain positions (u)
                  "steps": np.zeros((chains, 0)), "diverging": np.zeros((chains, 0), bool)}
            if seed_draws is not None:                     # an earlier run's draws, in these coordinates
                U = whitening.to_u(shares.from_w(self.to_w(seed_draws)))
                n = U.shape[1]
                ck.update(W=U, z=U[:, -1], steps=np.full((chains, n), np.nan), diverging=np.zeros((chains, n), bool))
                say(f"NUTS: going on from the saved draws ({n} a chain): warming up again from their last draws "
                    f"({num_warmup} steps) first")
            elif start is not None:                        # starting positions only: no draws
                ck.update(z=whitening.to_u(shares.from_w(self.to_w(start))))
                say("NUTS: the chains start from the given positions (no draws kept)")
        else:
            say(f"NUTS: going on from the checkpoint ({ck['warmup_done']} of {num_warmup} warm-up steps, "
                f"{ck['W'].shape[1]} of {num_samples} draws a chain)")
        shares = GainShares(self.model, ck["log_dose"])
        blocks = self.warmup_blocks(num_warmup, ck.get("stored_whitening", False))
        num_warmup = blocks[-1][1] if blocks else num_warmup        # the short schedule's (labels, clock, info)
        per_draws = min(block, num_samples)            # blocks with or without a checkpoint: limits checked between
        n_d = -(-num_samples // per_draws)
        clock = RunClock(num_warmup - ck["warmup_done"] + max(num_samples - ck["W"].shape[1], 0))
        say(f"NUTS: {chains} chains in parallel: {num_warmup} warm-up steps ({len(blocks)} block{'s' * (len(blocks) > 1)}"
            f": {', '.join(str(b - a) for a, b, _ in blocks)}), then {num_samples} draws a chain "
            f"({n_d} block{'s' * (n_d > 1)} of {per_draws})"
            + (f", then more until ESS >= {target_ess:.0f} and R-hat <= {max_r_hat}" if target_ess else "")
            + ("; a checkpoint after each block" if checkpoint else ""))
        warmed = False
        for i, (start, end, learn) in enumerate(blocks):
            if end <= ck["warmup_done"]:
                continue
            warmed = True
            n = end - max(start, ck["warmup_done"])
            label = (f"warm-up {end - n + 1}-{end} of {num_warmup} (block {i + 1} of {len(blocks)}: "
                     + ("learning the whitening)" if learn else "step size)"))
            say(f"NUTS: {label}")
            potential = partial(self.sampler_potential, shares, ck["whitening"])
            mcmc = MCMC(self.kernel(potential, ck["step_size"]), num_warmup=n, num_samples=1,
                        num_chains=chains, chain_method=method, progress_bar=progress)
            key, sub = jax.random.split(jnp.asarray(ck["rng"]))
            with ChainProgress.active(label):
                mcmc.warmup(sub, init_params=self.chain_values(ck["z"], chains), collect_warmup=True,
                            extra_fields=("num_steps", "z_grad"))
            state = jax.device_get(mcmc.post_warmup_state)
            ex = mcmc.get_extra_fields(group_by_chain=True)
            steps = np.asarray(ex["num_steps"]).reshape(chains, -1)
            step = np.asarray(state.adapt_state.step_size, float).reshape(chains)
            say(f"NUTS: warm-up block {i + 1}: " + self.tree_report(steps, step))
            if not learn:                  # the draws' step size: one for all chains (see pooled_step)
                state = self.pooled_step(state)
                say(f"NUTS: step size for the draws: {float(np.median(step)):.2g} (the chains' median)")
            ck.update(warmup_done=end, state=state, rng=np.asarray(key), z=np.asarray(state.z).reshape(chains, -1),
                      step_size=float(np.exp(np.mean(np.log(step)))))
            if learn:
                U = np.asarray(mcmc.get_samples(group_by_chain=True)).reshape(chains, n, -1)[:, n // 2:]
                G = np.asarray(ex["z_grad"]).reshape(chains, n, -1)[:, n // 2:]
                if U.shape[0] * U.shape[1] >= Whitening.MIN_DRAWS:
                    old = ck["whitening"]
                    whitening, stages = old.refined(U.reshape(-1, U.shape[-1]), G.reshape(-1, G.shape[-1]))
                    ck.update(whitening=whitening, z=whitening.to_u(old.to_v(ck["z"])),
                              W=whitening.to_u(old.to_v(ck["W"])))
                    say("NUTS: " + Whitening.describe(stages, U.shape[0] * U.shape[1]))
            self.save_checkpoint(checkpoint, ck, say, f"warm-up {ck['warmup_done']} of {num_warmup}", clock.after(n))
        if warmed:                                         # the next run (after new data) starts from it
            self.save_whitening(whitening_to, structure, ck["whitening"], say)
        potential = partial(self.sampler_potential, shares, ck["whitening"])
        kernel = self.kernel(potential, ck["step_size"])
        mcmc = None
        drawing, diag = time.time(), None
        while True:
            done = ck["W"].shape[1]
            if done >= num_samples:                       # the length asked for: done, or on to the ESS target
                if not target_ess:
                    break
                r_hat, ess, other = diag or self.diagnostics(shares, ck)
                if ess >= target_ess and r_hat <= max_r_hat:
                    say(f"NUTS: target reached at {done} draws a chain: smallest ESS {ess:.0f}, worst R-hat {r_hat:.3f}")
                    break
                if max_minutes is not None and time.time() - drawing >= 60 * max_minutes:
                    say(f"NUTS: time limit ({max_minutes:g} min) at {done} draws a chain: smallest ESS {ess:.0f}, "
                        f"worst R-hat {r_hat:.3f}")
                    break
                if done >= self.MAX_DRAWS_FACTOR * num_samples:
                    say(f"NUTS: stopped at {done} draws a chain ({self.MAX_DRAWS_FACTOR}x the {num_samples} asked "
                        f"for): smallest ESS {ess:.0f}, worst R-hat {r_hat:.3f}")
                    break
                say(f"NUTS: ESS target {target_ess:.0f} / R-hat {max_r_hat} not reached - {per_draws} more")
            n = per_draws if done >= num_samples else min(per_draws, num_samples - done)
            label = (f"draws {done + 1}-{done + n} of {num_samples} a chain (block {done // per_draws + 1} of {n_d})"
                     if done < num_samples else f"draws {done + 1}-{done + n} a chain (towards the ESS target)")
            say(f"NUTS: {label}")
            if mcmc is None or mcmc.num_samples != n:
                mcmc = MCMC(kernel, num_warmup=0, num_samples=n, num_chains=chains, chain_method=method,
                            progress_bar=progress)
            mcmc.post_warmup_state = jax.tree_util.tree_map(jnp.asarray, ck["state"])
            with ChainProgress.active(label):
                mcmc.run(mcmc.post_warmup_state.rng_key, init_params=self.chain_values(ck["z"], chains),
                         extra_fields=("num_steps", "diverging"))
            ex = mcmc.get_extra_fields(group_by_chain=True)
            state = jax.device_get(mcmc.last_state)
            steps = np.asarray(ex["num_steps"]).reshape(chains, -1)
            ck.update(state=state, z=np.asarray(state.z).reshape(chains, -1),
                      W=np.concatenate([ck["W"], np.asarray(mcmc.get_samples(group_by_chain=True)).reshape(chains, n, -1)],
                                       axis=1),
                      steps=np.concatenate([ck["steps"], steps], axis=1),
                      diverging=np.concatenate([ck["diverging"], np.asarray(ex["diverging"]).reshape(chains, -1)],
                                               axis=1))
            say("NUTS: " + self.tree_report(steps, np.asarray(state.adapt_state.step_size, float).reshape(chains)))
            diag = self.diagnostics(shares, ck)          # every block, with or without an ESS target
            say(f"NUTS: {ck['W'].shape[1]} draws a chain: smallest ESS {diag[1]:.0f}"
                + (f" of {target_ess:.0f}" if target_ess else "") + f", worst R-hat {diag[0]:.3f}"
                + (f" ({diag[2]})" if diag[2] else ""))
            self.save_checkpoint(checkpoint, ck, say, f"{ck['W'].shape[1]} draws a chain",
                                 clock.after(n, more=done + n < num_samples))
        W = self.draws_w(shares, ck)
        r_hat, ess, other = self.diagnostics(shares, ck)
        if other:
            say(f"NUTS: not judged: {other}")
        say("NUTS: write-out 1/2: R-hat and ESS of every parameter ...")   # (2026-11-13: two resets here; each
        r_each, ess_each = self.each(W)                                    # stage named first, to find the step)
        info = {"chains": chains, "mean_steps": float(np.nanmean(ck["steps"])),
                "divergences": int(np.sum(ck["diverging"])),
                "step_size": float(np.mean(ck["state"].adapt_state.step_size)),
                "r_hat": r_hat if chains > 1 else float("nan"), "min_ess": ess,
                "reached": not target_ess or bool(ess >= target_ess and r_hat <= max_r_hat),   # its ESS / R-hat target
                # per parameter and per chain (the app's Model page: physio_diagnostics.json)
                "r_hat_each": r_each, "ess_each": ess_each, "judged": self.judged(), "warmup": num_warmup,
                "divergences_chain": np.sum(ck["diverging"], axis=1).tolist(),
                "steps_chain": np.nanmean(ck["steps"], axis=1).tolist(),
                "at_limit_chain": np.mean(ck["steps"] >= self.MAX_STEPS, axis=1).tolist(),
                "step_size_chain": np.atleast_1d(np.asarray(ck["state"].adapt_state.step_size, float)).tolist(),
                "whitening_rank": ck["whitening"].rank}
        say(f"NUTS: write-out 2/2: {W.shape[0] * W.shape[1]} draws back to the model's parameters ...")
        return self.to_z(W.reshape(-1, W.shape[-1])), info

    @classmethod
    def warmup_blocks(cls, n: int, stored: bool = False) -> list[tuple[int, int, bool]]:
        """warm-up blocks (start, end, learns the whitening) for n steps: doubling learning blocks, then the step
        size alone (the draws' kernel is the last block's: never one whose whitening changed after it). `stored`:
        the run starts from the stored whitening - one learning block of at most STORED_BLOCK steps (it corrects
        what moved since), then the step size: 200 steps instead of 400."""
        tune = min(cls.TUNE_BLOCK, max(n // 4, 1)) if n > 1 else n
        if stored and n > 1:
            learn = min(cls.STORED_BLOCK, n - tune)
            return [(0, learn, True), (learn, learn + tune, False)]
        out, start, size = [], 0, cls.FIRST_BLOCK
        while start < n - tune:
            end = min(start + size, n - tune)
            out.append((start, end, True))
            start, size = end, min(2 * size, cls.WARMUP_BLOCK)
        return out + ([(start, n, False)] if tune else [])

    @staticmethod
    def pooled_step(state):
        """the warm-up's last state with every chain's step size set to the chains' median. The chains sample one
        posterior with one whitening: a chain's own tuned step says where it happened to be while tuning, and it
        carried that everywhere (2026-11-09: one chain at 0.024 vs ~0.07, 255 steps a draw vs 63 - every block
        waits for the slowest chain)."""
        a = state.adapt_state
        step = np.asarray(a.step_size, float)
        return state._replace(adapt_state=a._replace(step_size=np.full(step.shape, np.median(step), step.dtype)))

    def tree_report(self, steps: np.ndarray, step_size: np.ndarray) -> str:
        """steps a draw, share at the tree limit and step size per chain, and a warning when a chain is mostly at
        the limit or its step size is 2x off the others' (it tuned in another region: slower, may mix worse)."""
        limit = np.mean(steps >= self.MAX_STEPS, axis=1)
        text = (f"steps a draw {'/'.join(f'{x:.0f}' for x in steps.mean(1))}, at the tree limit "
                f"{'/'.join(f'{100 * x:.0f}' for x in limit)} %, step size {'/'.join(f'{x:.2g}' for x in step_size)}")
        odd = np.flatnonzero((limit > 0.5) | (np.abs(np.log(step_size / np.exp(np.mean(np.log(step_size))))) > np.log(2)))
        if odd.size:
            text += (f" - WARNING: chain{'s' * (odd.size > 1)} {', '.join(str(c + 1) for c in odd)} "
                     "at the tree limit or tuned to another step size: slower draws, maybe poorer mixing")
        return text

    def draws_w(self, shares: GainShares, ck: dict) -> np.ndarray:
        """the checkpoint's draws (chains, n, D), back from u to v to standardised w."""
        V_draws = ck["whitening"].to_v(ck["W"])
        return np.asarray(jax.vmap(jax.vmap(shares.w_of))(jnp.asarray(V_draws)))

    def diagnostics(self, shares: GainShares, ck: dict) -> tuple[float, float, str]:
        """worst split R-hat and smallest effective sample size of the draws so far over the judged parameters,
        and the same for the others as text ("" when all are judged)."""
        r, e = self.each(self.draws_w(shares, ck))
        keep = self.judged()
        other = (f"{'/'.join(self.UNJUDGED)}: ESS {np.nanmin(e[~keep]):.0f}, R-hat {np.nanmax(r[~keep]):.3f}"
                 if (~keep).any() else "")
        return float(np.nanmax(r[keep])), float(np.nanmin(e[keep])), other

    @staticmethod
    def each(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """split R-hat and effective sample size per parameter of draws (chains, n, D)"""
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
        r = (np.asarray(split_gelman_rubin(W)) if W.shape[0] > 1 and W.shape[1] >= 4
             else np.full(W.shape[-1], np.inf))
        return r, np.asarray(effective_sample_size(W))

    @staticmethod
    def kernel(potential, step_size: float = 1.0):
        """NUTS on u with an identity mass (the `Whitening` is the mass matrix); warm-up tunes the step size."""
        from numpyro.infer import NUTS
        return NUTS(potential_fn=potential, dense_mass=False, adapt_mass_matrix=False, step_size=step_size,
                    target_accept_prob=PhysioFit.TARGET_ACCEPT, max_tree_depth=int(PhysioFit.MAX_STEPS).bit_length())

    def sampler_potential(self, shares: GainShares, whitening: Whitening, u):
        """what NUTS samples: u with v = whitening.to_v(u), the potential at w(v) minus the log-determinant of dw/dv
        (the whitening's own is a constant)."""
        w, logdet = shares.to_w(whitening.to_v(u))
        return self.fast.potential(w) - logdet

    @staticmethod
    def chain_values(z: np.ndarray, chains: int):
        """per-chain values (chains, D) the way MCMC takes them (one chain: (D,))."""
        return jnp.asarray(z if chains > 1 else z[0])

    def save_whitening(self, path: Path | None, structure: str, whitening: Whitening, say=print) -> None:
        """the learned stages (all after the Hessian's, which belongs to one mode) with the parameter names, the
        model's `structure` and SAMPLER: what `load_whitening` checks."""
        if path is None:
            return
        tmp = Path(path).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"names": self.model.spec.names(), "structure": structure, "sampler": self.SAMPLER,
                         "stages": whitening.stages[1:]}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        say(f"NUTS: learned whitening stored for the next run ({path.name})")

    def load_whitening(self, path: Path | None, structure: str, say=print) -> tuple | None:
        """the stored learned stages on this model's parameters (`Whitening.renamed`), or None: no file, or one made
        by another model structure or sampler (said)."""
        st = self.load_checkpoint(path) if path is not None else None
        if not isinstance(st, dict) or "stages" not in st:
            return None
        if st.get("structure") != structure or st.get("sampler") != self.SAMPLER:
            say("NUTS: the stored whitening is from another model or sampler: not used")
            return None
        return Whitening.renamed(st["stages"], st["names"], self.model.spec.names())

    @staticmethod
    def load_checkpoint(path: Path | None) -> dict | None:
        """a NUTS run saved by `nuts` (our own file in processed/)."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, TypeError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
            return None

    @staticmethod
    def save_checkpoint(path: Path | None, ck: dict, say, what: str, timing: str = "") -> None:
        if path is None:
            return
        tmp = Path(path).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(ck, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())                 # on the disk before it replaces the last one (2026-11-13: a crash
            #                                      right after the swap left a checkpoint of zeros, the run lost)
        for wait in (0.5, 1, 2, 4, 8, None):     # whole or not at all: a stop mid-write keeps the last one
            try:
                tmp.replace(path)
                break
            except PermissionError:              # Windows: something reads the checkpoint right now - try again
                if wait is None:                 # still held: the run goes on, the next block saves (2026-11-04 a
                    say(f"NUTS: checkpoint not saved ({what}): {path} is held open; the next block tries again")
                    return                       # reader of the checkpoint stopped a 3-hour run here)
                time.sleep(wait)
        say(f"NUTS: checkpoint saved ({what})" + (f" - {timing}" if timing else ""))
