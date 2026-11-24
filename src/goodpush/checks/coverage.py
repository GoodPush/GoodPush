"""Is a 90 % range right 90 % of the time? A coverage check of the physio model's posterior ranges (review 2026-11-10):
per backtest cut-off a posterior (mode + NUTS) on the data before it, then a predictive sample per draw for every
threshold reading and ride best power of the HORIZON days after it, from the training actually ridden; each
observation's PIT (the share of its predictive samples below it) shows whether the ranges are honest."""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..physio import KINDS, QUANTITIES, PhysioData, PhysioFit, PhysioModel
from ..physio.model import ALL_OUT_SD, MIN_READING_SD, MIN_SHAPE, VERSION, ParamSpec
from .backtest import HORIZON, PerformanceMarker, PointBacktest


class PosteriorCoverage:
    """`goodpush ladder coverage physio`: per cut-off (PointBacktest's) the posterior mode on data.until(cut) (warm-
    started from the fold before; the first from the priors - the stored fit has seen the future), then NUTS with the
    refresh settings (the stored whitening read-only and its short warm-up; no checkpoint, nothing written to the
    athlete's fit files). PREDICT_DRAWS evenly spaced draws, each on the model of data.until(cut + HORIZON) with the
    weeks after the cut drawn from the prior (future unexplained changes belong in the range), simulated with the
    training ridden; one predictive sample per draw and held-out observation: a reading (not a one-sided bound) =
    log quantity + its source's bias + Normal(0, its sd); a best power = the model's mixture for its ride kind
    (all-out with its probability around the fatigued curve, else the Gamma shortfall below capacity). Per
    observation the PIT; `summary` groups them. Saved to processed/ladder/physio_coverage.json after every fold; a
    rerun skips the folds already there (same cut dates, model VERSION and SAMPLER, else it starts over)."""
    FILE = "physio_coverage.json"
    HORIZON = HORIZON
    PREDICT_DRAWS = 400
    WARMUP = 400                     # with the stored whitening: its short schedule (200 steps)
    DRAWS = 800                      # the refresh's settings (PhysioRun.REFRESH_*), longer time cap
    ESS = 200
    R_HAT = 1.02
    MINUTES = 15
    MAXITER = 15000
    LEVELS = (0.5, 0.9)
    AHEAD = 14                       # days in a "days ahead" group (ranges too narrow only later = the weekly noise)
    DURATIONS = (("<= 60 s", 0.0, 60.0), ("3-20 min", 180.0, 1200.0), (">= 40 min", 2400.0, np.inf))

    def __init__(self, data: PhysioData, folder: Path, whitening: Path | None = None, structure: str = "",
                 folds: int = 24, every: int = 30, say=print):
        """folder: where the result goes (processed/ladder); whitening: the stored whitening (read only)."""
        self.data, self.folder, self.whitening, self.structure = data, Path(folder), whitening, structure
        self.folds, self.every, self.say = folds, every, say
        self.durations = np.unique(np.asarray(data.effort_s, float))     # every fold's model: the same parameters

    @property
    def path(self) -> Path:
        return self.folder / self.FILE

    def cuts(self) -> list[int]:
        return PointBacktest(None, PerformanceMarker(self.data), self.folds, self.every, self.HORIZON).cuts()

    def key(self, cuts: list[int]) -> dict:
        """what makes a stored result the same run: cut dates, model version, sampler, settings."""
        return {"cuts": [str(self.data.days[c]) for c in cuts], "version": VERSION, "sampler": PhysioFit.SAMPLER,
                "horizon": self.HORIZON, "draws": self.DRAWS, "predict_draws": self.PREDICT_DRAWS}

    def stored(self, key: dict) -> dict:
        """the result so far when it was made by this run (`key`), else a fresh one."""
        try:
            out = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out = {}
        return out if out.get("key") == key else {"key": key, "folds": []}

    def save(self, out: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=True), encoding="utf-8")
        tmp.replace(self.path)

    def whitening_note(self) -> str | None:
        """a warning when the stored whitening (the last full NUTS run's) can't be used: missing, or made by another
        model / sampler - each fold then learns its own in the full warm-up (slower)."""
        st = PhysioFit.load_checkpoint(self.whitening) if self.whitening is not None else None
        if not isinstance(st, dict) or "stages" not in st:
            return (f"WARNING: no stored whitening (no full NUTS run on model v{VERSION}): each fold learns its own "
                    f"in the full {self.WARMUP}-step warm-up (slower)")
        made = str(st.get("structure", "")).rpartition("|")[2] or "?"
        if st.get("structure") != self.structure:
            return (f"WARNING: the stored whitening is from the NUTS run on model {made}, not v{VERSION}: each fold "
                    f"learns its own in the full {self.WARMUP}-step warm-up (slower); a full NUTS run on v{VERSION} "
                    f"first gives the short one")
        if st.get("sampler") != PhysioFit.SAMPLER:
            return (f"WARNING: the stored whitening is from another sampler ({st.get('sampler')}): each fold learns "
                    f"its own in the full {self.WARMUP}-step warm-up (slower)")
        return None

    def run(self) -> dict:
        note = self.whitening_note()
        if note:
            self.say(note)
        cuts = self.cuts()
        key = self.key(cuts)
        out = self.stored(key)
        done = {f["cut"] for f in out["folds"]}
        for f in out["folds"]:                         # a result from before the days ahead were stored
            f.update({} if "effort_ahead" in f else self.ahead(cuts[key["cuts"].index(f["cut"])], f))
        last = (out["last"]["names"], np.asarray(out["last"]["z"])) if out.get("last") else None
        clock = time.time()
        for i, cut in enumerate(cuts):                 # a couple of dozen folds
            day = str(self.data.days[cut])
            if day in done:
                continue
            self.say(f"coverage fold {i + 1} of {len(cuts)}: cut {day} ...")
            fold, last = self.fold(cut, last)
            out["folds"].append(fold)
            out["last"] = {"names": last[0], "z": np.asarray(last[1]).tolist()}
            out["summary"] = self.summary(out["folds"])
            self.save(out)
            left = sum(str(self.data.days[c]) not in {f["cut"] for f in out["folds"]} for c in cuts)
            self.say(f"coverage fold {i + 1} of {len(cuts)} done in {fold['minutes']:.1f} min (R-hat {fold['r_hat']:.3f}, "
                     f"ESS {fold['min_ess']:.0f}, {fold['n_readings']} readings, {fold['n_efforts']} best powers); "
                     f"{(time.time() - clock) / 60:.0f} min so far, {left} folds left")
        out["summary"] = self.summary(out["folds"])
        self.save(out)
        return out

    def fold(self, cut: int, last) -> tuple[dict, tuple]:
        """one cut-off: the posterior on the data before it, the PITs of the HORIZON days after it."""
        t0 = time.time()
        model = PhysioModel(self.data.until(cut), durations=self.durations)
        fitter = PhysioFit(model)
        res = fitter.map(None if last is None else model.spec.align(*last)[0], maxiter=self.MAXITER)
        draws, info = fitter.nuts(res.z, num_warmup=self.WARMUP, num_samples=self.DRAWS, seed=cut, say=self.say,
                                  progress=True, target_ess=self.ESS, max_r_hat=self.R_HAT, max_minutes=self.MINUTES,
                                  whitening_from=self.whitening, structure=self.structure)
        pick = np.linspace(0, len(draws) - 1, min(len(draws), self.PREDICT_DRAWS)).astype(int)
        pit = self.pits(cut, model.spec.names(), draws[pick], seed=cut)
        return ({"cut": str(self.data.days[cut]), "minutes": (time.time() - t0) / 60, "r_hat": float(info["r_hat"]),
                 "min_ess": float(info["min_ess"]), "divergences": int(info["divergences"]),
                 "n_readings": len(pit["reading_pit"]), "n_efforts": len(pit["effort_pit"])} | pit,
                (model.spec.names(), res.z))

    def held_out(self, cut: int) -> tuple[np.ndarray, np.ndarray]:
        """indices of the efforts and the (two-sided) readings in days cut .. cut + HORIZON - 1."""
        d, end = self.data, cut + self.HORIZON
        bound = np.array([s.bound for s in d.sources], bool)[np.asarray(d.reading_source, int)] if len(d.sources) \
            else np.zeros(len(d.reading_w), bool)
        e = np.flatnonzero((d.effort_day >= cut) & (d.effort_day < end))
        r = np.flatnonzero((d.reading_day >= cut) & (d.reading_day < end) & ~bound)
        return e, r

    def ahead(self, cut: int, fold: dict) -> dict:
        """a stored fold's days ahead, read from the data again (they are the held-out observations in order); when
        the data changed since (other counts) every observation counts as day 0."""
        e, r = self.held_out(cut)
        same = len(e) == len(fold["effort_pit"]) and len(r) == len(fold["reading_pit"])
        return {"effort_ahead": (self.data.effort_day[e] - cut).tolist() if same else [0] * len(fold["effort_pit"]),
                "reading_ahead": (self.data.reading_day[r] - cut).tolist() if same else [0] * len(fold["reading_pit"])}

    def pits(self, cut: int, names: list[str], draws: np.ndarray, seed: int = 0) -> dict:
        """the held-out observations' PITs under draws (n, D) of the fold's posterior (named `names`)."""
        d = self.data
        full = PhysioModel(d.until(cut + self.HORIZON), durations=self.durations)
        rng = np.random.default_rng(seed)
        Z = full.spec.align(names, draws, rng=rng)              # weeks after the cut: from the prior
        e, r = self.held_out(cut)
        src = np.asarray(d.reading_source, int)[r]
        quantity = np.array([QUANTITIES.index(s.quantity) for s in d.sources], int)[src] if len(d.sources) else src
        t = np.searchsorted(full.effort_s, np.asarray(d.effort_s, float)[e])
        obs = {"e_day": jnp.asarray(d.effort_day[e], int), "e_s": jnp.asarray(d.effort_s[e], float),
               "e_x": jnp.asarray(np.asarray(full.duration_x, float)[t]), "e_kind": jnp.asarray(d.effort_kind[e], int),
               "r_day": jnp.asarray(d.reading_day[r], int), "r_src": jnp.asarray(src, int),
               "r_q": jnp.asarray(quantity, int)}
        keys = jax.random.split(jax.random.PRNGKey(seed), len(Z))
        E, R = self.samples(jnp.asarray(Z), keys, full.spec.layout, full.arrays, obs)     # (n, efforts), (n, readings)
        E, R = np.asarray(E), np.asarray(R)
        return {"effort_pit": np.round(np.mean(E < np.log(d.effort_w[e]), axis=0), 4).tolist(),
                "effort_ahead": (d.effort_day[e] - cut).tolist(), "reading_ahead": (d.reading_day[r] - cut).tolist(),
                "effort_s": np.asarray(d.effort_s[e], float).tolist(),
                "effort_kind": [KINDS[k] for k in d.effort_kind[e]],
                "reading_pit": np.round(np.mean(R < np.log(d.reading_w[r]), axis=0), 4).tolist(),
                "reading_quantity": [QUANTITIES[q] for q in quantity]}

    @staticmethod
    def predictive(z, key, layout, a, obs):
        """one predictive sample (log W) of each held-out effort and reading for parameters z (the likelihood's own
        distributions: PhysioModel.effort_terms / reading_terms)."""
        p = ParamSpec.split(layout, z)
        X, F, _ = PhysioModel.simulate_of(z, layout, a)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        day, kind = obs["e_day"], obs["e_kind"]
        log_cap = jnp.log(jnp.maximum(PhysioModel.curve(X[day], obs["e_s"]), 1.0))
        log_fat = log_cap - jnp.exp(p["log_s_f"][0]) * F[day]
        all_out = jax.random.uniform(k1, day.shape) < jax.nn.sigmoid(p["logit_pi"])[kind]
        shape = MIN_SHAPE + jnp.exp(p["log_shape_m2"][0])
        scale = jnp.exp(p["log_deficit"][0] + p["log_deficit"][1] * obs["e_x"]
                        + jnp.r_[0.0, p["log_scale_kind"]][kind]) / shape
        deficit = jax.random.gamma(k2, shape, day.shape) * scale
        effort = jnp.where(all_out, log_fat + ALL_OUT_SD * jax.random.normal(k3, day.shape), log_cap - deficit)
        Q = PhysioModel.quantities(X[obs["r_day"]])
        mean = jnp.log(Q[obs["r_q"], jnp.arange(obs["r_day"].shape[0])]) + p["bias"][obs["r_src"]]
        sd = (MIN_READING_SD + jnp.exp(p["log_sd"]))[obs["r_src"]]
        return effort, mean + sd * jax.random.normal(k4, mean.shape)

    samples = staticmethod(jax.jit(jax.vmap(predictive.__func__, in_axes=(0, 0, None, None, None)), static_argnums=2))

    @classmethod
    def table(cls, pit: np.ndarray) -> dict:
        """n, the share inside the central 50 % and 90 % predictive ranges, the mean PIT (0.5 if unbiased), the share
        above the 95th percentile."""
        pit = np.asarray(pit, float)
        if not pit.size:
            return {"n": 0}
        return {"n": int(pit.size)} | {f"inside_{int(100 * L)}": float(np.mean(np.abs(pit - 0.5) <= L / 2))
                                       for L in cls.LEVELS} | {"mean_pit": float(pit.mean()),
                                                               "above_95": float(np.mean(pit > 0.95))}

    @classmethod
    def summary(cls, folds: list[dict]) -> dict:
        """the folds' PITs grouped: readings per quantity, best powers per ride kind and per duration group; and per
        fold its convergence and time."""
        rp = np.array([v for f in folds for v in f["reading_pit"]], float)
        rq = np.array([v for f in folds for v in f["reading_quantity"]], str)
        ep = np.array([v for f in folds for v in f["effort_pit"]], float)
        ek = np.array([v for f in folds for v in f["effort_kind"]], str)
        es = np.array([v for f in folds for v in f["effort_s"]], float)
        ea = np.array([v for f in folds for v in f["effort_ahead"]], float)
        ra = np.array([v for f in folds for v in f["reading_ahead"]], float)
        ahead = [(f"days {lo}-{lo + cls.AHEAD - 1}", lo) for lo in range(0, cls.HORIZON, cls.AHEAD)]
        return {"readings": {"all": cls.table(rp)} | {q: cls.table(rp[rq == q]) for q in QUANTITIES if (rq == q).any()},
                "readings_by_days_ahead": {g: cls.table(rp[(ra >= lo) & (ra < lo + cls.AHEAD)]) for g, lo in ahead},
                "efforts_by_kind": {"all": cls.table(ep)} | {k: cls.table(ep[ek == k]) for k in KINDS if (ek == k).any()},
                "efforts_by_duration": {g: cls.table(ep[(es >= lo) & (es <= hi)]) for g, lo, hi in cls.DURATIONS},
                "efforts_by_each_duration": {f"{s:.0f} s": cls.table(ep[es == s]) for s in np.unique(es)},
                "efforts_by_days_ahead": {g: cls.table(ep[(ea >= lo) & (ea < lo + cls.AHEAD)]) for g, lo in ahead},
                "folds": [{k: f[k] for k in ("cut", "r_hat", "min_ess", "divergences", "minutes", "n_readings",
                                             "n_efforts")} for f in folds]}

    @staticmethod
    def lines(summary: dict) -> list[str]:
        """the printed table."""
        head = f"{'':26s}{'n':>6s}{'in 50 %':>9s}{'in 90 %':>9s}{'mean PIT':>10s}{'> 95th':>8s}"
        out = []
        for title, part in (("Threshold readings", "readings"), ("Threshold readings by days ahead", "readings_by_days_ahead"),
                            ("Best powers by ride kind", "efforts_by_kind"),
                            ("Best powers by duration", "efforts_by_duration"),
                            ("Best powers by each duration", "efforts_by_each_duration"),
                            ("Best powers by days ahead", "efforts_by_days_ahead")):
            out += [title, head]
            for name, r in summary[part].items():
                out.append(f"  {name:24s}{r['n']:6d}" + (f"{100 * r['inside_50']:8.0f}%{100 * r['inside_90']:8.0f}%"
                                                        f"{r['mean_pit']:10.2f}{100 * r['above_95']:7.0f}%" if r["n"] else ""))
        out += ["Folds", f"  {'cut':12s}{'R-hat':>7s}{'ESS':>7s}{'div':>6s}{'min':>7s}{'readings':>10s}{'efforts':>9s}"]
        out += [f"  {f['cut']:12s}{f['r_hat']:7.3f}{f['min_ess']:7.0f}{f['divergences']:6d}{f['minutes']:7.1f}"
                f"{f['n_readings']:10d}{f['n_efforts']:9d}" for f in summary["folds"]]
        return out
