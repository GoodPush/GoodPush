"""How well the physio model forecasts (kept from the model ladder, 2026-11-13; the comparison models are gone):

* `PerformanceMarker`: the performance criterion - a day's best 20-min-equivalent power (max of P20, 0.82 x P5), kept
  on near-maximal days (>= 92 % of the previous 90 days' best).
* `PointBacktest`: rolling origin - FOLDS cut-offs EVERY_DAYS apart ending HORIZON days before the data's end; per fold
  the model's prediction of the held-out marker days after the cut; error = measured / predicted - 1.
* `PhysioBaseline`: the physio model's posterior mode per cut-off, forecasting the marker and its threshold readings;
  `PhysioCalibrated`: the same rescaled to the year's near-max days (the Model page's "Forecast error").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..physio import PhysioData

YEARS = 3.0             # a fold sees this many years before its cut-off
FOLDS = 8
EVERY_DAYS = 91
HORIZON = 42


class PerformanceMarker:
    NEAR_MAX = 0.92
    PRIOR_DAYS = 90
    P5_FACTOR = 0.82
    CP_DURATIONS = (180, 300, 600, 1200)          # the effort durations inside 3-20 min

    def __init__(self, data: PhysioData):
        self.days = data.days
        n = len(data.days)
        self.durations = np.unique(np.asarray(data.effort_s, float))
        best = np.full((n, len(self.durations)), np.nan)
        np.fmax.at(best, (np.asarray(data.effort_day, int), np.searchsorted(self.durations, data.effort_s)),
                   np.asarray(data.effort_w, float))
        self.best = best                                            # (days, durations) daily best power
        self.hist_h, self.power_w = data.hist_h, np.asarray(data.power_w, float)   # time at each power, per day
        eq = self.of_powers(self.column(300), self.column(1200))
        prior = pd.Series(eq).rolling(self.PRIOR_DAYS, min_periods=1).max().shift(1).to_numpy()
        self.value = np.where(eq >= self.NEAR_MAX * np.nan_to_num(prior, nan=0.0), eq, np.nan)

    def column(self, s: float) -> np.ndarray:
        """the daily best at duration s (all NaN if the data have no efforts of that length)."""
        k = int(np.searchsorted(self.durations, s))
        return self.best[:, k] if k < len(self.durations) and self.durations[k] == s else np.full(len(self.best), np.nan)

    @classmethod
    def of_powers(cls, p300, p1200):
        return np.fmax(p1200, cls.P5_FACTOR * p300)

    def window(self, cut: int, years: float = YEARS) -> np.ndarray:
        """the marker with only the days in [cut - years, cut) kept."""
        d = np.arange(len(self.value))
        return np.where((d >= cut - int(round(years * 365.25))) & (d < cut), self.value, np.nan)



class PointBacktest:
    """FOLDS cut-offs; per fold the model's prediction of the held-out marker days after the cut. Scores: MAE and
    RMSE (%), bias; `paired` compares two models on the same days with a bootstrap over *folds* (folds overlap: days
    aren't independent); `coverage` checks the error bands leave-one-fold-out."""

    def __init__(self, model, marker: PerformanceMarker, folds: int = FOLDS, every: int = EVERY_DAYS,
                 horizon: int = HORIZON, next_n: int = 0, since: int | None = None):
        """next_n > 0: score the first next_n marker days after each cut-off however late they come (the build-season
        question: what will the next real efforts be?) instead of the marker days within `horizon`.
        since (a day index): the prospective hold-out - cut-offs every `every` days from `since` on (data after it were
        never looked at while choosing models: notes/model_ladder.md, frozen 2026-11-06), however few days they score."""
        self.model, self.marker = model, marker
        self.folds, self.every, self.horizon, self.next_n, self.since = folds, every, horizon, next_n, since

    def cuts(self) -> list[int]:
        n = len(self.marker.value)
        if self.since is not None:
            return [int(c) for c in range(self.since, n, self.every)]
        end = n - self.horizon
        return [int(c) for c in end - self.every * np.arange(self.folds)[::-1] if c > 0]

    @staticmethod
    def errors(out: dict) -> np.ndarray:
        """absolute errors of a backtest's held-out days, in fold order (nan where nothing was predicted)."""
        o = np.array([v for f in out["folds"] for v in f["observed"]], float)
        p = np.array([np.nan if v is None else v for f in out["folds"] for v in f["predicted"]], float)
        return np.abs(o / p - 1.0)

    @staticmethod
    def fold_of(out: dict) -> np.ndarray:
        """each held-out day's fold index, in errors() order."""
        return np.repeat(np.arange(len(out["folds"])), [len(f["observed"]) for f in out["folds"]])

    @classmethod
    def paired(cls, a: dict, b: dict, draws: int = 5000, seed: int = 0) -> dict:
        """b's error minus a's on the same held-out days (percentage points), with a 90 % bootstrap range over whole folds
        (a day can be in two folds and neighbouring days share a fit: a bootstrap over days was too narrow, 2026-11-06)."""
        ea, eb = cls.errors(a), cls.errors(b)
        if ea.size != eb.size or not ea.size:
            raise ValueError("paired backtests: not the same held-out days (run both with the same --folds / --every)")
        d = 100 * (eb - ea)
        ok = np.isfinite(d)
        fold = cls.fold_of(a)
        F = len(a["folds"])
        S = np.bincount(fold[ok], d[ok], minlength=F)
        N = np.bincount(fold[ok], minlength=F).astype(float)
        idx = np.random.default_rng(seed).integers(0, F, (draws, F))
        m = S[idx].sum(axis=1) / np.maximum(N[idx].sum(axis=1), 1.0)
        return {"a": a["rung"], "b": b["rung"], "n": int(ok.sum()), "folds": F, "diff_pp": float(d[ok].mean()),
                "p05": float(np.percentile(m, 5)), "p95": float(np.percentile(m, 95))}

    @classmethod
    def coverage(cls, out: dict, level: float = 0.8) -> dict:
        """leave-one-fold-out band check: for each fold the band is the `level` quantile of |error| over the *other*
        folds; the share of its days inside it (about `level` if the bands are honest; less: too narrow)."""
        e = cls.errors(out)
        fold = cls.fold_of(out)
        ok = np.isfinite(e)
        inside = []
        for f in range(len(out["folds"])):          # a couple of dozen folds
            other = ok & (fold != f)
            mine = ok & (fold == f)
            if other.any() and mine.any():
                inside.append(e[mine] <= np.quantile(e[other], level))
        x = np.concatenate(inside) if inside else np.zeros(0, bool)
        return {"level": level, "n": int(x.size), "inside": float(x.mean()) if x.size else None}

    TAPER_SHARE = 0.6                # a scored day "after a taper": its last 7 days' load < 60 % of the 5 weeks before

    def taper_split(self, folds: list[dict]) -> dict:
        """the error on scored days after a low-load week vs after normal training (the model's load; none: {})."""
        load = getattr(self.model, "load", None)
        if load is None or not folds:
            return {}
        c = np.concatenate([[0.0], np.cumsum(load)])
        day = np.concatenate([np.searchsorted(self.marker.days, np.array(f["date"], dtype="datetime64[D]")) for f in folds])
        last7 = (c[day] - c[np.maximum(day - 7, 0)]) / 7
        before = (c[np.maximum(day - 7, 0)] - c[np.maximum(day - 42, 0)]) / 35
        taper = last7 < self.TAPER_SHARE * before
        o = np.array([v for f in folds for v in f["observed"]], float)
        p = np.array([np.nan if v is None else v for f in folds for v in f["predicted"]], float)
        err = o / p - 1.0
        out = {}
        for k, m in (("taper", taper), ("normal", ~taper)):
            e = err[m & np.isfinite(err)]
            out |= {f"{k}_n": int(e.size), f"{k}_mae_pct": float(100 * np.abs(e).mean()) if e.size else None,
                    f"{k}_bias_pct": float(100 * e.mean()) if e.size else None}
        return out

    def run(self, say=print) -> dict:
        folds, errs = [], []
        dates = self.marker.days
        for cut in self.cuts():                      # a handful of independent fits
            day = (np.flatnonzero(np.isfinite(self.marker.value[cut:]))[:self.next_n] if self.next_n
                   else np.flatnonzero(np.isfinite(self.marker.value[cut:cut + self.horizon]))) + cut
            pred, params = self.model.predict(cut, day)
            obs = self.marker.value[day]
            err = obs / pred - 1.0
            errs.append(err[np.isfinite(err)])
            folds.append({"cut": str(dates[cut - 1]), "n": int(np.isfinite(err).sum()),
                          "mae_pct": float(100 * np.nanmean(np.abs(err))) if np.isfinite(err).any() else None,
                          "bias_pct": float(100 * np.nanmean(err)) if np.isfinite(err).any() else None,
                          "date": [str(dates[d]) for d in day], "observed": np.round(obs, 1).tolist(),
                          "predicted": [None if not np.isfinite(p) else round(float(p), 1) for p in pred],
                          "params": {k: float(v) for k, v in params.items()}})
        e = np.concatenate(errs) if errs else np.zeros(0)
        out = {"rung": self.model.name, "horizon": self.horizon, "next_n": self.next_n, "since": self.since,
               "folds": folds, "n": int(e.size),
               "mae_pct": float(100 * np.abs(e).mean()) if e.size else None,
               "rmse_pct": float(100 * np.sqrt(np.mean(e ** 2))) if e.size else None,
               "bias_pct": float(100 * e.mean()) if e.size else None} | self.taper_split(folds)
        out["coverage"] = self.coverage(out) if e.size else None
        readings = getattr(self.model, "readings_summary", None)
        if readings is not None:
            out |= readings(folds)
        tp = (f"; after a taper {out['taper_mae_pct']:.1f} % (bias {out['taper_bias_pct']:+.1f}, n {out['taper_n']}), "
              f"otherwise {out['normal_mae_pct']:.1f} % (bias {out['normal_bias_pct']:+.1f})" if out.get("taper_n") else "")
        cv = out["coverage"] or {}
        tp += (f"; RMSE {out['rmse_pct']:.1f} %, 80 % band holds {100 * cv['inside']:.0f} % of days (leave-one-fold-out)"
               if e.size and cv.get("inside") is not None else "")
        say(f"backtest {self.model.name}: marker error {out['mae_pct']:.1f} % (bias {out['bias_pct']:+.1f} %) over "
            f"{out['n']} held-out near-max days" + (f" (the next {self.next_n} after each cut-off)" if self.next_n else "")
            + tp if e.size else f"backtest {self.model.name}: no held-out marker days")
        rd = [f"{q} readings {out[f'{q}_readings_fit_pct']:.1f} % with bias / {out[f'{q}_readings_raw_pct']:.1f} % "
              f"without (n {out[f'{q}_readings_n']})" for q in ("lt1", "cp") if f"{q}_readings_n" in out]
        bs = [f"{k[5:-7]} {out[k]:.1f}" for k in sorted(out) if k.startswith("bias_") and k.endswith("_max_sd")]
        if rd or bs:
            say("  held-out " + "; ".join(rd) + ("; largest bias (prior sds): " + ", ".join(bs) if bs else ""))
        changes = getattr(self.model, "changes_summary", None)
        if changes is not None:
            out |= changes(folds)
            ch = [f"{q.upper()} {out[f'{q}_change_mae_pp']:.1f} pp (no change: {out[f'{q}_change_none_pp']:.1f}"
                  + (f"; corr {out[f'{q}_change_corr']:+.2f}, slope {out[f'{q}_change_slope']:.2f}"
                     if out[f"{q}_change_corr"] is not None else "") + f"; n {out[f'{q}_change_n']})"
                  for q in ("lt1", "cp", "map") if f"{q}_change_n" in out]
            if ch:
                say(f"  {self.model.CHANGE_DAYS}-day change from training alone vs the latest fit: " + "; ".join(ch))
        return out



class PhysioBaseline:
    """The big physio model (goodpush.physio, models/physio.py) on the same yardstick (does its training
    response forecast better than the envelope?): per cut-off its posterior mode (L-BFGS, no NUTS) on all data before
    the cut, warm-started from the previous fold (the first from the priors), then its all-out 5- and 20-min power on
    each held-out day from the training actually ridden after the cut (fatigue included) -> the marker.
    Also its held-out threshold readings (not the one-sided ramp bounds) in the same days, per quantity: the error with
    each source's fitted bias, and against the state alone - a bias that soaks up a wrong level hides it from the
    first (2026-11-09: shared noise put LT1 45 % under alpha1, the marker never saw it)."""
    name = "physio"
    NEEDS_DATA = True
    MAXITER = 15000
    CHECKED = ("lt1", "cp")                  # reading quantities scored (QUANTITIES)

    def __init__(self, marker: PerformanceMarker, tss: np.ndarray, data: PhysioData, start=None, say=print):
        """start: (names, z) of a fit to warm-start the first fold from - only one made on data before it (the saved
        fit has seen every fold's future: a leak, 2026-11-06); default the priors, later folds from the fold before."""
        self.marker, self.data, self.say = marker, data, say
        self.load = np.nan_to_num(np.asarray(tss, float))
        self.last = start                                                  # (names, z) of the latest fit

    def predict(self, cut: int, days: np.ndarray) -> tuple[np.ndarray, dict]:
        import time

        from ..physio import PhysioFit, PhysioModel
        t0 = time.time()
        model = PhysioModel(self.data.until(cut), durations=np.unique(np.asarray(self.data.effort_s, float)))
        z0 = None if self.last is None else model.spec.align(*self.last)[0]
        res = PhysioFit(model).map(z0, maxiter=self.MAXITER)
        self.last, self.last_cut = (model.spec.names(), res.z), cut
        n = int(days.max() - cut + 1) if len(days) else 1
        d = self.data
        r = d.ridden(cut, cut + n)
        P = np.asarray(model.ahead(res.z, r["hist_h"], r["long_h"], np.array([300.0, 1200.0]), r["hr_hist_h"]))
        pred = PerformanceMarker.of_powers(P[days - cut, 0], P[days - cut, 1]) if len(days) else np.zeros(0)
        self.say(f"  physio fold to {d.days[cut - 1]}: mode in {res.iterations} iterations ({res.message}), "
                 f"{time.time() - t0:.0f} s")
        return pred, ({"iterations": float(res.iterations), "neg_log_post": float(res.neg_log_post),
                       "p300_next": float(P[0, 0]), "p1200_next": float(P[0, 1])} | self.reading_errors(model, res.z, cut, n)
                      | self.change_forecast(model, res.z, cut))

    CHANGE_DAYS = 84                         # the block-change yardstick: this far ahead, from training alone
    CHANGED = ("lt1", "cp", "map")

    def change_forecast(self, model, z, cut: int) -> dict:
        """the fold's thresholds (log W) on the morning of the cut and CHANGE_DAYS later, forecast from the training
        ridden in between (no unexplained weekly changes); {} when the data end before that."""
        from ..physio.model import QUANTITIES, PhysioModel
        d, h = self.data, self.CHANGE_DAYS
        if cut + h >= len(d.days):
            return {}
        X, _ = model.forecast(z, **d.ridden(cut, cut + h + 1))
        Q = np.log(np.asarray(PhysioModel.quantities(X)))
        return {"change_day": float(cut)} | {f"{q}_{k}": float(Q[QUANTITIES.index(q), i])
                                             for q in self.CHANGED for k, i in (("at_cut", 0), ("ahead", h))}

    def changes_summary(self, folds: list[dict]) -> dict:
        """does training alone forecast how a block changes you? Per CHANGED threshold, over the folds whose
        CHANGE_DAYS lie before the latest fold's cut: the forecast change (`change_forecast`) against the change the
        latest fit - which has seen those days' efforts and readings - puts between the same two mornings. n, their
        correlation, the slope of realised on forecast (above 1: forecasts too flat), the mean absolute error
        (percentage points) and the same for "no change" (a forecast must beat it). The realised change is the same
        model's reading of the later data, and blocks of neighbouring folds overlap: a yardstick between model
        versions, not proof."""
        from ..physio.model import QUANTITIES, PhysioModel
        last = getattr(self, "last_cut", 0)
        cut = np.array([f["params"]["change_day"] for f in folds if "change_day" in f["params"]], int)
        ok = cut + self.CHANGE_DAYS < last
        if not ok.any():
            return {}
        model = PhysioModel(self.data.until(last), durations=np.unique(np.asarray(self.data.effort_s, float)))
        R = np.log(np.asarray(PhysioModel.quantities(model.simulate(self.last[1])[0])))
        out = {}
        for q in self.CHANGED:
            p = np.array([f["params"][f"{q}_ahead"] - f["params"][f"{q}_at_cut"] for f in folds if "change_day" in f["params"]])[ok]
            r = R[QUANTITIES.index(q), cut[ok] + self.CHANGE_DAYS] - R[QUANTITIES.index(q), cut[ok]]
            both = p.size >= 3 and p.std() > 0 and r.std() > 0
            out |= {f"{q}_change_n": int(p.size), f"{q}_change_mae_pp": float(100 * np.abs(r - p).mean()),
                    f"{q}_change_none_pp": float(100 * np.abs(r).mean()),
                    f"{q}_change_forecast_sd_pct": float(100 * p.std()), f"{q}_change_realised_sd_pct": float(100 * r.std()),
                    f"{q}_change_corr": float(np.corrcoef(p, r)[0, 1]) if both else None,
                    f"{q}_change_slope": float(np.polyfit(p, r, 1)[0]) if both else None}
        return out

    def reading_errors(self, model, z, cut: int, n: int) -> dict:
        """per CHECKED quantity, over the held-out readings in days cut .. cut + n - 1: their count, summed |log error|
        with the source's fitted bias ("_fit") and without it ("_raw"); and each checked source's fitted bias."""
        from ..physio.model import QUANTITIES, PhysioModel
        d = self.data
        X, _ = model.forecast(z, **d.ridden(cut, cut + n))
        Q = np.log(np.asarray(PhysioModel.quantities(X)))                       # (4, n)
        bias = np.asarray(model.spec.unpack(z)["bias"], float)
        src = np.asarray(d.reading_source, int)
        q = np.array([QUANTITIES.index(s.quantity) for s in d.sources], int)[src] if len(d.sources) else src
        bound = np.array([s.bound for s in d.sources], bool)[src] if len(d.sources) else src.astype(bool)
        day = np.asarray(d.reading_day, int)
        out = {}
        for name in self.CHECKED:
            k = (day >= cut) & (day < cut + n) & ~bound & (q == QUANTITIES.index(name))
            raw = np.log(np.asarray(d.reading_w, float)[k]) - Q[q[k], day[k] - cut]
            out |= {f"{name}_n": float(k.sum()), f"{name}_raw": float(np.abs(raw).sum()),
                    f"{name}_fit": float(np.abs(raw - bias[src[k]]).sum())}
        for j, s in enumerate(d.sources):
            if s.quantity in self.CHECKED and not s.bound:
                out[f"bias_{s.name}"] = float(bias[j])
        return out

    def readings_summary(self, folds: list[dict]) -> dict:
        """the held-out reading errors over all folds (%), and the largest fitted bias per source in its prior sds."""
        out = {}
        for name in self.CHECKED:
            n = sum(f["params"].get(f"{name}_n", 0.0) for f in folds)
            if n:
                out |= {f"{name}_readings_n": int(n)} | {
                    f"{name}_readings_{k}_pct": float(100 * sum(f["params"][f"{name}_{k}"] for f in folds) / n)
                    for k in ("fit", "raw")}
        sd = {s.name: s.bias_sd for s in self.data.sources}
        for key in {k for f in folds for k in f["params"] if k.startswith("bias_")}:
            v = [f["params"][key] for f in folds if key in f["params"]]
            out[f"{key}_max_sd"] = float(np.max(np.abs(v)) / max(sd[key[5:]], 1e-9))
        return out


class PhysioCalibrated(PhysioBaseline):
    """PhysioBaseline rescaled by one number per cut-off: the median of (marker / the model's own marker) over the
    year's near-max days before the cut (the model's curve is all-out capacity; near-max days are mostly below it:
    raw, it forecast ~20 % high, 2026-11-05). Judges whether it predicts *changes*, not its level."""
    name = "physio_calibrated"
    CALIBRATION_DAYS = 365

    def predict(self, cut: int, days: np.ndarray) -> tuple[np.ndarray, dict]:
        from ..physio import PhysioModel
        pred, info = super().predict(cut, days)
        names, z = self.last
        model = PhysioModel(self.data.until(cut), durations=np.unique(np.asarray(self.data.effort_s, float)))
        X, F, _ = model.simulate(z)
        y = self.marker.window(cut, years=self.CALIBRATION_DAYS / 365.25)
        d = np.flatnonzero(np.isfinite(y))
        if not d.size:
            return pred, info
        P = np.asarray(model.all_out(z, np.asarray(X)[d][:, None, :], np.asarray(F)[d][:, None],
                                     np.array([300.0, 1200.0])))
        ratio = float(np.median(y[d] / PerformanceMarker.of_powers(P[:, 0], P[:, 1])))
        return pred * ratio, info | {"calibration": ratio}
