"""How well goodpush's analysis finds a synthetic rider's physiology again: its folder analysed as anyone's
(`goodpush run`, then the physio model's best fit, PhysioRun.current), then scored against synth/truth.*:

* the physio model's daily capacities (CP, LT1, MAP, 1-h power, W', Pmax) against the true ones: typical error,
  bias, and whether it follows the true changes (the correlation of 6-week changes);
* the training response's parameters (model truth only), in prior sds from the true ones;
* the heart rates goodpush estimates (maximum, threshold, resting), alpha1 and ramp-test thresholds;
* the illnesses and breaks it finds, and whether it learned illness markers.

One synthetic rider is one case: a check of the machinery, not proof it suits every rider."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_project
from .rider import SyntheticRider
from .truth import ModelTruth


class SynthCheck:
    FILE = "check.json"                      # in synth/
    STATES = ("cp", "lt1", "map", "ftp", "w_prime", "pmax")
    SHARED = ("lt1", "map", "ftp")           # what any curve defines (5-min, 1-h power): the simple truth's CP, W'
    #                                          and Pmax belong to its own curve, not the model's
    CHANGE_D = 42                            # changes over this many days for the "follows the truth" correlation
    RECENT_D = 90
    FOUND_D = 7                              # an illness / break found within this many days of the true one

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.synth = self.folder / SyntheticRider.FOLDER
        self.truth = pd.read_parquet(self.synth / "truth.parquet")
        self.info = json.loads((self.synth / "truth.json").read_text(encoding="utf-8"))

    def run(self, progress=print) -> dict:
        """the analysis as `goodpush run` does it, the physio model's best fit, then the scores (saved in synth/)."""
        from ..cli import cmd_run
        from ..models.physio import PhysioRun
        cmd_run(Namespace(root=str(self.folder), athlete=None, workers=0, force=False))
        model, z = PhysioRun(load_project(self.folder)).current(progress)
        out = self.score(dict(zip(model.spec.names(), z)), dict(zip(model.spec.names(), model.spec.sd)))
        (self.synth / self.FILE).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
        return out

    def score(self, fitted: dict, prior_sd: dict) -> dict:
        proj = load_project(self.folder)
        from ..models.physio import PhysioRun
        states = pd.read_parquet(proj.processed / PhysioRun.STATES)
        res = json.loads((proj.reports / "results.json").read_text(encoding="utf-8"))
        return {"truth": self.info["truth"], "days": len(self.truth), "states": self.states(states),
                "params": self.params(fitted, prior_sd), "heart": self.heart(res.get("athlete") or {}),
                "thresholds": self.thresholds(proj.processed), "illness": self.illness(res),
                "breaks": self.breaks(proj.processed), "markers": self.markers(proj.processed)}

    def states(self, fit: pd.DataFrame) -> dict:
        """per capacity: median |error| %, median error % (bias), the last day's and the last RECENT_D days' |error|,
        and the correlation of CHANGE_D-day changes (fit vs truth)."""
        cols = list(self.STATES if self.info["truth"] == ModelTruth.KIND else self.SHARED)
        f, t = fit[cols].align(self.truth[cols], join="inner", axis=0)
        e = f / t - 1.0
        df, dt = np.log(f).diff(self.CHANGE_D), np.log(t).diff(self.CHANGE_D)
        recent = e.index >= e.index[-1] - pd.Timedelta(days=self.RECENT_D)
        follows = df.corrwith(dt)
        return {c: {"abs_pct": 100 * float(e[c].abs().median()), "bias_pct": 100 * float(e[c].median()),
                    "recent_abs_pct": 100 * float(e[c][recent].abs().median()), "last_pct": 100 * float(e[c].iloc[-1]),
                    "follows_r": None if pd.isna(follows[c]) else float(follows[c])} for c in cols}

    def params(self, fitted: dict, prior_sd: dict) -> dict:
        """model truth: each training-response parameter's fitted value less the true one, in prior sds."""
        if self.info["truth"] != ModelTruth.KIND:
            return {}
        true = self.info["params"]
        keep = [n for n in true if n.split("[")[0] in ModelTruth.DRAWN and n in fitted]
        return {n: (fitted[n] - true[n]) / prior_sd[n] for n in keep}

    def heart(self, athlete: dict) -> dict:
        """goodpush's heart rates (estimated unless the profile entered them) against the rider's (bpm off)."""
        return {k: {"found": athlete.get(k), "true": self.info[k], "source": athlete.get(f"{k}_source") or "estimated",
                    "off": None if athlete.get(k) is None else float(athlete[k]) - self.info[k]}
                for k in ("max_hr", "lthr", "resting_hr")}

    def thresholds(self, processed: Path) -> dict:
        """alpha1's LT1 on ordinary rides and the ramp tests' LT1 / LT2 against the true LT1 / CP that day:
        median error %, how many."""
        try:
            a = pd.read_csv(processed / "alpha_rides.csv", parse_dates=["date"])
        except OSError:
            return {}
        day = a["date"].dt.normalize()
        true = self.truth.reindex(day)
        out = {}
        for name, col, q in (("alpha1_lt1", "hrvt1_w", "lt1"), ("ramp_lt1", "ramp_lt1_w", "lt1"),
                             ("ramp_lt2", "ramp_lt2_w", "cp")):
            if col not in a:
                continue
            e = (a[col].to_numpy(float) / true[q].to_numpy(float) - 1.0) * 100
            e = e[np.isfinite(e)]
            out[name] = {"n": int(e.size), "median_pct": float(np.median(e)) if e.size else None}
        return out

    def matched(self, true_days: list, found_days: list) -> dict:
        """how many true dates have a found one within FOUND_D days, and how many found ones match none."""
        t = pd.DatetimeIndex(true_days)
        f = pd.DatetimeIndex(found_days)
        if not len(t) or not len(f):
            return {"true": len(t), "found": len(f), "hit": 0, "false": len(f)}
        d = np.abs((t.values[:, None] - f.values[None, :]) / np.timedelta64(1, "D"))
        return {"true": len(t), "found": len(f), "hit": int((d.min(1) <= self.FOUND_D).sum()),
                "false": int((d.min(0) > self.FOUND_D).sum())}

    def illness(self, res: dict) -> dict:
        found = [e["onset"] for e in (res.get("illness") or {}).get("episodes") or [] if e.get("onset")]
        return self.matched([e["onset"] for e in self.info["illnesses"]], found)

    def breaks(self, processed: Path) -> dict:
        """the breaks goodpush detects (models.dips) against the true breaks and illnesses."""
        try:
            dips = json.loads((processed / "dips.json").read_text(encoding="utf-8")).get("dips") or []
        except (OSError, ValueError):
            dips = []
        true = [e["start"] for e in self.info["breaks"]] + [e["onset"] for e in self.info["illnesses"]]
        return self.matched(true, [d.get("start") for d in dips if d.get("start")])

    def markers(self, processed: Path) -> dict:
        try:
            m = json.loads((processed / "illness_markers.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {"episodes": m.get("episodes"), "chosen": [c.get("key") if isinstance(c, dict) else c
                                                          for c in m.get("chosen") or []]}

    @staticmethod
    def text(out: dict) -> str:
        """the scores, short, for the terminal."""
        lines = [f"Synthetic rider ({out['truth']} truth, {out['days']} days) - goodpush's analysis against the truth:",
                 "  capacity   typical off   bias    last 90 d   follows changes"]
        for c, s in out["states"].items():
            r = "-" if s["follows_r"] is None else f"{s['follows_r']:.2f}"
            lines.append(f"  {c:<9}  {s['abs_pct']:5.1f} %     {s['bias_pct']:+5.1f} %  {s['recent_abs_pct']:5.1f} %     r {r}")
        if out["params"]:
            worst = sorted(out["params"].items(), key=lambda kv: -abs(kv[1]))[:5]
            lines.append("  training response, furthest from the truth (prior sds): "
                         + ", ".join(f"{n} {v:+.1f}" for n, v in worst))
        for k, h in out["heart"].items():
            off = "-" if h["off"] is None else f"{h['off']:+.0f} bpm"
            lines.append(f"  {k}: {h['found']} vs {h['true']} ({off}; {h['source']})")
        for k, t in out["thresholds"].items():
            med = "-" if t["median_pct"] is None else f"{t['median_pct']:+.1f} %"
            lines.append(f"  {k}: {t['n']} readings, median {med} from the truth")
        for k in ("illness", "breaks"):
            m = out[k]
            lines.append(f"  {k}: {m['hit']} of {m['true']} found, {m['false']} found that weren't")
        if out["markers"]:
            lines.append(f"  illness markers: {', '.join(out['markers']['chosen']) or 'none learned'} "
                         f"({out['markers']['episodes']} episodes)")
        return "\n".join(lines)
