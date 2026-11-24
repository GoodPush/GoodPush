"""Write reports: dashboard.html, profile.md, results.json and CSV tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import dashboard, profile_md
from .results import build_results


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)


def write_reports(proj, r: dict, progress=print) -> dict:
    res = build_results(proj, r)
    out = proj.reports
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, cls=_Enc, indent=1), encoding="utf-8")
    (out / "profile.md").write_text(profile_md.render(res), encoding="utf-8")
    (out / "dashboard.html").write_text(dashboard.render(r, res), encoding="utf-8")
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    keep = ["id", "date", "name", "type", "sport", "indoor", "duration_s", "distance_m",
            "elev_gain", "workout", "type_name", "np_used", "avg_power", "ftp", "if_", "tss", "tss_source", "avg_hr",
            "ef", "decoupling", "hard", "long", "mmp_5", "mmp_60", "mmp_300", "mmp_1200",
            "mmp_3600", "mood_bad", "gear"]
    acts = r["acts"]
    acts[[c for c in keep if c in acts]].to_csv(tables / "activities.csv", index=False)
    r["daily"].to_csv(tables / "daily.csv")
    r["weekly"].to_csv(tables / "weekly.csv")
    for name in ("periods", "declines", "episodes"):
        df = r.get(name)
        if df is not None:
            df.to_csv(tables / f"{name}.csv", index=False)
    progress(f"reports written to {out}")
    return res
