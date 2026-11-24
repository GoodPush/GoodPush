"""Athlete power profile and phenotype (Coggan power-profile style)."""
from __future__ import annotations

import numpy as np

# W/kg ranges (approx. "untrained" -> "world class") from Coggan's power profile chart
PROFILE_RANGES = {
    "male": {"5s": (10.17, 24.04), "1min": (5.64, 11.50), "5min": (2.33, 7.60), "ftp": (1.86, 6.40)},
    "female": {"5s": (8.28, 19.42), "1min": (4.71, 9.29), "5min": (1.89, 6.61), "ftp": (1.50, 5.69)},
}
CATEGORIES = [(0, "Untrained"), (15, "Fair"), (30, "Moderate"), (45, "Good"),
              (60, "Very good"), (75, "Excellent"), (88, "Exceptional"), (97, "World class")]


def profile_scores(wkg: dict, sex: str = "male") -> dict:
    """Map W/kg at 5s/1min/5min/FTP to a 0-100 score on the population scale."""
    rng = PROFILE_RANGES.get(sex, PROFILE_RANGES["male"])
    out = {}
    for k, (lo, hi) in rng.items():
        v = wkg.get(k)
        if v is None or not np.isfinite(v):
            out[k] = np.nan
            continue
        out[k] = float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))
    return out


def category(score: float) -> str:
    if not np.isfinite(score):
        return "n/a"
    label = CATEGORIES[0][1]
    for thr, name in CATEGORIES:
        if score >= thr:
            label = name
    return label


def phenotype(scores: dict) -> tuple[str, str]:
    """Classify the rider from the *shape* of the profile (relative strengths)."""
    s = {k: v for k, v in scores.items() if np.isfinite(v)}
    if len(s) < 3:
        return "unknown", "not enough data across durations"
    mean = np.mean(list(s.values()))
    spread = max(s.values()) - min(s.values())
    top = max(s, key=s.get)
    low = min(s, key=s.get)
    if spread < 10:
        return "All-rounder", "balanced profile - no duration stands out by >10 points"
    if top in ("5s", "1min") and s.get("ftp", mean) < mean:
        return "Sprinter", f"strongest at {top}, relatively weaker aerobically"
    if top == "5min":
        return "Pursuiter / puncheur", "VO2max-type efforts (3-8 min) are the relative strength"
    if top == "ftp":
        return "Time-trialist / diesel", f"sustained threshold power is the relative strength; weakest at {low}"
    return "All-rounder", f"leaning to {top}, weakest at {low}"
