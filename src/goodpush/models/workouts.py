"""Classify each ride by its *workout type* from the 1 Hz power stream.

Intervals are detected on 30 s-smoothed power relative to the FTP of the
day, then the time spent in sustained efforts of each intensity decides the
type. Workout names from training apps (Zwift, SYSTM/Sufferfest, TrainerRoad)
are used to check the classifier and as a fallback.

Types (in priority order):
  test            - name says test / ramp / 4DP / FTP protocol
  race            - name says race / cat / open / crit, or race-like power
  vo2max          - >= 8 min in efforts of 2-8 min at >= 106 % FTP (or 30/30-style blocks)
  threshold       - >= 15 min in efforts >= 5 min at 95-105 % FTP (incl. over-unders)
  sweetspot       - >= 20 min in efforts >= 8 min at 88-94 % FTP
  tempo           - >= 30 min in efforts >= 10 min at 76-87 % FTP
  anaerobic       - >= 5 efforts of 15 s-2 min at >= 120 % FTP
  long_endurance  - >= 3 h, nothing above
  recovery        - IF < 0.65 and < 75 min
  endurance       - everything else
"""
from __future__ import annotations

import re

import numpy as np

TYPES = ["recovery", "endurance", "long_endurance", "tempo", "sweetspot", "threshold",
         "vo2max", "anaerobic", "race", "test"]
QUALITY = {"tempo", "sweetspot", "threshold", "vo2max", "anaerobic", "race", "test"}

_NAME_RULES = [
    ("test", r"\btest\b|ramp test|\b4dp\b|full frontal|half monty|ftp protocol|power profile|\blsct\b"),
    ("race", r"\brace\b|\bcat ?[1-5]\b|\bopen\b|\bcrit\b|24 ?hrs|gravel worlds|\bcrit\b"),
    ("vo2max", r"vo2|supra-?threshold|microburst|30 ?[/:] ?30|30 ?[/:] ?15|40 ?[/:] ?20|max aerobic"),
    ("anaerobic", r"anaerobic capacity|lactate tolerance(?!.*endurance)|sprint|neuromuscular"),
    ("threshold", r"threshold|lactate clearance|over-?\s?unders?|ftp builder - threshold|\b\d+x(8|10|12|15|20)m?\b"),
    ("sweetspot", r"sweet ?spot|muscular endurance"),
    ("tempo", r"\btempo\b"),
    ("recovery", r"recovery|\bchill|\beasy\b|spin\b"),
    ("endurance", r"endurance|\bz2\b|foundation|steady"),
]


def name_label(name: str) -> str | None:
    n = (name or "").lower()
    for label, pat in _NAME_RULES:
        if re.search(pat, n):
            return label
    return None


def _segments(mask: np.ndarray, min_len: int, max_gap: int = 20):
    """Runs of True (allowing short gaps) of at least min_len samples -> list of (start, end)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    segs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            if prev - s + 1 >= min_len:
                segs.append((s, prev + 1))
            s = i
        prev = i
    if prev - s + 1 >= min_len:
        segs.append((s, prev + 1))
    return segs


def _smooth(p, w):
    if p.size < w:
        return p.copy()
    c = np.concatenate([[0.0], np.cumsum(p)])
    out = np.empty_like(p)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    out[:w - 1] = p[:w - 1]
    return out


HR_GATE = {"vo2": 0.92, "thr": 0.90, "ss": 0.85, "tempo": 0.80}


def interval_summary(power: np.ndarray, ftp: float, hr=None, lthr: float | None = None) -> dict:
    """Sustained efforts by intensity. If heart rate and LTHR are given, an effort only
    counts when HR in its second half also reached the zone (guards against an FTP
    estimate that is too low, which would make easy rides look like intervals)."""
    p = np.nan_to_num(np.asarray(power, float))
    h = None
    if hr is not None and lthr:
        h = np.asarray(hr, float)
        if np.isfinite(h).sum() < 0.5 * h.size:
            h = None

    def hr_ok(s, e, key):
        if h is None:
            return True
        tail = h[s + (e - s) // 2: e]
        tail = tail[np.isfinite(tail)]
        return tail.size == 0 or tail.mean() >= HR_GATE[key] * lthr
    p30 = _smooth(p, 30) / ftp
    p10 = _smooth(p, 10) / ftp
    res = {"vo2_s": 0, "thr_s": 0, "ss_s": 0, "tempo_s": 0, "anaer_reps": 0,
           "vo2_reps": 0, "thr_reps": 0, "ss_reps": 0, "longest_effort_s": 0,
           "intermittent_s": 0}
    # sustained efforts from 30 s-smoothed power: start at >= 0.76 FTP, at least 2 min
    for s, e in _segments(p30 >= 0.76, 120):
        seg = p30[s:e]
        dur = e - s
        m = float(seg.mean())
        res["longest_effort_s"] = max(res["longest_effort_s"], dur)
        if m >= 1.06 and dur <= 600:
            if hr_ok(s, e, "vo2"):
                res["vo2_s"] += dur
                res["vo2_reps"] += 1
        elif m >= 0.95 and dur >= 300:
            if hr_ok(s, e, "thr"):
                res["thr_s"] += dur
                res["thr_reps"] += 1
        elif m >= 0.88 and dur >= 480:
            if hr_ok(s, e, "ss"):
                res["ss_s"] += dur
                res["ss_reps"] += 1
        elif m >= 0.76 and dur >= 600:
            if hr_ok(s, e, "tempo"):
                res["tempo_s"] += dur
        elif m >= 1.06 and hr_ok(s, e, "thr"):  # long hard block (race / test-like)
            res["thr_s"] += dur
    # intermittent VO2 (30/30, 30/15, 40/20): 5-min windows averaging >= 0.95 FTP with
    # high within-window variability
    if p.size >= 300:
        p5 = _smooth(p, 300) / ftp
        sd = np.sqrt(np.maximum(_smooth((p / ftp) ** 2, 300) - p5 ** 2, 0))
        win = (p5 >= 0.95) & (sd >= 0.35)
        if h is not None and win.any():
            h5 = _smooth(np.nan_to_num(h, nan=0.0), 300)
            win &= h5 >= HR_GATE["thr"] * lthr
        res["intermittent_s"] = int(win.sum())
    # anaerobic reps from 10 s-smoothed power
    res["anaer_reps"] = len(_segments(p10 >= 1.2, 15, max_gap=5))
    return res


def classify(power, ftp: float, name: str = "", duration_s: float | None = None,
             np_w: float | None = None, hr=None, lthr: float | None = None,
             indoor: bool | None = None) -> dict:
    """Return {'type', 'type_stream', 'type_name', ...interval summary}."""
    nl = name_label(name)
    out = {"type_name": nl}
    if power is None or ftp is None or not np.isfinite(ftp) or ftp <= 0:
        out.update(type_stream=None, type=nl or "endurance")
        return out
    p = np.nan_to_num(np.asarray(power, float))
    s = interval_summary(p, ftp, hr, lthr)
    out.update(s)
    dur = duration_s or p.size
    npw = np_w if np_w else float(np.mean(_smooth(p, 30) ** 4) ** 0.25)
    if_ = npw / ftp
    vi = npw / max(p.mean(), 1)
    out.update(if_=if_, vi=vi)
    if s["vo2_s"] >= 480 or s["intermittent_s"] >= 600:
        t = "vo2max"
    elif s["thr_s"] >= 900:
        t = "threshold"
    elif s["ss_s"] >= 1200:
        t = "sweetspot"
    elif s["tempo_s"] >= 1800:
        t = "tempo"
    elif s["anaer_reps"] >= 5 and (indoor or nl == "anaerobic"):
        t = "anaerobic"   # outdoors, short surges are terrain, not an anaerobic session
    elif if_ >= 0.85 and vi >= 1.12 and s["anaer_reps"] >= 8:
        t = "race"
    elif dur >= 3 * 3600:
        t = "long_endurance"
    elif if_ < 0.65 and dur < 75 * 60:
        t = "recovery"
    else:
        t = "endurance"
    out["type_stream"] = t
    # names win for tests and races (not detectable reliably from power) and when the
    # stream finds nothing structured but the name says what the session was meant to be
    if nl in ("test", "race"):
        t = nl
    elif nl in QUALITY and t in ("endurance", "long_endurance", "recovery", "tempo"):
        t = nl if (s["vo2_s"] + s["thr_s"] + s["ss_s"] + s["intermittent_s"]) > 0 or \
            nl == "tempo" else t
    out["type"] = t
    return out
