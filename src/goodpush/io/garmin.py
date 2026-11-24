"""Garmin wellness data: sleep, resting HR, HRV, stress, readiness, weight.

Where each metric lives in a Garmin Connect export (unzipped into the garmin folder):

| metric                         | file(s)                                                          | from  |
|--------------------------------|------------------------------------------------------------------|-------|
| sleep duration/stages/score    | DI_CONNECT/DI-Connect-Wellness/*_sleepData.json                  | 2018  |
| sleep stress, respiration      |   (same; score & stress only on newer watches)                  | 2023  |
| resting HR, stress, body batt. | DI_CONNECT/DI-Connect-Aggregator/UDSFile_*.json                  | 2018  |
| nightly HRV (+ baseline)       | DI_CONNECT/DI-Connect-Wellness/*_healthStatusData.json           | 2025  |
| nightly HRV (+ baseline)       | DI_CONNECT/DI-Connect-Uploaded-Files/UploadedFiles_*.zip -> FIT  |       |
|                                |   files of type 68 (HRV status summary message 370)             | 2023  |
| HRV 7-day avg, readiness score | DI_CONNECT/DI-Connect-Metrics/TrainingReadinessDTO_*.json        | 2023  |
| weight, Garmin FTP, VO2max     | DI_CONNECT/DI-Connect-Wellness/*_userBioMetrics.json             | 2018  |

Loose watch FIT files (sleep / monitoring / HRV) anywhere in the garmin folder are read too.
The zip scan is cached in data/processed (keyed by zip size + mtime).
"""
from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import fit
from ..progress import Progress

COLUMNS = ["sleep_h", "deep_h", "light_h", "rem_h", "awake_h", "sleep_score", "sleep_stress",
           "respiration", "bedtime_h", "rhr", "stress", "body_battery_max", "body_battery_min",
           "steps", "hrv", "hrv_weekly", "hrv_base_low", "hrv_base_high", "readiness",
           "weight", "vo2max_garmin", "ftp_garmin", "ftp_garmin_auto"]

FIT_HRV_TYPE = 68
FIT_SLEEP_TYPE = 49
FIT_MONITORING_TYPE = 32


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _date(v):
    if isinstance(v, str):
        return v[:10]
    if isinstance(v, (int, float)):  # epoch ms
        return pd.Timestamp(v, unit="ms").strftime("%Y-%m-%d")
    return None


# --- JSON parsers --------------------------------------------------------------
def parse_sleep(recs, tz):
    for r in recs:
        d = _date(r.get("calendarDate"))
        if not d or r.get("deepSleepSeconds") is None:
            continue
        parts = [r.get(k) or 0 for k in ("deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds")]
        row = {"date": d, "sleep_h": sum(parts) / 3600,
               "deep_h": parts[0] / 3600, "light_h": parts[1] / 3600,
               "rem_h": (r.get("remSleepSeconds") / 3600) if r.get("remSleepSeconds") is not None else np.nan,
               "awake_h": (r.get("awakeSleepSeconds") or 0) / 3600,
               "sleep_score": (r.get("sleepScores") or {}).get("overallScore"),
               "sleep_stress": r.get("avgSleepStress"), "respiration": r.get("averageRespiration")}
        st = r.get("sleepStartTimestampGMT")
        if st:
            t = pd.Timestamp(st, tz="UTC").tz_convert(tz)
            h = t.hour + t.minute / 60
            row["bedtime_h"] = h if h >= 12 else h + 24  # 23.5 = 11:30pm, 25 = 1am
        yield row


def parse_uds(recs, tz=None):
    for r in recs:
        d = _date(r.get("calendarDate"))
        if not d:
            continue
        row = {"date": d, "rhr": r.get("currentDayRestingHeartRate") or r.get("restingHeartRate"),
               "steps": r.get("totalSteps")}
        for a in (r.get("allDayStress") or {}).get("aggregatorList", []) or []:
            if a.get("type") == "TOTAL":
                row["stress"] = a.get("averageStressLevel")
        for b in (r.get("bodyBattery") or {}).get("bodyBatteryStatList", []) or []:
            if b.get("bodyBatteryStatType") == "HIGHEST":
                row["body_battery_max"] = b.get("statsValue")
            elif b.get("bodyBatteryStatType") == "LOWEST":
                row["body_battery_min"] = b.get("statsValue")
        yield row


def parse_health_status(recs, tz=None):
    for r in recs:
        d = _date(r.get("calendarDate"))
        for m in r.get("metrics", []) or []:
            if m.get("type") == "HRV" and m.get("value"):
                yield {"date": d, "hrv": m["value"],
                       "hrv_base_low": m.get("baselineLowerLimit") or np.nan,
                       "hrv_base_high": m.get("baselineUpperLimit") or np.nan}


def parse_readiness(recs, tz=None):
    # several records per day; the first (morning, after waking) is the one that matters
    recs = sorted([r for r in recs if r.get("calendarDate")],
                  key=lambda r: (r["calendarDate"], r.get("timestamp", "")))
    seen = set()
    for r in recs:
        d = _date(r["calendarDate"])
        if d in seen:
            continue
        seen.add(d)
        yield {"date": d, "readiness": r.get("score"), "hrv_weekly": r.get("hrvWeeklyAverage")}


def parse_biometrics(recs, tz=None):
    for r in recs:
        d = _date((r.get("metaData") or {}).get("calendarDate"))
        if not d:
            continue
        row = {"date": d}
        w = r.get("weight")
        if isinstance(w, dict) and w.get("weight"):
            row["weight"] = w["weight"] / 1000.0
        if r.get("vo2MaxCycling"):
            row["vo2max_garmin"] = r["vo2MaxCycling"]
        if r.get("functionalThresholdPower"):
            row["ftp_garmin"] = r["functionalThresholdPower"]
        if "ftpAutoDetected" in r:                  # Garmin's own estimate (1) or typed in (0)
            row["ftp_garmin_auto"] = int(bool(r["ftpAutoDetected"]))
        if len(row) > 1:
            yield row


PARSER_VERSION = 3  # bump when a parser changes: cached rows are re-read
JSON_SOURCES = [  # (filename pattern, parser, label)
    ("*sleepData*.json", parse_sleep, "sleep"),
    ("UDSFile_*.json", parse_uds, "daily summary (RHR, stress, body battery)"),
    ("*healthStatusData*.json", parse_health_status, "health status (nightly HRV)"),
    ("TrainingReadinessDTO_*.json", parse_readiness, "training readiness (HRV weekly avg)"),
    ("*userBioMetrics.json", parse_biometrics, "biometrics (weight, VO2max, FTP)"),
]


# --- FIT parsers -----------------------------------------------------------------
def _fit_day(ts, tz):
    return pd.Timestamp(ts + fit.FIT_EPOCH_OFFSET, unit="s", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d")


def fit_file_type(data: bytes):
    try:
        for _, m in fit.iter_messages(data, (0,)):
            return m.get(0)
    except Exception:
        return None
    return None


def parse_activity_ftp(data: bytes, tz: str):
    """The FTP a ride's FIT file says the device used (zones_target; Garmin's own estimate when auto-detect
    is on). It comes before the first record, so the file is read only up to there."""
    made = ftp = None
    try:
        for num, m in fit.iter_messages(data, (0, 7, 20)):
            if num == 0 and m.get(4):
                made = made or m[4]                  # file_id.time_created
            elif num == 7 and m.get(3):
                ftp = m[3]                           # zones_target.functional_threshold_power
            elif num == 20:
                break
    except Exception:
        return
    if made and ftp:
        yield {"date": _fit_day(made, tz), "ftp_garmin": ftp}


def parse_fit_wellness(data: bytes, tz: str):
    """Rows from a watch FIT file: HRV status (type 68), sleep (49), monitoring RHR (32), and the FTP
    an activity file says the device used (zones_target; Garmin's own estimate when auto-detect is on)."""
    try:
        msgs = fit.read_messages(data, (0, 7, 211, 275, 346, 370, 371))
    except Exception:
        return
    made = next((m[4] for m in msgs.get(0, []) if m.get(4)), None)       # file_id.time_created
    ftp = next((m[3] for m in msgs.get(7, []) if m.get(3)), None)        # zones_target.functional_threshold_power
    if made and ftp:
        yield {"date": _fit_day(made, tz), "ftp_garmin": ftp}
    for m in msgs.get(370, []):  # hrv_status_summary, values in ms * 128
        if m.get(253) is None or not m.get(1):
            continue
        yield {"date": _fit_day(m[253], tz), "hrv": m[1] / 128,
               "hrv_weekly": (m.get(0) or 0) / 128 or np.nan,
               "hrv_base_low": (m.get(4) or 0) / 128 or np.nan,
               "hrv_base_high": (m.get(5) or 0) / 128 or np.nan}
    if not msgs.get(370) and msgs.get(371):
        by = defaultdict(list)
        for m in msgs[371]:
            if m.get(253) is not None and m.get(0):
                by[_fit_day(m[253], tz)].append(m[0] / 128)
        for d, v in by.items():
            yield {"date": d, "hrv": float(np.mean(v))}
    levels = sorted((m[253], m.get(0)) for m in msgs.get(275, []) if m.get(253) is not None)
    if levels:
        ts = np.array([x[0] for x in levels], float)
        lv = np.array([-1 if x[1] is None else x[1] for x in levels])
        dt = np.clip(np.diff(np.append(ts, ts[-1] + 60)), 0, 600)
        secs = {k: float(dt[lv == k].sum()) for k in (1, 2, 3, 4)}
        asleep = secs[2] + secs[3] + secs[4]
        if asleep > 0:
            row = {"date": _fit_day(ts[-1], tz), "sleep_h": asleep / 3600, "light_h": secs[2] / 3600,
                   "deep_h": secs[3] / 3600, "rem_h": secs[4] / 3600, "awake_h": secs[1] / 3600}
            for a in msgs.get(346, []):
                if a.get(6) is not None:
                    row["sleep_score"] = a[6]
            yield row
    for m in msgs.get(211, []):
        if m.get(253) is not None and m.get(1):
            yield {"date": _fit_day(m[253], tz), "rhr": m[1]}


def scan_uploaded_zips(garmin_dir: Path, tz: str, cache_dir: Path | None, progress=print):
    """HRV status files inside DI-Connect-Uploaded-Files/*.zip (cached per zip)."""
    rows = []
    zips = sorted(Path(garmin_dir).rglob("UploadedFiles*.zip"))
    if not zips:
        return rows
    cache = {}
    cfile = cache_dir / "garmin_zip_hrv.json" if cache_dir else None
    if cfile and cfile.exists():
        try:
            cache = json.loads(cfile.read_text())
        except Exception:
            cache = {}
    changed = False
    for z in zips:
        st = z.stat()
        key = f"{z.name}|{st.st_size}|{int(st.st_mtime)}|{tz}"
        if key in cache:
            rows.extend(cache[key])
            continue
        progress(f"  scanning {z.name} for HRV files ...")
        got = []
        with zipfile.ZipFile(z) as zz:
            for n in zz.namelist():
                if not n.lower().endswith(".fit"):
                    continue
                data = zz.read(n)
                if fit_file_type(data) != FIT_HRV_TYPE:
                    continue
                got.extend(parse_fit_wellness(data, tz))
        cache = {k: v for k, v in cache.items() if not k.startswith(z.name + "|")}
        cache[key] = got
        rows.extend(got)
        changed = True
    if changed and cfile:
        cfile.parent.mkdir(parents=True, exist_ok=True)
        cfile.write_text(json.dumps(cache, default=lambda o: None if o != o else o))
    return rows


# --- main entry ---------------------------------------------------------------------
def load_wellness(garmin_dir: str | Path, tz: str = "UTC", fit_dirs=(), cache_dir=None,
                  progress=print, scan_zips=True) -> tuple[pd.DataFrame, dict]:
    """Return (daily wellness DataFrame indexed by date, inventory dict)."""
    gdir = Path(garmin_dir)
    inv = {"sources": {}, "json_files": 0, "fit_files": 0}
    frames = []  # (priority, DataFrame); lower priority value wins on conflicts
    if gdir.exists():
        import fnmatch
        import os
        import pickle
        # one walk of the folder (instead of one per file pattern) ...
        listing = []
        for root, dirs, files in os.walk(gdir):
            if Path(root) == gdir:             # activities/: ride files (API / export): only Garmin's FTP in them
                dirs[:] = [d for d in dirs if d != "activities"]
            for fn in files:
                listing.append(os.path.join(root, fn))
        listing.sort()
        rides = sorted(str(f) for f in (gdir / "activities").glob("*.fit"))
        # ... and parsed rows cached per file (re-parsed only when size / time change)
        cpath = Path(cache_dir) / "wellness_rows.pkl" if cache_dir else None
        try:
            fcache = pickle.loads(cpath.read_bytes()) if cpath and cpath.exists() else {}
        except Exception:
            fcache = {}
        used, dirty = {}, False

        def rows_for(path, parser, fit=False):
            nonlocal dirty
            st = os.stat(path)
            key = (st.st_size, st.st_mtime_ns, parser.__name__, tz, PARSER_VERSION)
            hit = fcache.get(path)
            if hit is not None and hit[0] == key:
                used[path] = hit
                return hit[1]
            if fit:
                rows = parser(Path(path).read_bytes(), tz)
            else:
                recs = _load(Path(path))
                rows = parser(recs if isinstance(recs, list) else [recs], tz)
            used[path] = (key, rows)
            dirty = True
            return rows
        for prio, (pat, parser, label) in enumerate(JSON_SOURCES):
            files = [f for f in listing if fnmatch.fnmatch(os.path.basename(f), pat)]
            rows = []
            for f in files:
                rows.extend(rows_for(f, parser))
            inv["json_files"] += len(files)
            if rows:
                frames.append((prio, pd.DataFrame(rows)))
                inv["sources"][label] = {"files": len(files), "records": len(rows)}
        fit_rows = []
        fits = [f for f in listing if f.lower().endswith(".fit")] + \
            [str(q) for d in fit_dirs for q in Path(d).rglob("*.fit")]
        for i, f in enumerate(fits, 1):
            Progress.step(progress, "Garmin health files", i, len(fits))
            inv["fit_files"] += 1
            fit_rows.extend(rows_for(f, parse_fit_wellness, fit=True))
        for i, f in enumerate(rides, 1):
            Progress.step(progress, "Garmin's FTP in ride files", i, len(rides))
            fit_rows.extend(rows_for(f, parse_activity_ftp, fit=True))
        if cpath and (dirty or set(used) != set(fcache)):
            try:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_bytes(pickle.dumps(used))
            except Exception:  # noqa: BLE001 - cache is optional
                pass
        zip_rows = scan_uploaded_zips(gdir, tz, Path(cache_dir) if cache_dir else None, progress) \
            if scan_zips else []
        if zip_rows:
            inv["sources"]["uploaded FIT HRV files"] = {"records": len(zip_rows)}
        if fit_rows or zip_rows:
            frames.append((len(JSON_SOURCES), pd.DataFrame(fit_rows + zip_rows)))
    empty = pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"))
    if not frames:
        return empty, inv
    daily = None
    for _, df in sorted(frames, key=lambda x: x[0]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        g = df.groupby("date").last()        # last non-missing value per column (C, not Python)
        daily = g if daily is None else daily.combine_first(g)
    for c in COLUMNS:
        if c not in daily:
            daily[c] = np.nan
    daily = daily[COLUMNS].apply(pd.to_numeric, errors="coerce").sort_index()
    daily.index.name = "date"
    # plausibility filters
    daily.loc[(daily["sleep_h"] < 1) | (daily["sleep_h"] > 16), "sleep_h"] = np.nan
    daily.loc[(daily["rhr"] < 25) | (daily["rhr"] > 120), "rhr"] = np.nan
    daily.loc[(daily["hrv"] <= 5) | (daily["hrv"] > 300), "hrv"] = np.nan
    w = daily["weight"]
    med = w.median()
    daily.loc[(w - med).abs() > 0.15 * med, "weight"] = np.nan  # drop default/placeholder weights
    daily = daily.dropna(how="all")
    inv["coverage"] = {c: _coverage(daily[c]) for c in COLUMNS if daily[c].notna().any()}
    return daily, inv


def _coverage(s: pd.Series) -> str:
    s = s.dropna()
    return f"{len(s)} days {s.index.min():%Y-%m-%d} -> {s.index.max():%Y-%m-%d}"
