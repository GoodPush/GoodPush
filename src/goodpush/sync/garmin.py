"""Incremental Garmin Connect download (sleep, resting HR, HRV, readiness, weight).

Preferred route now: the official Personal API (sync/garmin_api.py, `goodpush garmin-key`).
This module keeps the older routes: the community `garminconnect`
package (install with `pip install -e goodpush[garmin]`). It logs in like the
Garmin Connect web site; tokens are cached in ~/.goodpush/garmin_tokens so you
only type your password (and MFA code) once a year or so.

`goodpush sync garmin` fetches every day after the newest one you already have,
stores the raw responses in data/raw/garmin/api/raw/<date>.json, and rewrites
export-shaped files next to them so io.garmin reads them exactly like the bulk
export:
  data/raw/garmin/api/api_sleepData.json, UDSFile_api.json,
  api_healthStatusData.json, TrainingReadinessDTO_api.json, api_userBioMetrics.json

If Garmin changes its private API and this breaks, a fresh bulk export
(Account > Data management > Export your data) always works as a fallback.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..io.stamps import in_seconds, stamp
from . import FIRST_SYNC_DAYS, secrets_dir

API_SUBDIR = "api"
TOKENS = "garmin_tokens"


def _import_garmin():
    try:
        from garminconnect import Garmin
    except ImportError:
        raise RuntimeError("Garmin sync needs the garminconnect package:  "
                           "pip install -e \"path/to/goodpush[garmin]\"") from None
    return Garmin


def _version() -> tuple:
    try:
        from importlib.metadata import version
        return tuple(int(x) for x in version("garminconnect").split(".")[:2])
    except Exception:
        return (0, 0)


def _has_curl_cffi() -> bool:
    try:
        import curl_cffi  # noqa: F401
        return True
    except Exception:
        return False


def status() -> dict:
    """What goodpush knows about the Garmin connection (no network calls)."""
    import time as _t
    store_dir, store_txt = _token_paths()
    tokens = (store_dir.exists() and any(store_dir.iterdir())) or store_txt.exists()
    until = _blocked_until()
    try:
        from importlib.metadata import version
        gc = version("garminconnect")
    except Exception:
        gc = None
    return {"garminconnect": gc, "curl_cffi": _has_curl_cffi(), "tokens": bool(tokens),
            "token_dir": str(store_dir),
            "blocked_hours_left": round((until - _t.time()) / 3600, 1) if until else 0}


def _token_paths():
    d = secrets_dir()
    return d / TOKENS, d / (TOKENS + ".txt")


def _save_tokens(api) -> bool:
    """Persist the login session. garminconnect's API changed across versions
    (api.garth / api.client, dump(dir) / dumps()), so try the known variants."""
    store_dir, store_txt = _token_paths()
    for holder in (getattr(api, "garth", None), getattr(api, "client", None), api):
        if holder is None:
            continue
        dump = getattr(holder, "dump", None)
        if callable(dump):
            try:
                store_dir.mkdir(parents=True, exist_ok=True)
                dump(str(store_dir))
                return True
            except Exception:
                pass
        dumps = getattr(holder, "dumps", None)
        if callable(dumps):
            try:
                store_txt.write_text(dumps())
                return True
            except Exception:
                pass
    return False


def _friendly(e: Exception) -> str:
    msg = str(e)
    if "429" in msg or "TooManyRequests" in type(e).__name__:
        tip = "" if _has_curl_cffi() else (
            " curl_cffi is NOT installed, so the login looked like a script rather than a "
            "browser - install it (pip install --upgrade garminconnect curl_cffi) before the "
            "next try.")
        return ("Garmin is rate-limiting logins (HTTP 429). These blocks often last 48-72 h and "
                "every retry can extend them, so goodpush won't try again for 48 h." + tip +
                " Meanwhile `goodpush garmin-snippet` gets the same data from your browser "
                "without logging in.")
    if "401" in msg or "Authentication" in type(e).__name__:
        return ("Garmin rejected the login (401). Check the password by logging in at "
                "connect.garmin.com. Note: after a burst of 429s Garmin sometimes answers "
                "401 even for the right password - wait an hour, then try once more.")
    return f"{type(e).__name__}: {msg}"


def logout(progress=print) -> None:
    """Delete the cached Garmin session (Garmin offers no remote revoke for it)."""
    import shutil
    store_dir, store_txt = _token_paths()
    gone = False
    if store_dir.exists():
        shutil.rmtree(store_dir, ignore_errors=True)
        gone = True
    if store_txt.exists():
        store_txt.unlink()
        gone = True
    progress("Garmin: cached session deleted." if gone else "Garmin: no cached session found.")


def _client(interactive=True):
    Garmin = _import_garmin()
    store_dir, store_txt = _token_paths()
    token = str(store_dir) if store_dir.exists() and any(store_dir.iterdir()) else \
        (store_txt.read_text() if store_txt.exists() else None)
    if token:
        try:
            api = Garmin()
            api.login(token)
            return api
        except Exception as e:  # expired / revoked tokens
            if not interactive:
                raise RuntimeError(f"Garmin session expired ({_friendly(e)}) - "
                                   "run `goodpush auth garmin`") from None
    if not interactive:
        raise RuntimeError("Garmin not connected - run `goodpush auth garmin` first")
    return login()


BLOCK_FILE = "garmin_block.json"
BLOCK_HOURS = 48


def _blocked_until():
    """When Garmin last rate-limited us, refuse new login attempts for BLOCK_HOURS:
    every retry during a block tends to extend it."""
    import json as _j
    import time as _t
    p = secrets_dir() / BLOCK_FILE
    if not p.exists():
        return None
    try:
        until = _j.loads(p.read_text())["until"]
    except Exception:
        return None
    return until if until > _t.time() else None


def _note_block():
    import json as _j
    import time as _t
    (secrets_dir() / BLOCK_FILE).write_text(_j.dumps({"until": _t.time() + BLOCK_HOURS * 3600}))


def login(force: bool = False):
    """Interactive login; asks for email, password and (if enabled) the MFA code."""
    import getpass
    import time as _t
    until = _blocked_until()
    if until and not force:
        hrs = (until - _t.time()) / 3600
        raise SystemExit(
            f"Garmin rate-limited the last login. Not trying again for another {hrs:.0f} h, "
            "because retries during a block extend it (blocks of 48-72 h are common since "
            "March 2026). Use a Garmin bulk export meanwhile (`goodpush import`). "
            "Override with `goodpush auth garmin --force`.")
    Garmin = _import_garmin()
    if _version() >= (0, 3) and not _has_curl_cffi() and not force:
        raise SystemExit(
            "Install curl_cffi first:  pip install --upgrade garminconnect curl_cffi\n"
            "Without it garminconnect can't present a browser-like connection, which is the "
            "usual reason Garmin answers 429. (Skip this check with --force.)")
    email = input("Garmin Connect email: ").strip()
    pw = getpass.getpass("Garmin Connect password (hidden, not stored): ")
    api = Garmin(email=email, password=pw, prompt_mfa=lambda: input("MFA code: ").strip())
    store_dir, _ = _token_paths()
    store_dir.mkdir(parents=True, exist_ok=True)
    try:
        if _version() >= (0, 3):
            # 0.3+: login(tokenstore) logs in with the credentials and saves the session there
            api.login(str(store_dir))
            saved = any(store_dir.iterdir())
        else:
            # 0.2.x (garth based): plain login, then dump the garth session
            api.login()
            saved = _save_tokens(api)
    except Exception as e:
        msg = _friendly(e)
        if "429" in str(e) or "TooManyRequests" in type(e).__name__ or "rate-limit" in msg:
            _note_block()
        raise SystemExit(msg) from None
    (secrets_dir() / BLOCK_FILE).unlink(missing_ok=True)
    if saved:
        print(f"Garmin connected; session saved in {store_dir}")
    else:
        print("Garmin connected, but the session could not be saved - you'll be asked to log "
              "in on each sync.")
    return api


# --- normalisers: live API responses -> bulk-export record shapes ------------------------
def _iso_ms(v):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return pd.Timestamp(v, unit="ms").strftime("%Y-%m-%dT%H:%M:%S.0")
    return str(v)


def norm_sleep(resp: dict, day: str):
    s = (resp or {}).get("dailySleepDTO") or {}
    if s.get("deepSleepSeconds") is None and s.get("sleepTimeSeconds") is None:
        return None
    sc = s.get("sleepScores") or {}
    overall = sc.get("overall") if isinstance(sc.get("overall"), dict) else {}
    return {"calendarDate": s.get("calendarDate") or day,
            "sleepStartTimestampGMT": _iso_ms(s.get("sleepStartTimestampGMT")),
            "sleepEndTimestampGMT": _iso_ms(s.get("sleepEndTimestampGMT")),
            "deepSleepSeconds": s.get("deepSleepSeconds"),
            "lightSleepSeconds": s.get("lightSleepSeconds"),
            "remSleepSeconds": s.get("remSleepSeconds"),
            "awakeSleepSeconds": s.get("awakeSleepSeconds"),
            "averageRespiration": s.get("averageRespirationValue"),
            "avgSleepStress": s.get("avgSleepStress"),
            "sleepScores": {"overallScore": overall.get("value") or sc.get("overallScore")}}


def norm_summary(resp: dict, day: str):
    s = resp or {}
    if not s.get("restingHeartRate") and not s.get("totalSteps"):
        return None
    return {"calendarDate": s.get("calendarDate") or day,
            "restingHeartRate": s.get("restingHeartRate"),
            "currentDayRestingHeartRate": s.get("restingHeartRate"),
            "totalSteps": s.get("totalSteps"),
            "allDayStress": {"aggregatorList": [{"type": "TOTAL",
                                                 "averageStressLevel": s.get("averageStressLevel")}]},
            "bodyBattery": {"bodyBatteryStatList": [
                {"bodyBatteryStatType": "HIGHEST", "statsValue": s.get("bodyBatteryHighestValue")},
                {"bodyBatteryStatType": "LOWEST", "statsValue": s.get("bodyBatteryLowestValue")}]}}


def norm_hrv(resp: dict, day: str):
    h = (resp or {}).get("hrvSummary") or {}
    if not h.get("lastNightAvg"):
        return None
    b = h.get("baseline") or {}
    return {"calendarDate": h.get("calendarDate") or day,
            "metrics": [{"type": "HRV", "value": h.get("lastNightAvg"),
                         "baselineLowerLimit": b.get("balancedLow"),
                         "baselineUpperLimit": b.get("balancedUpper"),
                         "weeklyAvg": h.get("weeklyAvg")}]}


def norm_readiness(resp, day: str):
    items = resp if isinstance(resp, list) else ([resp] if isinstance(resp, dict) and resp else [])
    out = []
    for r in items:
        if r.get("score") is None:
            continue
        out.append({"calendarDate": r.get("calendarDate") or day, "timestamp": r.get("timestamp", ""),
                    "score": r.get("score"), "hrvWeeklyAverage": r.get("hrvWeeklyAverage"),
                    "sleepScore": r.get("sleepScore")})
    return out


def norm_weight(resp: dict):
    out = []
    for s in (resp or {}).get("dailyWeightSummaries") or []:  # get_weigh_ins() layout
        w = (s.get("latestWeight") or {})
        if w.get("weight"):
            d = s.get("summaryDate") or w.get("calendarDate")
            out.append({"metaData": {"calendarDate": f"{d}T00:00:00.0"},
                        "weight": {"weight": w["weight"], "sourceType": "API"}})
    for w in (resp or {}).get("dateWeightList") or []:
        if w.get("weight"):
            d = w.get("calendarDate") or _iso_ms(w.get("date"))[:10]
            out.append({"metaData": {"calendarDate": f"{d}T00:00:00.0"},
                        "weight": {"weight": w["weight"], "sourceType": "API"}})
    return out


# --- sync ---------------------------------------------------------------------------
def last_known_day(garmin_dir: Path) -> pd.Timestamp | None:
    """Newest day that already has sleep or resting-HR data (export or earlier syncs)."""
    from ..io.garmin import load_wellness
    daily, _ = load_wellness(garmin_dir, progress=lambda *a: None, cache_dir=None, scan_zips=False)
    if daily.empty:
        return None
    have = daily[["sleep_h", "rhr"]].dropna(how="all")
    return have.index.max() if len(have) else None


def sync(garmin_dir: str | Path, api=None, since: str | None = None, overlap_days: int = 2,
         pause: float = 0.6, progress=print) -> int:
    garmin_dir = Path(garmin_dir)
    out = garmin_dir / API_SUBDIR
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().normalize()
    if since:
        start = pd.Timestamp(since)
    else:
        last = last_known_day(garmin_dir)
        start = (last - pd.Timedelta(days=overlap_days)) if last is not None else \
            today - pd.Timedelta(days=FIRST_SYNC_DAYS)
    days = pd.date_range(start, today, freq="D")
    api = api or _client()
    progress(f"Garmin: fetching {len(days)} days ({start:%Y-%m-%d} -> {today:%Y-%m-%d})")
    def first(*names):
        """Call the first method that exists (names differ between garminconnect versions)."""
        def f(d):
            for n in names:
                m = getattr(api, n, None)
                if callable(m):
                    return m(d)
            raise AttributeError(f"none of {names} in this garminconnect version")
        return f

    calls = [("sleep", first("get_sleep_data")),
             ("summary", first("get_user_summary", "get_stats")),
             ("hrv", first("get_hrv_data")),
             ("readiness", first("get_training_readiness", "get_morning_training_readiness"))]
    n = 0
    for day in days:
        ds = day.strftime("%Y-%m-%d")
        rec = {}
        for name, fn in calls:
            try:
                rec[name] = fn(ds)
            except Exception as e:  # missing data for a day is normal (no watch worn)
                if "429" in str(e) or "TooManyRequests" in type(e).__name__:
                    progress(f"Garmin is rate-limiting data requests; stopped at {ds}. "
                             "Run the sync again later - it continues from here.")
                    rebuild(garmin_dir)
                    return n
                rec[name] = None
                rec.setdefault("_errors", {})[name] = str(e)[:200]
            time.sleep(pause)
        (raw / f"{ds}.json").write_text(json.dumps(rec))
        n += 1
        if n % 7 == 0:
            progress(f"  {ds}")
    try:
        a, b = start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        wt = api.get_weigh_ins(a, b) if hasattr(api, "get_weigh_ins") \
            else api.get_body_composition(a, b)
        (raw / f"weight_{start:%Y-%m-%d}.json").write_text(json.dumps({"resp": wt}))
    except Exception as e:
        progress(f"  (weight not fetched: {e})")
    rebuild(garmin_dir)
    progress(f"Garmin: {n} days saved")
    return n


# --- browser route (no login, uses your signed-in Garmin Connect tab) --------------------
BROWSER_SOURCE = "goodpush-garmin-browser"


def browser_snippet(garmin_dir: str | Path, since: str | None = None, overlap_days: int = 0,
                    strava_dir: str | Path | None = None) -> str:
    """JavaScript to paste into the console of a signed-in connect.garmin.com tab. It downloads
    the same data as `sync` into one JSON file for `goodpush import`."""
    from importlib.resources import files
    if since:
        start = pd.Timestamp(since)
    else:
        last = last_known_day(Path(garmin_dir))
        start = (last - pd.Timedelta(days=overlap_days)) if last is not None else \
            pd.Timestamp.now().normalize() - pd.Timedelta(days=FIRST_SYNC_DAYS)
    js = files("goodpush.sync").joinpath("garmin_browser.js").read_text(encoding="utf-8")
    act_start = last_activity_day(Path(garmin_dir), strava_dir)
    act_start = (act_start - pd.Timedelta(days=overlap_days)) if act_start is not None else start
    if since:
        act_start = start
    return js.replace("__START__", start.strftime("%Y-%m-%d")) \
             .replace("__ACT_START__", act_start.strftime("%Y-%m-%d"))


def last_activity_day(garmin_dir: Path, strava_dir: Path | None) -> pd.Timestamp | None:
    """Newest activity date we already have (Strava export/API + Garmin downloads)."""
    from ..io.strava import load_activities_csv
    try:
        acts = load_activities_csv(strava_dir or garmin_dir, garmin_dir)
    except Exception:
        return None
    t = acts["start_utc"].dropna()
    return t.max().tz_convert(None).normalize() if len(t) else None


def is_browser_file(path: Path) -> bool:
    path = Path(path)
    if path.suffix.lower() != ".json" or not path.is_file():
        return False
    with open(path, "rb") as f:
        head = f.read(200).decode("utf-8", "ignore")
    return BROWSER_SOURCE in head


# Garmin activity type -> Strava-style type (what the rest of goodpush understands)
GARMIN_TYPES = {
    "cycling": "Ride", "road_biking": "Ride", "track_cycling": "Ride", "cyclocross": "Ride",
    "recumbent_cycling": "Ride", "bmx": "Ride", "virtual_ride": "Virtual Ride",
    "indoor_cycling": "Virtual Ride", "mountain_biking": "Mountain Bike Ride",
    "gravel_cycling": "Gravel Ride", "e_bike_fitness": "E-Bike Ride", "e_bike_mountain": "E-Mountain Bike Ride",
    "running": "Run", "trail_running": "Trail Run", "treadmill_running": "Run", "track_running": "Run",
    "indoor_running": "Run", "walking": "Walk", "hiking": "Hike", "strength_training": "Weight Training",
    "yoga": "Yoga", "pilates": "Pilates", "lap_swimming": "Swim", "open_water_swimming": "Swim",
    "resort_skiing_snowboarding_ws": "Alpine Ski", "backcountry_skiing_snowboarding_ws": "Backcountry Ski",
    "cross_country_skiing_ws": "Nordic Ski", "rowing": "Rowing", "indoor_rowing": "Rowing",
    "elliptical": "Elliptical", "hiit": "HIIT", "fitness_equipment": "Workout",
}


def garmin_activity_row(a: dict, filename: str = "") -> dict:
    """Garmin Connect activity summary -> one row in goodpush's normalised activity columns."""
    tk = ((a.get("activityType") or {}).get("typeKey") or "").lower()
    typ = GARMIN_TYPES.get(tk) or ("Ride" if ("cycl" in tk or "bik" in tk) else
                                   "Run" if "run" in tk else tk.replace("_", " ").title() or "Workout")
    start = a.get("startTimeGMT") or ""
    if start and "T" not in start:
        start = start.replace(" ", "T") + "Z"
    return {
        "id": f"garmin_{a.get('activityId')}", "start_utc": start, "name": a.get("activityName") or "",
        "type": typ, "description": a.get("description") or "", "private_note": "",
        "elapsed_s": a.get("elapsedDuration") or a.get("duration"),
        "moving_s_csv": a.get("movingDuration") or a.get("duration"),
        "distance_m": a.get("distance"), "elev_gain": a.get("elevationGain"),
        "avg_hr_csv": a.get("averageHR"), "max_hr_csv": a.get("maxHR"),
        "avg_watts_csv": a.get("avgPower"), "wap_csv": a.get("normPower"),
        "max_watts_csv": a.get("maxPower"), "relative_effort": None,
        "perceived_exertion": None, "athlete_weight": None, "gear": "",
        "commute": False, "filename": filename, "avg_temp": a.get("averageTemperature"),
        "calories": a.get("calories"),
    }


def _save_activity_file(dest: Path, aid: str, blob: bytes) -> str:
    """Garmin sends the original upload as a zip (normally one .fit). Returns the file name."""
    import io
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    if blob[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            pick = next((n for n in names if n.lower().endswith(".fit")), names[0] if names else None)
            if pick is None:
                return ""
            ext = Path(pick).suffix.lower() or ".fit"
            if ext not in (".fit", ".tcx", ".gpx"):
                return ""
            name = f"{aid}{ext}"
            (dest / name).write_bytes(z.read(pick))
            return name
    if blob[8:12] == b".FIT":
        (dest / f"{aid}.fit").write_bytes(blob)
        return f"{aid}.fit"
    return ""


def import_activities(garmin_dir: str | Path, items: list, progress=print) -> int:
    """Add activities (browser download or Personal API) to data/raw/garmin/activities/.
    Each item: {"summary": <activity list entry>, "file_b64": str | "file_bytes": bytes}."""
    import base64
    from ..io.strava import API_COLUMNS, GARMIN_TABLE
    d = Path(garmin_dir) / "activities"
    d.mkdir(parents=True, exist_ok=True)
    table = d / GARMIN_TABLE
    old = pd.read_csv(table, dtype=str, keep_default_na=False) if table.exists() else \
        pd.DataFrame(columns=API_COLUMNS)
    rows, n_files = [], 0
    for it in items or []:
        a = it.get("summary") or {}
        if not a.get("activityId"):
            continue
        aid = f"garmin_{a['activityId']}"
        fname = ""
        blob = it.get("file_bytes") or (base64.b64decode(it["file_b64"]) if it.get("file_b64") else None)
        if blob:
            try:
                fname = _save_activity_file(d, aid, blob)
                n_files += bool(fname)
            except Exception as e:  # noqa: BLE001 - keep the summary even if the file is odd
                progress(f"  {aid}: file not saved ({e})")
        if not fname:   # keep a file we got earlier
            prev = old.loc[old["id"].eq(aid), "filename"]
            fname = prev.iloc[0] if len(prev) else ""
        rows.append(garmin_activity_row(a, fname))
    if not rows:
        return 0
    new = pd.DataFrame(rows, columns=API_COLUMNS).astype(str).replace({"None": "", "nan": ""})
    allr = pd.concat([old[~old["id"].isin(new["id"])], new], ignore_index=True)
    allr.sort_values("start_utc").to_csv(table, index=False)
    progress(f"Garmin activities: {len(rows)} saved ({n_files} with ride files), {len(allr)} in total")
    return len(rows)


def import_browser(garmin_dir: str | Path, path: str | Path, progress=print) -> int:
    """Store a file made by the browser snippet as raw API days, then rebuild."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("source") != BROWSER_SOURCE:
        raise ValueError(f"{path}: not a goodpush Garmin browser download")
    raw = Path(garmin_dir) / API_SUBDIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 0
    for ds, rec in sorted((data.get("days") or {}).items()):
        if len(ds) != 10 or not any(rec.get(k) for k in ("sleep", "summary", "hrv", "readiness")):
            continue
        (raw / f"{ds}.json").write_text(json.dumps(rec))
        n += 1
    if data.get("weight"):
        (raw / f"weight_{data.get('start', 'browser')}.json").write_text(json.dumps({"resp": data["weight"]}))
    rebuild(garmin_dir)
    progress(f"Garmin (browser download): {n} days of sleep / HRV / resting HR saved")
    if data.get("activities"):
        import_activities(garmin_dir, data["activities"], progress=progress)
    return n


def rebuild(garmin_dir: str | Path) -> None:
    """Regenerate the export-shaped files from all raw API responses."""
    out = Path(garmin_dir) / API_SUBDIR
    raw = out / "raw"
    sleep, uds, hrv, ready = [], [], [], []
    for f in sorted(raw.glob("????-??-??.json")):
        day = f.stem
        r = json.loads(f.read_text())
        x = norm_sleep(r.get("sleep"), day)
        if x:
            sleep.append(x)
        x = norm_summary(r.get("summary"), day)
        if x:
            uds.append(x)
        x = norm_hrv(r.get("hrv"), day)
        if x:
            hrv.append(x)
        ready.extend(norm_readiness(r.get("readiness"), day))
    weights = []
    for f in sorted(raw.glob("weight*.json")):
        weights.extend(norm_weight(json.loads(f.read_text()).get("resp")))
    files = {"api_sleepData.json": sleep, "UDSFile_api.json": uds,
             "api_healthStatusData.json": hrv, "TrainingReadinessDTO_api.json": ready,
             "api_userBioMetrics.json": weights}
    for name, rows in files.items():
        (out / name).write_text(json.dumps(rows))


# --- Garmin account export (`goodpush import`): the activities ------------------------------
EXPORT_INDEX = "export_activity_files.json"   # activities/: the activity FIT files found in the export's zips
EXPORT_MATCH_S = 600                          # a FIT file's creation time vs its activity's start
# summarizedActivities field -> the API's activity-list field and the unit change (ms -> s, cm -> m)
EXPORT_FIELDS = {"duration": ("duration", 1e-3), "elapsedDuration": ("elapsedDuration", 1e-3),
                 "movingDuration": ("movingDuration", 1e-3), "distance": ("distance", 1e-2),
                 "elevationGain": ("elevationGain", 1e-2), "averageHR": ("avgHr", 1), "maxHR": ("maxHr", 1),
                 "avgPower": ("avgPower", 1), "normPower": ("normPower", 1), "maxPower": ("maxPower", 1)}


def export_summary(a: dict) -> dict:
    """one summarizedActivities entry as an activity-list entry of Garmin's API (garmin_activity_row)."""
    out = {"activityId": a["activityId"], "activityName": a.get("name") or "",
           "activityType": {"typeKey": a.get("activityType") or ""},
           "startTimeGMT": pd.Timestamp(a["startTimeGmt"], unit="ms").strftime("%Y-%m-%d %H:%M:%S")}
    out.update({k: a[src] * f for k, (src, f) in EXPORT_FIELDS.items() if a.get(src) is not None})
    return out


def export_summaries(garmin_dir: str | Path) -> list[dict]:
    """the activities of a Garmin account export (DI-Connect-Fitness/*summarizedActivities*.json)."""
    out = []
    for f in sorted(Path(garmin_dir).rglob("*summarizedActivities*.json")):      # files: a handful
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for block in d if isinstance(d, list) else [d]:
            out += [export_summary(a) for a in (block or {}).get("summarizedActivitiesExport") or []
                    if a.get("activityId") and a.get("startTimeGmt")]
    return out


def export_activity_files(garmin_dir: str | Path) -> pd.DataFrame:
    """the activity FIT files (file_id type 4) in the export's UploadedFiles*.zip: zip, member, created
    (unix s). Each zip is read once (only each file's first message) and remembered in EXPORT_INDEX."""
    import zipfile
    from ..io.fit import FIT_EPOCH_OFFSET, iter_messages
    garmin_dir = Path(garmin_dir)
    idx = garmin_dir / "activities" / EXPORT_INDEX
    try:
        cache = json.loads(idx.read_text())
    except (OSError, ValueError):
        cache = {}
    rows, changed = [], False
    for zp in sorted(garmin_dir.rglob("UploadedFiles*.zip")):                   # zips: a handful
        key = stamp(zp.stat())
        if in_seconds((cache.get(zp.name) or {}).get("key") or "") != key:
            files = []
            with zipfile.ZipFile(zp) as z:
                for n in z.namelist():                                          # files: each read once
                    if not n.lower().endswith(".fit"):
                        continue
                    try:
                        fid = next((m for _, m in iter_messages(z.read(n), (0,))), {})
                    except Exception:  # noqa: BLE001 - a broken file is skipped
                        continue
                    if fid.get(0) == 4 and fid.get(4):
                        files.append([n, int(fid[4]) + FIT_EPOCH_OFFSET])
            cache[zp.name] = {"key": key, "files": files}
            changed = True
        rows += [(str(zp), n, t) for n, t in cache[zp.name]["files"]]
    if changed:
        idx.parent.mkdir(parents=True, exist_ok=True)
        idx.write_text(json.dumps(cache))
    return pd.DataFrame(rows, columns=["zip", "member", "created"])


def import_export_activities(garmin_dir: str | Path, progress=print) -> int:
    """the activities of a Garmin account export, stored like the API's (import_activities): each summary
    with its FIT file (the one created within EXPORT_MATCH_S of its start), or without one. Activities
    already stored with a file are skipped, so it is cheap to run on every update. Returns how many were
    added."""
    import zipfile
    import numpy as np
    from ..io.strava import GARMIN_TABLE
    summ = export_summaries(garmin_dir)
    if not summ:
        return 0
    table = Path(garmin_dir) / "activities" / GARMIN_TABLE
    t = pd.read_csv(table, dtype=str, keep_default_na=False) if table.exists() else pd.DataFrame(columns=["id", "filename"])
    with_file, known = set(t.loc[t["filename"].ne(""), "id"]), set(t["id"])
    todo = [a for a in summ if f"garmin_{a['activityId']}" not in with_file]
    if not todo:
        return 0
    files = export_activity_files(garmin_dir).sort_values("created").reset_index(drop=True)
    start = pd.to_datetime([a["startTimeGMT"] for a in todo]).as_unit("s").asi8
    match = np.full(len(todo), -1)
    if len(files):
        c = files["created"].to_numpy()
        i = np.clip(np.searchsorted(c, start), 1, max(len(c) - 1, 1))
        j = np.where(np.abs(c[i - 1] - start) <= np.abs(c[np.minimum(i, len(c) - 1)] - start), i - 1, np.minimum(i, len(c) - 1))
        d = np.abs(c[j] - start)
        ok = d <= EXPORT_MATCH_S
        best = pd.Series(d[ok], index=np.flatnonzero(ok)).groupby(j[ok]).idxmin()   # each file goes to one activity
        match[best.to_numpy()] = best.index.to_numpy()
    n = 0
    for zp, g in files.iloc[match[match >= 0]].assign(a=np.flatnonzero(match >= 0)).groupby("zip"):   # zips: a handful
        with zipfile.ZipFile(zp) as z:
            n += import_activities(garmin_dir, [{"summary": todo[a], "file_bytes": z.read(m)}
                                                for a, m in zip(g["a"], g["member"])], progress=progress)
    bare = [{"summary": a} for a, m in zip(todo, match) if m < 0 and f"garmin_{a['activityId']}" not in known]
    n += import_activities(garmin_dir, bare, progress=progress) if bare else 0
    progress(f"Garmin export: {n} activities added ({int((match >= 0).sum())} with their files)")
    return n
