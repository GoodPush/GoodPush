"""Garmin Connect Personal API (official, read-only, your own data).

Create a key once in Garmin Connect (Settings > Personal API Key, valid 30-365 days),
then `goodpush garmin-key` stores it in ~/.goodpush/garmin_personal_api.json (never in
the project folder). Each sync swaps the key for an access token (valid 24 h, cached)
and pulls only the days and activities you don't have yet:

  sleep, resting HR, steps, stress, Body Battery  (one call each per day)
  HRV, the night's summary and every 5-min reading (one call per day)
  weight                                          (one call for the whole range)
  activities + their original .fit files          (list, then one download per new ride)

Responses are stored exactly like the browser download (data/raw/garmin/api/raw/<day>.json
and activities/activities_garmin.csv), so everything downstream is unchanged.
Garmin's own Training Readiness score is not part of the Personal API; goodpush's
readiness is computed from HRV / resting HR / sleep / load and doesn't need it.

Docs: https://developer.garmin.com/gc-developer-program/personal/personal-api/
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import FIRST_SYNC_DAYS, read_secret, secrets_dir, write_secret

BASE = "https://personal.connectapi.garmin.com"
TOKEN_URL = BASE + "/personalapitoken-service/personal-api/personal-api-key/token"
KEY_FILE = "garmin_personal_api.json"
TOKEN_FILE = "garmin_personal_token.json"
SYNC_FILE = "garmin_personal_sync.json"      # the last sync: when, what, days to fetch again
UA = "goodpush (personal use)"
PAUSE = 0.15          # seconds between calls; we only fetch a handful of days at a time
NETWORK_ERROR = 599   # status for a dropped / failed connection: retried like a server error


class GarminApiError(RuntimeError):
    pass


class RateLimited(GarminApiError):
    pass


class Unavailable(GarminApiError):
    """a Garmin service didn't answer (dropped connections, after retrying)."""


# --- key + token ---------------------------------------------------------------------------
def save_key(key: str) -> Path:
    key = (key or "").strip()
    if key.lower().startswith("apikey "):
        key = key[7:].strip()
    if len(key) < 16 or any(c.isspace() for c in key):
        raise ValueError("That doesn't look like a Garmin Personal API key.")
    p = write_secret(KEY_FILE, {"key": key, "saved": pd.Timestamp.now().isoformat(timespec="seconds")})
    (secrets_dir() / TOKEN_FILE).unlink(missing_ok=True)
    return p


def has_key() -> bool:
    return bool(read_secret(KEY_FILE).get("key"))


def forget_key() -> None:
    for n in (KEY_FILE, TOKEN_FILE):
        (secrets_dir() / n).unlink(missing_ok=True)


def key_info() -> dict:
    k = read_secret(KEY_FILE)
    return {"configured": bool(k.get("key")), "saved": k.get("saved"),
            "last_sync": read_secret(SYNC_FILE)}


def _request(url: str, headers: dict, method="GET", timeout=60):
    # no body on the token POST (like Garmin's curl example), so no Content-Type is sent
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b"", dict(e.headers or {})
    except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException) as e:
        # e.g. "Remote end closed connection without response": usually gone on a second try
        return NETWORK_ERROR, f"network error: {getattr(e, 'reason', e)}".encode(), {}


def _find(d, names):
    """First value whose key matches one of names (searching nested dicts)."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() in names and v not in (None, ""):
                return v
        for v in d.values():
            r = _find(v, names)
            if r not in (None, ""):
                return r
    return None


def _parse_token(body: bytes):
    txt = body.decode("utf-8", "ignore").strip()
    try:
        d = json.loads(txt)
    except ValueError:
        d = txt.strip('"')
    if isinstance(d, str):
        return d, None
    tok = _find(d, {"access_token", "accesstoken", "token", "jwt"})
    exp = _find(d, {"expires_in", "expiresin"})
    at = _find(d, {"expires_at", "expiresat", "expiration", "expirationtime", "expiry"})
    t_exp = None
    try:
        if exp is not None:
            t_exp = time.time() + float(exp)
        elif at is not None:
            v = float(at) if str(at).replace(".", "").isdigit() else pd.Timestamp(at).timestamp()
            t_exp = v / 1000 if v > 1e11 else v
    except (ValueError, TypeError):
        t_exp = None
    return (str(tok) if tok else None), t_exp


def access_token(force: bool = False) -> str:
    key = read_secret(KEY_FILE).get("key")
    if not key:
        raise GarminApiError("No Garmin Personal API key yet - run `goodpush garmin-key` "
                             "(or paste it in the app under Settings).")
    cached = read_secret(TOKEN_FILE)
    if not force and cached.get("token") and cached.get("exp", 0) > time.time() + 300:
        return cached["token"]
    # Accept */*: Garmin answers 406 to "Accept: application/json" here (the token isn't JSON-typed)
    code, body, _ = _request(TOKEN_URL, {"Authorization": f"ApiKey {key}", "Accept": "*/*"},
                             method="POST")
    if code in (401, 403):
        raise GarminApiError("Garmin refused the Personal API key (expired or deleted?). Create a new "
                             "one in Garmin Connect > Settings > Personal API Key, then `goodpush garmin-key`.")
    if code == 429:
        raise RateLimited("Garmin is rate-limiting token requests; try again in a while.")
    if code == NETWORK_ERROR:
        raise GarminApiError(f"Garmin token request: {body.decode()}")
    if code >= 400:
        raise GarminApiError(f"token request failed (HTTP {code}): {body[:200]!r}")
    tok, exp = _parse_token(body)
    if not tok:
        raise GarminApiError("Garmin's token response had no token in it - the API may have changed.")
    write_secret(TOKEN_FILE, {"token": tok, "exp": exp or time.time() + 23 * 3600})
    return tok


class Client:
    def __init__(self, pause: float = PAUSE):
        self.pause = pause
        self.token = access_token()
        self.calls = 0
        self.down = set()        # services that didn't answer this run: not asked again until the next one
        self.skipped = []        # what an `optional` request couldn't get

    @staticmethod
    def service(path: str) -> str:
        """the Garmin service a path belongs to, e.g. "usersummary-service"."""
        return path.strip("/").split("/")[0]

    def _get(self, path: str, params: dict | None = None, accept="application/json"):
        if self.service(path) in self.down:
            raise Unavailable(f"{self.service(path)} didn't answer earlier in this update")
        url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        refreshed = False
        for attempt in range(4):
            if self.calls:
                time.sleep(self.pause)
            self.calls += 1
            code, body, hdr = _request(url, {"Authorization": f"Bearer {self.token}", "Accept": accept})
            if code == 406 and accept != "*/*":
                accept = "*/*"
                continue
            if code == 401 and not refreshed:
                self.token, refreshed = access_token(force=True), True
                continue
            if code == 429:
                raise RateLimited("Garmin is rate-limiting requests; the next update continues from here.")
            if code >= 500 and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            if code in (204, 404) or not body:
                return None
            if code == NETWORK_ERROR:
                self.down.add(self.service(path))
                raise Unavailable(f"GET {path}: {body.decode()} (gave up after retrying)")
            if code >= 400:
                raise GarminApiError(f"GET {path} failed (HTTP {code}): {body[:200]!r}")
            return body
        return None

    def optional(self, get, path, *args, **kw):
        """get(path, ...), or None when Garmin doesn't answer (noted in `skipped`, fetched next update)."""
        try:
            return get(path, *args, **kw)
        except Unavailable:
            self.skipped.append(path)
            return None

    def json(self, path, **params):
        b = self._get(path, params or None)
        if b is None:
            return None
        try:
            return json.loads(b)
        except ValueError:
            return None

    def file(self, path):
        return self._get(path, None, accept="application/octet-stream, application/zip, */*")


# --- sync ---------------------------------------------------------------------------------
def _start_day(garmin_dir: Path, since, overlap_days) -> pd.Timestamp:
    from .garmin import last_known_day
    today = pd.Timestamp.now().normalize()
    if since:
        return pd.Timestamp(since)
    last = last_known_day(garmin_dir)
    if last is None:
        return today - pd.Timedelta(days=FIRST_SYNC_DAYS)
    retry = (read_secret(SYNC_FILE) or {}).get("retry") or []     # days a dropped request left incomplete
    return min([last - pd.Timedelta(days=overlap_days), today] + [pd.Timestamp(d) for d in retry])


def fetch_day(c: Client, ds: str) -> dict:
    """the day's sleep, summary and HRV; a part Garmin doesn't answer is None (and in c.skipped)."""
    sleep = c.optional(c.json, "/sleep-service/sleep/dailySleep", date=ds)
    if isinstance(sleep, dict) and "dailySleepDTO" not in sleep:
        sleep = {"dailySleepDTO": sleep}          # personal API returns the DTO itself
    summary = c.optional(c.json, "/usersummary-service/usersummary/daily", calendarDate=ds)
    hrv = c.optional(c.json, f"/hrv-service/hrv/{ds}")
    return {"sleep": sleep, "summary": summary,
            "hrv": {"hrvSummary": hrv.get("hrvSummary"), "hrvReadings": hrv.get("hrvReadings")}   # every 5-min reading
            if isinstance(hrv, dict) else None,
            "_source": "personal-api"}


def _merge_day(path: Path, rec: dict) -> dict:
    """Keep anything an earlier download had that this one lacks (e.g. readiness)."""
    if path.exists():
        try:
            old = json.loads(path.read_text())
            for k, v in old.items():
                if rec.get(k) in (None, {}, []) and v not in (None, {}, []):
                    rec[k] = v
        except ValueError:
            pass
    return rec


def sync(garmin_dir: str | Path, strava_dir: str | Path | None = None, since: str | None = None,
         overlap_days: int = 1, progress=print) -> dict:
    """Pull new days + activities. Returns {"days": n, "activities": n, "files": n}."""
    from .garmin import API_SUBDIR, import_activities, last_activity_day, rebuild
    from ..io.strava import GARMIN_TABLE
    garmin_dir = Path(garmin_dir)
    raw = garmin_dir / API_SUBDIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().normalize()
    start = _start_day(garmin_dir, since, overlap_days)
    days = pd.date_range(start, today, freq="D")
    c = Client()
    out = {"days": 0, "activities": 0, "files": 0, "start": str(start.date())}
    progress(f"Garmin: {len(days)} day(s) of sleep / HRV / resting HR ({start:%Y-%m-%d} -> {today:%Y-%m-%d})")
    stopped, retry = None, []
    try:
        for d in days:
            ds = d.strftime("%Y-%m-%d")
            n = len(c.skipped)
            rec = fetch_day(c, ds)
            if len(c.skipped) > n:
                retry.append(ds)
            if any(rec.get(k) for k in ("sleep", "summary", "hrv")):
                p = raw / f"{ds}.json"
                p.write_text(json.dumps(_merge_day(p, rec)))
                out["days"] += 1
        wt = c.optional(c.json, f"/weight-service/weight/range/{start:%Y-%m-%d}/{today:%Y-%m-%d}", includeAll="true")
        if wt:
            (raw / f"weight_{start:%Y-%m-%d}.json").write_text(json.dumps({"resp": wt}))
    except RateLimited as e:
        stopped = str(e)
    rebuild(garmin_dir)

    # activities: list since the newest one we have, download files only for new ones
    if not stopped:
        a0 = last_activity_day(garmin_dir, Path(strava_dir) if strava_dir else None)
        a0 = min(a0 - pd.Timedelta(days=overlap_days), start) if a0 is not None else start   # start: incl. retried days
        if since:
            a0 = min(a0, pd.Timestamp(since))
        table = garmin_dir / "activities" / GARMIN_TABLE
        have = set()
        if table.exists():
            t = pd.read_csv(table, dtype=str, keep_default_na=False)
            have = set(t.loc[t["filename"].ne(""), "id"])
        items = []
        try:
            lst, s0 = [], 0
            while True:
                page = c.optional(c.json, "/activitylist-service/activities/search/activities",
                              startDate=f"{a0:%Y-%m-%d}", endDate=f"{today:%Y-%m-%d}", start=s0, limit=100)
                if not isinstance(page, list) or not page:
                    break
                lst += page
                if len(page) < 100:
                    break
                s0 += 100
            for a in lst:
                aid = a.get("activityId")
                if not aid:
                    continue
                it = {"summary": a}
                if f"garmin_{aid}" not in have:
                    n = len(c.skipped)
                    it["file_bytes"] = c.optional(c.file, f"/download-service/files/activity/{aid}")
                    if len(c.skipped) > n:
                        retry.append(str(a.get("startTimeLocal") or a.get("startTimeGMT") or today)[:10])
                    progress(f"  {str(a.get('startTimeLocal', ''))[:16]}  {a.get('activityName', '')}"
                             + ("  (file: Garmin didn't answer, next update)" if len(c.skipped) > n
                                else "" if it["file_bytes"] else "  (no file)"))
                items.append(it)
        except RateLimited as e:
            stopped = str(e)
        if items:
            out["activities"] = import_activities(garmin_dir, items, progress=progress)
            out["files"] = sum(bool(i.get("file_bytes")) for i in items)
        else:
            progress("Garmin activities: nothing new")
    if stopped:
        progress(stopped)
    out["stopped"] = stopped
    out["calls"] = c.calls
    out["skipped"], out["retry"] = len(c.skipped), sorted(set(retry))
    if c.skipped:
        progress(f"Garmin didn't answer {len(c.skipped)} request(s) ({', '.join(sorted(c.down))}): "
                 "skipped, fetched again next update")
    write_secret(SYNC_FILE, {"at": pd.Timestamp.now().isoformat(timespec="seconds"), **out})
    progress(f"Garmin: {out['days']} day(s) and {out['activities']} activit(ies) updated "
             f"({c.calls} requests)")
    return out
