"""Incremental Strava download via the official Strava API (stdlib only).

One-time setup (`goodpush auth strava`):
  1. Create an API application at https://www.strava.com/settings/api
     - "Authorization Callback Domain": localhost
  2. Run `goodpush auth strava`, paste the Client ID and Client Secret, approve in
     the browser. Tokens are stored in ~/.goodpush/strava.json and refreshed
     automatically.

`goodpush sync strava` then fetches every activity newer than the newest one you
already have (export or earlier syncs), with description/private note and the
second-by-second streams, and appends them to data/raw/strava/api/:
  activities_api.csv          - one row per activity (same columns as the export)
  streams/<activity id>.json  - streams in Strava's JSON format (read by ingest)

Rate limits (default 100 read requests / 15 min, 1000 / day) are respected:
when hit, the sync waits for the next 15-minute window or stops cleanly and
resumes where it left off next time.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

import pandas as pd

from . import FIRST_SYNC_DAYS, read_secret, write_secret
from ..io.strava import API_COLUMNS, API_SUBDIR, API_TABLE, load_activities_csv

API = "https://www.strava.com/api/v3"
TOKEN_URL = "https://www.strava.com/oauth/token"
AUTH_URL = "https://www.strava.com/oauth/authorize"
SECRET_FILE = "strava.json"  # pragma: allowlist secret (a file name)
REDIRECT_PORT = 8765
STREAM_KEYS = ["time", "watts", "heartrate", "cadence", "velocity_smooth", "altitude", "distance"]

# sport_type -> the activity-type names used in the bulk export
TYPE_MAP = {"VirtualRide": "Virtual Ride", "EBikeRide": "E-Bike Ride", "GravelRide": "Gravel Ride",
            "MountainBikeRide": "Mountain Bike Ride", "EMountainBikeRide": "E-Mountain Bike Ride",
            "WeightTraining": "Weight Training", "TrailRun": "Trail Run", "VirtualRun": "Virtual Run",
            "AlpineSki": "Alpine Ski", "NordicSki": "Nordic Ski", "BackcountrySki": "Backcountry Ski",
            "RockClimbing": "Rock Climb", "StandUpPaddling": "Stand Up Paddling"}


class RateLimited(Exception):
    pass


# --- HTTP --------------------------------------------------------------------------
def _request(url, data=None, token=None, method=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "null"), dict(r.headers)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited(dict(e.headers)) from None
        msg = e.read().decode(errors="replace")[:300]
        if e.code == 403 and "Inactive" in msg:
            raise RuntimeError(
                "Strava says your API application is 'Inactive'. Since July 2026 Strava only "
                "serves API data to apps whose owner has an active Strava subscription. Subscribe "
                "(or check your subscription is active) and press 'reactivate' on "
                "strava.com/settings/api, then run the sync again. Meanwhile a Strava bulk "
                "export unzipped into data/raw/strava still works.") from None
        raise RuntimeError(f"Strava API {e.code} for {url.split('?')[0]}: {msg}") from None


# --- auth --------------------------------------------------------------------------
def authorize(client_id: str | None = None, client_secret: str | None = None, open_browser=True):
    """Interactive one-time OAuth: opens the browser, catches the redirect on localhost."""
    import getpass
    import re
    sec = read_secret(SECRET_FILE)
    client_id = client_id or sec.get("client_id") or input("Strava Client ID: ").strip()
    if not client_secret and not sec.get("client_secret"):
        print("Paste the Client Secret from strava.com/settings/api (click 'show'). It is the\n"
              "40-character code, not the Access Token or Refresh Token.")
    client_secret = client_secret or sec.get("client_secret") or \
        getpass.getpass("Strava Client Secret (hidden; right-click to paste in cmd): ").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", client_secret):
        raise SystemExit(f"That doesn't look like a Strava Client Secret (got {len(client_secret)} "
                         "characters; expected 40 hex characters). Ctrl+V into a hidden prompt "
                         "sometimes inserts junk in Windows cmd - try right-click paste.")
    if not client_id.isdigit():
        raise SystemExit(f"Client ID should be a number (got {client_id!r}).")
    redirect = f"http://localhost:{REDIRECT_PORT}/exchange"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id, "response_type": "code", "redirect_uri": redirect,
        "approval_prompt": "auto", "scope": "read,activity:read_all,profile:read_all"})
    got = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in q.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            ok = "code" in got
            self.wfile.write((b"<h3>goodpush: Strava connected - you can close this tab.</h3>" if ok
                              else b"<h3>goodpush: authorization failed.</h3>"))

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("localhost", REDIRECT_PORT), H)
    th = threading.Thread(target=srv.handle_request, daemon=True)
    th.start()
    print("Opening Strava in your browser to approve access...\n" + url)
    if open_browser:
        webbrowser.open(url)
    th.join(timeout=300)
    srv.server_close()
    if "code" not in got:
        raise RuntimeError(f"no authorization code received ({got.get('error', 'timeout')})")
    try:
        tok, _ = _request(TOKEN_URL, {"client_id": client_id, "client_secret": client_secret,
                                      "code": got["code"], "grant_type": "authorization_code"})
    except RuntimeError as e:
        if "401" in str(e):
            raise SystemExit(
                "Strava rejected the Client ID / Client Secret pair (401 'Application invalid').\n"
                f"Check on strava.com/settings/api that the Client ID is {client_id} and copy the\n"
                f"Client Secret again (yours started {client_secret[:3]}... and ended "
                f"...{client_secret[-3:]}). If you clicked 'refresh' on the secret there, use the "
                "new one. Then run `goodpush auth strava` again.") from None
        raise
    sec = {"client_id": client_id, "client_secret": client_secret,
           "access_token": tok["access_token"], "refresh_token": tok["refresh_token"],
           "expires_at": tok["expires_at"], "athlete_id": (tok.get("athlete") or {}).get("id"),
           "scope": got.get("scope", "")}
    p = write_secret(SECRET_FILE, sec)
    if "activity:read_all" not in sec["scope"]:
        print("warning: 'View data about your private activities' was not granted - private "
              "rides will be missing. Re-run `goodpush auth strava` and tick it.")
    print(f"Strava authorized; tokens saved to {p}")


DEAUTH_URL = "https://www.strava.com/oauth/deauthorize"


def deauthorize(progress=print) -> None:
    """Revoke goodpush's access on Strava and delete the locally stored tokens."""
    from . import secrets_dir
    p = secrets_dir() / SECRET_FILE
    sec = read_secret(SECRET_FILE)
    if not sec:
        progress("Strava: no stored connection - nothing to remove.")
        return
    revoked = False
    try:
        _request(DEAUTH_URL, {"access_token": access_token()}, method="POST")
        revoked = True
    except Exception as e:  # e.g. app already 'Inactive' or token expired
        progress(f"Strava did not confirm the revoke ({str(e)[:120]}).")
    p.unlink(missing_ok=True)
    progress("Strava: local tokens and client secret deleted." + (
        " Access revoked on Strava." if revoked else
        " To be sure, remove goodpush under strava.com/settings/apps."))
    progress("You can now delete the API application at strava.com/settings/api "
             "(scroll to the bottom, 'Delete application').")


def access_token() -> str:
    sec = read_secret(SECRET_FILE)
    if not sec.get("refresh_token"):
        raise RuntimeError("Strava not connected - run `goodpush auth strava` first")
    if sec.get("expires_at", 0) < time.time() + 120:
        tok, _ = _request(TOKEN_URL, {"client_id": sec["client_id"],
                                      "client_secret": sec["client_secret"],
                                      "refresh_token": sec["refresh_token"],
                                      "grant_type": "refresh_token"})
        sec.update(access_token=tok["access_token"], refresh_token=tok["refresh_token"],
                   expires_at=tok["expires_at"])
        write_secret(SECRET_FILE, sec)
    return sec["access_token"]


# --- conversion ------------------------------------------------------------------
def activity_row(a: dict, filename: str = "") -> dict:
    """Strava API (Detailed)Activity -> one row in the export's normalised columns."""
    st = a.get("sport_type") or a.get("type") or ""
    return {
        "id": str(a["id"]), "start_utc": a.get("start_date", ""), "name": a.get("name", ""),
        "type": TYPE_MAP.get(st, st), "description": a.get("description") or "",
        "private_note": a.get("private_note") or "", "elapsed_s": a.get("elapsed_time"),
        "moving_s_csv": a.get("moving_time"), "distance_m": a.get("distance"),
        "elev_gain": a.get("total_elevation_gain"), "avg_hr_csv": a.get("average_heartrate"),
        "max_hr_csv": a.get("max_heartrate"), "avg_watts_csv": a.get("average_watts"),
        "wap_csv": a.get("weighted_average_watts"), "max_watts_csv": a.get("max_watts"),
        "relative_effort": a.get("suffer_score"), "perceived_exertion": a.get("perceived_exertion"),
        "athlete_weight": None, "gear": (a.get("gear") or {}).get("name") or a.get("gear_id") or "",
        "commute": bool(a.get("commute")), "filename": filename,
        "avg_temp": a.get("average_temp"), "calories": a.get("calories"),
    }


def streams_to_json(a: dict, streams: dict) -> dict:
    """Streams (key_by_type) -> Strava export JSON layout understood by io.streams."""
    start = pd.Timestamp(a["start_date"]).timestamp()
    t = (streams.get("time") or {}).get("data") or []
    cols = {"watts": "watts", "heartrate": "heartrate", "cadence": "cadence",
            "velocity_smooth": "speed", "altitude": "altitude", "distance": "distance"}
    have = [k for k in cols if streams.get(k, {}).get("data")]
    fields = ["time"] + [cols[k] for k in have]
    values = []
    for i, dt in enumerate(t):
        row = [start + dt]
        for k in have:
            d = streams[k]["data"]
            row.append(d[i] if i < len(d) else None)
        values.append(row)
    return {"metadata": {"source": "strava api", "activity_id": a["id"],
                         "start_date": a["start_date"], "activity_type": a.get("sport_type")},
            "data": [{"fields": fields, "values": values}]}


# --- sync --------------------------------------------------------------------------
class Client:
    """Thin API client that tracks the rate-limit headers."""

    def __init__(self, token: str, wait=True, progress=print):
        self.token, self.wait, self.progress = token, wait, progress

    def get(self, path, **params):
        url = f"{API}{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        for _ in range(3):
            try:
                data, h = _request(url, token=self.token)
                self._check(h)
                return data
            except RateLimited:
                if not self.wait:
                    raise
                secs = 900 - (time.time() % 900) + 5
                self.progress(f"  rate limit reached - waiting {secs / 60:.0f} min for the next window")
                time.sleep(secs)
        raise RateLimited({})

    def _check(self, h):
        usage = h.get("X-ReadRateLimit-Usage") or h.get("X-RateLimit-Usage")
        limit = h.get("X-ReadRateLimit-Limit") or h.get("X-RateLimit-Limit")
        if usage and limit:
            u15, uday = (int(x) for x in usage.split(",")[:2])
            l15, lday = (int(x) for x in limit.split(",")[:2])
            if uday >= lday - 2:
                raise RateLimited({"daily": True})
            if u15 >= l15 - 2 and self.wait:
                secs = 900 - (time.time() % 900) + 5
                self.progress(f"  near the 15-min rate limit - pausing {secs / 60:.0f} min")
                time.sleep(secs)


def sync(strava_dir: str | Path, client: Client | None = None, since: str | None = None,
         overlap_days: int = 3, progress=print) -> int:
    """Append activities newer than what we have. Returns the number of new activities."""
    strava_dir = Path(strava_dir)
    out = strava_dir / API_SUBDIR
    (out / "streams").mkdir(parents=True, exist_ok=True)
    table = out / API_TABLE
    have = pd.read_csv(table, dtype=str, keep_default_na=False) if table.exists() \
        else pd.DataFrame(columns=API_COLUMNS)
    try:
        known = load_activities_csv(strava_dir)
        known_ids = set(known["id"])
        last = known["start_utc"].max()
    except FileNotFoundError:
        known_ids, last = set(), None
    if since:
        after = pd.Timestamp(since, tz="UTC")
    elif last is not None and not pd.isna(last):
        after = last - pd.Timedelta(days=overlap_days)
    else:
        after = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=FIRST_SYNC_DAYS)
    client = client or Client(access_token(), progress=progress)
    progress(f"Strava: looking for activities since {after:%Y-%m-%d}")
    summaries, page = [], 1
    while True:
        batch = client.get("/athlete/activities", after=int(after.timestamp()), page=page,
                           per_page=100)
        if not batch:
            break
        summaries.extend(batch)
        page += 1
    new = [a for a in summaries if str(a["id"]) not in known_ids]
    progress(f"  {len(summaries)} activities in range, {len(new)} new")
    rows = have.to_dict("records")
    added = 0
    try:
        for a in sorted(new, key=lambda x: x["start_date"]):
            detail = client.get(f"/activities/{a['id']}", include_all_efforts="false")
            fname = ""
            if not detail.get("manual"):
                st = client.get(f"/activities/{a['id']}/streams", keys=",".join(STREAM_KEYS),
                                key_by_type="true")
                if isinstance(st, list):  # older API shape
                    st = {s["type"]: s for s in st}
                if st and (st.get("time") or {}).get("data"):
                    fname = f"{API_SUBDIR}/streams/{a['id']}.json"
                    (strava_dir / fname).write_text(json.dumps(streams_to_json(detail, st)))
            rows = [r for r in rows if str(r["id"]) != str(a["id"])]
            rows.append(activity_row(detail, fname))
            added += 1
            progress(f"  + {detail.get('start_date', '')[:10]} {detail.get('name', '')[:50]}")
            if added % 10 == 0:
                _save(rows, table)
    except RateLimited:
        progress("  daily rate limit reached - saved what we have; run sync again tomorrow")
    _save(rows, table)
    return added


def _save(rows, table: Path):
    df = pd.DataFrame(rows)
    for c in API_COLUMNS:
        if c not in df:
            df[c] = ""
    df[API_COLUMNS].to_csv(table, index=False)
