"""Google Health API (Fitbit's successor; the Fitbit Web API switches off on 2 Nov 2026).

Each person uses their own free Google Cloud project, so nothing is shared and nothing costs:

1. In the Google Health app, sign in with a personal Google account (and move the Fitbit account
   to it when asked).
2. console.cloud.google.com: new project -> enable "Google Health API" -> OAuth consent screen
   (External, add yourself as a test user, then "Publish app": unverified apps may have up to
   100 users, and in "Testing" Google expires the sign-in every 7 days) -> Credentials ->
   OAuth client ID, type "Desktop app".
3. goodpush app > Settings > Fitbit / Google Health: paste the client ID and secret, Connect,
   approve. The client and the refresh token stay in ~/.goodpush (never in the project).

Each sync pulls the daily summaries and sleep sessions since the last sync (a year the first
time) and keeps every data point by its resource name in data/raw/google_health/<type>.json;
io/google_health.py turns them into the daily wellness columns.

Docs: https://developers.google.com/health (endpoints, data types, OAuth setup).
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from . import FIRST_SYNC_DAYS, read_secret, secrets_dir, write_secret

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://health.googleapis.com/v4/users/me"
SCOPES = ["https://www.googleapis.com/auth/googlehealth.sleep.readonly",
          "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly"]
CLIENT_FILE, TOKEN_FILE, SYNC_FILE = "google_health_client.json", "google_health_token.json", "google_health_sync.json"
# REST data type -> its DataPoint field (daily summaries; filtered on <field>.date)
DAILY = {"daily-resting-heart-rate": "dailyRestingHeartRate",
         "daily-heart-rate-variability": "dailyHeartRateVariability",
         "daily-respiratory-rate": "dailyRespiratoryRate",
         "daily-oxygen-saturation": "dailyOxygenSaturation",
         "daily-sleep-temperature-derivations": "dailySleepTemperatureDerivations"}
OVERLAP_DAYS = 3
UA = "goodpush (personal use)"


class GoogleHealthError(RuntimeError):
    pass


class GoogleHealth:
    """Your Google Health connection: the OAuth client you created, sign-in, and API requests."""

    def __init__(self):
        self.pending = {}          # state -> (PKCE verifier, redirect URI) while you approve in Google

    # --- connection ---------------------------------------------------------------------
    @staticmethod
    def save_client(client_id: str, client_secret: str) -> None:
        cid, sec = (client_id or "").strip(), (client_secret or "").strip()
        if not cid.endswith(".apps.googleusercontent.com") or not sec:
            raise ValueError("Paste the OAuth client ID (…apps.googleusercontent.com) and its secret.")
        write_secret(CLIENT_FILE, {"client_id": cid, "client_secret": sec,
                                   "saved": pd.Timestamp.now().isoformat(timespec="seconds")})
        (secrets_dir() / TOKEN_FILE).unlink(missing_ok=True)

    @staticmethod
    def connected() -> bool:
        return bool(read_secret(TOKEN_FILE).get("refresh_token"))

    @staticmethod
    def info() -> dict:
        c = read_secret(CLIENT_FILE)
        return {"client": bool(c.get("client_id")), "connected": GoogleHealth.connected(),
                "saved": c.get("saved"), "last_sync": read_secret(SYNC_FILE)}

    @staticmethod
    def forget() -> None:
        for n in (CLIENT_FILE, TOKEN_FILE, SYNC_FILE):
            (secrets_dir() / n).unlink(missing_ok=True)

    def auth_url(self, redirect_uri: str) -> str:
        """Google's consent page; it sends you back to `redirect_uri` (this app, on this PC)."""
        c = read_secret(CLIENT_FILE)
        if not c.get("client_id"):
            raise GoogleHealthError("Save your OAuth client ID and secret first.")
        state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        self.pending = {state: (verifier, redirect_uri)}
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        q = {"client_id": c["client_id"], "redirect_uri": redirect_uri, "response_type": "code",
             "scope": " ".join(SCOPES), "access_type": "offline", "prompt": "consent", "state": state,
             "code_challenge": challenge, "code_challenge_method": "S256"}
        return AUTH_URL + "?" + urllib.parse.urlencode(q)

    def finish(self, code: str, state: str) -> None:
        """Swap the code Google sent back for a refresh token (kept in ~/.goodpush)."""
        if state not in self.pending:
            raise GoogleHealthError("That sign-in link is stale - press Connect again.")
        verifier, redirect_uri = self.pending.pop(state)
        c = read_secret(CLIENT_FILE)
        tok = self.token_request({"code": code, "client_id": c["client_id"], "client_secret": c["client_secret"],
                                  "redirect_uri": redirect_uri, "grant_type": "authorization_code",
                                  "code_verifier": verifier})
        if not tok.get("refresh_token"):
            raise GoogleHealthError("Google gave no refresh token - remove goodpush's access in your Google "
                                    "account (Security > Third-party access) and connect again.")
        write_secret(TOKEN_FILE, {"refresh_token": tok["refresh_token"], "access_token": tok.get("access_token"),
                                  "exp": time.time() + float(tok.get("expires_in", 3600))})

    @staticmethod
    def token_request(form: dict) -> dict:
        req = urllib.request.Request(TOKEN_URL, data=urllib.parse.urlencode(form).encode(), method="POST",
                                     headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = (e.read() or b"")[:300].decode("utf-8", "ignore")
            if "invalid_grant" in msg:
                (secrets_dir() / TOKEN_FILE).unlink(missing_ok=True)
                raise GoogleHealthError("Google no longer accepts the sign-in (expired or revoked; apps left "
                                        "in 'Testing' expire after 7 days) - press Connect again.") from e
            raise GoogleHealthError(f"Google sign-in failed (HTTP {e.code}): {msg}") from e

    def access_token(self, force: bool = False) -> str:
        t = read_secret(TOKEN_FILE)
        if not t.get("refresh_token"):
            raise GoogleHealthError("Not connected to Google Health yet (Settings > Fitbit / Google Health).")
        if not force and t.get("access_token") and t.get("exp", 0) > time.time() + 120:
            return t["access_token"]
        c = read_secret(CLIENT_FILE)
        tok = self.token_request({"client_id": c["client_id"], "client_secret": c["client_secret"],
                                  "refresh_token": t["refresh_token"], "grant_type": "refresh_token"})
        t.update(access_token=tok["access_token"], exp=time.time() + float(tok.get("expires_in", 3600)))
        write_secret(TOKEN_FILE, t)
        return t["access_token"]

    # --- data ---------------------------------------------------------------------------
    def get(self, path: str, params: dict) -> dict:
        url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
        for attempt in range(3):
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.access_token(force=attempt > 0)}",
                                                       "Accept": "application/json", "User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:
                body = (e.read() or b"")[:300].decode("utf-8", "ignore")
                if e.code == 401 and attempt == 0:
                    continue
                if "ACCOUNT_NOT_LINKED" in body:
                    raise GoogleHealthError("Your Google account isn't linked to Google Health yet: open the "
                                            "Google Health app and sign in with it first.") from e
                if e.code == 429:
                    raise GoogleHealthError("Google is rate-limiting requests; the next update continues.") from e
                if e.code >= 500 and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise GoogleHealthError(f"GET {path} failed (HTTP {e.code}): {body}") from e
        raise GoogleHealthError(f"GET {path} failed")

    def points(self, data_type: str, flt: str, page_size: int) -> list[dict]:
        """Every data point of a type matching the filter (follows the pages)."""
        out, token = [], None
        while True:
            q = {"filter": flt, "pageSize": page_size, **({"pageToken": token} if token else {})}
            r = self.get(f"dataTypes/{data_type}/dataPoints", q)
            out += r.get("dataPoints") or []
            token = r.get("nextPageToken")
            if not token:
                return out

    def sync(self, folder: str | Path, since: str | None = None, progress=print) -> dict:
        """Pull new data points into folder/<type>.json (merged by resource name). Returns counts."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        last = (read_secret(SYNC_FILE).get("at") or "")[:10]
        today = pd.Timestamp.now().normalize()
        start = pd.Timestamp(since) if since else (pd.Timestamp(last) - pd.Timedelta(days=OVERLAP_DAYS) if last
                                                   else today - pd.Timedelta(days=FIRST_SYNC_DAYS))
        progress(f"Google Health: {start:%Y-%m-%d} -> {today:%Y-%m-%d}")
        jobs = [(t, f'{f}.date >= "{start:%Y-%m-%d}"', 1000) for t, f in DAILY.items()]
        jobs.append(("sleep", f'sleep.interval.civil_end_time >= "{start:%Y-%m-%d}"', 25))
        out = {"start": str(start.date()), "new": 0, "types": {}}
        for data_type, flt, size in jobs:
            got = self.points(data_type, flt, size)
            p = folder / f"{data_type}.json"
            have = json.loads(p.read_text()) if p.exists() else {}
            new = {x.get("name") or json.dumps(x, sort_keys=True): x for x in got}
            out["new"] += sum(k not in have or have[k] != v for k, v in new.items())
            p.write_text(json.dumps({**have, **new}))
            out["types"][data_type] = len(got)
        write_secret(SYNC_FILE, {"at": pd.Timestamp.now().isoformat(timespec="seconds"), **out})
        progress(f"Google Health: {out['new']} new or changed data point(s)")
        return out
