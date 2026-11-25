"""intervals.icu: send the plan's sessions to your intervals.icu calendar as planned workouts.

Uses your own free API key (intervals.icu Settings -> Developer Settings), sent as HTTP Basic auth with user
"API_KEY"; athlete id 0 is the key's own athlete. Sessions go in one bulk upsert keyed by `external_id`
(goodpush_<session key>), so sending again updates them instead of adding copies; sessions sent earlier that the
plan no longer has (from today on) are deleted. Past days are left alone, as are workouts you made yourself.
The workout is the session's intervals.icu text (report/plan_out.session_icu, the gp_*.txt next to the .fit).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from ..report.plan_out import outbound_title
from . import read_secret, secrets_dir, write_secret


class IntervalsIcuError(RuntimeError):
    pass


class IntervalsIcu:
    """The intervals.icu API for one athlete's key (in their credentials folder, athletes.Athletes.secrets)."""
    BASE = "https://intervals.icu/api/v1/athlete/0"
    KEY_FILE = "intervals_icu.json"            # {"key", "saved"}
    SENT_FILE = "intervals_icu_sent.json"      # {"at", "ids": {external_id: date}, "sent", "deleted"}: the last push
    PREFIX = "goodpush_"
    DAYS = 14                                  # today and the next 13 days
    UA = "goodpush (personal use)"
    TIMEOUT = 60
    RETRIES = 2                                # after a 429 (waits Retry-After) or a dropped connection

    def __init__(self, key: str | None = None):
        self.key = key if key is not None else read_secret(self.KEY_FILE).get("key", "")

    # --- the key -------------------------------------------------------------------------------
    @classmethod
    def save_key(cls, key: str) -> str:
        """Check the key with intervals.icu, then store it; returns the athlete's name there."""
        key = (key or "").strip()
        if len(key) < 10 or any(c.isspace() for c in key):
            raise ValueError("That doesn't look like an intervals.icu API key.")
        name = cls(key).athlete_name()
        write_secret(cls.KEY_FILE, {"key": key, "saved": pd.Timestamp.now().isoformat(timespec="seconds")})
        return name

    @classmethod
    def forget(cls) -> None:
        for n in (cls.KEY_FILE, cls.SENT_FILE):
            (secrets_dir() / n).unlink(missing_ok=True)

    @classmethod
    def info(cls) -> dict:
        k, sent = read_secret(cls.KEY_FILE), read_secret(cls.SENT_FILE)
        return {"configured": bool(k.get("key")), "saved": k.get("saved"), "days": cls.DAYS,
                "last_push": {x: sent.get(x) for x in ("at", "sent", "deleted")} if sent else None}

    def athlete_name(self) -> str:
        return str((self.request("GET", "") or {}).get("name") or "")

    # --- the plan -> events ----------------------------------------------------------------------
    @classmethod
    def needs_files(cls, plan_dir: Path) -> bool:
        """The plan was written before it had intervals.icu text (an older goodpush): write it again first."""
        p = Path(plan_dir) / "plan.json"
        rows = json.loads(p.read_text(encoding="utf-8")).get("sessions") or [] if p.exists() else []
        return any(r.get("fit") and not (Path(plan_dir) / "workouts" / str(r.get("icu") or "")).is_file()
                   for r in rows)

    @classmethod
    def events(cls, plan_dir: Path, today: pd.Timestamp) -> list[dict]:
        """The plan's sessions from today on (DAYS days) that have a workout file, as intervals.icu events."""
        plan_dir = Path(plan_dir)
        rows = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8")).get("sessions") or []
        first, last = str(today.date()), str((today + pd.Timedelta(days=cls.DAYS - 1)).date())
        out = []
        for r in rows:                                          # a few weeks of sessions
            f = plan_dir / "workouts" / str(r.get("icu") or "")
            if not (r.get("icu") and first <= r["date"] <= last and f.is_file()):
                continue
            out.append({"category": "WORKOUT", "type": "Ride", "start_date_local": f"{r['date']}T00:00:00",
                        "name": outbound_title(r["title"]), "description": f.read_text(encoding="utf-8"),
                        "moving_time": int(r["minutes"]) * 60, "icu_training_load": int(r["tss"]),
                        "external_id": cls.PREFIX + str(r.get("key") or r["date"])})
        return out

    def push(self, plan_dir: Path, today: pd.Timestamp, progress=print, rewrite=None) -> dict:
        """Send the next DAYS days of the plan; delete what an earlier push sent that the plan no longer has.
        `rewrite()` writes the plan again, called first when it was written without intervals.icu text."""
        if not self.key:
            raise IntervalsIcuError("No intervals.icu API key yet: add it under Settings > API keys.")
        if rewrite is not None and self.needs_files(plan_dir):
            progress("intervals.icu: writing the plan's workout text first")
            rewrite()
        evs = self.events(plan_dir, today)
        if not evs and self.needs_files(plan_dir):
            raise IntervalsIcuError("The plan has no intervals.icu workout text yet: update the plan, then send again.")
        progress(f"intervals.icu: sending {len(evs)} workout(s)")
        if evs:
            self.request("POST", "/events/bulk?upsert=true", evs)
        ids = {e["external_id"]: e["start_date_local"][:10] for e in evs}
        before = read_secret(self.SENT_FILE).get("ids") or {}
        gone = [x for x, d in before.items() if x not in ids and d >= str(today.date())]
        if gone:
            progress(f"intervals.icu: removing {len(gone)} workout(s) the plan no longer has")
            self.request("PUT", "/events/bulk-delete", [{"external_id": x} for x in gone])
        kept = {x: d for x, d in before.items() if d < str(today.date())}      # past ones: theirs to keep
        res = {"at": pd.Timestamp.now().isoformat(timespec="seconds"), "sent": len(evs), "deleted": len(gone)}
        write_secret(self.SENT_FILE, {**res, "ids": {**kept, **ids}})
        progress(f"intervals.icu: {len(evs)} workout(s) on your calendar, {len(gone)} removed")
        return res

    # --- HTTP ---------------------------------------------------------------------------------
    def request(self, method: str, path: str, body=None):
        auth = base64.b64encode(f"API_KEY:{self.key}".encode()).decode()
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Basic {auth}", "User-Agent": self.UA, "Accept": "application/json",
                   **({"Content-Type": "application/json"} if data is not None else {})}
        for attempt in range(self.RETRIES + 1):
            req = urllib.request.Request(self.BASE + path, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                    txt = r.read().decode("utf-8", "replace")
                    return json.loads(txt) if txt.strip() else None
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.RETRIES:
                    time.sleep(min(float(e.headers.get("Retry-After") or 5), 60))
                    continue
                if e.code in (401, 403):
                    raise IntervalsIcuError("intervals.icu didn't accept the API key (Settings > intervals.icu).")
                msg = e.read().decode("utf-8", "replace")[:300]
                raise IntervalsIcuError(f"intervals.icu answered {e.code}: {msg}") from None
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt < self.RETRIES:
                    time.sleep(2)
                    continue
                raise IntervalsIcuError(f"Couldn't reach intervals.icu: {getattr(e, 'reason', e)}") from None
        return None
