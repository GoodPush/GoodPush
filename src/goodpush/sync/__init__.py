"""Pull the latest data straight from Strava and Garmin Connect.

Credentials and tokens live in ~/.goodpush/ (outside the project folder, so the
project can be shared or put in git without leaking them). Override with the
GOODPUSH_SECRETS environment variable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# a first sync (nothing from that source yet) reaches this far back; older history comes from the
# services' account exports (`goodpush import`), not from their APIs
FIRST_SYNC_DAYS = 90


def secrets_dir() -> Path:
    p = Path(os.environ.get("GOODPUSH_SECRETS") or Path.home() / ".goodpush")
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_secret(name: str) -> dict:
    p = secrets_dir() / name
    return json.loads(p.read_text()) if p.exists() else {}


def write_secret(name: str, data: dict) -> Path:
    p = secrets_dir() / name
    p.write_text(json.dumps(data, indent=1))
    try:
        os.chmod(p, 0o600)
    except OSError:  # pragma: no cover - Windows
        pass
    return p
