"""Pull the latest data straight from Strava and Garmin Connect.

Credentials and tokens live in this computer's machine folder (folders.Folders: outside the project folder, so
the project can be shared or put in git without leaking them); each athlete's in its athletes/<name>/
(GOODPUSH_SECRETS, set by athletes.Athletes.use).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..folders import Folders

# a first sync (nothing from that source yet) reaches this far back; older history comes from the
# services' account exports (`goodpush import`), not from their APIs
FIRST_SYNC_DAYS = 90


def secrets_dir() -> Path:
    p = Path(os.environ.get("GOODPUSH_SECRETS") or Folders.machine())
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
