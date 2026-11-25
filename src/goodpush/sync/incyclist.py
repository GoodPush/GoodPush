"""Incyclist (desktop app; it keeps everything in local files): put the plan straight into
Incyclist's workout calendar, no cloud service in between.

Incyclist's calendar (incyclist-services: workouts/calendar/service.ts) is the file
<Incyclist data folder>/workouts/calendar.json - on Windows %APPDATA%/Incyclist. It is a list of
{day, workoutId, workout, updated, source}; `workout` uses the same JSON as Incyclist's workout
library (steps / repeated segments, power in % of FTP). Incyclist reads it when it starts and
shows this week's and the next 7 days' sessions under Workouts > Scheduled Workouts.

* Incyclist only reads the file at start-up: restart it after the plan changes.
* If an online calendar is connected in Incyclist for workout downloads, its sync replaces the
  list - use one or the other.
* Only entries with source "goodpush" are replaced; anything else in the file is kept. Past days
  of the last two weeks stay; the file is written whole and swapped in (never half-written).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


SOURCE = "goodpush"
CAL_FILE = ("workouts", "calendar.json")
KEEP_PAST_DAYS = 14


def default_folder() -> Path:
    """Incyclist's own folder (an Electron app's: %APPDATA%, ~/Library/Application Support, ~/.config)."""
    import platformdirs
    return Path(platformdirs.user_config_dir("Incyclist", appauthor=False, roaming=True))


def folder(cfg: dict) -> Path:
    f = (cfg.get("incyclist") or {}).get("folder") or ""
    return Path(f).expanduser() if f else default_folder()


def calendar_path(cfg: dict) -> Path:
    return folder(cfg).joinpath(*CAL_FILE)


def is_running() -> bool | None:
    """Incyclist open right now? (Windows only; None = can't tell.)"""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Incyclist.exe", "/NH"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return "incyclist.exe" in out.stdout.lower()
    except Exception:  # noqa: BLE001
        return None


def _read(p: Path) -> list:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:  # noqa: BLE001 - missing or unreadable: start a new list
        return []


def status(cfg: dict) -> dict:
    f, p = folder(cfg), calendar_path(cfg)
    ours = [e for e in _read(p) if e.get("source") == SOURCE]
    return {"folder": str(f), "default_folder": str(default_folder()), "found": (f / "workouts").is_dir(),
            "calendar": str(p), "entries": len(ours), "days": sorted(str(e.get("day", ""))[:10] for e in ours)[-14:],
            "running": is_running(), "sync": bool((cfg.get("incyclist") or {}).get("sync")),
            "written": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.exists() else None}


# --- plan session -> Incyclist workout JSON --------------------------------------------------
def _step(st) -> dict:
    ramp = st.kind in ("warmup", "cooldown") and st.hi > st.lo
    return {"type": "step", "duration": int(round(st.minutes * 60)), "text": st.name + (f" · {st.cue}" if st.cue else ""),
            "power": {"type": "pct of FTP", "min": round(st.lo * 100, 1), "max": round(st.hi * 100, 1)},
            "work": st.kind == "active", "steady": not ramp, "cooldown": st.kind == "cooldown"}


def _steps(items) -> list[dict]:
    """Steps and Repeats as Incyclist steps and (nested) segments; a ramp as 1-min steps."""
    out = []
    for x in items:                          # a workout's few dozen items; a repeat is what has reps
        if hasattr(x, "reps"):
            out.append({"type": "segment", "repeat": int(x.reps), "steps": _steps(x.steps)})
        else:
            out += [_step(p) for p in x.pieces()]
    return out


def workout(s, ds: str) -> dict:
    """A plan Session as an Incyclist workout definition (what its library and calendar store)."""
    from ..report.plan_out import WHY, outbound_title
    steps = _steps(s.steps)
    desc = " ".join(x for x in (WHY.get(s.type, ""), s.note, s.adjusted and f"Adjusted: {s.adjusted}") if x)
    digest = hashlib.sha1(json.dumps([s.title, steps], sort_keys=True).encode()).hexdigest()[:10]
    return {"type": "workout", "id": f"{SOURCE}-{ds}", "hash": f"{SOURCE}-{ds}-{digest}",
            "name": outbound_title(s.title), "description": desc, "category": {"name": "goodpush plan"}, "steps": steps}


def _day_utc(ds: str, tz: str) -> str:
    """Local midnight as JavaScript writes a Date into JSON (UTC, milliseconds, Z)."""
    t = pd.Timestamp(ds).tz_localize(tz).tz_convert("UTC")
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def entries(out: dict, tz: str) -> list[dict]:
    from ..report.plan_out import has_file
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z")
    res = []
    for s in out["sessions"]:
        if not has_file(s) or s.date < out["today"]:
            continue
        ds = str(s.date.date())
        wid = (s.key or ds).replace("#", "-")          # a second session that day: goodpush-<day>-2
        res.append({"day": _day_utc(ds, tz), "workoutId": f"{SOURCE}-{wid}", "workout": workout(s, wid),
                    "updated": now, "source": SOURCE})
    return res


def _local_day(e: dict, tz: str) -> str:
    try:
        return str(pd.Timestamp(e["day"]).tz_convert(tz).date())
    except Exception:  # noqa: BLE001
        return str(e.get("day", ""))[:10]


def sync(out: dict, cfg: dict, tz: str, progress=print) -> dict:
    """Write the plan (today on) into Incyclist's calendar, keeping everything not ours."""
    f = folder(cfg)
    if not f.is_dir():
        raise FileNotFoundError(f"no Incyclist data folder at {f} - is Incyclist installed? "
                                "(Set the folder in Settings > Incyclist.)")
    p = calendar_path(cfg)
    today = str(out["today"].date())
    keep_from = str((out["today"] - pd.Timedelta(days=KEEP_PAST_DAYS)).date())
    old = _read(p)
    others = [e for e in old if e.get("source") != SOURCE]
    past = [e for e in old if e.get("source") == SOURCE and keep_from <= _local_day(e, tz) < today]
    new = entries(out, tz)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.goodpush-tmp")
    tmp.write_text(json.dumps(others + past + new, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    running = is_running()
    progress(f"Incyclist: {len(new)} planned workouts written to {p}"
             + (" - restart Incyclist to see them" if running else ""))
    return {"written": len(new), "kept": len(others) + len(past), "running": running, "file": str(p)}
