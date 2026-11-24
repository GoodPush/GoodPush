"""Parse activity files into a cached per-activity summary table.

Only new or changed files are parsed (cache key = file size + CRC of first
64 KB + parser version), so re-running after a fresh Strava export is fast.
"""
from __future__ import annotations

import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Project
from .io import store
from .io.stamps import in_seconds, stamp
from .io.streams import read_any
from .io.strava import DeletedActivities, project_activities
from .metrics.power import MMP_DURATIONS, summarize_stream, to_1hz
from .progress import Progress

PARSER_VERSION = 3
SUMMARY_TABLE = "activity_streams"


def file_key(path: Path) -> str:
    st = path.stat()
    with open(path, "rb") as fh:
        head = fh.read(65536)
    return f"{st.st_size}-{zlib.crc32(head):08x}-v{PARSER_VERSION}"


def process_file(job: tuple) -> dict:
    """Worker: parse one activity file -> flat summary dict (picklable)."""
    act_id, path, key, max_power, stream_dir = job[:5]
    row = {"id": act_id, "source": str(path), "source_key": key, "error": "",
           "stat_key": job[5] if len(job) > 5 else ""}
    try:
        res = read_any(path)
        s = to_1hz(res["records"], max_power=max_power)
        row["fit_sport"] = res.get("sport") or ""
        row["fit_sub_sport"] = res.get("sub_sport") or ""
        if s is None:
            row["error"] = "no samples"
            return row
        row["start_unix"] = s["start"]
        summ = summarize_stream(s)
        mmp = summ.pop("mmp")
        for d, v in zip(MMP_DURATIONS, mmp):
            row[f"mmp_{d}"] = v
        for k in ("power_hist", "hr_hist"):
            v = summ.pop(k)
            row[k] = json.dumps(v) if v is not None else ""
        row.update(summ)
        if stream_dir:
            np.savez_compressed(
                Path(stream_dir) / f"{act_id}.npz",
                start=np.array([s["start"]]),
                power=s["power"].astype(np.float32), heart_rate=s["heart_rate"].astype(np.float32),
                cadence=s["cadence"].astype(np.float32), speed=s["speed"].astype(np.float32),
                altitude=s["altitude"].astype(np.float32), moving=s["moving"])
    except Exception as e:  # keep going on corrupt files
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    return row


def stat_index(paths) -> dict:
    """{path: stamp ("size-mtime")} for many files with one directory scan per folder (much faster
    than a stat() per file, especially on Windows / network drives). Missing files are absent."""
    by_dir = {}
    for p in paths:
        p = Path(p)
        by_dir.setdefault(p.parent, set()).add(p.name)
    out = {}
    for d, names in by_dir.items():
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name in names:
                        out[str(d / e.name)] = stamp(e.stat())
        except FileNotFoundError:
            continue
    return out


def forget(proj: Project, aids, progress=print) -> None:
    """Drop deleted activities from everything ingest keeps: their streams (streams.h5 is rewritten without
    them) and their rows in the parsed-file and alphaHRV caches. The analysis tables follow on its next run.
    ingest() calls it for any deleted activity still cached, so a delete is cleaned up even if its own job
    never ran."""
    from .alpha import INDEX
    from .io import streamstore
    aids = [str(a) for a in aids]
    streamstore.remove(proj.processed, aids, progress)
    cache = store.load(proj.processed, SUMMARY_TABLE)
    if cache is not None and cache["id"].astype(str).isin(aids).any():
        store.save(cache[~cache["id"].astype(str).isin(aids)].reset_index(drop=True), proj.processed, SUMMARY_TABLE)
    idx = proj.processed / INDEX
    if idx.exists():
        a1 = pd.read_csv(idx, dtype={"id": str, "stat_key": str})
        if a1["id"].isin(aids).any():
            a1[~a1["id"].isin(aids)].to_csv(idx, index=False)


def add_ride(proj: Project, key: str | None, ride_id: str, progress=print) -> None:
    """A ride `goodpush ride` just saved: ingested, and the plan's session `key` it was ridden for (today or
    earlier) marked done with it."""
    from .planning.state import PlanState
    ingest(proj, progress=progress)
    if key and key[:10] <= str(pd.Timestamp.today().date()):
        with PlanState.open(proj.processed / "plan_state.json") as st:
            if key in st.get("planned", {}):
                st.mark(key, "done", ride=ride_id)
                progress(f"Linked to the plan's {key} session.")


def ingest(proj: Project, workers: int | None = None, force: bool = False,
           progress=print, limit: int | None = None) -> pd.DataFrame:
    strava_dir = proj.path("strava")
    try:                                             # rides from a Garmin account export (cheap once done)
        from .sync.garmin import import_export_activities
        import_export_activities(proj.path("garmin"), progress)
    except Exception as e:  # noqa: BLE001 - the rest still ingests
        progress(f"Garmin export activities skipped: {e}")
    acts = project_activities(proj)
    cache = None if force else store.load(proj.processed, SUMMARY_TABLE)
    gone = DeletedActivities(proj.root).ids()
    if cache is not None and cache["id"].astype(str).isin(gone).any():
        forget(proj, gone, progress)
        cache = store.load(proj.processed, SUMMARY_TABLE)
    cached = {} if cache is None else dict(zip(cache["id"].astype(str), cache["source_key"]))
    cached_stat = {} if cache is None or "stat_key" not in cache else \
        dict(zip(cache["id"].astype(str), in_seconds(cache["stat_key"])))

    keep_streams = proj.cfg["ingest"].get("keep_streams", True)
    stream_dir = proj.processed / "streams"
    if keep_streams:
        stream_dir.mkdir(parents=True, exist_ok=True)
    max_power = proj.cfg["ingest"].get("max_power", 2000)

    jobs, missing, migrated = [], 0, {}
    files = {r.id: strava_dir / r.filename for r in acts.itertuples() if r.filename}
    stats = stat_index(files.values())
    for aid, p in files.items():
        sk = stats.get(str(p))
        if sk is None:
            missing += 1
            continue
        if cached_stat.get(str(aid)) == sk:          # unchanged since last time: no need to open it
            continue
        k = file_key(p)
        if cached.get(str(aid)) == k:                 # parsed before, just not stat-keyed yet
            migrated[str(aid)] = sk
            continue
        jobs.append((aid, str(p), k, max_power, str(stream_dir) if keep_streams else "", sk))

    total_todo = len(jobs)
    if limit:
        jobs = jobs[:limit]
    progress(f"{len(acts)} activities in activities.csv; {total_todo} files to parse "
             f"({len(cached)} cached, {missing} files missing)")
    rows = []

    if migrated and cache is not None:
        if "stat_key" not in cache:
            cache["stat_key"] = ""
        sid = cache["id"].astype(str)
        cache["stat_key"] = sid.map(migrated).fillna(cache["stat_key"].astype(str).replace("nan", ""))

    def checkpoint(final=False):
        nonlocal cache
        if final and not rows and not migrated and cache is not None:
            return cache                                  # nothing changed: skip the rewrite
        new = pd.DataFrame(rows)
        if cache is not None and not new.empty:
            keep = cache[~cache["id"].astype(str).isin(new["id"].astype(str))]
            merged = pd.concat([keep, new], ignore_index=True)
        elif cache is not None:
            merged = cache
        else:
            merged = new
        if merged.empty:
            merged = pd.DataFrame(columns=["id", "source", "source_key", "error"])
        merged["id"] = merged["id"].astype(str)
        merged = merged.reset_index(drop=True)
        store.save(merged, proj.processed, SUMMARY_TABLE)
        return merged

    if jobs:
        n = workers if workers is not None else proj.cfg["ingest"].get("workers", 0)
        n = n or max(1, (os.cpu_count() or 2) - 1)
        t0 = time.time()
        if n == 1 or len(jobs) < 8:
            for i, j in enumerate(jobs, 1):
                rows.append(process_file(j))
                Progress.step(progress, "Reading ride files", i, len(jobs))
                if i % 100 == 0:
                    progress(f"  parsed {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
                    checkpoint()
        else:
            with ProcessPoolExecutor(max_workers=n) as ex:
                futs = [ex.submit(process_file, j) for j in jobs]
                for i, f in enumerate(as_completed(futs), 1):
                    rows.append(f.result())
                    Progress.step(progress, "Reading ride files", i, len(jobs))
                    if i % 100 == 0:
                        progress(f"  parsed {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")
                        checkpoint()
        progress(f"parsed {len(jobs)} files in {time.time() - t0:.1f}s")

    out = checkpoint(final=True)
    errs = (out["error"].fillna("") != "").sum()
    if errs:
        progress(f"warning: {errs} files could not be parsed (see 'error' column)")
    try:   # DFA alpha1 from the alphaHRV Connect IQ field, where recorded
        from .alpha import ingest_alpha
        ingest_alpha(proj, acts, workers=workers, progress=progress, stats=stats)
    except Exception as e:  # noqa: BLE001 - optional extra, never block the main ingest
        progress(f"alphaHRV skipped: {e}")
    try:   # move the new per-ride .npz files into the single streams.h5
        from .io import streamstore
        streamstore.pack(proj.processed, progress=progress)
    except Exception as e:  # noqa: BLE001 - the .npz files still work on their own
        progress(f"streams.h5 not updated ({e}); using the .npz files")
    return out
