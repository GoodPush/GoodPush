"""Overnight HRV, every 5-min reading of every night, in streams.h5 (group "hrv", one entry per night).

Sources (whichever has the night; the one with more readings wins):
* Garmin export: HRV FIT files (hrv_value, mesg 371) in DI-Connect-Uploaded-Files/*.zip and loose .fit
* Garmin Personal API: hrvReadings in data/raw/garmin/api/raw/<day>.json (kept by sync/garmin_api.py)

A night is filed under the local date of its last reading (the morning you wake up), like Garmin's
calendarDate. `NightHrv.update` only reads sources that changed (processed/hrv_sources.json);
`NightHrv.quantiles` gives each night's min / 25th / median / 75th percentile / max in one pass.
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import fit, streamstore
from .stamps import in_seconds, stamp
from ..progress import Progress

INDEX = "hrv_sources.json"
MIN_READINGS = 6                  # fewer readings than this: no box for that night
QUANTILES = {"hrv_min": 0.0, "hrv_q25": 0.25, "hrv_median": 0.5, "hrv_q75": 0.75, "hrv_max": 1.0}


class NightHrv:
    """Every night's HRV readings: collect them into the stream store, summarise them per night."""

    @staticmethod
    def night(t: np.ndarray, tz: str) -> str:
        """The night's date: the local day of its last reading."""
        return pd.Timestamp(int(t.max()), unit="s", tz="UTC").tz_convert(tz).strftime("%Y-%m-%d")

    @staticmethod
    def from_fit(data: bytes) -> tuple[np.ndarray, np.ndarray] | None:
        """(unix seconds, ms) of the hrv_value messages in a FIT file, or None."""
        try:
            msgs = fit.read_messages(data, (371,)).get(371, [])
        except Exception:  # noqa: BLE001 - not a FIT file we can read
            return None
        m = [(x[253], x[0]) for x in msgs if x.get(253) is not None and x.get(0)]
        if not m:
            return None
        a = np.array(m, float)
        return (a[:, 0] + fit.FIT_EPOCH_OFFSET).astype(np.int64), a[:, 1] / 128     # value in ms * 128

    @staticmethod
    def from_api(day: dict) -> tuple[np.ndarray, np.ndarray] | None:
        r = ((day.get("hrv") or {}).get("hrvReadings")) or []
        r = [x for x in r if x.get("hrvValue") and x.get("readingTimeGMT")]
        if not r:
            return None
        t = (pd.to_datetime([x["readingTimeGMT"] for x in r], utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")
        return np.asarray(t, np.int64), np.array([x["hrvValue"] for x in r], float)

    @classmethod
    def fit_files(cls, path: Path):
        """FIT payloads of a loose .fit or of a (nested) zip of them."""
        if path.suffix.lower() == ".fit":
            yield path.read_bytes()
            return
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                low = name.lower()
                if low.endswith(".fit"):
                    yield zf.read(name)
                elif low.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(zf.read(name))) as inner:
                        yield from (inner.read(n) for n in inner.namelist() if n.lower().endswith(".fit"))

    @staticmethod
    def sources_read(index: dict, garmin_name: str) -> dict:
        """the sources index ({path in the Garmin folder: stamp}) from the stored one: earlier versions keyed
        it by the full path (one computer's, e.g. F:\\...\\garmin\\x.fit) with nanosecond stamps, which matched
        no file on another computer after a peer sync, so every night was read again."""
        if not index:
            return {}
        s = pd.Series(index, dtype=str)
        rel = s.index.str.replace("\\", "/", regex=False).str.rsplit(f"/{garmin_name}/", n=1).str[-1]
        return dict(zip(rel, in_seconds(s)))

    @classmethod
    def update(cls, garmin_dir: str | Path, processed: str | Path, tz: str = "UTC", progress=print) -> int:
        """Stage the nights of new / changed sources and pack them into streams.h5. Returns nights written."""
        garmin_dir, processed = Path(garmin_dir), Path(processed)
        if not garmin_dir.exists():
            return 0
        ip = processed / INDEX
        stored = json.loads(ip.read_text()) if ip.exists() else {}
        seen = cls.sources_read(stored, garmin_dir.name)
        sources = sorted([*garmin_dir.rglob("UploadedFiles*.zip"), *garmin_dir.rglob("*.fit"),
                          *(garmin_dir / "api" / "raw").glob("????-??-??.json")])
        nights, done = {}, {}
        for i, p in enumerate(sources, 1):
            Progress.step(progress, "Overnight HRV files", i, len(sources))
            rel, key = p.relative_to(garmin_dir).as_posix(), stamp(os.stat(p))
            if seen.get(rel) == key:
                continue
            got = [cls.from_api(json.loads(p.read_text()))] if p.suffix == ".json" else \
                [cls.from_fit(b) for b in cls.fit_files(p)]
            for r in got:
                if r is not None and r[0].size:
                    d = cls.night(r[0], tz)
                    if r[0].size > nights.get(d, (np.zeros(0),))[0].size:
                        nights[d] = r
            done[rel] = key
        if nights:
            have = streamstore.read_many(processed, list(nights), "hrv", ["t"])
            out = streamstore.stream_dir(processed)
            out.mkdir(parents=True, exist_ok=True)
            write = {d: r for d, r in nights.items() if r[0].size > have.get(d, {}).get("t", np.zeros(0)).size}
            for d, (t, v) in write.items():
                o = np.argsort(t)
                np.savez_compressed(out / f"{d}.hrv.npz", t=t[o], hrv=v[o].astype(np.float32))
            if write:
                streamstore.pack(processed, progress=progress)
                progress(f"overnight HRV: {len(write)} night(s) of 5-min readings stored")
            nights = write
        if done or seen.keys() != stored.keys():
            ip.write_text(json.dumps({**seen, **done}))
        return len(nights)

    @staticmethod
    def quantiles(processed: str | Path) -> pd.DataFrame:
        """Per night (index date): hrv_min, hrv_q25, hrv_median, hrv_q75, hrv_max, hrv_n. Vectorised:
        one sort of all readings by (night, value), then linear interpolation as numpy.quantile does."""
        keys, n, data = streamstore.read_all(processed, "hrv", ["hrv"])
        if not len(keys):
            return pd.DataFrame(columns=[*QUANTILES, "hrv_n"], index=pd.DatetimeIndex([], name="date"))
        v = data["hrv"].astype(float)
        gid = np.repeat(np.arange(len(keys)), n)
        ok = np.isfinite(v) & (v > 0)
        v, gid = v[ok], gid[ok]
        cnt = np.bincount(gid, minlength=len(keys))
        vs = np.r_[v[np.lexsort((v, gid))], np.nan]          # trailing NaN: index target for empty nights
        start = np.cumsum(cnt) - cnt
        out = {"hrv_n": cnt}
        for col, q in QUANTILES.items():
            pos = q * np.maximum(cnt - 1, 0)
            lo, hi = np.floor(pos).astype(np.int64), np.ceil(pos).astype(np.int64)
            empty = cnt == 0
            a, b = vs[np.where(empty, -1, start + lo)], vs[np.where(empty, -1, start + hi)]
            out[col] = np.where(cnt >= MIN_READINGS, a + (pos - lo) * (b - a), np.nan)
        df = pd.DataFrame(out, index=pd.DatetimeIndex(pd.to_datetime(keys), name="date")).sort_index()
        return df[[*QUANTILES, "hrv_n"]]
