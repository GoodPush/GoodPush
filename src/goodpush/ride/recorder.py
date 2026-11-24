"""The ride as 1-Hz samples, saved as a FIT activity (a lap per workout step) that ingest reads like any
other ride: data/raw/goodpush/<id>.fit + a row in activities_goodpush.csv. A copy of the samples is
written every minute (<id>.partial.npz) so a crash loses at most a minute; `recover` turns leftovers
into rides."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.fit_workout import FIT_EPOCH_OFFSET, _Writer
from ..io.strava import API_COLUMNS, RIDES_TABLE
from ..metrics.power import normalized_power

INVALID = {"uint8": 0xFF, "uint16": 0xFFFF, "uint32": 0xFFFFFFFF}
HRV_PER_MESSAGE = 5                 # beat-to-beat intervals per FIT hrv message (as Garmin's)


class RideRecorder:
    """Samples (time, power, cadence, hr, speed, step, running): step = workout step index (-1 = none),
    running = the workout clock was running (counts as timer time)."""

    COLUMNS = ("time", "power", "cadence", "hr", "speed", "step", "running")

    def __init__(self, start: float, title: str = "Ride", rows: list | None = None, beats: list | None = None):
        self.start = int(start)
        self.title = title
        self.rows = rows or []
        self.beats = beats or []            # (time heard, beat-to-beat interval s, beats it spans: > 1 = missed)

    @property
    def id(self) -> str:
        return f"goodpush_{self.start}"

    @property
    def seconds(self) -> int:
        return len(self.rows)

    def add(self, t: float, power: float, cadence: float, hr: float, speed: float, step: int,
            running: bool) -> None:
        self.rows.append((t, power, cadence, hr, speed, step, float(running)))

    def arrays(self) -> dict[str, np.ndarray]:
        a = np.array(self.rows, float).reshape(-1, len(self.COLUMNS))
        return dict(zip(self.COLUMNS, a.T))

    def summary(self) -> dict:
        a = self.arrays()
        p, hr, cad = a["power"], a["hr"], a["cadence"]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN columns (no HR strap): NaN
            return {"elapsed_s": len(p), "timer_s": int(a["running"].sum()),
                    "distance_m": float(np.nansum(a["speed"])),
                    "avg_power": float(np.nanmean(p)), "max_power": float(np.nanmax(p)),
                    "np": normalized_power(np.nan_to_num(p)) if np.isfinite(p).any() else np.nan,
                    "avg_hr": float(np.nanmean(hr)), "max_hr": float(np.nanmax(hr)),
                    "avg_cadence": float(np.nanmean(np.where(cad > 0, cad, np.nan)))}

    def fit_bytes(self) -> bytes:
        a = self.arrays()
        n = len(a["time"])
        ts = (a["time"].astype(np.int64) - FIT_EPOCH_OFFSET).astype(np.uint32)
        t0, t1 = int(ts[0]), int(ts[-1]) + 1
        dist = np.cumsum(np.nan_to_num(a["speed"]))
        cols = {"power": self.fit_int(a["power"], "uint16"), "hr": self.fit_int(a["hr"], "uint8"),
                "cadence": self.fit_int(a["cadence"], "uint8"),
                "speed": self.fit_int(a["speed"] * 1000, "uint16"), "dist": self.fit_int(dist * 100, "uint32")}
        w = _Writer()
        w.message(0, [(0, "enum", 4, None), (1, "uint16", 255, None), (2, "uint16", 0, None),
                      (3, "uint32z", (t0 & 0xFFFFFFF) or 1, None), (4, "uint32", t0, None)])
        w.message(21, [(253, "uint32", t0, None), (0, "enum", 0, None), (1, "enum", 0, None)])   # timer start
        w.messages(20, [(253, "uint32"), (7, "uint16"), (3, "uint8"), (4, "uint8"), (6, "uint16"), (5, "uint32")],
                   [ts, cols["power"], cols["hr"], cols["cadence"], cols["speed"], cols["dist"]])
        w.message(21, [(253, "uint32", t1, None), (0, "enum", 0, None), (1, "enum", 4, None)])   # stop all
        # laps: one per run of samples on the same step
        first = np.r_[0, np.flatnonzero(np.diff(a["step"])) + 1]
        last = np.r_[first[1:], n]
        lap = self.lap_stats(a, first, last, dist)
        w.messages(19, [(253, "uint32"), (2, "uint32"), (7, "uint32"), (8, "uint32"), (9, "uint32"),
                        (19, "uint16"), (20, "uint16"), (15, "uint8"), (16, "uint8"), (17, "uint8"),
                        (0, "enum"), (1, "enum"), (254, "uint16"), (71, "uint16")],          # 71: the workout step
                   [ts[last - 1] + 1, ts[first], (last - first) * 1000, lap["timer"] * 1000, lap["dist"],
                    lap["avg_power"], lap["max_power"], lap["avg_hr"], lap["max_hr"], lap["avg_cadence"],
                    np.full(len(first), 9), np.full(len(first), 1), np.arange(len(first)),
                    np.where(a["step"][first] >= 0, a["step"][first], 0xFFFF).astype(int)])
        rr = self.hrv_ms()
        if rr.size:                         # beat-to-beat intervals: FIT hrv messages, 5 a message (ms)
            w.messages(78, [(0, ("uint16", HRV_PER_MESSAGE))], [rr.reshape(-1, HRV_PER_MESSAGE)])
        s = self.summary()
        w.message(18, [(253, "uint32", t1, None), (2, "uint32", t0, None), (7, "uint32", n * 1000, None),
                       (8, "uint32", s["timer_s"] * 1000, None),
                       (9, "uint32", self.fit_int(s["distance_m"] * 100, "uint32"), None),
                       (5, "enum", 2, None), (6, "enum", 6, None),                      # cycling, indoor
                       (20, "uint16", self.fit_int(s["avg_power"], "uint16"), None),
                       (21, "uint16", self.fit_int(s["max_power"], "uint16"), None),
                       (34, "uint16", self.fit_int(s["np"], "uint16"), None),
                       (16, "uint8", self.fit_int(s["avg_hr"], "uint8"), None),
                       (17, "uint8", self.fit_int(s["max_hr"], "uint8"), None),
                       (18, "uint8", self.fit_int(s["avg_cadence"], "uint8"), None),
                       (0, "enum", 8, None), (1, "enum", 1, None),
                       (25, "uint16", 0, None), (26, "uint16", len(first), None)])
        local = t1 + (time.localtime(self.start).tm_gmtoff or 0)
        w.message(34, [(253, "uint32", t1, None), (0, "uint32", s["timer_s"] * 1000, None),
                       (1, "uint16", 1, None), (2, "enum", 0, None), (3, "enum", 26, None),
                       (4, "enum", 1, None), (5, "uint32", local, None)])
        return w.bytes()

    def hrv_ms(self) -> np.ndarray:
        """the beat-to-beat intervals in ms for FIT hrv messages (one over missed beats as it was: the reader's
        artifact filter sees it), padded to whole messages with 0xFFFF (invalid, as Garmin pads)."""
        ms = np.minimum(np.round(np.array([b[1] for b in self.beats], float) * 1000), 0xFFFE).astype(int)
        return np.r_[ms, np.full(-ms.size % HRV_PER_MESSAGE, 0xFFFF)]

    @staticmethod
    def fit_int(x, typ: str):
        """Rounded to FIT integers; NaN -> the type's invalid value."""
        x = np.asarray(x, float)
        bad = INVALID[typ]
        out = np.where(np.isfinite(x), np.clip(np.round(np.nan_to_num(x)), 0, bad - 1), bad).astype(np.int64)
        return int(out) if out.ndim == 0 else out

    def lap_stats(self, a: dict, first: np.ndarray, last: np.ndarray, dist: np.ndarray) -> dict:
        """Per lap (samples first..last-1): timer seconds, distance (cm), mean / max power, hr, cadence."""
        out = {"timer": np.add.reduceat(a["running"], first).astype(np.int64),
               "dist": self.fit_int((dist[last - 1] - np.r_[0, dist][first]) * 100, "uint32")}
        for name, typ in (("power", "uint16"), ("hr", "uint8"), ("cadence", "uint8")):
            x = a[name]
            ok = np.isfinite(x)
            k = np.add.reduceat(ok.astype(float), first)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.add.reduceat(np.where(ok, x, 0), first) / k
            top = np.maximum.reduceat(np.where(ok, x, -np.inf), first)
            out[f"avg_{name}"] = self.fit_int(np.where(k > 0, mean, np.nan), typ)
            out[f"max_{name}"] = self.fit_int(np.where(k > 0, top, np.nan), typ)
        return out

    def save(self, folder: Path) -> Path:
        """The FIT file + its row in the activities table; the partial copy is removed."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.id}.fit"
        path.write_bytes(self.fit_bytes())
        table = folder / RIDES_TABLE
        df = pd.DataFrame([self.row()])
        if table.exists():
            old = pd.read_csv(table, dtype=str, keep_default_na=False)
            df = pd.concat([old[old["id"] != self.id], df.astype(str)], ignore_index=True)
        df[API_COLUMNS].sort_values("start_utc").to_csv(table, index=False)
        self.partial_path(folder).unlink(missing_ok=True)
        return path

    def row(self, kind: str = "Virtual Ride", description: str = "") -> dict:
        """the ride's row in the activities table (API_COLUMNS); its FIT file is <id>.fit next to the table."""
        s = self.summary()
        row = {c: "" for c in API_COLUMNS}
        row.update(id=self.id, start_utc=pd.Timestamp(self.start, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                   name=self.title, type=kind, description=description, elapsed_s=s["elapsed_s"],
                   moving_s_csv=s["timer_s"], distance_m=round(s["distance_m"], 1), avg_hr_csv=s["avg_hr"],
                   max_hr_csv=s["max_hr"], avg_watts_csv=s["avg_power"], wap_csv=s["np"], max_watts_csv=s["max_power"],
                   commute=False, filename=f"{self.id}.fit")
        return row

    def partial_path(self, folder: Path) -> Path:
        return Path(folder) / f"{self.id}.partial.npz"

    def save_partial(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{self.id}.tmp.npz"
        np.savez(tmp, start=self.start, title=self.title, rows=np.array(self.rows, float),
                 beats=np.array(self.beats, float).reshape(-1, 3))
        tmp.replace(self.partial_path(folder))

    def discard(self, folder: Path) -> None:
        self.partial_path(folder).unlink(missing_ok=True)

    @classmethod
    def load_partial(cls, path: Path) -> RideRecorder:
        with np.load(path) as z:
            return cls(int(z["start"]), str(z["title"]), z["rows"].tolist(),
                       [tuple(b) for b in z["beats"]] if "beats" in z else [])

    @classmethod
    def recover(cls, folder: Path, min_seconds: int = 60) -> list[Path]:
        """Rides left as partial copies (a crash): saved as rides if they're long enough, else dropped."""
        out = []
        for p in sorted(Path(folder).glob("*.partial.npz")):
            r = cls.load_partial(p)
            if r.seconds >= min_seconds:
                out.append(r.save(folder))
            p.unlink(missing_ok=True)
        return out
