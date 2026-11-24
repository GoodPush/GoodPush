"""Strava bulk-export helpers (activities.csv)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

CYCLING_TYPES = {"Ride", "Virtual Ride", "VirtualRide", "E-Bike Ride", "Gravel Ride",
                 "Mountain Bike Ride", "Velomobile", "Handcycle", "E-Mountain Bike Ride"}

# Map of output column -> candidate csv headers (pandas renames duplicates with .1)
_COLS = {
    "id": ["Activity ID"],
    "date_raw": ["Activity Date"],
    "name": ["Activity Name"],
    "type": ["Activity Type"],
    "description": ["Activity Description"],
    "private_note": ["Activity Private Note"],
    "elapsed_s": ["Elapsed Time.1", "Elapsed Time"],
    "moving_s_csv": ["Moving Time"],
    "distance_m": ["Distance.1"],
    "distance_km": ["Distance"],
    "elev_gain": ["Elevation Gain"],
    "avg_hr_csv": ["Average Heart Rate"],
    "max_hr_csv": ["Max Heart Rate.1", "Max Heart Rate"],
    "avg_watts_csv": ["Average Watts"],
    "wap_csv": ["Weighted Average Power"],
    "max_watts_csv": ["Max Watts"],
    "relative_effort": ["Relative Effort.1", "Relative Effort"],
    "perceived_exertion": ["Perceived Exertion"],
    "athlete_weight": ["Athlete Weight"],
    "gear": ["Activity Gear"],
    "commute": ["Commute.1", "Commute"],
    "filename": ["Filename"],
    "avg_temp": ["Average Temperature"],
    "calories": ["Calories"],
}


API_SUBDIR = "api"                      # data/raw/strava/api/ - written by `goodpush sync strava`
API_TABLE = "activities_api.csv"
API_COLUMNS = ["id", "start_utc", "name", "type", "description", "private_note", "elapsed_s",
               "moving_s_csv", "distance_m", "elev_gain", "avg_hr_csv", "max_hr_csv",
               "avg_watts_csv", "wap_csv", "max_watts_csv", "relative_effort",
               "perceived_exertion", "athlete_weight", "gear", "commute", "filename", "avg_temp",
               "calories"]
_NUM = ["elapsed_s", "moving_s_csv", "distance_m", "elev_gain", "avg_hr_csv", "max_hr_csv",
        "avg_watts_csv", "wap_csv", "max_watts_csv", "relative_effort", "perceived_exertion",
        "athlete_weight", "avg_temp", "calories"]


def load_activities_csv(strava_dir: str | Path, garmin_dir: str | Path | None = None,
                        rides_dir: str | Path | None = None, deleted=()) -> pd.DataFrame:
    """All activities: the Strava bulk export, anything pulled by `goodpush sync strava`, rides
    recorded by `goodpush ride`, and activities downloaded from Garmin Connect, less the ids in `deleted`.

    One ride in more than one source (the trainer recording and the watch, Strava and Garmin; SameRide) is
    listed once: the Strava export row wins, then the Strava API row, then goodpush's own recording, then
    Garmin. The kept row's `also` lists the other copies' ids. Deleted ids go first: deleting the kept copy
    leaves its twin."""
    strava_dir = Path(strava_dir)
    deleted = set(deleted)
    tables = [_load_export(strava_dir) if (strava_dir / "activities.csv").exists() else None,
              load_api_table(strava_dir),
              load_rides_table(rides_dir, strava_dir) if rides_dir is not None else None,
              load_garmin_table(garmin_dir, strava_dir) if garmin_dir is not None else None]
    df = None
    for table in tables:
        if table is None or not len(table):
            continue
        table = table[~table["id"].isin(deleted)]
        if df is None:
            df = table.assign(also="").reset_index(drop=True)
            continue
        table = table[~table["id"].isin(set(df["id"]))]          # the Strava API's copy of an export row
        df = SameRide.merge(df, table)
    if df is None:
        raise FileNotFoundError(f"no activities in {strava_dir}: unzip your Strava export there, "
                                "or import Garmin activities (`goodpush garmin-snippet`)")
    return df.sort_values("start_utc").reset_index(drop=True)


class SameRide:
    """One ride recorded twice - by `goodpush ride` and the watch, or in Strava and Garmin: counted once (TSS,
    hours, the plan). The same ride when the starts are within NEAR, or the two overlap for at least OVERLAP of
    the longer one's elapsed time: both cover mostly the same time (a watch started minutes before the trainer
    recording; not a ride left recording for a day, which would swallow the next day's rides)."""
    NEAR = pd.Timedelta(minutes=5)
    OVERLAP = 0.5
    CANDIDATES = (0, 1, 2)          # rows of `have` checked before each new ride's end and around its start

    @classmethod
    def match(cls, table: pd.DataFrame, have: pd.DataFrame) -> np.ndarray:
        """for each row of `table`, the position in `have` of the same ride, or -1."""
        n = len(table)
        if not len(have) or not n:
            return np.full(n, -1)
        hs, he = cls.span(have)
        order = np.argsort(hs, kind="stable")
        hs, he = hs[order], he[order]
        ts, te = cls.span(table)
        last = np.searchsorted(hs, te, side="left") - 1       # the last ride of `have` starting before the end
        near = np.searchsorted(hs, ts)                        # ... and the ones around the start
        cand = np.clip(np.column_stack([last[:, None] - np.array(cls.CANDIDATES),
                                        near[:, None] - np.array([1, 0])]), 0, len(hs) - 1)
        overlap = np.minimum(te[:, None], he[cand]) - np.maximum(ts[:, None], hs[cand])
        longer = np.maximum((te - ts)[:, None], he[cand] - hs[cand])
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(longer > 0, overlap / np.where(longer > 0, longer, 1), 0.0)
        share = np.where(np.abs(ts[:, None] - hs[cand]) <= cls.NEAR.value, 2.0, share)   # near starts: always
        share[np.isnan(ts)] = 0.0
        best = share.argmax(1)
        hit = share[np.arange(n), best] >= cls.OVERLAP
        return np.where(hit, order[cand[np.arange(n), best]], -1)

    @staticmethod
    def span(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """start and end (float ns since the epoch; NaN without a start); no elapsed time = no length."""
        start = df["start_utc"].to_numpy(dtype="datetime64[ns]")
        s = np.where(np.isnat(start), np.nan, start.astype("int64").astype(float))
        length = pd.to_numeric(df["elapsed_s"], errors="coerce").fillna(0).clip(lower=0).to_numpy(float) * 1e9
        return s, s + length

    @classmethod
    def merge(cls, df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        """df (rows kept so far, with `also`) plus the rows of `table` that aren't the same ride as one of them;
        those that are go into the kept row's `also`."""
        m = cls.match(table, df)
        dup = m >= 0
        if dup.any():
            extra = pd.Series(table["id"].to_numpy()[dup], index=m[dup]).groupby(level=0).agg(",".join)
            cur = df["also"].to_numpy(object)[extra.index]
            df = df.copy()
            df.loc[df.index[extra.index], "also"] = np.where(cur == "", extra.to_numpy(object),
                                                             cur + "," + extra.to_numpy(object))
        return pd.concat([df, table[~dup].assign(also="")], ignore_index=True)


def project_activities(proj) -> pd.DataFrame:
    """load_activities_csv over the project's folders, without the activities you deleted."""
    return load_activities_csv(proj.path("strava"), proj.path("garmin"), proj.path("rides"),
                               DeletedActivities(proj.root).ids())


class DeletedActivities:
    """Activities you deleted in the app, kept in data/deleted_activities.json (your input, not a cache): left
    out from then on, so the next sync or export can't bring them back. A ride goodpush recorded also loses
    its FIT and its row; ingest drops their streams from the processed folder (ingest.forget)."""
    FILE = "deleted_activities.json"

    def __init__(self, root):
        self.path = Path(root) / "data" / self.FILE

    def ids(self) -> set[str]:
        try:
            return set(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return set()

    def delete(self, aid: str, rides_dir: str | Path) -> bool:
        """True when it was goodpush's own recording (its file removed too)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self.ids() | {aid}), indent=1), encoding="utf-8")
        table = Path(rides_dir) / RIDES_TABLE
        rides = pd.read_csv(table, dtype=str, keep_default_na=False) if table.exists() else None
        mine = rides is not None and rides["id"].eq(aid).any()
        if mine:
            for f in rides.loc[rides["id"].eq(aid), "filename"]:
                (table.parent / f).unlink(missing_ok=True)
            rides[rides["id"].ne(aid)].to_csv(table, index=False)
        return mine


class StravaUpload:
    """A ride goodpush recorded, sent to Strava by hand (uploads through Strava's API cost money): Strava's
    upload page opens in the browser and the file manager shows a hand-off folder holding the ride's FIT (to drag
    onto the page) and a PNG of its chart (to add to the activity as a photo; a FIT can't carry one)."""
    PAGE = "https://www.strava.com/upload/select"
    PNG_MAX = 5 * 1024 ** 2          # bytes: a chart picture is well under 1 MB
    FOLDER = "strava_upload"         # under data/processed: only the last ride sent

    def __init__(self, rides_dir: str | Path, processed: str | Path):
        self.table = Path(rides_dir) / RIDES_TABLE
        self.out = Path(processed) / self.FOLDER

    def file(self, aid: str) -> Path:
        """the FIT file of goodpush's own recording `aid`"""
        rides = pd.read_csv(self.table, dtype=str, keep_default_na=False) if self.table.exists() else None
        names = rides.loc[rides["id"].eq(aid), "filename"] if rides is not None else pd.Series(dtype=str)
        path = self.table.parent / names.iloc[0] if len(names) else None
        if path is None or not path.exists():
            raise FileNotFoundError("goodpush has no file of this ride (only its own recordings can be sent)")
        return path

    def hand_off(self, aid: str, png: bytes = b"") -> Path:
        """the hand-off folder with only this ride's FIT (+ its chart, if given); the FIT's path"""
        src = self.file(aid)
        if png and (len(png) > self.PNG_MAX or not png.startswith(b"\x89PNG")):
            raise ValueError("the chart picture didn't come through")
        self.out.mkdir(parents=True, exist_ok=True)
        for old in self.out.iterdir():                # the last ride sent
            old.unlink()
        fit = self.out / src.name
        shutil.copyfile(src, fit)
        if png:
            fit.with_suffix(".png").write_bytes(png)
        return fit

    def open(self, aid: str, png: bytes = b"") -> Path:
        fit = self.hand_off(aid, png)
        webbrowser.open(self.PAGE)
        self.reveal(fit)
        return fit

    @staticmethod
    def reveal(path: Path) -> None:
        """the file selected in Explorer / Finder (on Linux: its folder)"""
        cmd = (["explorer", f"/select,{path}"] if sys.platform == "win32" else
               ["open", "-R", str(path)] if sys.platform == "darwin" else ["xdg-open", str(path.parent)])
        subprocess.Popen(cmd)


GARMIN_TABLE = "activities_garmin.csv"   # data/raw/garmin/activities/ - written by `goodpush garmin-import`
RIDES_TABLE = "activities_goodpush.csv"   # data/raw/goodpush/ - written by `goodpush ride`


def load_garmin_table(garmin_dir: str | Path, strava_dir: str | Path) -> pd.DataFrame | None:
    return _load_local_table(Path(garmin_dir) / "activities", GARMIN_TABLE, strava_dir)


def load_rides_table(rides_dir: str | Path, strava_dir: str | Path) -> pd.DataFrame | None:
    return _load_local_table(Path(rides_dir), RIDES_TABLE, strava_dir)


def _load_local_table(d: Path, name: str, strava_dir: str | Path) -> pd.DataFrame | None:
    import os
    p = d / name
    if not p.exists():
        return None
    df = load_api_table_file(p)
    # file names are stored relative to the table's folder; ingest resolves them
    # against the strava folder
    df["filename"] = [os.path.relpath(d / f, strava_dir) if f else "" for f in df["filename"]]
    return df


def load_api_table(strava_dir: str | Path) -> pd.DataFrame | None:
    p = Path(strava_dir) / API_SUBDIR / API_TABLE
    if not p.exists():
        return None
    return load_api_table_file(p)


def load_api_table_file(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    for c in API_COLUMNS:
        if c not in df:
            df[c] = ""
    for c in _NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    df["commute"] = df["commute"].astype(str).str.lower().eq("true")
    df["is_cycling"] = df["type"].isin(CYCLING_TYPES)
    df["indoor"] = df["type"].isin({"Virtual Ride", "VirtualRide"})
    df["id"] = df["id"].astype(str)
    return df[API_COLUMNS + ["is_cycling", "indoor"]]


def _load_export(strava_dir: Path) -> pd.DataFrame:
    path = Path(strava_dir) / "activities.csv"
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = pd.DataFrame(index=raw.index)
    for out, cands in _COLS.items():
        col = next((c for c in cands if c in raw.columns), None)
        df[out] = raw[col] if col else ""
    num = ["elapsed_s", "moving_s_csv", "distance_m", "distance_km", "elev_gain",
           "avg_hr_csv", "max_hr_csv", "avg_watts_csv", "wap_csv", "max_watts_csv",
           "relative_effort", "perceived_exertion", "athlete_weight", "avg_temp", "calories"]
    for c in num:
        df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
    if df["distance_m"].isna().all():
        df["distance_m"] = df["distance_km"] * 1000
    df["start_utc"] = pd.to_datetime(df["date_raw"], format="%b %d, %Y, %I:%M:%S %p",
                                     errors="coerce", utc=True)
    bad = df["start_utc"].isna()
    if bad.any():
        df.loc[bad, "start_utc"] = pd.to_datetime(df.loc[bad, "date_raw"], errors="coerce",
                                                  utc=True)
    df["commute"] = df["commute"].str.lower().eq("true")
    df["is_cycling"] = df["type"].isin(CYCLING_TYPES)
    df["indoor"] = df["type"].isin({"Virtual Ride", "VirtualRide"})
    df["id"] = df["id"].astype(str)
    df = df.drop(columns=["date_raw", "distance_km"])
    return df


def sport_class(activity_type: str) -> str:
    t = (activity_type or "").lower()
    if activity_type in CYCLING_TYPES or "ride" in t or "bike" in t or "cycl" in t:
        return "ride"
    if "run" in t:
        return "run"
    if "weight" in t or "strength" in t or "crossfit" in t:
        return "strength"
    if "yoga" in t or "pilates" in t:
        return "yoga"
    return "other"


def weight_history(df: pd.DataFrame) -> pd.Series:
    """Athlete weight (kg) recorded with activities, indexed by date."""
    w = df[["start_utc", "athlete_weight"]].dropna()
    w = w[(w["athlete_weight"] > 35) & (w["athlete_weight"] < 200)]
    if w.empty:
        return pd.Series(dtype=float)
    s = w.set_index(w["start_utc"].dt.tz_convert(None).dt.normalize())["athlete_weight"]
    return s.groupby(level=0).last().sort_index()


FATIGUE_WORDS = ["tired", "fatigue", "exhausted", "wrecked", "dead legs", "legs dead",
                 "heavy legs", "no legs", "weak", "sick", "ill", "cold", "flu", "covid",
                 "injur", "pain", "sore", "lurg", "virus", "fever", "cramp", "bonk", "cooked", "toast", "struggl",
                 "hate", "awful", "terrible", "burn out", "burnout", "overtrain",
                 "no sleep", "slept bad", "slept poor", "jet lag"]
GOOD_WORDS = ["pr", "strong", "great", "good legs", "felt good", "best", "record",
              "fresh", "easy", "fun", "awesome", "flying"]


def mood_flags(text: str) -> tuple[int, int]:
    """Very crude keyword count of fatigue / positive words in a description."""
    t = f" {(text or '').lower()} "
    import re
    bad = sum(1 for w in FATIGUE_WORDS if re.search(rf"\b{re.escape(w)}", t))
    good = sum(1 for w in GOOD_WORDS if re.search(rf"\b{re.escape(w)}\b", t))
    return bad, good
