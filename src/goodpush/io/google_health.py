"""Daily wellness rows from Google Health (Fitbit) data points saved by sync/google_health.py.

Columns as the Garmin ones (sleep_h, deep_h, light_h, rem_h, awake_h, bedtime_h, rhr, hrv,
respiration) plus what only Fitbit gives: hrv_deep (RMSSD in deep sleep), spo2 (night average, %)
and skin_temp_dev (night skin temperature minus your baseline, C).

Merging with Garmin: Garmin wins wherever it has a value. HRV is never mixed: Fitbit's average
night RMSSD and Garmin's overnight HRV have different normal ranges, so Fitbit's fills `hrv` only
when there is no Garmin HRV at all (it always stays in hrv_deep).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = {"DEEP": "deep_h", "LIGHT": "light_h", "REM": "rem_h", "AWAKE": "awake_h"}
# REST data type -> (DataPoint field, {value field: column})
DAILY = {"daily-resting-heart-rate": ("dailyRestingHeartRate", {"beatsPerMinute": "rhr"}),
         "daily-heart-rate-variability": ("dailyHeartRateVariability",
                                          {"averageHeartRateVariabilityMilliseconds": "hrv",
                                           "deepSleepRootMeanSquareOfSuccessiveDifferencesMilliseconds": "hrv_deep"}),
         "daily-respiratory-rate": ("dailyRespiratoryRate", {"breathsPerMinute": "respiration"}),
         "daily-oxygen-saturation": ("dailyOxygenSaturation", {"averagePercentage": "spo2"})}
EXTRA = ["hrv_deep", "spo2", "skin_temp_dev"]


class GoogleHealthDays:
    """The saved data points -> one row per day."""

    @staticmethod
    def day(v) -> str | None:
        """YYYY-MM-DD from a Date {year, month, day}, a civil date-time {date: ...} or a string."""
        if isinstance(v, str):
            return v[:10] or None
        if isinstance(v, dict):
            if "year" in v:
                return f"{int(v['year']):04d}-{int(v.get('month', 1)):02d}-{int(v.get('day', 1)):02d}"
            return GoogleHealthDays.day(v.get("date"))
        return None

    @staticmethod
    def num(v) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    @classmethod
    def sleep_rows(cls, points: list, tz: str) -> list[dict]:
        rows = []
        for p in points:                                   # one per night (or nap)
            s = p.get("sleep") or {}
            iv, sm = s.get("interval") or {}, s.get("summary") or {}
            d = cls.day(iv.get("civilEndTime")) or cls.day(iv.get("endTime"))
            if not d or sm.get("minutesAsleep") is None:
                continue
            row = {"date": d, "sleep_h": cls.num(sm["minutesAsleep"]) / 60,
                   "awake_h": cls.num(sm.get("minutesAwake")) / 60}
            for st in sm.get("stagesSummary") or []:
                if st.get("type") in STAGES:
                    row[STAGES[st["type"]]] = cls.num(st.get("minutes")) / 60
            if iv.get("startTime"):
                t = pd.Timestamp(iv["startTime"])
                t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(tz)
                h = t.hour + t.minute / 60
                row["bedtime_h"] = h if h >= 12 else h + 24     # 23.5 = 11:30pm, 25 = 1am
            rows.append(row)
        return rows

    @classmethod
    def daily_rows(cls, data_type: str, points: list) -> list[dict]:
        if data_type == "daily-sleep-temperature-derivations":
            return [{"date": cls.day(x.get("date")),
                     "skin_temp_dev": cls.num(x.get("nightlyTemperatureCelsius")) - cls.num(x.get("baselineTemperatureCelsius"))}
                    for x in (p.get("dailySleepTemperatureDerivations") or {} for p in points)]
        field, cols = DAILY[data_type]
        return [{"date": cls.day(x.get("date")), **{c: cls.num(x.get(k)) for k, c in cols.items()}}
                for x in (p.get(field) or {} for p in points)]

    @classmethod
    def load(cls, folder: str | Path, tz: str = "UTC") -> pd.DataFrame:
        """Daily rows (index date) from folder/<type>.json; empty when nothing is synced."""
        folder = Path(folder)
        frames = []
        for f in sorted(folder.glob("*.json")) if folder.exists() else []:
            t = f.stem
            if t != "sleep" and t not in DAILY and t != "daily-sleep-temperature-derivations":
                continue
            pts = list(json.loads(f.read_text()).values())
            rows = cls.sleep_rows(pts, tz) if t == "sleep" else cls.daily_rows(t, pts)
            df = pd.DataFrame(rows)
            if len(df):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
                # the longest sleep of a day is the night (naps drop out)
                df = df.sort_values("sleep_h").groupby("date").last() if t == "sleep" else df.groupby("date").last()
                frames.append(df)
        if not frames:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        out = pd.concat(frames, axis=1).sort_index()
        out.index.name = "date"
        for c, lo, hi in (("sleep_h", 1, 16), ("rhr", 25, 120), ("hrv", 5, 300), ("hrv_deep", 5, 300),
                          ("spo2", 70, 100), ("respiration", 5, 40)):
            if c in out:
                out[c] = out[c].where(out[c].between(lo, hi))
        return out

    @staticmethod
    def merge(garmin: pd.DataFrame, gh: pd.DataFrame) -> pd.DataFrame:
        """Garmin wins per value; Fitbit fills the gaps and adds its own columns (see the module doc)."""
        if not len(gh):
            return garmin
        gh = gh.copy()
        if "hrv" in gh and "hrv" in garmin and garmin["hrv"].notna().any():
            gh = gh.drop(columns="hrv")
        out = garmin.combine_first(gh)
        return out[list(garmin.columns) + [c for c in out.columns if c not in garmin.columns]]
