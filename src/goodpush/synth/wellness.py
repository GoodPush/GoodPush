"""A synthetic rider's nights as a watch records them, in the Garmin Personal API's day files
(data/raw/garmin/api/raw/<day>.json, as sync/garmin_api.py keeps them): sleep and its stages, resting heart
rate, overnight HRV with every 5-min reading, steps, stress. Fatigue and illness show in them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Nights:
    SLEEP_SD_H = 0.6
    BED_H, BED_SD_H = 23.0, 0.6          # local bedtime
    STAGES = {"deepSleepSeconds": 0.18, "remSleepSeconds": 0.22, "awakeSleepSeconds": 0.05}   # light: the rest
    RHR_SD, HRV_SD = 1.5, 0.08           # bpm; log
    RHR_FATIGUE, HRV_FATIGUE = 1.5, 0.04   # per sd of the rider's fatigue
    ILL_RHR, ILL_HRV, ILL_SLEEP = 7.0, 0.3, -0.7   # at an illness's worst: bpm up, log HRV down, hours of sleep
    ILL_BEFORE_D = 2                     # the signs start this many days before the illness stops the riding
    READING_S = 300
    READING_SD = 0.15                    # a 5-min reading's scatter around the night's value (log)

    def __init__(self, profile, rng: np.random.Generator):
        self.p = profile
        self.rng = rng

    @classmethod
    def illness_level(cls, days: pd.DatetimeIndex, episodes: list) -> np.ndarray:
        """0..1 per day: rising from ILL_BEFORE_D days before each illness's onset to 1 on it, easing off over its
        days off and as long again."""
        if not episodes:
            return np.zeros(len(days))
        on = days.get_indexer(pd.DatetimeIndex([a for a, _ in episodes]))[:, None]
        length = np.array([k for _, k in episodes], float)[:, None]
        d = np.arange(len(days))[None, :] - on                   # (episodes, days)
        level = np.where(d <= 0, (d + cls.ILL_BEFORE_D) / cls.ILL_BEFORE_D, 1.0 - d / (2.0 * length))
        return np.clip(level, 0.0, 1.0).max(0)

    def write(self, folder: Path, days: pd.DatetimeIndex, fatigue: np.ndarray, ill: np.ndarray) -> int:
        """one file per day (the night before it) into `folder`; returns how many."""
        n = len(days)
        rng, p = self.rng, self.p
        f = (fatigue - fatigue.mean()) / (fatigue.std() or 1.0)
        sleep_h = np.clip(p.sleep_h + self.SLEEP_SD_H * rng.standard_normal(n) + self.ILL_SLEEP * ill, 4.0, 10.5)
        rhr = p.resting_hr + self.RHR_SD * rng.standard_normal(n) + self.RHR_FATIGUE * f + self.ILL_RHR * ill
        hrv = p.hrv_ms * np.exp(self.HRV_SD * rng.standard_normal(n) - self.HRV_FATIGUE * f - self.ILL_HRV * ill)
        bed = days - pd.Timedelta(days=1) + pd.to_timedelta(self.BED_H + self.BED_SD_H * rng.standard_normal(n), "h")
        start = bed.tz_localize(p.timezone, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
        end = start + pd.to_timedelta(sleep_h, "h")
        weekly = pd.Series(hrv).rolling(7, min_periods=1).mean().to_numpy()
        base = pd.Series(hrv).rolling(28, min_periods=7)
        lo = (base.mean() - base.std()).bfill().to_numpy()
        hi = (base.mean() + base.std()).bfill().to_numpy()
        steps = rng.integers(5000, 13000, n)
        stress = np.clip(25 + 6 * rng.standard_normal(n) + 4 * f + 15 * ill, 5, 90)
        folder.mkdir(parents=True, exist_ok=True)
        done = 0
        for i, day in enumerate(days):                      # one file a day: written as the API's are
            if pd.isna(start[i]):
                continue
            ds = day.strftime("%Y-%m-%d")
            (folder / f"{ds}.json").write_text(json.dumps(self.day(
                ds, start[i], end[i], sleep_h[i], rhr[i], hrv[i], weekly[i], lo[i], hi[i], steps[i], stress[i])))
            done += 1
        return done

    def day(self, ds: str, start, end, sleep_h, rhr, hrv, weekly, lo, hi, steps, stress) -> dict:
        s = sleep_h * 3600.0
        stages = {k: int(v * s) for k, v in self.STAGES.items()}
        stages["lightSleepSeconds"] = int(s - sum(stages.values()))
        t = pd.date_range(start, end, freq=f"{self.READING_S}s")
        vals = hrv * np.exp(self.READING_SD * self.rng.standard_normal(len(t)))
        readings = [{"hrvValue": int(round(v)), "readingTimeGMT": x.strftime("%Y-%m-%dT%H:%M:%S.0")}
                    for x, v in zip(t, vals)]
        score = int(np.clip(100 - 12 * abs(8.0 - sleep_h) - 0.3 * stress, 20, 100))
        return {"sleep": {"dailySleepDTO": {"calendarDate": ds, "sleepTimeSeconds": int(s - stages["awakeSleepSeconds"]),
                                            "sleepStartTimestampGMT": int(start.timestamp() * 1000),
                                            "sleepEndTimestampGMT": int(end.timestamp() * 1000), **stages,
                                            "averageRespirationValue": 14.0, "avgSleepStress": round(stress / 2, 1),
                                            "sleepScores": {"overall": {"value": score}}}},
                "summary": {"calendarDate": ds, "restingHeartRate": int(round(rhr)), "totalSteps": int(steps),
                            "averageStressLevel": int(stress), "bodyBatteryHighestValue": int(np.clip(110 - stress, 20, 100)),
                            "bodyBatteryLowestValue": int(np.clip(40 - stress / 2, 5, 60))},
                "hrv": {"hrvSummary": {"calendarDate": ds, "lastNightAvg": int(round(hrv)), "weeklyAvg": int(round(weekly)),
                                       "baseline": {"balancedLow": int(round(lo)), "balancedUpper": int(round(hi))}},
                        "hrvReadings": readings},
                "_source": "synthetic"}
