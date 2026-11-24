"""A synthetic rider's whole history, written as an athlete folder goodpush reads like anyone's: rides as FIT
files (data/raw/goodpush, as `goodpush ride` records them), the watch's nights (data/raw/garmin/api/raw), the tests
linked to their planned sessions (plan_state.json), goodpush.toml - and the truth in synth/ (profile.toml,
truth.parquet: every morning's true capacity and the day's session; truth.json: parameters, illnesses, breaks,
tests).

The rider trains to their own weekly habit (the profile), not to goodpush's plan: the planner's Scheduler plans
ahead from today on its own state, and a synthetic rider who already followed it would test it against itself."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..app.tomledit import set_values
from ..config import CONFIG_NAME, init_project, load_project
from ..io.strava import API_COLUMNS, RIDES_TABLE
from ..metrics.power import mean_max
from ..planning.state import PlanState
from ..planning.workouts import ALL_OUT_NAME, TestEfforts, WorkoutLibrary
from ..ride.recorder import RideRecorder
from .profile import DAY_NAMES, RiderProfile
from .riding import FTP_TEST_STEP, RideMaker
from .truth import TRUTHS
from .wellness import Nights


class SyntheticRider:
    FOLDER = "synth"
    VERSION = 1
    START_H = (7.0, 18.0)                 # rides start between these local hours
    ILL_DAYS, BREAK_DAYS = (4, 12), (7, 14)
    EPISODE_GAP_D = 21                    # illnesses and breaks at least this far apart (and from the start)
    COMEBACK_EASY_D = 3                   # only easy rides this long after an illness
    NOTES = ("Back after a cold", "First ride after the flu", "Easy one, still coughing after that virus",
             "Back on the bike after being sick")
    KEY_MINUTES = 75                      # a key day at least (longer days: the session with more Z2, as pros ride)
    LONG_SHARE, LONG_MIN, LONG_MAX = 0.35, 90, 330   # the long ride: this share of the week, within these minutes
    EASY_MIN = 45
    SETTLE_D = 28                         # the start is in balance with the first four weeks' riding (truth `settle`)
    LEVEL_START = 2
    FTP_START = 0.95                      # the rider's FTP at first: this x their true 1-h power
    FTP_OF_TEST = 0.95                    # ... after a 20-min test: this x its average
    FTP_OF_RAMP = 0.75                    # ... after a ramp test: this x its best minute
    TESTS = ("test", "ramp", "lsct")      # the order they're placed in (the first wins a shared day)
    INDOOR, OUTDOOR = "Virtual Ride", "Ride"

    def __init__(self, profile: RiderProfile, progress=print):
        self.p = profile.checked()
        self.progress = progress
        self.rng = np.random.default_rng(profile.seed)
        end = pd.Timestamp(profile.end) if profile.end else \
            pd.Timestamp.now(tz=profile.timezone).tz_localize(None).normalize() - pd.Timedelta(days=1)
        self.days = pd.date_range(end - pd.Timedelta(days=profile.days - 1), end)
        self.illnesses, self.breaks = self.episodes()
        self.truth = TRUTHS[profile.truth](profile, self.days, [on for on, _ in self.illnesses], self.rng)
        self.maker = RideMaker(profile, self.rng)

    # --- what happens when ---------------------------------------------------------------
    def episodes(self) -> tuple[list, list]:
        """(start, days) of the illnesses and of the breaks: about the profile's number a year, each at least
        EPISODE_GAP_D from the others and from the start (random slots spaced out, then shuffled between the two)."""
        p, rng, T = self.p, self.rng, len(self.days)
        years = T / 365.0
        n_ill, n_break = rng.poisson(p.illness_per_year * years), rng.poisson(p.breaks_per_year * years)
        slot = self.EPISODE_GAP_D + max(self.ILL_DAYS[1], self.BREAK_DAYS[1])
        n = min(n_ill + n_break, max((T - self.EPISODE_GAP_D) // slot, 0))
        n_ill = min(n_ill, n)
        room = T - self.EPISODE_GAP_D - n * slot
        start = np.sort(rng.integers(0, room + 1, n)) + np.arange(n) * slot + self.EPISODE_GAP_D
        ill = rng.permutation(n) < n_ill
        length = np.where(ill, rng.integers(self.ILL_DAYS[0], self.ILL_DAYS[1] + 1, n),
                          rng.integers(self.BREAK_DAYS[0], self.BREAK_DAYS[1] + 1, n))
        pairs = list(zip(self.days[start], length.tolist()))
        return [x for x, i in zip(pairs, ill) if i], [x for x, i in zip(pairs, ill) if not i]

    def spans(self, episodes: list) -> tuple[np.ndarray, np.ndarray]:
        """day indices (first, one past the last) of episodes."""
        a = self.days.get_indexer(pd.DatetimeIndex([s for s, _ in episodes]))
        return a, a + np.array([k for _, k in episodes], int)

    def schedule(self) -> pd.DataFrame:
        """per day: the session kind (rest, endurance, long, a key type, test, ramp, lsct), its minutes, whether
        it's outside, an all-out effort's length (0 = none) and a note."""
        p, rng = self.p, self.rng
        n = len(self.days)
        dow = self.days.day_of_week.to_numpy()
        week = (np.arange(n) + dow[0]) // 7
        hours = np.interp(np.arange(n), [0, n - 1], [p.hours_start, p.hours_end])
        ride = np.isin(dow, [DAY_NAMES.index(d) for d in set(p.ride_days) | set(p.key_days) | {p.long_day}])
        key = ride & np.isin(dow, [DAY_NAMES.index(d) for d in p.key_days])
        long = ride & (dow == DAY_NAMES.index(p.long_day))
        recovery = (week + 1) % p.recovery_every == 0 if p.recovery_every else np.zeros(n, bool)
        hours = np.where(recovery, hours * p.recovery_share, hours)
        rides_week = len(set(p.ride_days) | set(p.key_days) | {p.long_day})   # a whole week's (not a partial one's)
        long_min = np.clip(self.LONG_SHARE * hours * 60, self.LONG_MIN, self.LONG_MAX)
        day_min = (hours * 60 - long_min) / max(rides_week - 1, 1)            # the other days share the rest
        minutes = np.where(long, long_min, np.maximum(day_min, np.where(key, self.KEY_MINUTES, self.EASY_MIN)))
        mix = list(p.key_mix)
        w = np.array([p.key_mix[k] for k in mix], float)
        kind = np.where(long, "long", np.where(key, rng.choice(mix, n, p=w / w.sum()), "endurance")).astype(object)
        kind[key & recovery] = "tempo"
        a, b = self.spans(self.illnesses + self.breaks)
        off = RideMaker.covered(n, a, b)
        after = self.since_illness()
        kind[after < self.COMEBACK_EASY_D] = "endurance"
        k = np.flatnonzero(key)
        first_key = np.zeros(n, bool)
        first_key[k[np.unique(week[k], return_index=True)[1]]] = True
        open_day = ~off & ~recovery & (after >= self.COMEBACK_EASY_D)
        self.tests(kind, week, {"test": first_key & open_day, "ramp": first_key & open_day,
                                "lsct": ride & ~key & ~long & open_day})
        kind[~ride | off] = "rest"
        minutes = np.where(kind == "rest", 0, minutes)
        easy = np.isin(kind, ["endurance", "long"])
        outdoor = easy & (rng.random(n) < p.outdoor_share)
        rides_month = max(len(p.ride_days) * 30 / 7, 1)
        effort = np.where(easy & (rng.random(n) < p.all_out_per_month / rides_month), rng.choice(RideMaker.ALL_OUT_S, n), 0)
        note = self.comeback_notes(kind)
        surges = outdoor.copy()                     # drawn last and only when asked: older profiles keep their riders
        if p.surge_share > 0:
            surges |= easy & (rng.random(n) < p.surge_share)
        return pd.DataFrame({"kind": kind, "minutes": minutes, "outdoor": outdoor, "surges": surges,
                             "effort_s": effort, "note": note}, index=self.days)

    def since_illness(self) -> np.ndarray:
        """days since the last illness ended (large before the first, negative never)."""
        _, end = self.spans(self.illnesses)
        end = np.r_[-10_000, np.sort(end)]
        t = np.arange(len(self.days))
        return t - end[np.searchsorted(end, t, "right") - 1]

    def comeback_notes(self, kind: np.ndarray) -> np.ndarray:
        """the first ride after each illness says so (how goodpush finds illnesses in anyone's rides)."""
        note = np.full(len(kind), "", object)
        rides = np.flatnonzero(kind != "rest")
        _, end = self.spans(self.illnesses)
        j = np.searchsorted(rides, end)
        back = rides[j[j < rides.size]]
        note[back] = np.array(self.NOTES, object)[self.rng.integers(len(self.NOTES), size=back.size)]
        return note

    def tests(self, kind: np.ndarray, week: np.ndarray, candidates: dict) -> None:
        """tests into `kind`: each on the first candidate day of each block of its weeks (the profile's
        *_weeks), blocks of different tests offset so they fall in different weeks."""
        p = self.p
        every = {"test": p.ftp_test_weeks, "ramp": p.ramp_weeks, "lsct": p.lsct_weeks}
        taken = np.zeros(len(kind), bool)
        for j, name in enumerate(self.TESTS):        # three kinds of test
            if not every[name]:
                continue
            day = np.flatnonzero(candidates[name] & ~taken)
            block = (week[day] + j * every[name] // len(self.TESTS)) // every[name]
            pick = day[np.unique(block, return_index=True)[1]]
            kind[pick] = name
            taken[pick] = True

    # --- the days -----------------------------------------------------------------------
    def workout(self, date, kind: str, minutes: float, levels: dict, rotation: TestEfforts | None = None):
        """the session ridden: an easy ride, a test, or the key type at its next level, lengthened with Z2 to the
        day's minutes (Workout.resized); with `rotation` (profile test_efforts) opening with the all-out effort due
        (as the planner's, Scheduler.max_efforts: at most one a week)."""
        if kind in ("endurance", "long"):
            return WorkoutLibrary.easy(date, kind, minutes)
        if kind == "lsct":
            return WorkoutLibrary.lsct(date, max(minutes, self.EASY_MIN))
        if kind in ("test", "ramp"):
            return WorkoutLibrary.test(kind, date)
        levels[kind] = levels.get(kind, self.LEVEL_START) + 1
        w = WorkoutLibrary.intervals(date, kind, levels[kind] - 1)
        w = w.resized(minutes) if minutes > w.minutes + 5 else w
        if rotation is None or any(d is not None and (date - d).days < 7 for d in rotation.last.values()):
            return w
        sec = rotation.pick(kind, date)
        if sec:
            rotation.placed(sec, date)
            return WorkoutLibrary.max_effort(w, sec)
        return w

    def usual(self, sched: pd.DataFrame, ftp: float) -> list[list[np.ndarray]]:
        """the first SETTLE_D days' rides (1-Hz power each day) ridden at the start: the rider's usual training,
        which the truth's start is put in balance with (`settle`)."""
        cap, levels, out = self.truth.morning(), {}, []
        for date, s in sched.iloc[:self.SETTLE_D].iterrows():     # four weeks
            if s.kind == "rest":
                out.append([])
                continue
            w = self.workout(date, s.kind, float(s.minutes), levels)
            out.append([self.maker.ride(w, ftp, cap, self.truth.all_out, bool(s.outdoor),
                                        surges=bool(s.surges))["power"]])
        return out

    def run(self) -> dict:
        """every day in turn - each morning depends on the days before: the morning's truth, the day's ride (if
        any), the truth moved on. Returns the rides (FIT bytes, activities rows), the mornings and the tests."""
        sched = self.schedule()
        ftp = self.FTP_START * self.truth.morning().ftp
        self.truth.settle(self.usual(sched, ftp))
        levels, morning, rides, tests = {}, [], [], []
        rotation = TestEfforts() if self.p.test_efforts else None
        for i, (date, s) in enumerate(sched.iterrows()):   # days: sequential by nature (state feedback)
            cap = self.truth.morning()
            morning.append({"date": date, **cap.row(), "ftp_belief": ftp})
            if s.kind == "rest":
                self.truth.advance([])
                continue
            w = self.workout(date, s.kind, float(s.minutes), levels, rotation)
            r = self.maker.ride(w, ftp, cap, self.truth.all_out, bool(s.outdoor), int(s.effort_s) or None,
                                bool(s.surges))
            self.truth.advance([r["power"]])
            rides.append(self.recorded(r, w, date, cap, bool(s.outdoor), s.note))
            if s.kind in self.TESTS or w.title.startswith(ALL_OUT_NAME):     # linked to the plan: found as tests
                tests.append((w, rides[-1]["id"]))
            ftp = self.new_ftp(s.kind, r, ftp)
            if i % 60 == 0:
                self.progress(f"synthetic rider: day {i + 1} of {len(sched)}, {len(rides)} rides")
        return {"rides": rides, "morning": pd.DataFrame(morning).set_index("date"), "tests": tests,
                "schedule": sched}

    def new_ftp(self, kind: str, r: dict, ftp: float) -> float:
        """the rider's FTP after the day: moved by a 20-min test or a ramp."""
        if kind == "test":
            i = int(np.flatnonzero(r["names"] == FTP_TEST_STEP)[0])
            return self.FTP_OF_TEST * float(r["power"][r["step"] == i].mean())
        if kind == "ramp":
            return self.FTP_OF_RAMP * mean_max(r["power"], [60])[0]
        return ftp

    def recorded(self, r: dict, w, date: pd.Timestamp, cap, outdoor: bool, note: str) -> dict:
        """the ride as `goodpush ride` records it (RideRecorder): FIT bytes and its activities row."""
        local = date + pd.to_timedelta(self.rng.uniform(*self.START_H), "h")
        start = int(local.tz_localize(self.p.timezone, ambiguous=False, nonexistent="shift_forward").timestamp())
        p, heart = r["power"], r["heart"]
        hr = self.maker.heart.noisy(heart)
        n = len(p)
        speed = (np.maximum(p, 30.0 if outdoor else 0.0) / 0.25) ** (1 / 3)
        cad = np.where(p > 0, 88 + 3 * self.rng.standard_normal(n), 0.0)
        rows = np.c_[start + np.arange(n), p, cad, hr, speed, r["step"], np.ones(n)]
        beats = self.maker.beats.beats(heart, p, cap, float(start)) if self.rng.random() < self.p.strap_share else []
        rec = RideRecorder(start, w.title, rows.tolist(), beats)
        return {"id": rec.id, "row": rec.row(self.OUTDOOR if outdoor else self.INDOOR, note), "fit": rec.fit_bytes()}

    # --- the folder ------------------------------------------------------------------------
    def write(self, folder: str | Path):
        """the athlete folder (refused if it already has a goodpush.toml); returns its Project."""
        folder = Path(folder)
        if (folder / CONFIG_NAME).exists():
            raise FileExistsError(f"{folder} already has a {CONFIG_NAME}: pick a new folder")
        out = self.run()
        init_project(folder)
        self.settings(folder / CONFIG_NAME)
        proj = load_project(folder)
        self.progress(f"synthetic rider: writing {len(out['rides'])} rides")
        rides = proj.path("rides")
        rides.mkdir(parents=True, exist_ok=True)
        for r in out["rides"]:                           # one file per ride
            (rides / r["row"]["filename"]).write_bytes(r["fit"])
        pd.DataFrame([r["row"] for r in out["rides"]], columns=API_COLUMNS).to_csv(rides / RIDES_TABLE, index=False)
        if self.p.wellness:
            self.progress("synthetic rider: writing the nights")
            from ..sync import garmin
            ill = Nights.illness_level(self.days, self.illnesses)
            Nights(self.p, self.rng).write(proj.path("garmin") / garmin.API_SUBDIR / "raw", self.days,
                                           out["morning"]["fatigue"].to_numpy(float), ill)
            garmin.rebuild(proj.path("garmin"))
        self.link_tests(proj.processed / "plan_state.json", out["tests"])
        self.save_truth(folder / self.FOLDER, out)
        return proj

    def settings(self, path: Path) -> None:
        """goodpush.toml: who the rider is (heart rates only with enter_hr), the plan, no background sampling."""
        p = self.p
        ath = {"name": p.name, "sex": p.sex, "birth_year": p.birth_year, "timezone": p.timezone,
               "weight_kg": p.weight_kg}
        if p.enter_hr:
            ath.update(max_hr=p.max_hr, lthr=p.lthr, resting_hr=p.resting_hr)
        set_values(path, "athlete", ath)
        set_values(path, "plan", {"enabled": p.plan, "hours_per_week": round(p.hours_end, 1)})
        set_values(path, "physio", {"ranges_every_days": 0})

    @staticmethod
    def link_tests(path: Path, tests: list) -> None:
        """the tests as planned sessions done by their rides: how goodpush knows a ride was a ramp test or LSCT."""
        st = PlanState.load(path)
        st.record([w for w, _ in tests])
        for w, rid in tests:                             # a test every few weeks
            st.mark(str(w.date.date()), "done", ride=rid)
        st.save()

    def save_truth(self, folder: Path, out: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "profile.toml").write_text(self.p.text(), encoding="utf-8")
        out["morning"].join(out["schedule"]).to_parquet(folder / "truth.parquet")
        info = {"version": self.VERSION, "truth": self.p.truth, "params": self.truth.params(),
                "illnesses": [{"onset": str(a.date()), "days": k} for a, k in self.illnesses],
                "breaks": [{"start": str(a.date()), "days": k} for a, k in self.breaks],
                "tests": [{"date": str(w.date.date()), "type": w.type, "ride": rid} for w, rid in out["tests"]],
                "max_hr": self.p.max_hr, "lthr": self.p.lthr, "resting_hr": self.p.resting_hr}
        (folder / "truth.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
