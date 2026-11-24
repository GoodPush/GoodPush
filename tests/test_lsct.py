"""The LSCT: the workout, what the analysis reads from its ride, and its score."""
import numpy as np
import pandas as pd
import pytest

from goodpush.io.fit import FIT_EPOCH_OFFSET
from goodpush.models.lsct import LSCT
from goodpush.planning.scoring import SessionScorer
from goodpush.planning.workouts import LSCT_STAGES, LSCT_STOP, LSCT_STOP_MIN, WorkoutLibrary as WL
from goodpush.report.plan_out import step_watts

DAY = pd.Timestamp("2026-11-24")


def test_lsct_workout_stages_stop_and_easy_ride():
    w = WL.lsct(DAY, 60)
    st = w.expanded()
    assert w.type == "lsct" and w.minutes == pytest.approx(60) == w.total_minutes()
    assert [s.hr for s in st[:3]] == [f for _, _, f, _ in LSCT_STAGES]
    assert {s.mode for s in st[:3]} == {"resistance"}       # you hold the heart rate, not ERG
    assert [s.minutes for s in st[:3]] == [6, 6, 3]
    stop = st[3]
    assert (stop.name, stop.minutes, stop.hi, stop.kind) == (LSCT_STOP, LSCT_STOP_MIN, 0.0, "rest")
    assert st[4].name == "Z2" and st[4].hr is None
    short = w.resized(30)                                 # the stages never stretch: the easy part does
    assert [s.minutes for s in short.expanded()[:4]] == [6, 6, 3, LSCT_STOP_MIN] and short.minutes == 30
    d = step_watts(st[1], 250)                            # the heart-rate share goes to the ride
    assert d["hr"] == .80 and d["lo"] == round(.70 * 250)
    assert WL.rebuild(DAY, {"type": "lsct", "minutes": 45}).type == "lsct"


def ride_of(plan_steps, stage_w=(100, 180, 230), stage_hr=(110, 140, 160)):
    """laps (one per step, as the recorder writes them) and 1-s streams of an LSCT ridden as planned: power and
    heart rate flat per stage, heart rate falling 40 bpm over the minute sitting still."""
    start = 1_790_000_000.0
    secs = np.array([round(s.minutes * 60) for s in plan_steps])
    t0 = np.r_[0, np.cumsum(secs)[:-1]]
    n = int(secs.sum())
    p, hr = np.full(n, 120.0), np.full(n, 120.0)
    for i in range(3):
        p[t0[i]:t0[i] + secs[i]], hr[t0[i]:t0[i] + secs[i]] = stage_w[i], stage_hr[i]
    s = t0[3]
    p[s:s + secs[3]] = 0.0
    hr[s:s + secs[3]] = np.maximum(stage_hr[2] - 40 * np.arange(secs[3]) / 60, 100)
    laps = [{2: int(start + a - FIT_EPOCH_OFFSET), 8: int(b * 1000), 71: i} for i, (a, b) in enumerate(zip(t0, secs))]
    return laps, start, {"power": p, "heart_rate": hr}


def test_lsct_read_from_the_ride():
    w = WL.lsct(DAY, 45)
    laps, start, z = ride_of(w.expanded())
    r = LSCT.from_ride(laps, start, w, z)
    v = r.values()
    assert [v[f"lsct_w_{k}"] for k in (1, 2, 3)] == [100, 180, 230]
    assert [v[f"lsct_hr_{k}"] for k in (1, 2, 3)] == [110, 140, 160]
    assert v["lsct_hrr_60"] == 40                         # 160 at the stop, 120 a minute in
    no_stop = [x for x in laps if x[71] != 3]             # stopped the ride after the last stage
    assert LSCT.from_ride(no_stop, start, w, z).recovery() is None
    assert LSCT.from_ride(laps, start, WL.easy(DAY, "endurance", 45), z) is None   # not an LSCT


def test_lsct_scored_by_its_length():
    ride = pd.Series({"duration_s": 40 * 60, "workout": "endurance"})
    assert SessionScorer.result("lsct", ride, None, 45)["result"] == "done"
    ride["duration_s"] = 10 * 60
    assert SessionScorer.result("lsct", ride, None, 45)["result"] == "short"


def test_lsct_planned_rides():
    st = {"planned": {"2026-11-24": {"type": "lsct", "ride": "11"}, "2026-11-25": {"type": "lsct",
                                                                                 "compliance": {"ride": "12"}},
                      "2026-11-26": {"type": "ramp", "ride": "13"}}}
    assert LSCT.planned_rides(st) == {"11": "2026-11-24", "12": "2026-11-25"}
