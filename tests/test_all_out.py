"""All-out steps from the plan: found in the ride (models.all_out), counted as tests by the physio model's efforts,
and ridden at the curve by the synthetic rider."""
import numpy as np
import pandas as pd

from goodpush.io.fit import FIT_EPOCH_OFFSET
from goodpush.models.all_out import AllOutStep
from goodpush.models.physio import PhysioInputs
from goodpush.planning.workouts import ALL_OUT_NAME, MAX_EFFORTS, WorkoutLibrary as WL

DAY = pd.Timestamp("2026-11-24")


def ridden(w, effort_w: float, rest_w: float = 150.0):
    """laps (one per step, as the recorder writes them) and 1-s power of `w` ridden at rest_w, its all-out step at
    effort_w."""
    plan = w.expanded()
    secs = np.array([round(s.minutes * 60) for s in plan])
    t0 = np.r_[0, np.cumsum(secs)[:-1]]
    p = np.full(int(secs.sum()), rest_w)
    i = next(i for i, s in enumerate(plan) if s.name.startswith(ALL_OUT_NAME))
    p[t0[i]:t0[i] + secs[i]] = effort_w
    start = 1_790_000_000.0
    laps = [{2: int(start + a - FIT_EPOCH_OFFSET), 8: int(b * 1000), 71: j} for j, (a, b) in enumerate(zip(t0, secs))]
    return laps, start, p


def test_all_out_step_read_from_the_ride():
    w = WL.max_effort(WL.intervals(DAY, "threshold", 2), 180)
    laps, start, p = ridden(w, 900.0)
    r = AllOutStep.from_ride(laps, start, w, p)
    assert (r.seconds, r.watts) == (180, 900.0)
    assert AllOutStep.from_ride(laps, start, WL.intervals(DAY, "threshold", 2), p) is None      # no all-out step
    assert AllOutStep.from_ride([x for x in laps if x[71] != 7], start, w, p) is None          # its lap missing


def test_the_step_counts_as_a_test_where_it_is_the_ride_s_best():
    """the ride's best at the step's length is a test (kind 2) when it came from the step and reached the recent
    best (TEST_SHARE); a hard ride's other bests stay hard; an effort beaten later in the ride stays hard."""
    s = PhysioInputs.EFFORT_S
    days = pd.date_range("2026-03-01", periods=60)
    best = {t: 1000.0 * (t / 5.0) ** -0.25 for t in s}

    def ride(day, scale, step_s=np.nan, step_w=np.nan):   # noqa: E306  hygiene: ok (test-local builder)
        return {"date": days[day], "workout": "threshold", "moving_s": 7200.0, "duration_s": 7200.0,
                "power_coverage": 1.0, "all_out_s": step_s, "all_out_w": step_w,
                **{f"mmp_{t}": best[t] * scale.get(t, 0.6) for t in s}}
    rides = pd.DataFrame([ride(5, {t: 1.0 for t in s}),                          # the 90-day best
                          ride(30, {180: 0.98}, 180, best[180] * 0.98),             # the 3 min, from the step
                          ride(40, {180: 0.98}, 180, best[180] * 0.90),             # beaten later in the ride
                          ride(50, {180: 0.70}, 180, best[180] * 0.70)])            # far below: a failed one
    day, dur, _, kind = PhysioInputs.efforts(rides, np.array([5, 30, 40, 50]))
    k = {(int(a), float(b)): int(c) for a, b, c in zip(day, dur, kind)}
    assert k[(30, 180.0)] == 2 and k[(30, 5.0)] == 1 and k[(30, 60.0)] == 1
    assert k[(40, 180.0)] == 1 and k[(50, 180.0)] == 1


def test_synthetic_rider_rides_the_all_out_step_at_its_curve():
    from goodpush.physio import PowerCurve
    from goodpush.synth import RiderProfile
    from goodpush.synth.riding import RideMaker
    from goodpush.synth.truth import Capacity
    cap = Capacity(cp=215, w_prime=15000, pmax=820, fade=0.08, map=270, lt1=155, ftp=208, fatigue=0, form=1)
    all_out = lambda t: PowerCurve.power(np.asarray(t, float), cap.cp, cap.w_prime, cap.pmax, cap.fade)  # noqa: E731  hygiene: ok
    m = RideMaker(RiderProfile(), np.random.default_rng(0))
    for sec in MAX_EFFORTS:
        w = WL.max_effort(WL.intervals(DAY, "vo2max", 3), sec)
        r = m.ride(w, cap.ftp, cap, all_out, False)
        i = next(i for i, s in enumerate(w.expanded()) if s.name.startswith(ALL_OUT_NAME))
        got = r["power"][r["step"] == i].mean()
        assert abs(np.log(got / float(all_out(float(sec))))) < 0.1, sec            # at the curve (day's scatter)
