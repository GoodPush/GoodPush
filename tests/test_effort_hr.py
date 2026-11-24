"""Measured vs inferred: a best effort counts as all-out when heart rate reached your maximum then (2026-11-04)."""
import numpy as np
import pandas as pd
import pytest

from goodpush.metrics.effort_hr import EffortHeartRate as E


def ride(seconds, effort_at, effort_s, effort_w, hr_peak, base_w=150.0, coast=0):
    """1 Hz power / HR: easy riding, one effort; heart rate climbs through it to `hr_peak` (a lag after its end)."""
    p = np.full(seconds, base_w)
    h = np.full(seconds, 130.0)
    p[effort_at:effort_at + effort_s] = effort_w
    if coast:
        p[effort_at:effort_at + coast] = 0.0
    end = effort_at + effort_s
    h[effort_at:end] = np.linspace(140, hr_peak - 3, effort_s)
    h[end:end + 20] = hr_peak                                     # the peak lags the effort
    return p, h


def test_best_windows_and_the_heart_rate_at_their_end():
    rides = [ride(3600, 1200, 300, 300.0, 192), ride(3000, 600, 1200, 250.0, 180), ride(400, 100, 60, 500.0, 185)]
    n = [len(p) for p, _ in rides]
    P = np.concatenate([p for p, _ in rides])
    H = np.concatenate([h for _, h in rides])
    t = E.from_streams(["a", "b", "c"], n, P, H)
    assert t.loc[0, "effort_w_300"] == pytest.approx(300.0) and t.loc[0, "effort_hr_300"] == 192    # its own window
    assert t.loc[1, "effort_w_1200"] == pytest.approx(250.0) and t.loc[1, "effort_hr_1200"] == 180     # (5 s after the end)
    assert t.loc[2, "effort_hr_60"] == 185                        # 1 min: the peak after it counts (lag)
    assert np.isnan(t.loc[2, "effort_w_1200"])                    # a ride shorter than the effort: none
    assert t.loc[1, "effort_hr_60"] < 185                         # windows never reach into the next ride


def test_maxed_needs_your_maximum_then_and_a_steady_effort():
    days = pd.date_range("2025-01-01", periods=800)
    rides = pd.DataFrame({"date": days, "max_hr": np.where(days.year == 2025, 180.0, 176.0)})
    ref = E.max_hr_then(rides, 187.0)
    assert ref.iloc[100] == pytest.approx(180, abs=1) and ref.iloc[-1] < ref.iloc[100]    # it falls with age
    e = pd.DataFrame({"date": [days[100]] * 3, "max_hr": 180.0,
                      "effort_hr_300": [176.0, 173.0, 177.0], "effort_coast_300": [0.0, 0.0, 0.2],
                      "effort_hr_60": [171.0, 169.0, np.nan]})
    m = E.maxed(pd.concat([rides, e], ignore_index=True), 187.0).iloc[-3:]
    assert m[300].tolist() == [True, False, False]                # within 5 bpm; 7 short; coasting
    assert m[60].tolist() == [True, False, False]                 # 1 min: 10 bpm (heart rate lags)


def test_max_hr_seen_later_was_already_there():
    days = pd.date_range("2025-01-01", periods=400)
    rides = pd.DataFrame({"date": days, "max_hr": np.where(days.year == 2025, 180.0, 186.0)})
    ref = E.max_hr_then(rides, 180.0)
    assert ref.iloc[100] == pytest.approx(186, abs=1)              # a low year is a floor, not a lower maximum
    assert (np.diff(ref.to_numpy()) <= 1e-9).all()                 # never rises


def test_a_maxed_effort_is_all_out_for_the_model():
    from goodpush.models.physio import PhysioInputs
    from goodpush.physio import KINDS
    days = pd.date_range("2026-10-01", periods=3)
    cols = {f"mmp_{s}": [400.0 * (60 / max(s, 60)) ** 0.15] * 2 for s in PhysioInputs.EFFORT_S}
    rides = pd.DataFrame({"date": [days[1], days[1]], "workout": ["vo2max", ""], "moving_s": 4000.0,
                          "duration_s": 4000.0, "power_coverage": 1.0, "max_hr": 185.0,
                          "effort_hr_300": [184.0, 140.0], "effort_coast_300": 0.0, **cols})
    rides.loc[1, [f"mmp_{s}" for s in PhysioInputs.EFFORT_S]] *= 0.5        # the easy ride's bests are lower
    day, s, w, kind = PhysioInputs.efforts(rides, np.array([1, 1]), 185.0)
    k = dict(zip(s, kind))
    assert KINDS[k[300]] == "maxed" and KINDS[k[60]] == "hard"     # 5 min verified; the rest the day's kind
    day, s, w, kind = PhysioInputs.efforts(rides, np.array([1, 1]), None)
    assert "maxed" not in {KINDS[x] for x in kind}                  # no maximum heart rate known: none


def test_the_rider_page_shows_what_was_measured():
    from goodpush.models.thresholds import measured
    days = pd.date_range("2026-01-01", periods=300)
    acts = pd.DataFrame({"id": [str(i) for i in range(300)], "date": days, "sport": "ride", "max_hr": 184.0,
                         "effort_w_300": 250.0, "effort_hr_300": 160.0, "effort_coast_300": 0.0,
                         "effort_w_60": 400.0, "effort_hr_60": 150.0})
    acts.loc[200, ["effort_w_300", "effort_hr_300"]] = [300.0, 183.0]           # an honest 5 min, HR at max
    acts.loc[210, ["effort_w_300", "effort_hr_300"]] = [330.0, 165.0]           # higher, but not all-out
    m = measured(acts, days[-1])
    assert m["300"]["w"] == 300 and m["300"]["date"] == str(days[200].date()) and m["300"]["hr"] == 183
    assert m["60"] is None and m["1200"] is None                                  # nothing confirmed: inferred
