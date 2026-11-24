"""Heart-rate evidence that riding was severe (physio v16 z2 / z3 split): lag, drift, evidence, the stored shares, the
day histograms by evidence."""
import json

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from goodpush.metrics.severe_hr import SevereHeartRate, SevereSplit
from goodpush.models.physio import PhysioInputs
from goodpush.physio.model import EVIDENCE

KNOTS = np.array([0.70, 0.80, 0.90, 1.00])
SHARE = np.array([0.55, 0.70, 0.85, 1.0])            # power / CP at those heart rates / LTHR (CP at LTHR)


def ride(blocks, tau=50.0, drift_bpm_h=0.0, seed=0):
    """1 Hz power from (seconds, watts) blocks; heart rate = first-order lag (tau) of the HR that power asks for
    (lthr 180, CP 250; above CP it keeps rising), plus drift and noise."""
    rng = np.random.default_rng(seed)
    P = np.concatenate([np.full(s, w, float) for s, w in blocks])
    y = np.log(np.maximum(P, 1.0) / 250.0)
    x = np.interp(y, np.log(SHARE), KNOTS, left=0.6) + np.where(y > 0, y * 0.6, 0.0)
    a = 1.0 - np.exp(-1.0 / tau)
    target = 180.0 * x
    H = lfilter([a], [1.0, a - 1.0], target, zi=[target[0] * (1 - a)])[0]
    H += drift_bpm_h / 3600 * np.arange(P.size) + rng.normal(0, 1.0, P.size)
    return P, H


def measure(rides, rider=None):
    P = np.concatenate([r[0] for r in rides])
    H = np.concatenate([r[1] for r in rides])
    k = len(rides)
    return SevereHeartRate([r[0].size for r in rides], P, H, np.full(k, 190.0), np.full(k, 250.0), np.full(k, 180.0), rider)


WARM, EASY = (900, 160), (600, 150)


def sessions():
    return [ride([WARM] + [(300, 290), (180, 120)] * 5 + [EASY], seed=1),          # VO2max 5 x 5 min
            ride([WARM] + [(900, 225), (300, 140)] * 3 + [EASY], seed=2),          # sweet spot 3 x 15 min
            ride([WARM, (3600, 200), (1200, 230), EASY], seed=3)]                 # steady


def test_lag_found_and_vo2_reps_carry_evidence_sweet_spot_against():
    m = measure(sessions())
    e = m.evidence()
    assert abs(m.rider["tau_s"] - 50) <= 10 and np.isfinite(m.rider["scatter_bpm"])
    n = [r[0].size for r in sessions()]
    vo2, ss = e[:n[0]], e[n[0]:n[0] + n[1]]
    P = np.concatenate([r[0] for r in sessions()])
    assert np.median(vo2[P[:n[0]] == 290][-60:]) > 1          # late in a rep: heart rate well above threshold
    assert np.median(ss[P[n[0]:n[0] + n[1]] == 225]) < 0


def test_drift_taken_off_below_cp():
    """a long steady ride below CP with drift: no evidence for severe once the drift is taken off."""
    m = measure([ride([(7200, 230)], drift_bpm_h=8.0, seed=4), ride([(900, 160), (3600, 200)], seed=5)])
    e = m.evidence()
    assert m.info["drift_bpm_h"]["median"] > 3 and np.median(e[3600:7200]) < 0


def test_given_rider_values_are_used():
    m = measure(sessions(), rider={"tau_s": 40.0, "hr_per_cp_share": 90.0, "scatter_bpm": 6.0})
    m.fractions(150)
    assert m.rider == {"tau_s": 40.0, "hr_per_cp_share": 90.0, "scatter_bpm": 6.0} and "lag_r2" not in m.info
    assert m.info["width_bpm"] == SevereHeartRate.WIDTH_SHARE * 6.0


def test_fractions_conserve_time_and_seconds_without_heart_rate_have_none():
    P, H = sessions()[0]
    H = H.copy()
    H[:600] = np.nan                                        # the strap starts late
    m = SevereHeartRate([P.size], P, H, [190.0], [250.0], [180.0], rider={"scatter_bpm": 6.0})   # no steady riding of its own
    f = m.fractions(150)
    assert f.shape == (1, 150, len(EVIDENCE))
    b = np.minimum((P / 10).astype(int), 149)
    count = np.bincount(b, minlength=150)
    with_hr = np.bincount(b[600:], minlength=150)
    assert np.allclose(f[0].sum(1) * count, with_hr)        # each bin: its seconds with heart rate, no more


def rides_table(n):
    """rides as PhysioInputs reads them: one power histogram each (1 h at 250 W), no HR zones."""
    return pd.DataFrame({"power_hist": [json.dumps((np.eye(200, dtype=int)[25] * 3600).tolist())] * n, "ftp": 250.0,
                         **{f"hz{i}_s": np.nan for i in range(1, 8)}})


def test_day_histograms_by_evidence_conserve_hours():
    rides = rides_table(2)
    share = np.zeros((2, PhysioInputs.BINS, len(EVIDENCE)))
    share[0, 25, 9] = 0.75                                  # ride 0: three quarters with heart rate; ride 1: none
    hist = PhysioInputs.histograms(rides, np.array([0, 1]), 2)
    hr = PhysioInputs.histograms(rides, np.array([0, 1]), 2, share)
    assert hr.shape == (2, PhysioInputs.BINS, len(EVIDENCE)) and np.isclose(hist[0, 25], 1.0)
    assert np.isclose(hr[0].sum(), 0.75) and hr[1].sum() == 0 and np.all(hr.sum(2) <= hist + 1e-12)


def test_split_needs_a_fit_then_keeps_each_ride(tmp_path):
    """no stored fit: no evidence; with one: rides done once, kept in FILE, a new ride added with the rider's values
    learned before."""
    proc = tmp_path / "processed"
    (proc / "streams").mkdir(parents=True)
    rides = pd.DataFrame({"id": ["a", "b", "c"], "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])})
    for i, (P, H) in zip("abc", sessions()):
        np.savez(proc / "streams" / f"{i}.npz", power=P.astype("f4"), heart_rate=H.astype("f4"))
    split = SevereSplit(proc)
    assert not split.fractions(rides.iloc[:2], np.full(2, 180.0), 150).any() and not (proc / SevereSplit.FILE).exists()
    days = pd.date_range("2026-01-01", "2026-01-06", name="date")             # the fit ends before ride c
    pd.DataFrame({"lt1": 190.0, "cp": 250.0}, index=days).to_parquet(proc / SevereSplit.STATES)
    f = split.fractions(rides.iloc[:2], np.full(2, 180.0), 150)
    centre = (f * EVIDENCE).sum(2) / np.maximum(f.sum(2), 1e-12)
    assert centre[0, 29] > 0 and centre[1, 22] < 0              # VO2 reps at 290 W for severe, sweet spot 225 W against
    st = split.stored()
    assert st["ids"] == ["a", "b"] and abs(st["rider"]["tau_s"] - 50) <= 10           # rider learned from every stream
    assert np.array_equal(split.fractions(rides.iloc[:2], np.full(2, 180.0), 150), f)  # read back, not done again
    f3 = split.fractions(rides, np.full(3, 180.0), 150)
    assert np.array_equal(f3[:2], f) and f3[2].any()                           # c: the fit's last day's edges
