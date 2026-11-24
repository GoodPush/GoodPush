"""goodpush.checks: the physio model's forecast on held-out days (PerformanceMarker, PointBacktest, PhysioBaseline,
PhysioCalibrated). Kept from the model ladder's tests (2026-11-13: the comparison models are gone; a flat stand-in
model exercises the backtest)."""
import numpy as np
import pytest

from goodpush.checks import PerformanceMarker, PhysioBaseline, PhysioCalibrated, PointBacktest
from goodpush.physio import PhysioData


def marker_data(n_days=400, seed=0):
    """near-max 3- / 5- / 10- / 20-min efforts every 4th day on a CP 237 W, W' 15.6 kJ curve (20-min equivalent ~250 W)."""
    rng = np.random.default_rng(seed)
    durations = np.array([180.0, 300.0, 600.0, 1200.0])
    day = np.repeat(np.arange(0, n_days, 4), durations.size)
    s = np.tile(durations, day.size // durations.size)
    w = (237.0 + 15600.0 / s) * (1 + 0.02 * rng.standard_normal(day.size))
    return PhysioData(days=np.datetime64("2024-01-01") + np.arange(n_days), power_w=np.array([100.0]),
                      hist_h=np.zeros((n_days, 1)), long_h=np.zeros(n_days), recovery_z=np.zeros((n_days, 4)),
                      week=np.arange(n_days) // 7, effort_day=day, effort_s=s, effort_w=w,
                      effort_kind=np.zeros(day.size, int),
                      reading_day=np.zeros(0, int), reading_w=np.zeros(0), reading_source=np.zeros(0, int))


class FlatModel:
    """a stand-in for the backtest: the median marker before the cut-off, times `scale`, flat ahead."""

    def __init__(self, marker: PerformanceMarker, scale: float = 1.0, name: str = "flat"):
        self.marker, self.scale, self.name = marker, scale, name

    def predict(self, cut: int, days: np.ndarray):
        level = float(np.nanmedian(self.marker.window(cut)))
        return np.full(len(days), self.scale * level), {"level": level}


def test_marker_keeps_near_max_days_and_the_backtest_never_sees_the_future():
    mk = PerformanceMarker(marker_data())
    v = mk.value[np.isfinite(mk.value)]
    assert v.size > 50 and np.all(np.abs(v / 250 - 1) < 0.1)        # max(P20, 0.82 x P5) ~ 250 W
    model, days = FlatModel(mk), np.arange(300, 340)
    before, _ = model.predict(300, days)
    mk.value[300:] *= 1.5                                           # the future changes: the prediction doesn't
    after, _ = model.predict(300, days)
    mk.value[300:] /= 1.5
    assert np.allclose(before, after) and np.all(np.abs(before / 250 - 1) < 0.05)
    out = PointBacktest(model, mk, folds=3, every=30, horizon=28).run(say=lambda s: None)
    assert out["n"] > 0 and out["mae_pct"] < 6 and len(out["folds"]) == 3


def test_backtest_pairs_bootstrap_coverage_and_prospective_cuts():
    mk = PerformanceMarker(marker_data())
    a = PointBacktest(FlatModel(mk), mk, folds=6, every=30, horizon=28).run(say=lambda s: None)
    b = PointBacktest(FlatModel(mk, 1.03, "high"), mk, folds=6, every=30, horizon=28).run(say=lambda s: None)
    assert a["rmse_pct"] >= a["mae_pct"] > 0 and b["bias_pct"] < a["bias_pct"]   # 3 % high: biased low after
    c = PointBacktest.paired(a, b)
    assert c["folds"] == 6 and c["p05"] <= c["diff_pp"] <= c["p95"]
    with pytest.raises(ValueError):                                  # other days: no pairing
        PointBacktest.paired(a, PointBacktest(FlatModel(mk), mk, folds=2, every=30, horizon=28).run(say=lambda s: None))
    cov = a["coverage"]
    assert cov["n"] == a["n"] and 0.4 < cov["inside"] <= 1.0          # leave-one-fold-out 80 % band
    pb = PointBacktest(FlatModel(mk), mk, every=30, since=350)
    assert pb.cuts() == [350, 380] and pb.run(say=lambda s: None)["since"] == 350   # only after the freeze


def test_check_results_stay_on_this_machine():
    from goodpush.sync.peer import PeerRules
    assert PeerRules.kind("data/processed/ladder/physio_calibrated.json") == "skip"


def test_physio_baseline_fits_before_the_cut_only_and_calibrates():
    from test_physio import synthetic
    d = synthetic(84)
    mk = PerformanceMarker(d)
    days = np.arange(60, 84)
    for M in (PhysioBaseline, PhysioCalibrated):
        m = M(mk, np.zeros(84), d, say=lambda s: None)
        m.MAXITER = 200
        before, info = m.predict(60, days)
        d.effort_w[d.effort_day >= 60] *= 1.5                         # the future changes: the prediction doesn't
        m2 = M(mk, np.zeros(84), d, say=lambda s: None)
        m2.MAXITER = 200
        after, _ = m2.predict(60, days)
        d.effort_w[d.effort_day >= 60] /= 1.5
        assert np.allclose(before, after, rtol=1e-6) and np.all(np.isfinite(before)) and info["iterations"] > 0


def test_physio_backtest_scores_held_out_readings_with_and_without_their_bias():
    """2026-11-09: a fitted bias can soak up a wrong LT1 level, so held-out readings are scored both ways."""
    from dataclasses import replace
    from test_physio import synthetic
    d = replace(synthetic(84), reading_day=np.array([10, 40, 50, 70]), reading_w=np.array([245.0, 248, 190, 150]))
    m = PhysioBaseline(PerformanceMarker(d), np.zeros(84), d, say=lambda s: None)
    m.MAXITER = 200
    _, info = m.predict(60, np.arange(60, 84))
    assert info["lt1_n"] == 1 and info["cp_n"] == 0                 # only the reading after the cut
    names, z = m.last
    bias = z[names.index("bias[1]")]
    assert info["bias_alpha1_lt1"] == bias and "bias_garmin_ftp" not in info     # ftp readings aren't checked
    assert any(np.isclose(info["lt1_fit"], abs(sign * info["lt1_raw"] - bias)) for sign in (1, -1))   # one reading
    s = m.readings_summary([{"params": info}, {"params": info}])
    assert s["lt1_readings_n"] == 2 and np.isclose(s["lt1_readings_raw_pct"], 100 * info["lt1_raw"])
    assert np.isclose(s["bias_alpha1_lt1_max_sd"], abs(bias) / 0.08)


def test_physio_backtest_scores_the_block_change_from_training_alone():
    """2026-11-10: training alone gave the direction of 12-week changes but under half their size - the yardstick a
    change to the dose response is judged on: each fold's forecast change against the latest fit's."""
    from test_physio import synthetic
    d = synthetic(224)
    m = PhysioBaseline(PerformanceMarker(d), np.zeros(224), d, say=lambda s: None)
    m.MAXITER = 200
    _, first = m.predict(100, np.arange(100, 110))
    assert first["change_day"] == 100 and np.isfinite([first["cp_at_cut"], first["cp_ahead"], first["lt1_ahead"]]).all()
    _, late = m.predict(190, np.arange(190, 200))
    assert "change_day" not in late                                   # its 84 days run past the data
    s = m.changes_summary([{"params": first}, {"params": late}])
    assert s["cp_change_n"] == s["map_change_n"] == 1 and s["cp_change_corr"] is None     # one block: no correlation
    assert np.isfinite(s["cp_change_mae_pp"]) and s["cp_change_none_pp"] >= 0
    assert m.changes_summary([{"params": late}]) == {}
