"""The fitted FTP model: recovers a known FTP path from synthetic rides, caches its fit, falls back."""
import json

import numpy as np
import pandas as pd
import pytest

from goodpush.models import ftp_model
from goodpush.models.ftp_model import DURATIONS, FtpModel

RATIO = dict(zip(DURATIONS, (1.25, 1.15, 1.05, 0.97, 0.90, 0.86)))     # best power / FTP per duration


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(3)
    days = pd.date_range("2024-01-01", periods=500)
    ftp = pd.Series(np.interp(np.arange(500), [0, 150, 300, 500], [220, 250, 230, 240]), index=days)
    ride_days = days[rng.random(500) < 0.6]
    rows = []
    for d in ride_days:
        f = ftp[d]
        effort = rng.uniform(0.85, 1.0)                    # not every ride is all-out
        rows.append({"date": d, "duration_s": 5400, "avg_power": 0.7 * f, "np_used": 0.75 * f, "power_coverage": 1.0,
                     "avg_hr": 50 + (180 - 50) * 0.7 * f / f * rng.uniform(0.97, 1.03),     # 70 % of FTP at 70 % HRR
                     **{f"mmp_{k}": r * f * effort for k, r in RATIO.items()}})
    rides = pd.DataFrame(rows)
    labels = ftp.iloc[::25].round()
    return days, rides, ftp, labels


def test_fit_recovers_a_known_ftp(synthetic):
    days, rides, ftp, labels = synthetic
    lb, hr = FtpModel.log_bests(rides, days), FtpModel.hr_threshold(rides, days, pd.Series(50.0, index=days), 180.0)
    m = FtpModel.fit(lb, hr, days, labels, 180.0)
    pred = pd.Series(m.predict_from(lb, hr), index=days)
    assert m.mae < 6 and m.lthr == 180.0                  # LTHR is what it was given, not fitted
    h = pd.Series(hr, index=days)[days[100:]]             # power at threshold HR: ~ FTP on these rides
    assert np.nanmean(np.abs(h - ftp[days[100:]])) < 15
    late = days[100:]                                     # after the first weeks of rides
    assert np.nanmean(np.abs(pred[late] - ftp[late])) < 10
    assert np.isnan(pred.iloc[0])                         # day 1: no rides before it


def test_labels_and_cached_fit(synthetic, tmp_path, monkeypatch):
    days, rides, ftp, labels = synthetic
    w = pd.DataFrame({"ftp_garmin": [220.0, 220.0, 225.0, 230.0], "ftp_garmin_auto": [1, 1, 0, 1]},
                     index=pd.to_datetime(["2024-01-05", "2024-01-06", "2024-02-01", "2024-03-01"]))
    lab = FtpModel.labels(w, {"2024-03-01": 250, "2024-04-01": 240})
    assert lab.to_dict() == {pd.Timestamp("2024-01-05"): 220.0, pd.Timestamp("2024-03-01"): 250.0,
                             pd.Timestamp("2024-04-01"): 240.0}   # changes only, auto only, yours win
    lb, hr = FtpModel.log_bests(rides, days), FtpModel.hr_threshold(rides, days, pd.Series(50.0, index=days), 180.0)
    assert FtpModel.cached(tmp_path, lab, (lb, hr, days), 180.0, progress=lambda *a: None) is None   # < 10 labels
    m = FtpModel.cached(tmp_path, labels, (lb, hr, days), 180.0, progress=lambda *a: None)
    assert json.loads((tmp_path / ftp_model.CACHE).read_text())["labels_key"] == m.labels_key
    monkeypatch.setattr(FtpModel, "fit", classmethod(lambda *a, **k: pytest.fail("refitted")))
    assert FtpModel.cached(tmp_path, labels, (lb, hr, days), 180.0, progress=lambda *a: None) == m     # from the cache
    monkeypatch.undo()
    m2 = FtpModel.cached(tmp_path, labels, (lb, hr, days), 175.0, progress=lambda *a: None)
    assert m2.lthr == 175.0 and m2.labels_key != m.labels_key                                    # new LTHR: refit


def test_ramp_tests_count_more_and_the_model_lt2_can_be_the_ftp(synthetic, tmp_path):
    """2026-11-03: tests are the cornerstone of validation - a ramp test is an FTP value for the fit, weighted
    above Garmin's, at the physio model's LT2 that day (the ramp's own alpha1 LT2 reads high); auto_method "model"
    makes FTP the model's LT2 itself."""
    from types import SimpleNamespace

    from goodpush.analyze import build_ftp
    from goodpush.metrics.power import MMP_DURATIONS
    days, rides, ftp, labels = synthetic
    test_days = pd.to_datetime(["2024-06-01", "2024-09-01"])
    r = pd.DataFrame({"date": list(test_days) + [pd.Timestamp("2024-07-01")], "ramp_lt2_w": [270.0, 280.0, np.nan]})
    cp = pd.Series([246.0], index=pd.to_datetime(["2024-06-01"]))
    t = FtpModel.tests(r, cp)
    assert t.to_dict() == {pd.Timestamp("2024-06-01"): 246.0, pd.Timestamp("2024-09-01"): 280.0}   # model's, else raw
    lb, hr = FtpModel.log_bests(rides, days), FtpModel.hr_threshold(rides, days, pd.Series(50.0, index=days), 180.0)
    off = pd.Series(ftp[test_days].to_numpy() + 30, index=test_days)          # two tests well above Garmin's labels
    lab = pd.concat([labels, off]).sort_index()
    plain = FtpModel.fit(lb, hr, days, lab, 180.0)
    heavy = FtpModel.fit(lb, hr, days, lab, 180.0, weights=pd.Series(10.0, index=test_days))
    at = days.get_indexer(test_days + pd.Timedelta(days=1))
    assert (heavy.predict_from(lb, hr)[at] - off.to_numpy()).__abs__().mean() < \
        (plain.predict_from(lb, hr)[at] - off.to_numpy()).__abs__().mean()        # weighted tests pull harder
    pd.DataFrame({"cp": np.linspace(240.0, 250.0, 30)},
                 index=pd.date_range("2026-03-02", periods=30, name="date")).to_parquet(tmp_path / ftp_model.STATES)
    d2 = pd.date_range("2026-01-01", "2026-03-31")
    acts = pd.DataFrame({"date": d2[::3], "sport": "ride", "has_power": True,
                         **{f"mmp_{d}": 250.0 * (60 / max(d, 60)) ** 0.07 for d in MMP_DURATIONS}})
    proj = SimpleNamespace(processed=tmp_path, cfg={"ftp": {"mode": "auto", "auto_method": "model", "window_days": 90}})
    out, _ = build_ftp(proj, acts, d2)
    assert out.loc["2026-03-02", "ftp"] == 240 and out.loc["2026-03-31", "ftp"] == 250        # the model's LT2
    assert out.loc["2026-01-31", "ftp"] == out.loc["2026-01-31", "ftp_auto"] > 0              # before it: the rest


def test_a_ramp_lt2_at_exhaustion_is_only_a_loose_cap(tmp_path):
    """alpha1 crossed 0.5 within a few watts of a ramp's top and a beat of its max HR - not a threshold
    (LT2 sits well below the top). The model still takes it, as the upper bound every ramp threshold is (a loose
    one here, harmless); it isn't an FTP value for the fitted FTP model, and the pages say why."""
    from goodpush.models.physio import PhysioInputs
    from goodpush.models.ramp import RampTest
    r = pd.DataFrame({"date": pd.to_datetime(["2026-11-03", "2026-12-15", "2027-01-20"]),
                      "ramp_top_w": [284.0, 290.0, 295.0], "ramp_max_hr": [180.0, 181.0, 181.0],
                      "ramp_lt2_w": [281.0, 250.0, 252.0], "ramp_lt2_hr": [179.4, 168.0, 179.0]})
    why = RampTest.lt2_check(r)
    assert "top" in why[0] and why[1] == "" and "highest heart rate" in why[2]
    assert FtpModel.tests(r).to_dict() == {pd.Timestamp("2026-12-15"): 250.0}
    days = pd.date_range("2026-11-01", "2027-01-31")
    daily = pd.DataFrame(index=days)
    d, w, src = PhysioInputs.readings(r, days.get_indexer(r["date"]), daily, None)
    from goodpush.models.physio import SOURCES
    names = [SOURCES[i].name for i in src]
    assert sorted(zip(names, w)) == [("ramp_lt2", 250.0), ("ramp_lt2", 252.0), ("ramp_lt2", 281.0)]   # the top: an effort
