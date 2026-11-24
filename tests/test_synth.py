"""goodpush.synth: synthetic riders - their truth, their rides and nights, and the check of goodpush's analysis."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from goodpush.synth import RiderProfile, SyntheticRider
from goodpush.synth.riding import RideMaker
from goodpush.synth.truth import Capacity, SimpleTruth


def short_rider(tmp_path=None, **kw) -> SyntheticRider:
    """a quick rider: a few months, one illness and one break likely, tests often."""
    p = RiderProfile(**{"days": 120, "end": "2026-06-30", "seed": 4, "illness_per_year": 6, "breaks_per_year": 3,
                        "ftp_test_weeks": 4, "ramp_weeks": 4, "lsct_weeks": 3, **kw})
    return SyntheticRider(p, progress=lambda *a: None)


def test_profile_round_trip_and_checks(tmp_path):
    p = RiderProfile(cp_w=250, map_w=310, key_mix={"vo2max": 1.0})
    f = tmp_path / "p.toml"
    f.write_text(p.text(), encoding="utf-8")
    assert RiderProfile.load(f) == p
    f.write_text("cp_watts = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        RiderProfile.load(f)
    with pytest.raises(ValueError, match="LT1 < CP"):
        RiderProfile(cp_w=300, map_w=290).checked()


def test_model_truth_is_the_physio_model_on_the_same_rides():
    """day by day with the pipeline's own histograms (PhysioInputs.power_hours) = PhysioModel.simulate of them."""
    from goodpush.io.fit import read_messages
    from goodpush.metrics.power import power_histogram
    from goodpush.models.physio import PhysioInputs
    from goodpush.physio import PhysioModel
    r = short_rider(days=60, illness_per_year=12)
    assert r.illnesses                                   # an illness week's larger noise is in it too
    out = r.run()
    t = r.truth
    days = r.days
    H = np.zeros((len(days), PhysioInputs.BINS))
    L = np.zeros(len(days))
    ids = pd.DatetimeIndex(out["schedule"].index[out["schedule"]["kind"] != "rest"])
    for k, ride in zip(days.get_indexer(ids), out["rides"]):     # sixty days
        rec = read_messages(ride["fit"], [20])[20]               # the power as the FIT file holds it
        p = np.array([m.get(7, 0) for m in rec], float)
        H[k] = PhysioInputs.power_hours(power_histogram(p)[None])[0]
        L[k] = PhysioInputs.long_hours([len(p)])[0]
    model = PhysioModel(replace(t.data, hist_h=H, long_h=L), PhysioInputs.EFFORT_S)
    X, F, _ = model.simulate(t.z, t.carry0)                     # from the settled start (pool, fatigue)
    cp = np.asarray(PhysioModel.capacity(X)["cp"])
    np.testing.assert_allclose(cp, out["morning"]["cp"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(F), out["morning"]["fatigue"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert set(t.params()) <= set(model.spec.names()) and not any(n.startswith("eta") for n in t.params())


@pytest.mark.parametrize("truth", ["model", "simple"])
def test_a_high_volume_rider_starts_in_balance(truth):
    """a pro's start is held by their usual training (truth `settle`): at steady hours CP stays near the profile's,
    instead of running off because their hours are far above what the prior's centre sustains."""
    p = RiderProfile(truth=truth, days=120, end="2026-06-30", seed=2, cp_w=400, map_w=470, lt1_w=300, pmax_w=1450,
                     fade=0.05, max_hr=192, lthr=174, resting_hr=42, hours_start=20, hours_end=20,
                     ride_days=list(("mon", "tue", "wed", "thu", "fri", "sat", "sun")), key_days=["tue", "thu", "sat"],
                     illness_per_year=0, breaks_per_year=0, draw_dynamics=False)
    r = SyntheticRider(p, progress=lambda *a: None)
    out = r.run()
    cp = out["morning"]["cp"]
    assert cp.iloc[0] == pytest.approx(400) and abs(cp.iloc[-1] / 400 - 1) < 0.08
    s = out["schedule"]
    assert s["minutes"][s["kind"] == "long"].max() <= SyntheticRider.LONG_MAX
    keys = s.index[s["kind"].isin(["threshold", "vo2max", "sweetspot", "anaerobic"])]
    w = r.workout(keys[0], s.loc[keys[0], "kind"], float(s.loc[keys[0], "minutes"]), {})
    assert w.minutes > SyntheticRider.KEY_MINUTES + 30                    # the session with more Z2 around it


def test_simple_truth_trains_and_detrains():
    p = RiderProfile(truth="simple")
    days = pd.date_range("2026-01-01", periods=150)
    t = SimpleTruth(p, days, [], np.random.default_rng(0))
    start = t.morning()
    assert start.cp == pytest.approx(p.cp_w) and start.lt1 == pytest.approx(p.lt1_w)
    assert start.map == pytest.approx(p.map_w, rel=1e-6)                  # its own curve puts 5-min power there
    for _ in range(60):
        t.advance([np.full(3 * 3600, 0.85 * p.cp_w)])                      # three hours of threshold-ish work a day
    trained = t.morning()
    for _ in range(60):
        t.advance([])
    rested = t.morning()
    assert trained.cp > start.cp > rested.cp
    assert trained.form < rested.form                                      # tired after the block, fresh after rest


def test_beats_give_the_alpha1_their_intensity_sets():
    """metrics.dfa on the generated beats (as FIT stores them, in ms): ~1 easy, 0.75 at LT1, 0.5 at CP."""
    from goodpush.metrics import dfa
    m = RideMaker(RiderProfile(), np.random.default_rng(0))
    cap = Capacity(cp=215, w_prime=15000, pmax=820, fade=0.08, map=270, lt1=155, ftp=208, fatigue=0, form=1)
    for watts, want in ((100, 1.02), (155, 0.75), (215, 0.5)):
        power = np.full(1800, float(watts))
        b = np.array(m.beats.beats(m.heart.hr(power, cap, 0.0), power, cap, 0.0))
        a, _ = dfa.series(np.round(b[:, 1], 3), 0.0, np.arange(600, 1800, 30.0))
        assert np.nanmedian(a) == pytest.approx(want, abs=0.06)
    hr = m.heart.hr(np.full(600, 215.0), cap, 0.0)
    assert hr[-1] == pytest.approx(166 + 600 / 3600 * 4, abs=1.5)        # held at CP: threshold heart rate
    hr = m.heart.hr(np.full(3 * 3600, 215.0), cap, 0.0)                    # drift over the first hour, then level
    assert hr[3600] - hr[600] == pytest.approx(4 * 5 / 6, abs=0.1) and hr[-1] == pytest.approx(hr[3600])


def test_reps_last_only_as_long_as_w_prime():
    """RideMaker.paced: reps above CP run W' down (models.stamina.RideStamina); once it's spent the rest of that rep
    is ridden just under CP, so no stretch goes past empty - and riding below CP is left as it is."""
    from goodpush.models.stamina import RideStamina
    m = RideMaker(RiderProfile(), np.random.default_rng(0))
    cap = Capacity(cp=215, w_prime=15000, pmax=820, fade=0.08, map=270, lt1=155, ftp=208, fatigue=0, form=1)
    rep = np.r_[np.full(240, 290.0), np.full(120, 120.0)]                  # 4 min at 290 W (75 W over CP), 2 min easy
    p = np.r_[np.full(600, 150.0), np.tile(rep, 5)]
    out = m.paced(p, cap)
    assert (RideStamina(out, cap.cp, cap.w_prime, cap.lt1).w_balance() >= -1.0).all()
    assert out[:600].tolist() == p[:600].tolist()                         # the warm-up as it was
    hard = out[600:].reshape(5, -1)[:, :240]
    assert hard[0].mean() > hard[-1].mean() and hard[-1].min() == pytest.approx(m.EASE * cap.cp)
    assert out.max() <= p.max()


def test_a_ramp_never_beats_the_curve():
    """RideMaker.ramp: the ramp ends where its mean over some effort duration reaches all-out for it (2026-11-22: it
    ended at the 2-min power and its last 5-20 min sat above the 5-20 min curve)."""
    from goodpush.models.physio import PhysioInputs
    from goodpush.physio import PowerCurve
    m = RideMaker(RiderProfile(), np.random.default_rng(0))
    cap = Capacity(cp=215, w_prime=15000, pmax=820, fade=0.08, map=270, lt1=155, ftp=208, fatigue=0, form=1)
    all_out = lambda t: PowerCurve.power(np.asarray(t, float), cap.cp, cap.w_prime, cap.pmax, cap.fade)  # noqa: E731  hygiene: ok
    p = np.r_[np.full(600, 120.0), np.repeat(100.0 + 20.0 * np.arange(30), 60)]    # 10 min easy, +20 W a minute
    m.ramp(p, 600, p.size, cap.ftp, all_out)
    c = np.r_[0.0, np.cumsum(p[600:])]
    for d in PhysioInputs.EFFORT_S[:8]:
        assert ((c[d:] - c[:-d]) / d).max() <= float(all_out(float(d))) * np.exp(3 * m.ALL_OUT_SD)
    assert p[-1] == pytest.approx(m.EASY_AFTER * cap.ftp) and p[600:].max() > cap.cp


def test_an_all_out_effort_outdoors_stays_at_the_curve():
    """2026-11-22: the easy ride's all-out effort was placed before the terrain and surges, which scaled it up to 19 %
    over the curve (3-5 min 'maxed' bests above it); placed last, nothing moves it."""
    from goodpush.physio import PowerCurve
    from goodpush.planning.workouts import WorkoutLibrary
    cap = Capacity(cp=215, w_prime=15000, pmax=820, fade=0.08, map=270, lt1=155, ftp=208, fatigue=0, form=1)
    all_out = lambda t: PowerCurve.power(np.asarray(t, float), cap.cp, cap.w_prime, cap.pmax, cap.fade)  # noqa: E731  hygiene: ok
    m = RideMaker(RiderProfile(), np.random.default_rng(1))
    w = WorkoutLibrary.easy(pd.Timestamp("2026-06-01"), "long", 180)
    worst = 0.0
    for _ in range(20):                                       # a few rides' terrain
        p = m.ride(w, cap.ftp, cap, all_out, True, 300, True)["power"]
        c = np.r_[0.0, np.cumsum(p)]
        worst = max(worst, ((c[300:] - c[:-300]) / 300).max() / float(all_out(300.0)))
    assert worst <= np.exp(3 * m.ALL_OUT_SD)


def test_indoor_surges_reach_near_the_curve():
    """surge_share: indoor easy rides get the outdoor surges (2026-11-22: an outdoor rider's few near-curve efforts
    pinned the fitted ceiling, the indoor rider's CP floated +18 %); no surge beats the curve for its length (they
    beat the 15-30 s curve: W' -19 %, Pmax +28 %); 0 leaves the schedule as it was."""
    from goodpush.physio import PowerCurve
    all_out = lambda t: PowerCurve.power(np.asarray(t, float), 215.0, 15000.0, 820.0, 0.08)  # noqa: E731  hygiene: ok
    m = RideMaker(RiderProfile(), np.random.default_rng(0))
    p = m.surged(np.full(3 * 3600, 150.0), all_out)
    c = np.r_[0.0, np.cumsum(p)]
    for d in (5, 10, 15, 20):
        assert ((c[d:] - c[:-d]) / d).max() <= float(all_out(float(d))) * np.exp(3 * m.ALL_OUT_SD)
    assert p.max() >= 0.5 * float(all_out(20.0)) and (p == 150.0).mean() > 0.9
    s0, s1 = short_rider().schedule(), short_rider(surge_share=1.0).schedule()
    pd.testing.assert_frame_equal(s0.drop(columns="surges"), s1.drop(columns="surges"))
    easy = s1["kind"].isin(["endurance", "long"])
    assert (s0["surges"] == s0["outdoor"]).all() and s1["surges"][easy].all() and not s1["surges"][~easy].any()


def test_schedule_tests_illness_and_comeback():
    r = short_rider()
    s = r.schedule()
    a, b = r.spans(r.illnesses + r.breaks)
    off = RideMaker.covered(len(s), a, b)
    assert (s["kind"][off] == "rest").all() and (s["minutes"][off] == 0).all()
    for kind in ("test", "ramp", "lsct"):
        d = s.index[s["kind"] == kind]
        assert len(d) >= 2 and (np.diff(d).astype("timedelta64[D]").astype(int) >= 7).all()
    notes = s[s["note"] != ""]
    assert len(notes) == len(r.illnesses)
    for (start, k), (day, _) in zip(r.illnesses, notes.iterrows()):       # the first ride back says so
        assert day >= start + pd.Timedelta(days=k) and s.loc[day, "kind"] == "endurance"


def test_shared_pieces_unchanged():
    """the parts split out for the synthetic rider work as before: hours per bin, a ride's activities row."""
    from goodpush.models.physio import PhysioInputs
    from goodpush.ride.recorder import RideRecorder
    H = np.zeros((1, 200))
    H[0, [0, 5, 149, 160, 199]] = [600, 3600, 360, 360, 36]
    h = PhysioInputs.power_hours(H)
    assert h.shape == (1, 150) and h[0, 0] == 0 and h[0, 5] == 1.0 and h[0, 149] == pytest.approx(0.21)
    assert PhysioInputs.long_hours([3600, 3 * 3600]).tolist() == [0.0, 1.0]
    rec = RideRecorder(1_700_000_000, "Test", [(1_700_000_000 + i, 200.0, 90.0, 140.0, 9.0, 0, 1.0) for i in range(60)])
    row = rec.row("Ride", "note")
    assert row["id"] == rec.id and row["type"] == "Ride" and row["description"] == "note"
    assert row["filename"] == f"{rec.id}.fit" and row["avg_watts_csv"] == 200.0


def test_synthetic_rider_through_goodpush_and_scored(tmp_path):
    """a short rider written, analysed like anyone's (goodpush run + the physio model's best fit) and scored."""
    from goodpush.synth.check import SynthCheck
    r = short_rider(plan=False, days=90, illness_per_year=0, breaks_per_year=0)
    proj = r.write(tmp_path / "rider")
    with pytest.raises(FileExistsError):
        r.write(tmp_path / "rider")
    assert (proj.root / "synth" / "truth.parquet").exists() and list(proj.path("garmin").rglob("????-??-??.json"))
    out = SynthCheck(proj.root).run(progress=lambda *a: None)
    assert set(out["states"]) == set(SynthCheck.STATES) and out["params"]
    assert out["states"]["ftp"]["abs_pct"] < 25                          # in the right range, not a fixed number
    assert out["heart"]["resting_hr"]["found"] is not None
    assert out["thresholds"]["alpha1_lt1"]["n"] > 0 and out["thresholds"]["ramp_lt1"]["n"] > 0
    acts = pd.read_parquet(proj.processed / "activities.parquet")
    info = SynthCheck(proj.root).info
    assert acts["ramp_top_w"].notna().sum() == sum(t["type"] == "ramp" for t in info["tests"]) > 0
    assert "capacity" in SynthCheck.text(out)
