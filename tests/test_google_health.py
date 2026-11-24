"""Google Health (Fitbit): parsing the saved data points, merging with Garmin, sync and sign-in.
The data points follow the shapes in the Google Health API reference (no live account needed)."""
import json
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from goodpush.io.google_health import GoogleHealthDays
from goodpush.sync import google_health as gh_mod
from goodpush.sync.google_health import GoogleHealth, GoogleHealthError

D = lambda s: {"year": int(s[:4]), "month": int(s[5:7]), "day": int(s[8:10])}  # noqa: E731


def points():
    return {
        "sleep": [{"name": f"users/me/dataTypes/sleep/dataPoints/{i}", "sleep": {
            "interval": {"startTime": start, "civilEndTime": {"date": D(end)}},
            "summary": {"minutesAsleep": asleep, "minutesAwake": 30, "stagesSummary": [
                {"type": "DEEP", "minutes": 80, "count": 4}, {"type": "LIGHT", "minutes": 250, "count": 20},
                {"type": "REM", "minutes": 90, "count": 5}, {"type": "AWAKE", "minutes": 30, "count": 12}]}}}
            for i, (start, end, asleep) in enumerate([("2026-10-20T06:30:00Z", "2026-10-21", 420),
                                                      ("2026-10-21T13:00:00Z", "2026-10-21", 40),   # nap
                                                      ("2026-10-21T06:00:00Z", "2026-10-22", 450)])],
        "daily-resting-heart-rate": [{"name": f"r{i}", "dailyRestingHeartRate": {"date": D(d), "beatsPerMinute": v}}
                                     for i, (d, v) in enumerate([("2026-10-21", 52), ("2026-10-22", "54")])],
        "daily-heart-rate-variability": [{"name": "h1", "dailyHeartRateVariability": {
            "date": D("2026-10-21"), "averageHeartRateVariabilityMilliseconds": 48.5,
            "deepSleepRootMeanSquareOfSuccessiveDifferencesMilliseconds": 61.0}}],
        "daily-respiratory-rate": [{"name": "b1", "dailyRespiratoryRate": {"date": D("2026-10-21"), "breathsPerMinute": 14.2}}],
        "daily-oxygen-saturation": [{"name": "o1", "dailyOxygenSaturation": {"date": D("2026-10-22"), "averagePercentage": 96.4}}],
        "daily-sleep-temperature-derivations": [{"name": "t1", "dailySleepTemperatureDerivations": {
            "date": D("2026-10-22"), "nightlyTemperatureCelsius": 34.9, "baselineTemperatureCelsius": 34.4}}],
    }


def save(folder, pts):
    folder.mkdir(parents=True, exist_ok=True)
    for t, xs in pts.items():
        (folder / f"{t}.json").write_text(json.dumps({x["name"]: x for x in xs}))


def test_days_from_google_health_points(tmp_path):
    save(tmp_path, points())
    d = GoogleHealthDays.load(tmp_path, "America/Los_Angeles")
    r21, r22 = d.loc["2026-10-21"], d.loc["2026-10-22"]
    assert r21["sleep_h"] == 7 and r21["deep_h"] == pytest.approx(80 / 60) and r21["awake_h"] == 0.5   # nap ignored
    assert r21["bedtime_h"] == pytest.approx(23.5)          # 06:30 UTC = 23:30 the evening before
    assert (r21["rhr"], r22["rhr"], r21["hrv"], r21["hrv_deep"], r21["respiration"]) == (52, 54, 48.5, 61, 14.2)
    assert r22["sleep_h"] == 7.5 and r22["spo2"] == 96.4 and r22["skin_temp_dev"] == pytest.approx(0.5)
    assert len(GoogleHealthDays.load(tmp_path / "none")) == 0


def test_garmin_wins_and_hrv_is_never_mixed(tmp_path):
    save(tmp_path, points())
    gh = GoogleHealthDays.load(tmp_path)
    days = pd.DatetimeIndex(["2026-10-21", "2026-10-22"], name="date")
    garmin = pd.DataFrame({"sleep_h": [6.0, np.nan], "rhr": [50.0, np.nan], "hrv": [np.nan, 40.0]}, index=days)
    m = GoogleHealthDays.merge(garmin, gh)
    assert list(m["sleep_h"]) == [6.0, 7.5] and list(m["rhr"]) == [50.0, 54.0]       # Garmin wins, Fitbit fills
    assert np.isnan(m.loc["2026-10-21", "hrv"]) and m.loc["2026-10-21", "hrv_deep"] == 61   # Garmin has HRV: no mixing
    only = GoogleHealthDays.merge(garmin.drop(columns="hrv"), gh)                    # no Garmin HRV: Fitbit's
    assert only.loc["2026-10-21", "hrv"] == 48.5 and "spo2" in only


def test_sync_pages_and_merges_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    calls = []
    pts = points()

    def fake_get(self, path, params):
        calls.append((path, params))
        t = path.split("/")[1]
        xs = pts.get(t, [])
        if t == "sleep" and not params.get("pageToken"):                    # two pages
            return {"dataPoints": xs[:2], "nextPageToken": "p2"}
        return {"dataPoints": xs[2:] if t == "sleep" else xs}
    monkeypatch.setattr(GoogleHealth, "get", fake_get)
    folder = tmp_path / "gh"
    out = GoogleHealth().sync(folder, since="2026-10-01", progress=lambda *a: None)
    assert out["new"] == 9 and out["types"]["sleep"] == 3
    assert ('dataTypes/daily-resting-heart-rate/dataPoints',
            {"filter": 'dailyRestingHeartRate.date >= "2026-10-01"', "pageSize": 1000}) in calls
    assert any(p.get("filter") == 'sleep.interval.civil_end_time >= "2026-10-01"' and p["pageSize"] == 25 for _, p in calls)
    assert len(json.loads((folder / "sleep.json").read_text())) == 3
    assert GoogleHealth().sync(folder, since="2026-10-01", progress=lambda *a: None)["new"] == 0   # nothing new
    assert GoogleHealth.info()["last_sync"]["start"] == "2026-10-01"


def test_sign_in_with_pkce(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path))
    with pytest.raises(ValueError):
        GoogleHealth.save_client("not-a-client", "x")
    g = GoogleHealth()
    with pytest.raises(GoogleHealthError):
        g.auth_url("http://127.0.0.1:8765/oauth/google")                    # no client yet
    GoogleHealth.save_client("123-abc.apps.googleusercontent.com", "s3cret")
    q = urllib.parse.parse_qs(urllib.parse.urlparse(g.auth_url("http://127.0.0.1:8765/oauth/google")).query)
    assert q["code_challenge_method"] == ["S256"] and q["access_type"] == ["offline"] and len(q["scope"][0].split()) == 2
    sent = {}

    def fake_token(form):
        sent.update(form)
        return {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}
    monkeypatch.setattr(GoogleHealth, "token_request", staticmethod(fake_token))
    with pytest.raises(GoogleHealthError):
        g.finish("code", "wrong-state")
    g.finish("the-code", q["state"][0])
    assert sent["code_verifier"] and sent["redirect_uri"] == "http://127.0.0.1:8765/oauth/google"
    assert GoogleHealth.connected() and g.access_token() == "at"
    GoogleHealth.forget()
    assert not GoogleHealth.connected() and not (tmp_path / gh_mod.CLIENT_FILE).exists()
