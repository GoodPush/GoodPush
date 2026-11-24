"""Overnight HRV readings: from FIT files and Garmin API days into streams.h5, then per-night quantiles."""
import json
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest

from goodpush.io import fit, streamstore
from goodpush.io.hrv import NightHrv


def hrv_fit(t0_unix: int, values) -> bytes:
    """Tiny FIT file of hrv_value messages (371: 253 timestamp, 0 value in ms * 128), 5 min apart."""
    recs = bytearray([0x40, 0, 0]) + struct.pack("<H", 371) + bytes([2, 253, 4, 0x86, 0, 2, 0x84])
    for i, v in enumerate(values):
        recs += bytes([0x00]) + struct.pack("<IH", t0_unix - fit.FIT_EPOCH_OFFSET + 300 * i, int(v * 128))
    return struct.pack("<BBHI4sH", 14, 0x20, 2100, len(recs), b".FIT", 0) + bytes(recs) + b"\x00\x00"


def api_day(start: str, values) -> dict:
    t = pd.date_range(start, periods=len(values), freq="5min")
    return {"hrv": {"hrvSummary": {"lastNightAvg": 70},
                    "hrvReadings": [{"hrvValue": v, "readingTimeGMT": x.strftime("%Y-%m-%dT%H:%M:%S.0")}
                                    for v, x in zip(values, t)]}}


@pytest.fixture
def garmin(tmp_path):
    g = tmp_path / "garmin"
    (g / "api" / "raw").mkdir(parents=True)
    rng = np.random.default_rng(1)
    night1 = np.round(rng.normal(70, 12, 80), 1)                  # in the export (zipped FIT)
    night2 = np.round(rng.normal(60, 8, 75), 1)                   # from the API
    t1 = int(pd.Timestamp("2026-10-20 23:30", tz="UTC").timestamp())
    with zipfile.ZipFile(g / "UploadedFiles_0-_Part1.zip", "w") as z:
        z.writestr("hrv_night1.fit", hrv_fit(t1, night1))
        z.writestr("not_hrv.fit", b"junk")
    (g / "api" / "raw" / "2026-10-22.json").write_text(json.dumps(api_day("2026-10-21 23:40", night2)))
    return g, night1, night2


def test_nights_into_the_store_and_their_quantiles(garmin, tmp_path):
    g, night1, night2 = garmin
    proc = tmp_path / "processed"
    assert NightHrv.update(g, proc, "UTC", progress=lambda *a: None) == 2
    assert streamstore.ids(proc, "hrv") == {"2026-10-21", "2026-10-22"}    # filed under the wake-up date
    assert streamstore.ids(proc, "ride") == set()                          # .hrv.npz isn't a ride
    q = NightHrv.quantiles(proc)
    for day, v in (("2026-10-21", night1), ("2026-10-22", night2)):
        exp = np.quantile(v, [0, .25, .5, .75, 1])
        got = q.loc[day, ["hrv_min", "hrv_q25", "hrv_median", "hrv_q75", "hrv_max"]].to_numpy(float)
        assert np.allclose(got, exp, atol=0.01) and q.loc[day, "hrv_n"] == len(v)
    assert NightHrv.update(g, proc, "UTC", progress=lambda *a: None) == 0    # nothing changed: nothing read


def test_more_readings_win_and_short_nights_have_no_box(garmin, tmp_path):
    g, night1, _ = garmin
    proc = tmp_path / "processed"
    NightHrv.update(g, proc, "UTC", progress=lambda *a: None)
    # the API later returns the 21st with fewer readings: kept as it is; a 3-reading night gets no box
    (g / "api" / "raw" / "2026-10-21.json").write_text(json.dumps(api_day("2026-10-20 23:30", night1[:40])))
    (g / "api" / "raw" / "2026-10-23.json").write_text(json.dumps(api_day("2026-10-23 01:00", [50, 55, 60])))
    assert NightHrv.update(g, proc, "UTC", progress=lambda *a: None) == 1
    q = NightHrv.quantiles(proc)
    assert q.loc["2026-10-21", "hrv_n"] == 80 and np.isnan(q.loc["2026-10-23", "hrv_median"])
    assert len(NightHrv.quantiles(tmp_path / "empty")) == 0


def test_sources_synced_from_another_computer_are_not_read_again(garmin, tmp_path, monkeypatch):
    """peer sync brings the other computer's index (its full paths, nanosecond stamps) and the files with
    their mtime in whole seconds: they still count as read."""
    from goodpush.io.stamps import in_seconds
    g, _, _ = garmin
    proc = tmp_path / "processed"
    NightHrv.update(g, proc, "UTC", progress=lambda *a: None)
    index = json.loads((proc / "hrv_sources.json").read_text())
    elsewhere = {"C:\\goodpush\\athletes\\athlete_name\\data\\raw\\garmin\\" + rel.replace("/", "\\"):   # privacy: ok, hygiene: ok (made up)
                 f"{key}123456789" for rel, key in index.items()}        # ns: the seconds and a fraction
    (proc / "hrv_sources.json").write_text(json.dumps(elsewhere))
    monkeypatch.setattr(NightHrv, "fit_files", lambda p: pytest.fail(f"read {p} again"))
    assert NightHrv.update(g, proc, "UTC", progress=lambda *a: None) == 0
    assert json.loads((proc / "hrv_sources.json").read_text()) == index     # stored in the new form
    assert in_seconds("2048-1793764609391587000") == "2048-1793764609"
    assert in_seconds("2048-1793764609391587000-a3") == "2048-1793764609-a3"
    assert in_seconds("2048-1793764609-a3") == "2048-1793764609-a3"
    assert list(in_seconds(pd.Series(["5-1793764609000000001", "5-1793764609"]))) == ["5-1793764609"] * 2
