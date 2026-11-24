import asyncio
import json
import struct
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from goodpush.io import fit
from goodpush.io.fit_workout import fit_crc
from goodpush.io.strava import load_activities_csv
from goodpush.ride import ant
from goodpush.ride.ant import AntLink
from goodpush.ride.recorder import RideRecorder
from goodpush.ride.sensors import STALE, Device, SensorHub
from goodpush.ride.session import GAP_S, RideSession, WorkoutRun
from goodpush.ride.sim import SimLink

T0 = 1_790_000_000
STEPS = [{"name": "Warm up", "min": 2, "lo": 100, "hi": 140, "kind": "warmup"},
         {"name": "Threshold", "min": 1, "lo": 220, "hi": 240, "kind": "active"},
         {"name": "Sprint, all-out", "min": 0.25, "lo": 400, "hi": 600, "kind": "active"},
         {"name": "Cool down", "min": 1, "lo": 90, "hi": 110, "kind": "cooldown"}]


def recorder(n=300, hr=True):
    r = RideRecorder(T0, "Test ride")
    i = np.arange(n)
    power = 150 + 50 * np.sin(i / 20)
    for k in range(n):                       # one sample per second, as the ride adds them
        r.add(T0 + k, power[k], 90, 140 + k % 10 if hr else np.nan, 8.0, k // 100, True)
    return r


def test_sensor_hub_sources_and_staleness():
    hub = SensorHub({"trainer": "ant:fec:1", "power": "ant:power:2", "hr": "ant:hr:3"})
    hub.put("ant:fec:1", "power", 300, t=10)          # the power meter wins for power
    hub.put("ant:power:2", "power", 250, t=10)
    hub.put("ant:fec:1", "cadence", 85, t=10)         # no cadence sensor: the power meter, then the trainer
    hub.put("ant:hr:3", "hr", 140, t=10)
    assert hub.get("power", 11) == 250
    assert hub.get("cadence", 11) == 85               # the power meter sends none: the trainer's
    hub.put("ant:power:2", "cadence", 88, t=10)
    assert hub.get("cadence", 11) == 88               # the power meter outranks the trainer for cadence
    hub.put("ant:fec:1", "power", 260, t=13)          # the power meter's channel goes quiet: the trainer fills in
    assert hub.get("power", 13) == 250 and hub.get("power", 14) == 260
    assert hub.get("hr", 12) == 140 and np.isnan(hub.get("hr", 14))    # stale after 3 s
    assert [d.key for d in hub.devices()] == ["ant:fec:1", "ant:power:2", "ant:hr:3"]
    assert Device.from_key("ant:fec:12345").label == "Smart trainer #12345"


def test_ant_pages_feed_the_hub():
    hub = SensorHub({"trainer": "ant:fec:1", "hr": "ant:hr:2", "cadence": "ant:cadence:3"})
    link = AntLink(hub)                               # no stick needed: just the page mapping
    link.on_data("ant:fec:1", 25, "standard_power", SimpleNamespace(instantaneous_power=210, cadence=255))
    link.on_data("ant:fec:1", 16, "general_fe", SimpleNamespace(speed=9.5))
    link.on_data("ant:hr:2", 0, "heart_rate", SimpleNamespace(heart_rate=150, beat_count=1, beat_time=10.0,
                                                               previous_heart_beat_time=-1))
    link.on_data("ant:cadence:3", 0, "bike_cadence", SimpleNamespace(calculate_cadence=lambda: 92.0))
    assert (hub.get("power"), hub.get("speed"), hub.get("hr"), hub.get("cadence")) == (210, 9.5, 150, 92)


def test_fit_activity_roundtrip():
    r = recorder(hr=False)
    data = r.fit_bytes()
    assert fit_crc(data) == 0                          # file CRC checks out
    a = fit.read_activity(data)
    rec = a["records"]
    p = np.array(r.arrays()["power"])
    assert rec["time"][0] == T0 and len(rec["time"]) == 300
    assert np.array_equal(rec["power"], np.round(p).astype(int))
    assert set(rec["cadence"]) == {90} and set(rec["heart_rate"]) == {None}    # no strap: invalid
    assert rec["distance"][-1] == pytest.approx(2400)
    assert (a["sport"], a["sub_sport"]) == ("cycling", "indoor_cycling")
    laps = fit.read_messages(data, [19])[19]
    assert len(laps) == 3 and [l[7] for l in laps] == [100_000] * 3            # one lap per step
    assert laps[1][19] == round(p[100:200].mean())
    assert [l[71] for l in laps] == [0, 1, 2]                                    # the workout step each lap was


def test_save_table_and_ingest_priority(tmp_path):
    rides, garmin = tmp_path / "goodpush", tmp_path / "garmin" / "activities"
    garmin.mkdir(parents=True)
    pd.DataFrame([{"id": "garmin_1", "start_utc": "2026-09-21T14:15:20Z", "type": "Virtual Ride",
                   "filename": "garmin_1.fit"}]).to_csv(garmin / "activities_garmin.csv", index=False)
    r = recorder()
    path = r.save(rides)
    assert path.exists() and not list(rides.glob("*.npz"))
    acts = load_activities_csv(tmp_path / "strava", tmp_path / "garmin", rides)
    assert list(acts["id"]) == ["goodpush_1790000000"]  # the watch's copy of the same ride is dropped
    assert acts["indoor"].all() and acts["avg_watts_csv"].iloc[0] == pytest.approx(r.arrays()["power"].mean())


def test_deleted_rides_stay_gone(tmp_path):
    """the app's Delete: goodpush's own recording loses its FIT and row; a synced ride is only left out
    (a re-sync rewrites its table, the deleted list still drops it)."""
    from types import SimpleNamespace
    from goodpush.io.strava import DeletedActivities, project_activities
    rides, garmin = tmp_path / "goodpush", tmp_path / "garmin" / "activities"
    garmin.mkdir(parents=True)
    pd.DataFrame([{"id": "garmin_1", "start_utc": "2026-08-01T14:15:20Z", "type": "Ride",
                   "filename": "garmin_1.fit"}]).to_csv(garmin / "activities_garmin.csv", index=False)
    path = recorder().save(rides)
    proj = SimpleNamespace(root=tmp_path, path=lambda k: tmp_path / {"rides": "goodpush"}.get(k, k))
    assert set(project_activities(proj)["id"]) == {"garmin_1", "goodpush_1790000000"}
    gone = DeletedActivities(tmp_path)
    assert gone.delete("goodpush_1790000000", rides) and not path.exists()
    assert not gone.delete("garmin_1", rides) and gone.ids() == {"garmin_1", "goodpush_1790000000"}
    assert (garmin / "activities_garmin.csv").exists()                  # Garmin's copy is left alone
    assert project_activities(proj).empty
    pd.DataFrame([{"id": "garmin_2", "start_utc": "2026-08-02T14:15:20Z", "type": "Ride", "filename": "g2.fit"},
                  {"id": "garmin_1", "start_utc": "2026-08-01T14:15:20Z", "type": "Ride", "filename": "garmin_1.fit"}]
                 ).to_csv(garmin / "activities_garmin.csv", index=False)  # a later sync brings it back ...
    assert list(project_activities(proj)["id"]) == ["garmin_2"]          # ... still left out


def test_strava_upload_hands_over_the_fit(tmp_path, monkeypatch):
    """Send to Strava: the upload page opens and the hand-off folder (the FIT + the chart PNG, only the last ride)
    is shown; a picture that isn't a PNG is refused; other rides have no file here"""
    from goodpush.io import strava
    from goodpush.io.strava import StravaUpload
    opened = []
    monkeypatch.setattr(strava.webbrowser, "open", opened.append)
    monkeypatch.setattr(strava.subprocess, "Popen", opened.append)
    rides, processed = tmp_path / "goodpush", tmp_path / "processed"
    src = recorder().save(rides)
    up = StravaUpload(rides, processed)
    (processed / StravaUpload.FOLDER).mkdir(parents=True)
    (processed / StravaUpload.FOLDER / "older.fit").write_bytes(b"x")
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    fit = up.open("goodpush_1790000000", png)
    assert opened[0] == StravaUpload.PAGE and str(fit.parent) in " ".join(opened[1])     # Linux: the folder
    assert fit.read_bytes() == src.read_bytes() and fit.with_suffix(".png").read_bytes() == png
    assert sorted(p.name for p in fit.parent.iterdir()) == [fit.name, fit.with_suffix(".png").name]
    assert [p.name for p in up.hand_off("goodpush_1790000000").parent.iterdir()] == [fit.name]
    with pytest.raises(ValueError):
        up.hand_off("goodpush_1790000000", b"<svg/>")
    with pytest.raises(FileNotFoundError):
        up.file("garmin_1")


def test_deleted_ride_leaves_the_processed_folder(tmp_path):
    """ingest.forget: its stream and its rows in the parsed-file and alphaHRV caches go; others stay"""
    from types import SimpleNamespace
    from goodpush.alpha import INDEX
    from goodpush.ingest import SUMMARY_TABLE, forget
    from goodpush.io import store
    proj = SimpleNamespace(processed=tmp_path / "processed")
    (proj.processed / "streams").mkdir(parents=True)
    for a in ("garmin_1", "garmin_2"):
        np.savez(proj.processed / "streams" / f"{a}.npz", start=np.array([T0]), power=np.ones(5, np.float32))
    store.save(pd.DataFrame({"id": ["garmin_1", "garmin_2"], "source_key": ["k1", "k2"]}), proj.processed, SUMMARY_TABLE)
    pd.DataFrame({"id": ["garmin_1", "garmin_2"], "source_key": ["k1", "k2"]}).to_csv(proj.processed / INDEX, index=False)
    forget(proj, ["garmin_1"], progress=lambda *a: None)
    assert [p.name for p in (proj.processed / "streams").iterdir()] == ["garmin_2.npz"]
    assert list(store.load(proj.processed, SUMMARY_TABLE)["id"]) == ["garmin_2"]
    assert list(pd.read_csv(proj.processed / INDEX, dtype=str)["id"]) == ["garmin_2"]


def test_partial_copy_recovered(tmp_path):
    r = recorder(120)
    r.save_partial(tmp_path)
    RideRecorder(T0 + 999).save_partial(tmp_path)     # empty: dropped
    saved = RideRecorder.recover(tmp_path)
    assert [p.name for p in saved] == ["goodpush_1790000000.fit"] and not list(tmp_path.glob("*.npz"))


def test_all_out_steps_leave_erg_and_come_back():
    """2026-11-22: a primer and an all-out step ask for resistance: from ERG the trainer switches for them (the
    primer is there to find your gear) and back to ERG after; in resistance or sim already, it stays; a mode you
    pick yourself in between is kept."""
    steps = [{"name": "Warm up", "min": 1, "lo": 100, "hi": 140, "kind": "warmup"},
             {"name": "Primer", "min": 1, "lo": 200, "hi": 220, "kind": "active", "mode": "resistance"},
             {"name": "All-out 15 s", "min": 0.25, "lo": 600, "hi": 800, "kind": "active", "mode": "resistance"},
             {"name": "Recover", "min": 2, "lo": 100, "hi": 120, "kind": "rest"},
             {"name": "Threshold", "min": 1, "lo": 220, "hi": 240, "kind": "active"}]
    run = WorkoutRun("T", steps)
    assert run.mode == "erg" and run.command() == ("power", 120)
    seen = []
    for _ in range(5 * 60):                              # a second at a time through the session
        run.advance(1, riding=True)
        seen.append((run.index, run.mode))
    mode = dict(seen)
    assert mode[1] == mode[2] == "resistance" and mode[3] == mode[4] == "erg"
    run.seek(0)
    run.choose_mode("sim")                               # yours: the all-out steps leave it alone
    run.seek(70)
    assert run.index == 1 and run.mode == "sim"
    run.seek(200)
    assert run.mode == "sim"
    run.seek(0)
    run.choose_mode("erg")
    run.seek(70)
    run.choose_mode("erg")                               # back to ERG yourself in the primer: kept after it too
    run.seek(200)
    assert run.mode == "erg" and run.mode_before is None


def test_workout_run_steps_targets_and_controls(tmp_path):
    run = WorkoutRun("T", STEPS)
    assert run.total == 4.25 * 60 and run.index == 0 and run.target() == 120
    run.advance(150, riding=True)
    assert run.index == 1 and run.step_left == 30 and run.next_step["name"].startswith("Sprint")
    run.advance(10, riding=False)                       # not pedalling: the clock waits
    assert run.elapsed == 150
    run.nudge(5)
    assert run.target() == pytest.approx(241.5)            # 230 W + 5 %
    run.skip()
    assert run.index == 2 and run.target() == pytest.approx(525)   # all-out: still ERG (500 W + 5 %)
    run.mode = "resistance"
    assert run.target() is None and run.command() == ("resistance", 30)   # only you switch modes
    run.nudge(5)
    assert run.resistance == 35
    run.skip(); assert run.target() is None
    run.mode = "sim"
    run.nudge(1); run.nudge(1)
    assert run.command() == ("grade", 2.0) and run.resistance == 35
    run.nudge(-50)
    assert run.grade == -10                              # a downhill, not a cliff
    run.next_mode()
    assert run.mode == "erg"
    run.advance(1000, riding=True)
    assert run.done and run.index == -1 and run.target() == run.free_w      # after it: ERG holds free_w
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"ftp": 250, "sessions": [
        {"date": "2026-10-27", "key": "2026-10-27", "title": "Rest", "steps": []},
        {"date": "2026-10-27", "key": "2026-10-27#2", "title": "Openers", "steps": STEPS}]}))
    r = WorkoutRun.from_plan(plan, today=pd.Timestamp("2026-10-27"))
    assert (r.title, r.key, len(r.steps)) == ("Openers", "2026-10-27#2", 4)
    assert WorkoutRun.from_plan(plan, key="2026-10-27") is None


def test_ride_session_with_simulated_trainer(tmp_path):
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "sim:hr:2"})
    sim = SimLink(hub)
    ride = RideSession(sim, hub, WorkoutRun("T", STEPS), tmp_path, now=T0)
    for t in T0 + np.arange(1, 160 * 4) / 4:             # 160 s at 4 ticks / s, the overlay's rate
        sim.tick(t)
        ride.tick(t)
    assert ride.recorder.seconds == 159
    assert ride.run.index == 1 and sim.target == 230      # ERG target of the threshold step
    assert 220 < ride.power3() < 240
    assert (tmp_path / "goodpush_1790000000.partial.npz").exists()   # minute copies
    sim.pedalling = False
    for t in T0 + 160 + np.arange(0, 60) / 4:
        sim.tick(t)
        ride.tick(t)
    stopped = ride.run.elapsed
    for t in T0 + 175 + np.arange(0, 20) / 4:
        sim.tick(t)
        ride.tick(t)
    assert ride.run.elapsed == stopped and ride.run.index == 1        # auto-pause
    sim.pedalling = True
    for t in T0 + 180 + np.arange(0, 60 * 4) / 4:
        sim.tick(t)
        ride.tick(t)
    assert ride.run.index == 3 and sim.target == 100      # past the sprint, cooling down
    path = ride.finish()
    a = fit.read_activity(path.read_bytes())
    assert len(a["records"]["time"]) == ride.recorder.seconds == 239
    assert len(fit.read_messages(path.read_bytes(), [19])[19]) == 4         # warm up, threshold, sprint, cool down


def test_clocks_stop_when_you_do_not_when_readings_drop_out(tmp_path):
    hub = SensorHub({"trainer": "sim:fec:1"})
    ride = RideSession(SimLink(hub), hub, WorkoutRun("T", STEPS), tmp_path, now=T0)

    def ride_for(seconds, power=200, cadence=90):
        for t in ride.second + 1 + np.arange(seconds):
            if power is not None:
                hub.put("sim:fec:1", "power", power, t)
            if cadence is not None:
                hub.put("sim:fec:1", "cadence", cadence, t)
            ride.tick(t)

    hub.put("sim:fec:1", "power", 200, T0)
    ride.tick(T0)                                       # the first pedal stroke starts the ride
    ride_for(20)
    assert ride.moving == ride.run.elapsed == 20
    ride_for(20, power=None)                            # the power readings drop out, cadence carries on
    assert ride.moving == ride.run.elapsed == 40
    ride_for(8, power=None, cadence=None)               # every reading drops out: as it was, for up to GAP_S
    assert ride.moving == 48
    ride_for(10, power=None, cadence=None)              # ... then the clocks stop
    assert ride.moving == 40 + STALE + GAP_S            # the cadence went stale at 43 s
    ride_for(5)
    m = ride.moving
    ride_for(5, power=0, cadence=0)                     # you stop pedalling: they stop at once
    assert ride.moving == m and ride.recorder.seconds == 68     # still recorded, as stopped time


def test_workout_seek():
    run = WorkoutRun("T", STEPS)
    run.seek(130)
    assert run.index == 1 and run.step_left == 50
    run.seek(-5)
    assert run.elapsed == 0
    run.seek(1e6)
    assert run.done


def test_device_channels_track():
    sent = []

    class Device:                                     # openant's default: a background-scanning channel
        node = SimpleNamespace(ant=SimpleNamespace(write_message=lambda m: sent.append((m._id, list(m._data)))))
        channel = SimpleNamespace(id=1)

        def __init__(self, device_type):
            self.device_type = device_type
            self.data = {"batteries": [None] * 15}    # openant's: batteries 0-14

        def open_channel(self, extended=True, channel_type=None, ext_assign=0x01):
            self.ext_assign = ext_assign

    trainer, strap = ant.tracking(Device)(17), ant.tracking(Device)(120)
    trainer.open_channel()
    assert trainer.ext_assign is None and not sent and ant.tracking(Device) is ant.tracking(Device)
    strap.open_channel()                              # a sensor searches at low priority only
    assert sent == [(0x44, [1, 0]), (0x63, [1, 0xFF])]
    assert len(strap.data["batteries"]) == ant.BATTERY_IDS   # a device reporting battery 15 doesn't raise


def test_bad_message_keeps_the_sensors_going():
    calls = []

    def start():                                      # openant's loop: a message it can't handle ends it
        calls.append(1)
        if len(calls) == 1:
            raise IndexError("list assignment index out of range")

    AntLink.run_node(SimpleNamespace(start=start))
    assert len(calls) == 2                            # back to the next messages, then the node stopped


def test_trainer_commands_checked(monkeypatch):
    sent, refuse = [], []

    def deliver(page):                                # the trainer acknowledging (or not) each page
        if refuse:
            return ant.TX_FAILED
        sent.append(page)
        return ant.TX_DONE

    monkeypatch.setattr(ant, "RETRY_S", 0.001)
    link = AntLink(SensorHub({"trainer": "ant:fec:1"}))
    fe = SimpleNamespace(resistance_mode=None, resistance=0.0)          # openant's copy of page 71
    link.trainer = SimpleNamespace(command_status=None, data={"fe": fe})
    monkeypatch.setattr(link, "deliver", deliver)
    assert AntLink.power_page(250) == [0x31, 255, 255, 255, 255, 255, 0xE8, 0x03]          # 1000 x 0.25 W
    assert AntLink.resistance_page(30)[7] == 60
    stop = threading.Event()
    threading.Thread(target=link.send_loop, args=(stop,), daemon=True).start()
    refuse.append(1)
    link.set_power(250)
    time.sleep(0.2)                                   # missed again and again: shown on the bar
    assert link.trouble == "Trainer isn't taking commands" and not sent
    refuse.clear()                                    # the trainer hears it: the same target goes again
    time.sleep(0.1)
    assert sent == [AntLink.power_page(250), ant.STATUS_PAGE] and link.trouble == ""
    fe.resistance_mode, fe.resistance = SimpleNamespace(name="TargetPower"), 250.0
    assert link.holds(("power", 250)) and not link.holds(("power", 180)) and not link.holds(("grade", 2))
    link.trainer.command_status = SimpleNamespace(name="NotSupported")          # its page 71 reply
    assert link.trouble.startswith("Trainer refused")
    stop.set()


def test_cadence_glitches_dropped():
    hub = SensorHub({"cadence": "ant:speed_cadence:3"})
    link = AntLink(hub)
    link.on_data("ant:speed_cadence:3", 0, "bike_cadence", SimpleNamespace(calculate_cadence=lambda: 25000.0))
    assert np.isnan(hub.get("cadence"))               # the first message: the sensor's lifetime revolutions
    link.on_data("ant:speed_cadence:3", 0, "bike_cadence", SimpleNamespace(calculate_cadence=lambda: 90.0))
    assert hub.get("cadence") == 90


def test_free_ride_erg_holds_a_power():
    run = WorkoutRun("Free ride", [])
    assert run.command() == ("power", 150)            # ERG with no step: free_w
    run.nudge(1)
    run.nudge(1)
    assert run.command() == ("power", 160)            # 5 W a click
    run.mode = "resistance"
    assert run.command()[0] == "resistance"


def test_sim_mode_pages_and_session(tmp_path):
    assert AntLink.track_page(0) == [0x33, 255, 255, 255, 255, 0x20, 0x4E, 80]      # 0 %: 20000, Crr 0.004
    assert AntLink.track_page(-2.5)[5:7] == [0x26, 0x4D]                          # 19750
    page = AntLink.user_page(68.04, 9)                     # 6804 (0.01 kg), 180 (0.05 kg), 70 cm
    assert page[:3] == [0x37, 0x94, 0x1A] and page[4] >> 4 | page[5] << 4 == 180 and page[6] == 70
    hub = SensorHub({"trainer": "sim:fec:1"})
    sim = SimLink(hub)
    ride = RideSession(sim, hub, WorkoutRun("T", STEPS), tmp_path, now=T0)
    ride.run.mode = "sim"
    ride.run.grade = 4
    ride.tick(T0 + 1)
    assert sim.grade == 4 and sim.target is None


def test_ride_starts_at_the_first_pedal_stroke(tmp_path):
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "ant:hr:2"})
    ride = RideSession(SimLink(hub), hub, WorkoutRun("T", STEPS), tmp_path, now=T0)
    ride.tick(T0 + 120.5)                               # two minutes in the device picker: not recorded
    for t in T0 + 121.5 + np.arange(600):               # ten minutes by the trainer, heart rate on, no power
        hub.put("sim:fec:1", "power", 0, t)
        hub.add_beats("ant:hr:2", [(1.0, 1)], t=t)
        ride.tick(t)
    assert ride.recorder.seconds == 0 and ride.run.elapsed == 0 and not ride.recorder.beats and ride.sent[0] is not None
    for t in T0 + 721.5 + np.arange(3):
        hub.put("sim:fec:1", "power", 150, t)
        ride.tick(t)
    assert ride.recorder.seconds == 2 and ride.recorder.id == f"goodpush_{T0 + 721}"


def test_overlay_draws_offscreen(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from goodpush.ride.overlay import Overlay
    _app = QApplication.instance() or QApplication([])          # kept alive for the windows
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "sim:hr:2"})
    sim = SimLink(hub)
    ride = RideSession(sim, hub, WorkoutRun("T", STEPS), tmp_path)
    saved = {}
    bar = Overlay(ride, {"opacity": .5}, 250, saved.update, None)
    sim.tick(ride.start + 1)
    bar.tick()
    img = bar.grab().toImage()
    assert img.width() == 640 and img.height() == 116
    bar.set_value("scale", 1.25)
    assert saved == {"scale": 1.25} and bar.width() == 800
    bar.timer.stop()
    bar.tray.hide()
    ride.discard()


def test_tray_icon_draws_the_logo_bike(monkeypatch):
    """The tray (and the app icon) draw app/static/mark.svg: the shipped file is found and painted in its colour."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QApplication
    from goodpush.ride.overlay import MARK, Overlay
    _app = QApplication.instance() or QApplication([])
    colour = QColor(MARK.read_text(encoding="utf-8").split('stroke="')[1][:7])
    img = Overlay.icon().pixmap(64, 64).toImage()
    pixels = [img.pixelColor(x, y) for x in range(64) for y in range(64)]
    assert any(p.alpha() == 255 and p.rgb() == colour.rgb() for p in pixels)
    assert img.pixelColor(0, 0).alpha() == 0                       # no tile behind the bike in the tray



def test_overlay_close_x_asks_save_or_discard(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
    QTest = pytest.importorskip("PySide6.QtTest").QTest
    from goodpush.ride import overlay
    _app = QApplication.instance() or QApplication([])          # kept alive for the windows
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "sim:hr:2"})
    sim = SimLink(hub)
    ride = RideSession(sim, hub, WorkoutRun("T", STEPS), tmp_path)
    bar = overlay.Overlay(ride, {"opacity": .5}, 250, lambda v: None, None)
    bar.show()
    asked = []
    answers = iter([QMessageBox.Cancel, QMessageBox.Discard, QMessageBox.Save])
    monkeypatch.setattr(overlay.Overlay, "ask", lambda self, text, buttons=None: asked.append(text) or next(answers))
    corner = QPoint(overlay.W - 11, 11)
    monkeypatch.setattr(type(ride.recorder), "seconds", property(lambda self: 30))     # half a minute ridden
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, corner)          # the x: carry on (Cancel) ...
    assert bar.isVisible() and bar.outcome == "save" and "too short" in asked[0]
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, corner)          # ... then throw it away
    assert not bar.isVisible() and bar.outcome == "discard"
    bar = overlay.Overlay(ride, {"opacity": .5}, 250, lambda v: None, None)
    bar.show()
    monkeypatch.setattr(type(ride.recorder), "seconds", property(lambda self: 1800))   # a real ride: save or discard
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, corner)
    assert "save it or throw it away" in asked[2] and bar.outcome == "save" and not bar.isVisible()
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, QPoint(300, 50))   # elsewhere on the bar: no question
    assert len(asked) == 3
    ride.discard()


def test_question_over_the_bar_stays_on_top(monkeypatch, tmp_path):
    """The save / discard question is a stay-on-top window from the moment it opens (Windows: else a full-screen
    trainer app can cover it), and the button clicked is the answer."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from goodpush.ride import overlay
    app = QApplication.instance() or QApplication([])
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "sim:hr:2"})
    ride = RideSession(SimLink(hub), hub, WorkoutRun("T", STEPS), tmp_path)
    bar = overlay.Overlay(ride, {"opacity": .5}, 250, lambda v: None, None)
    on_top = []
    QTimer.singleShot(50, lambda: on_top.append(bool(app.activeModalWidget().windowFlags() & Qt.WindowStaysOnTopHint)))
    QTimer.singleShot(100, lambda: app.activeModalWidget().button(QMessageBox.Save).click())
    ans = bar.ask("Finish?", QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
    assert on_top == [True] and ans == QMessageBox.Save
    bar.timer.stop()
    bar.tray.hide()
    ride.discard()


def test_cancelled_dialog_keeps_the_ride_going(monkeypatch, tmp_path):
    """Cancel on a dialog (the devices picker, the save question) mid-ride: the ride carries on with the
    devices it had; only the bar closing ends it. Qt doesn't count the bar (a tool window) as open."""
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog
    from goodpush.ride import overlay
    app = QApplication.instance() or QApplication([])
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "sim:hr:2"})
    ride = RideSession(SimLink(hub), hub, WorkoutRun("T", STEPS), tmp_path)
    window = overlay.RideWindow(ride, {}, 250, lambda v: None, lambda c: None)
    monkeypatch.setattr(overlay, "DevicePicker", lambda link, choice: QDialog())
    reconnected = []
    monkeypatch.setattr(ride, "reconnect", reconnected.append)
    monkeypatch.setattr(overlay.Overlay, "ask", lambda self, text, buttons=None: overlay.QMessageBox.Yes)
    steps = []
    bar = lambda: next(w for w in app.topLevelWidgets() if isinstance(w, overlay.Overlay) and w.isVisible())
    QTimer.singleShot(50, lambda: bar().devices())                               # Devices ...
    QTimer.singleShot(150, lambda: app.activeModalWidget().reject())             # ... Cancel
    QTimer.singleShot(400, lambda: steps.append("still riding") or bar().discard())
    QTimer.singleShot(5000, app.quit)                                # a hang fails below, not forever
    assert window.run(pick=False) == (None, hub.choice)              # thrown away, not saved
    assert steps == ["still riding"] and reconnected[-1] == hub.choice


def test_ride_started_from_the_app(tmp_path, monkeypatch):
    """the app's Ride it now: a ride window that dies at once comes back as the error (its last lines); one
    still running is open. A stand-in program, never the real overlay (it would open on the screen)."""
    import sys
    from types import SimpleNamespace
    from goodpush.app import server
    from goodpush.app.server import RideLauncher
    proj = SimpleNamespace(processed=tmp_path, root=tmp_path, logs=tmp_path)
    assert RideLauncher.command(proj, "2026-10-29")[-4:] == ["ride", "--save-only", "--session", "2026-10-29"]
    monkeypatch.setattr(RideLauncher, "command", staticmethod(
        lambda proj, key="": [sys.executable, "-c", "import sys; print('no ANT+ stick found'); sys.exit(2)"]))
    with pytest.raises(ValueError, match="didn't start.*no ANT\\+ stick found"):
        RideLauncher.start(proj, "2026-10-29")
    monkeypatch.setattr(RideLauncher, "command", staticmethod(lambda proj, key="": [sys.executable, "-c", "import time; time.sleep(30)"]))
    monkeypatch.setattr(server, "RIDE_STARTUP_S", 0.5)
    said = []
    RideLauncher.start(proj, "2026-10-29", said.append)
    try:
        assert "Ride window open" in said[0]
        with pytest.raises(ValueError, match="already open"):
            RideLauncher.start(proj, "2026-10-29")
    finally:
        RideLauncher.proc.kill()


def test_closing_the_ride_window_ends_the_ride(tmp_path, monkeypatch):
    """the window closes, its process ends: the app gets the ride it saved (to add as a job) and another ride can
    start at once, not after the analysis. A stand-in program, never the real overlay."""
    import sys
    import threading
    from types import SimpleNamespace
    from goodpush.app import server
    from goodpush.app.server import RideLauncher
    proj = SimpleNamespace(processed=tmp_path, root=tmp_path, logs=tmp_path)
    monkeypatch.setattr(RideLauncher, "command", staticmethod(lambda proj, key="": [
        sys.executable, "-c", "import time; time.sleep(1); print('Saved /rides/goodpush_1793745166.fit')"]))
    monkeypatch.setattr(server, "RIDE_STARTUP_S", 0.3)
    got, done = [], threading.Event()
    RideLauncher.start(proj, "2026-11-03#2", lambda *a: None, lambda rid: (got.append(rid), done.set()))
    assert done.wait(10) and got == ["goodpush_1793745166"]
    RideLauncher.proc.wait(5)
    monkeypatch.setattr(RideLauncher, "command", staticmethod(lambda proj, key="": [
        sys.executable, "-c", "print('Nothing saved.')"]))
    with pytest.raises(ValueError, match="didn't start"):         # not "already open": the last ride ended
        RideLauncher.start(proj, "2026-11-03#2")


def test_ant_stick_in_use_or_silent_is_explained(monkeypatch):
    """macOS: the app holding the stick is named (from the USB registry); a silent stick: replug it."""
    from types import SimpleNamespace
    from goodpush.ride import ant
    reg = ('+-o Root\n+-o ANT USBStick2@14200000\n  |   "UsbExclusiveOwner" = "pid 4242, TrainerRoad"\n'
           '  |   "idProduct" = 4104\n  |   "idVendor" = 4047\n+-o Headset@1\n  |   "idVendor" = 1452\n')
    monkeypatch.setattr(ant.sys, "platform", "darwin")
    monkeypatch.setattr(ant.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=reg))
    assert ant.AntLink.stick_owner() == "TrainerRoad"
    assert "in use by TrainerRoad" in ant.AntLink.why_not_open(OSError("Access denied"))
    monkeypatch.setattr(ant.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=reg.replace("4242", str(ant.os.getpid()))))
    assert ant.AntLink.stick_owner() == ""                               # goodpush itself holding it: not "another app"
    assert ant.AntLink.why_not_open(TimeoutError("Timed out while waiting for message")).startswith(ant.NO_ANSWER)
    assert ant.AntLink.why_not_open(ValueError("Device not found")).startswith("No ANT+ USB stick found")


def test_overlay_buttons_and_dragging_the_marker(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication
    QTest = pytest.importorskip("PySide6.QtTest").QTest
    from goodpush.ride import overlay
    _app = QApplication.instance() or QApplication([])          # kept alive for the windows
    hub = SensorHub({"trainer": "sim:fec:1"})
    ride = RideSession(SimLink(hub), hub, WorkoutRun("T", STEPS), tmp_path)
    bar = overlay.Overlay(ride, {"opacity": .5}, 250, lambda v: None, None)
    bar.show()
    at = lambda name: overlay.BUTTONS[name].center().toPoint()
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("harder"))
    assert ride.run.target() == pytest.approx(121.2)             # 120 W + 1 %
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("easier"))
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("easier"))
    assert ride.run.target() == pytest.approx(118.8)
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("mode"))
    assert ride.run.mode == "resistance" and ride.run.target() is None
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("harder"))
    assert ride.run.resistance == 31
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("mode"))
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("harder"))
    assert ride.run.mode == "sim" and ride.run.command() == ("grade", 1.0)
    QTest.mouseClick(bar, Qt.LeftButton, Qt.NoModifier, at("mode"))
    assert ride.run.mode == "erg"
    box, y = overlay.PROFILE, int(overlay.PROFILE.center().y())
    x = lambda f: int(box.x() + f * box.width())
    QTest.mousePress(bar, Qt.LeftButton, Qt.NoModifier, QPoint(x(.5), y))
    assert ride.run.elapsed == pytest.approx(.5 * ride.run.total, abs=1)
    QTest.mouseMove(bar, QPoint(x(.25), y))
    QTest.mouseRelease(bar, Qt.LeftButton, Qt.NoModifier, QPoint(x(.25), y))
    assert ride.run.elapsed == pytest.approx(.25 * ride.run.total, abs=1)
    QTest.mouseMove(bar, QPoint(x(.75), y))                       # no button held: no seeking
    assert ride.run.elapsed == pytest.approx(.25 * ride.run.total, abs=1)
    bar.grab().toImage().save(str(tmp_path / "bar.png"))
    bar.timer.stop()
    bar.tray.hide()
    ride.discard()


def test_ramp_targets_move_through_the_step():
    steps = [{"name": "Ramp", "min": 2, "lo": 100, "hi": 120, "lo_end": 200, "hi_end": 220, "kind": "active"},
             {"name": "Hold", "min": 1, "lo": 200, "hi": 220, "kind": "active", "cadence": 90}]
    run = WorkoutRun("R", steps)
    assert run.target() == 110
    run.seek(60)
    assert run.target() == pytest.approx(160)            # half way up the ramp
    run.seek(119)
    assert run.target() == pytest.approx(209.2, abs=.1)
    run.seek(130)
    assert run.target() == 210 and run.watts(run.step) == 210


def test_ramp_test_summary():
    """a ramp to failure from its laps: stopped 5 s into a step (the one before is held: within LENIENCY_S), then easy
    riding and skipped steps that aren't part of it; laps without their step (older rides) are taken in order."""
    from goodpush.io.fit import FIT_EPOCH_OFFSET
    from goodpush.models.ramp import RampTest
    from goodpush.planning.workouts import RAMP_STEPS, WorkoutLibrary
    ftp, t0 = 200.0, 1_000_000.0
    w = WorkoutLibrary.ramp_test(pd.Timestamp("2026-11-03"))
    held = 6                                          # 50-80 % in full, 86 % 5 s short (held), stopped 5 s into 92 %
    secs = [600] + [120] * held + [115] + [5] + [300]          # warm-up, held steps (last one 5 s short), stop, easy
    watts = np.r_[np.full(600, 100.0), np.repeat(RAMP_STEPS[:held + 1] * ftp, 120)[:120 * held + 115],
                  np.full(5, RAMP_STEPS[held + 1] * ftp), np.full(300, 90.0)]
    hr = np.r_[np.full(600, 110.0), 0.4 * watts[600:600 + 120 * held + 120] + 80, np.full(300, np.nan)]
    hr[-300:] = np.linspace(hr[600 + 120 * held + 119], 120, 300)                  # recovering
    starts = np.r_[0, np.cumsum(secs)[:-1]]
    laps = [{2: int(t0 + a) - FIT_EPOCH_OFFSET, 8: s * 1000} for a, s in zip(starts, secs)]
    z = {"power": watts, "heart_rate": hr, "cadence": np.full(watts.size, 90.0)}
    r = RampTest.from_ride(laps, t0, w, ftp, z)
    assert list(r.steps["held"]) == [True] * (held + 1) + [False]    # the easy lap after the stop isn't a ramp step
    s = r.summary({"FTP": 140.0, "LT2": 999.0}, ftp)
    assert s["last_held"] == "Ramp 86 %" and s["top_w"] == round(.86 * ftp + 5 / 120 * .06 * ftp)
    assert s["hr_at"]["FTP"]["hr"] == round(0.4 * 140 + 80) and s["hr_at"]["LT2"]["hr"] is None     # outside the steps
    assert s["recovery"][60] > 0 and s["max_hr"] == round(hr[600:].max())


def test_beat_intervals_from_an_ant_strap():
    """ANT+ heart-rate pages -> beat-to-beat intervals: one beat a message, two split at page 4's earlier beat, more
    as one interval over the beats it spans (exact, a missed-beat artifact), the 64 s rollover, a repeated
    message adding nothing, and a fresh start after a long silence."""
    b = ant.AntBeats()
    assert b.heard(0, 10, 63.5, -1, now=0) == []                                   # the first message: a reference
    assert b.heard(0, 11, 0.3, -1, now=1) == [(pytest.approx(0.8), 1)]             # over the rollover
    assert b.heard(0, 11, 0.3, -1, now=1.25) == []                                 # the same beat again
    assert b.heard(4, 13, 1.9, 1.1, now=2) == [(pytest.approx(0.8), 1)] * 2       # one message lost: split at it
    assert b.heard(0, 16, 4.3, -1, now=4) == [(pytest.approx(2.4), 3)]             # three beats in one: exact span
    assert b.heard(0, 30, 5.0, -1, now=70) == []                                   # silent past the rollover: afresh
    assert b.heard(0, 31, 5.7, -1, now=71) == [(pytest.approx(0.7), 1)]


def test_beats_recorded_and_written_to_the_fit(tmp_path):
    """the chosen strap's intervals go to the recording (another strap's don't), survive the crash copy and come
    back from the FIT file's hrv messages (padding left out; a missed-beat interval as it was)."""
    hub = SensorHub({"hr": "ant:hr:2"})
    hub.add_beats("ant:hr:2", [(0.8, 1), (0.79, 1), (1.6, 2), (0.81, 1)], t=T0)
    hub.add_beats("ant:hr:9", [(0.5, 1)], t=T0)                            # not the chosen strap
    r = recorder(hr=True)
    r.beats += hub.take_beats()
    assert hub.take_beats() == [] and len(r.beats) == 4
    r.save_partial(tmp_path)
    back = RideRecorder.load_partial(r.partial_path(tmp_path))
    assert [b[1:] for b in back.beats] == [(0.8, 1), (0.79, 1), (1.6, 2), (0.81, 1)]
    data = back.fit_bytes()
    assert fit_crc(data) == 0
    assert np.allclose(fit.read_rr(data), [0.8, 0.79, 1.6, 0.81])


def test_alpha1_from_a_goodpush_recordings_beats(tmp_path):
    """an indoor ride's alpha1 is goodpush's own, from the beat intervals in its FIT file (alpha.extract)."""
    from goodpush.alpha import extract
    r = recorder(hr=True)                                   # 5 min
    rng = np.random.default_rng(3)
    r.beats = [(T0 + i * 0.5, 0.5 + 0.01 * x, 1) for i, x in enumerate(rng.standard_normal(600))]
    f = tmp_path / "ride.fit"
    f.write_bytes(r.fit_bytes())
    row = extract(("goodpush_1", str(f), "k", str(tmp_path), "s"))
    assert row["has_alpha"] and row["source"] == "own", row
    z = np.load(tmp_path / "goodpush_1.a1.npz")
    a = z["a1"][np.isfinite(z["a1"])]
    assert a.size and 0.35 < np.median(a) < 0.8                   # uncorrelated beats: about DFA's 0.58
    assert extract(("none", str(tmp_path / "x.gpx"), "k", str(tmp_path), "s"))["has_alpha"] is False


def test_live_alpha1_during_the_ride(tmp_path):
    """the bar's alpha1: from the last 2 min of the strap's beats, worked out every 5 s; until the beats fill those
    2 min, how far along; then the value with its artifact share, however large."""
    hub = SensorHub({"trainer": "sim:fec:1", "hr": "ant:hr:2"})
    ride = RideSession(SimLink(hub), hub, WorkoutRun("T", STEPS), tmp_path, now=T0)
    rng = np.random.default_rng(4)
    for t in T0 + np.arange(0, 181):
        hub.put("sim:fec:1", "power", 150, t)
        hub.add_beats("ant:hr:2", [(0.5 + 0.01 * x, 1) for x in rng.standard_normal(2)], t=t)   # 120 bpm
        ride.tick(t)
        if t == T0 + 60:
            a, err, filled = ride.alpha1
            assert np.isnan(a) and 0.45 < filled < 0.55       # half of the 2 min
    a, err, filled = ride.alpha1
    assert 0.4 < a < 0.8 and err < 0.02 and filled == 1       # uncorrelated beats: about DFA's 0.58
    assert ride.alpha_at == T0 + 180
    ride.recorder.beats[-30::3] = [(t, 1.3, 1)] * 10          # 10 odd beats in the last 2 min: 4 % artifacts
    a, err, _ = ride.live_alpha1()
    assert np.isfinite(a) and 0.03 < err < 0.2


def test_same_ride_from_two_sources_counts_once(tmp_path):
    """the watch started 12 minutes before the trainer recording: past the old 5-minute start rule, but the two
    overlap - one ride, the goodpush copy kept and the Garmin one in `also`; deleting the kept copy leaves its twin;
    rides that only touch stay apart."""
    from goodpush.io.strava import SameRide
    rides, garmin = tmp_path / "goodpush", tmp_path / "garmin" / "activities"
    rides.mkdir()
    garmin.mkdir(parents=True)
    pd.DataFrame([{"id": "goodpush_1", "start_utc": "2026-11-10T17:12:00Z", "type": "Virtual Ride",
                   "elapsed_s": 3600, "filename": "a.fit"},
                  {"id": "goodpush_2", "start_utc": "2026-11-11T17:00:00Z", "type": "Virtual Ride",
                   "elapsed_s": 1800, "filename": "b.fit"}]).to_csv(rides / "activities_goodpush.csv", index=False)
    pd.DataFrame([{"id": "garmin_1", "start_utc": "2026-11-10T17:00:00Z", "type": "Virtual Ride", "elapsed_s": 4500,
                   "filename": "g1.fit"},
                  {"id": "garmin_2", "start_utc": "2026-11-11T17:30:00Z", "type": "Ride", "elapsed_s": 3600,
                   "filename": "g2.fit"},
                  {"id": "garmin_3", "start_utc": "2026-11-12T08:00:00Z", "type": "Ride", "elapsed_s": 3600,
                   "filename": "g3.fit"}]).to_csv(garmin / "activities_garmin.csv", index=False)
    acts = load_activities_csv(tmp_path / "strava", tmp_path / "garmin", rides)
    assert list(acts["id"]) == ["goodpush_1", "goodpush_2", "garmin_2", "garmin_3"]
    assert list(acts["also"]) == ["garmin_1", "", "", ""]
    again = load_activities_csv(tmp_path / "strava", tmp_path / "garmin", rides, deleted={"goodpush_1"})
    assert "garmin_1" in set(again["id"]) and "goodpush_1" not in set(again["id"])
    starts = pd.Series(pd.to_datetime(["2026-11-10T17:14:00Z", "2026-11-10T20:00:00Z", None], utc=True))
    table = pd.DataFrame({"start_utc": starts, "elapsed_s": [np.nan, 600, 600]})
    have = acts[["start_utc", "elapsed_s"]]
    assert list(SameRide.match(table, have)) == [0, -1, -1]       # near start without a length; apart; no start


LSCT_STEPS = [{"name": "LSCT 60 % HRmax", "min": 6, "lo": 112, "hi": 138, "kind": "active", "hr": .6,
               "mode": "resistance"},
              {"name": "LSCT 80 % HRmax", "min": 6, "lo": 175, "hi": 190, "kind": "active", "hr": .8,
               "mode": "resistance"},
              {"name": "Stop, sit still", "min": 1.5, "lo": 0, "hi": 0, "kind": "rest"},
              {"name": "Z2", "min": 5, "lo": 138, "hi": 162, "kind": "active"}]


def test_heart_rate_steps_ride_in_resistance_and_the_sitting_still_runs_on():
    run = WorkoutRun("LSCT", LSCT_STEPS, max_hr=190)
    assert run.hr_target() == 114 and run.mode == "resistance"     # it opens in resistance: you hold the heart rate
    assert run.target() is None and run.command() == ("resistance", 30)
    assert WorkoutRun("T", STEPS).mode == "erg" and WorkoutRun("Free", []).mode == "erg"
    assert run.ends_at(1000.0) == 1000.0 + 18.5 * 60                # riding on without a stop
    run.advance(60, riding=True)
    assert run.ends_at(1000.0) == 1000.0 + 17.5 * 60
    assert WorkoutRun("Free", []).ends_at(1000.0) is None
    run.seek(6 * 60)
    assert run.hr_target() == 152
    run.seek(12 * 60)
    run.mode = "erg"
    assert run.still and run.target() == 0 and run.hr_target() is None
    run.advance(60, riding=False)                          # not pedalling, and the clock runs
    assert run.elapsed == 13 * 60
    run.advance(60, riding=False)                          # ... until the step ends; then it waits for you again
    run.advance(60, riding=False)
    assert run.index == 3 and not run.still and run.elapsed == 13.5 * 60
    assert run.target() == 150                             # Z2 in ERG
    assert WorkoutRun("LSCT", LSCT_STEPS).hr_target() is None    # max HR unknown: no bpm to show


def test_bluetooth_device_keys():
    d = Device("ble", "hr", "C0:FF:EE:12:34:56", name="Strap 1")
    assert d.key == "ble:hr:C0:FF:EE:12:34:56" and Device.from_key(d.key) == d       # the name isn't in the key
    assert d.label == "Heart rate Strap 1 (Bluetooth)"
    assert Device.from_key(d.key).label == "Heart rate 34:56 (Bluetooth)"           # saved: no name
    assert Device.from_key("ant:fec:12345").number == 12345


def test_bluetooth_messages_feed_the_hub():
    from goodpush.ride.ble import BleLink, CrankCadence
    hub = SensorHub({"trainer": "ble:ftms:A", "power": "ble:power:B", "hr": "ble:hr:C"})
    link = BleLink(hub)
    # indoor bike data: speed 36 km/h, cadence 90 rpm (0.5), power 250 W
    bike = struct.pack("<HHHh", 0x0044, 3600, 180, 250)
    assert BleLink.indoor_bike(bike) == {"speed": 10.0, "cadence": 90.0, "power": 250}
    link.on_bike("ble:ftms:A", None, bytearray(bike))
    assert hub.get("speed") == 10.0
    # cycling power: balance (bit 0) and crank data (bit 5)
    cp = struct.pack("<HhBHH", 0x21, 230, 50, 100, 1024)
    assert BleLink.cycling_power(cp) == (230, (100, 1024))
    assert BleLink.csc(struct.pack("<BIHHH", 3, 5, 0, 7, 2048)) == (7, 2048)       # wheel data first
    # heart rate: uint16 bpm, energy expended, two intervals
    assert BleLink.heart_rate(struct.pack("<BHHHH", 0x19, 150, 9, 820, 410)) == (150, [820 / 1024, 410 / 1024])
    link.on_hr("ble:hr:C", None, bytearray(struct.pack("<BBH", 0x10, 140, 512)))
    assert hub.get("hr") == 140 and [b[1:] for b in hub.take_beats()] == [(0.5, 1)]
    c = CrankCadence()
    assert c.heard(100, 0, 0.0) is None                       # the first message: nothing to compare with
    assert c.heard(101, 683, 0.7) == pytest.approx(60 / (683 / 1024))
    assert c.heard(101, 683, 1.5) is None and c.heard(101, 683, 4.0) == 0.0     # stopped pedalling
    assert c.heard(102, 65000, 4.5) == pytest.approx(60 / ((65000 - 683) / 1024))  # (a long pause, within 64 s)
    assert c.heard(103, 100, 70.0) is None                    # past the rollover: afresh


class FakeTrainer:
    """a Bluetooth trainer's control point: answers each write with the result code in `codes` (default success;
    a callable gets the message; None: no answer)."""

    def __init__(self, link, codes: dict):
        self.link, self.codes, self.sent = link, codes, []

    async def write_gatt_char(self, uuid, message, response=None):
        from goodpush.ride import ble
        self.sent.append(bytes(message))
        code = self.codes.get(message[0], ble.SUCCESS)
        if callable(code):
            code = code(bytes(message))
        if code is not None:
            self.link.on_answer(None, bytearray([ble.ANSWER, message[0], code]))

    async def ride(self):
        from goodpush.ride import ble
        link = self.link
        assert await link.deliver(self, ("power", 200)) == ble.SUCCESS
        assert [m[0] for m in self.sent] == [ble.REQUEST_CONTROL, ble.START, ble.POWER]    # control first, once
        assert await link.deliver(self, ("resistance", 50)) == ble.SUCCESS                # one byte, FTMS 1.0's way
        assert self.sent[-1] == bytes([0x04, 100]) and link.level_bytes == 1
        self.codes[ble.POWER] = 0x02
        assert await link.deliver(self, ("power", 300)) == 0x02
        self.codes[ble.POWER] = None                             # no answer at all
        assert await link.deliver(self, ("power", 300)) is None


def test_bluetooth_trainer_commands(monkeypatch):
    from goodpush.ride import ble
    from goodpush.ride.ble import BleLink
    monkeypatch.setattr(ble, "ACK_WAIT_S", 0.05)
    link = BleLink(SensorHub({"trainer": "ble:ftms:A"}))
    link.level = (0.0, 20.0)
    assert link.payload(("power", 250)) == bytes([0x05, 0xFA, 0x00])
    assert link.payload(("resistance", 50)) == struct.pack("<Bh", 0x04, 100)       # level 10 of 0-20 (0.1)
    assert link.payload(("grade", 4.5)) == struct.pack("<BhhBB", 0x11, 0, 450, 40, 51)
    asyncio.run(FakeTrainer(link, {ble.RESISTANCE: lambda m: ble.INVALID if len(m) == 3 else ble.SUCCESS}).ride())
    link.misses, link.refused = 0, "NotSupported"
    assert link.trouble == "Trainer refused the command (NotSupported)"
    link.misses = ble.MISSES
    assert link.trouble == "Trainer isn't taking commands"


class FakeLink:
    """an ANT+ / Bluetooth link for RideLinks: finds `found` (or fails with `fails`), records what it's told."""

    def __init__(self, found=(), fails=""):
        self.found, self.fails, self.connected, self.sent, self.closed = list(found), fails, [], [], False
        self.trouble = ""

    def scan(self, seconds, on_found=None, stop=None):
        if self.fails:
            raise RuntimeError(self.fails)
        return self.found

    def connect(self, devices):
        self.connected += devices

    def set_power(self, watts):
        self.sent.append(watts)

    def close(self):
        self.closed = True


def test_ant_and_bluetooth_in_one_ride():
    from goodpush.ride.links import RideLinks
    hr, trainer = Device("ble", "hr", "AA:BB"), Device("ant", "fec", 7)
    hub = SensorHub({"trainer": trainer.key, "hr": hr.key})
    ant_, ble_ = FakeLink([trainer]), FakeLink([hr])
    links = RideLinks(hub, {"ant": ant_, "ble": ble_})
    assert sorted(d.key for d in links.scan(0.1)) == sorted([trainer.key, hr.key])
    links.connect(hub.devices())
    assert ant_.connected == [trainer] and ble_.connected == [hr]
    links.set_power(220)
    assert ant_.sent == [220] and ble_.sent == []                  # commands to the trainer's link only
    ant_.trouble = "Trainer isn't taking commands"
    assert links.trouble == ant_.trouble
    links.close()
    assert ant_.closed and ble_.closed
    no_stick = RideLinks(hub, {"ant": FakeLink(fails="No ANT+ USB stick found."), "ble": FakeLink([hr])})
    assert no_stick.scan(0.1) == [hr]                              # Bluetooth alone is fine
    with pytest.raises(RuntimeError, match="No ANT"):
        RideLinks(hub, {"ant": FakeLink(fails="No ANT+ USB stick found."), "ble": FakeLink()}).scan(0.1)
