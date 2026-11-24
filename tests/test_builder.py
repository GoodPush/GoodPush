"""Workouts you write: the text format both ways, the type it suits, your library, and adding one to the plan."""
import pandas as pd
import pytest

from goodpush.planning import LADDERS, PlanState
from goodpush.planning.builder import MyWorkouts, WorkoutText
from goodpush.planning.steps import Repeat, Step, flat

TEXT = """# threshold day
Warm up 10m 50%>70%
3x
  Threshold 8m 95-100%
  4m 55%
5x 30s 120%, 30s 50%
Tempo 20m Z3 90rpm
Steady 1h30m 200-210W
Cool down 1:30 50%
"""


def test_parse_every_form():
    wt = WorkoutText(ftp=250)
    items = wt.parse(TEXT)
    assert not wt.errors
    warm, thr, micro, tempo, steady, cool = items
    assert warm.kind == "warmup" and warm.ramp and (warm.mid, warm.mid_end) == pytest.approx((.5, .7))
    assert isinstance(thr, Repeat) and thr.reps == 3 and thr.steps[0].name == "Threshold"
    assert (thr.steps[0].lo, thr.steps[0].hi) == pytest.approx((.95, 1.0))
    assert thr.steps[1].kind == "rest" and thr.steps[1].name == "Recover"        # easy step in a repeat: recovery
    assert micro.reps == 5 and [s.minutes for s in micro.steps] == pytest.approx([.5, .5]) and micro.steps[1].kind == "rest"
    assert (tempo.lo, tempo.hi, tempo.cadence) == (.76, .90, 90)
    assert steady.minutes == 90 and (steady.lo, steady.hi) == pytest.approx((.8, .84))
    assert cool.kind == "cooldown" and cool.minutes == pytest.approx(1.5)
    assert sum(s.minutes for s in flat(items)) == pytest.approx(10 + 36 + 5 + 20 + 90 + 1.5)


def test_blank_line_ends_an_unindented_repeat_and_errors_name_the_line():
    wt = WorkoutText()
    items = wt.parse("4x\n1m 120%\n1m 50%\n\n10m 60%\nwhat 90%\nTempo 10m\n2x\n")
    assert isinstance(items[0], Repeat) and len(items[0].steps) == 2 and items[1].minutes == 10
    assert [e.line for e in wt.errors] == [6, 7, 8]
    w = WorkoutText()
    assert w.parse("10m 200W") == [] and w.errors[0].message == "watts need your FTP"


def test_text_round_trip():
    wt = WorkoutText(ftp=250)
    items = wt.parse(TEXT)
    again = WorkoutText().parse(WorkoutText.format(items))
    assert [x.to_dict() for x in again] == [x.to_dict() for x in items]
    assert WorkoutText.duration_text(91.5) == "1h31m30s" and WorkoutText.format([Step("Easy", 5, .5, .5)]) == "Easy 5m 50%"
    z2 = WorkoutText().parse(WorkoutText.format([Step("Z2", 21.25, .55, .693)]))[0]   # a step named like a zone
    assert (z2.name, z2.minutes, z2.lo, z2.hi) == ("Z2", 21.25, .55, .693)


def test_guess_type():
    g = lambda t: WorkoutText.guess_type(WorkoutText().parse(t))        # noqa: E731
    assert g("10m 50%\n5x\n  4m 115%\n  4m 50%") == "vo2max"
    assert g("2x\n  20m 97%\n  5m 50%") == "threshold"
    assert g("3x\n  15m 90%\n  5m 50%") == "sweetspot"
    assert g("2x\n  20m 80%\n  5m 50%") == "tempo"
    assert g("8x\n  30s 150%\n  2m 50%") == "anaerobic"
    assert g("90m 65%") == "endurance" and g("3h 65%") == "long" and g("45m 50%") == "recovery"


def test_my_workouts_library(tmp_path):
    lib = MyWorkouts(tmp_path)
    steps = [x.to_dict() for x in WorkoutText().parse("10m 60%\n3x\n  8m 95%\n  4m 55%")]
    a = lib.save("My Threshold!", "threshold", steps)
    b = lib.save("My Threshold!", "threshold", steps)
    assert (a, b) == ("my-threshold", "my-threshold-2")
    w = lib.load()[a]
    assert w["minutes"] == 46 and w["text"].startswith("Endurance 10m 60%") and w["type"] == "threshold"
    lib.save("Renamed", "tempo", steps, wid=a)
    assert lib.load()[a]["title"] == "Renamed"
    lib.delete(b)
    assert list(lib.load()) == [a]
    with pytest.raises(ValueError):
        lib.save("x", "threshold", [])
    with pytest.raises(ValueError):
        lib.save("x", "sprints", steps)


def test_add_your_workout_to_the_plan():
    today = pd.Timestamp("2026-10-22")
    st = PlanState({"levels": {k: 2 for k in LADDERS}, "planned": {
        "2026-10-24": {"type": "endurance", "title": "Endurance", "minutes": 90},
        "2026-10-20": {"type": "rest", "title": "Rest"}}})
    d = {"type": "threshold", "title": "Mine", "steps": [x.to_dict() for x in WorkoutText().parse("10m 60%\n2x\n  20m 97%\n  5m 50%")]}
    k = st.add(d, "2026-10-24", today)
    assert k == "2026-10-24#2" and st["manual_plan"][k]["minutes"] == 60 and st["manual_plan"][k]["why_change"] == "your workout"
    assert st.add(d, "2026-10-24", today, key="2026-10-24") == "2026-10-24"          # in place of the plan's session
    assert st["manual_plan"]["2026-10-24"]["type"] == "threshold"
    past = st.add(d, "2026-10-20", today)                                             # a past day: recorded, to be scored
    assert st["planned"][past]["type"] == "threshold"
    with pytest.raises(ValueError):
        st.add({**d, "steps": []}, "2026-10-25", today)
    with pytest.raises(ValueError):
        st.add(d, "2026-10-25", today, key="2026-10-24")


def wait_for_jobs(url: str, seconds: float = 120) -> None:
    """until the app's background jobs (and the one queued) are done."""
    import json
    import time
    import urllib.request
    end = time.time() + seconds
    while time.time() < end:
        j = json.loads(urllib.request.urlopen(url).read())
        if not j.get("running") and not j.get("queued"):
            return
        time.sleep(0.2)
    raise TimeoutError("the app's job did not finish")


def test_app_easier_harder_for_a_planned_session_and_your_own_workout(tmp_path):
    """The calendar's easier / harder moves the session's format a step from now on (nothing saved at the limit);
    the builder's moves your workout from the steps as built; the calendar tells which way is still open."""
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from goodpush.app.server import calendar_data, make_handler
    from goodpush.config import init_project, load_project
    from goodpush.planning import WorkoutLibrary as WL
    init_project(tmp_path)
    proj = load_project(tmp_path)
    sp = proj.processed / "plan_state.json"
    day = pd.Timestamp.now(tz=proj.tz).tz_localize(None).normalize() + pd.Timedelta(days=30)
    key = str(day.date())
    fmt = WL.format_of(day, "vo2max")
    record = {"type": "vo2max", "level": 1, "title": WL.intervals(day, "vo2max", 1).title, "minutes": 70}
    with PlanState.open(sp) as st:
        st["planned"][key] = dict(record)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, "k" * 20, lan=False))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/api/action"
    post = lambda b: json.loads(urllib.request.urlopen(urllib.request.Request(  # noqa: E731
        url, json.dumps(b).encode(), {"Content-Type": "application/json", "X-GoodPush": "1"})).read())
    try:
        assert calendar_data(proj, None, None)["planned"][key]["adjust"] == \
            {"format": fmt, "offset": 0, "easier": True, "harder": True}
        seen = post({"action": "adjust_preview", "key": key, "step": -2})    # the open window: shown, nothing saved
        assert seen["adjust"]["offset"] == -2 and seen["steps"] and seen["blocks"] and seen["title"]
        assert PlanState.load(sp).offsets() == {}
        post({"action": "adjust_session", "key": key, "step": -1})
        wait_for_jobs(url.replace("/action", "/job"))          # the re-plan it started
        assert PlanState.load(sp).offsets() == {"vo2max": {fmt: -1}}
        with PlanState.open(sp) as st:
            st.set_offset("vo2max", fmt, -99)                  # as easy as it gets for this session
            st["planned"][key] = dict(record)
        adj = calendar_data(proj, None, None)["planned"][key]["adjust"]
        assert adj["offset"] < 0 and not adj["easier"] and adj["harder"]
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"action": "adjust_session", "key": key, "step": -1})
        assert "as easy as" in json.loads(e.value.read())["error"]
        assert PlanState.load(sp).offsets() == {"vo2max": {fmt: -99}}       # nothing saved
        steps = [x.to_dict() for x in WorkoutText().parse("Warm up 10m 50%\n4x\n  Hard 4m 110%\n  Recover 3m 50%")]
        j = post({"action": "workout_adjust", "steps": steps, "offset": -1})     # a rep fewer first
        assert j["offset"] == -1 and j["easier"] and j["harder"] and "3x" in j["text"] and "3m45s" not in j["text"]
        assert "2x" in post({"action": "workout_adjust", "steps": steps, "offset": -5})["text"]
        j = post({"action": "workout_adjust", "steps": steps, "offset": 9})
        assert j["offset"] == 1 and not j["harder"] and j["steps"][1]["reps"] == 4      # shorter rests; never power
    finally:
        srv.shutdown()
