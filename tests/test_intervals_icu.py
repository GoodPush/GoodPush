import json

import pandas as pd
import pytest

from goodpush.sync.intervals_icu import IntervalsIcu, IntervalsIcuError


class FakeIcu(IntervalsIcu):
    """intervals.icu without the network: records the calls."""

    def __init__(self, key="k" * 20):
        super().__init__(key)
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return {"name": "Rider"} if path == "" else None


def plan(tmp_path, sessions):
    d = tmp_path / "plan"
    (d / "workouts").mkdir(parents=True, exist_ok=True)
    for s in sessions:
        if s.get("icu"):
            (d / "workouts" / s["icu"]).write_text(f"- 10m 55%\n# {s['title']}\n", encoding="utf-8")
    (d / "plan.json").write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    return d


def session(date, title, key=None, icu=True):
    return {"date": date, "key": key or date, "title": title, "minutes": 60, "tss": 50,
            "icu": f"gp_{date}_{title}.txt" if icu else ""}


def test_push_upserts_the_coming_days_and_removes_dropped_ones(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    today = pd.Timestamp("2026-11-23")
    d = plan(tmp_path, [session("2026-11-22", "Past"), session("2026-11-23", "Z2"), session("2026-11-23", "Second", "2026-11-23#2"),
                        session("2026-11-24", "Rest", icu=False), session("2026-11-25", "Threshold"),
                        session("2026-12-20", "Far")])
    icu = FakeIcu()
    res = icu.push(d, today, progress=lambda m: None)
    (method, path, evs), = icu.calls                       # nothing sent before: nothing to delete
    assert (method, path) == ("POST", "/events/bulk?upsert=true") and res["sent"] == 3
    assert [e["external_id"] for e in evs] == ["goodpush_2026-11-23", "goodpush_2026-11-23#2", "goodpush_2026-11-25"]
    e = evs[0]
    assert e["category"] == "WORKOUT" and e["type"] == "Ride" and e["start_date_local"] == "2026-11-23T00:00:00"
    assert e["name"] == "GoodPush - Z2"
    assert e["description"].startswith("- 10m 55%") and e["moving_time"] == 3600 and e["icu_training_load"] == 50
    # the plan moves the threshold session and drops the second one: they are deleted, the rest updated
    d = plan(tmp_path, [session("2026-11-23", "Z2"), session("2026-11-26", "Threshold")])
    icu = FakeIcu()
    res = icu.push(d, today, progress=lambda m: None)
    assert [c[:2] for c in icu.calls] == [("POST", "/events/bulk?upsert=true"), ("PUT", "/events/bulk-delete")]
    assert icu.calls[1][2] == [{"external_id": "goodpush_2026-11-23#2"}, {"external_id": "goodpush_2026-11-25"}]
    assert res == {**res, "sent": 2, "deleted": 2} and IntervalsIcu.info()["last_push"]["deleted"] == 2
    # a day later yesterday's workout stays in intervals.icu (past days are left alone)
    icu = FakeIcu()
    icu.push(plan(tmp_path, [session("2026-11-26", "Threshold")]), today + pd.Timedelta(days=1), progress=lambda m: None)
    assert [c[0] for c in icu.calls] == ["POST"]


def test_key_is_checked_stored_and_forgotten(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    monkeypatch.setattr(IntervalsIcu, "request", FakeIcu.request)
    monkeypatch.setattr(IntervalsIcu, "calls", [], raising=False)
    with pytest.raises(ValueError):
        IntervalsIcu.save_key("short key")
    assert IntervalsIcu.save_key("  abcdefghijklmnop  ") == "Rider"
    assert IntervalsIcu().key == "abcdefghijklmnop" and IntervalsIcu.info()["configured"]
    IntervalsIcu.forget()
    assert not IntervalsIcu.info()["configured"]
    with pytest.raises(IntervalsIcuError):
        IntervalsIcu().push(tmp_path, pd.Timestamp("2026-11-23"))


def test_a_plan_written_without_intervals_text_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    old = {**session("2026-11-24", "Z2"), "fit": "gp_2026-11-24_Z2.fit"}
    del old["icu"]                                          # written by a goodpush without intervals.icu text
    d = plan(tmp_path, [old])
    assert IntervalsIcu.needs_files(d) and not IntervalsIcu.needs_files(tmp_path / "nothing")
    with pytest.raises(IntervalsIcuError, match="update the plan"):
        FakeIcu().push(d, pd.Timestamp("2026-11-23"), progress=lambda m: None)
    d = plan(tmp_path, [{**session("2026-11-24", "Z2"), "fit": "gp_2026-11-24_Z2.fit"}])
    assert not IntervalsIcu.needs_files(d)


def test_push_writes_an_old_plan_again_first(tmp_path, monkeypatch):
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path / "secrets"))
    old = {**session("2026-11-24", "Z2"), "fit": "gp_2026-11-24_Z2.fit"}
    del old["icu"]
    d = plan(tmp_path, [old])
    icu = FakeIcu()
    res = icu.push(d, pd.Timestamp("2026-11-23"), progress=lambda m: None,
                   rewrite=lambda: plan(tmp_path, [{**session("2026-11-24", "Z2"), "fit": "gp_2026-11-24_Z2.fit"}]))
    assert res["sent"] == 1 and icu.calls[0][2][0]["external_id"] == "goodpush_2026-11-24"
