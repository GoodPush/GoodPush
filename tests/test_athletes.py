import os

import pytest

from goodpush.athletes import Athletes
from goodpush.config import load_project


@pytest.fixture
def home(tmp_path, monkeypatch):
    """a machine with one athlete's sign-ins from before there were athletes (~/.goodpush)."""
    for k in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(k, str(tmp_path))
    for k in ("GOODPUSH_SECRETS", "GOODPUSH_ATHLETE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(Athletes, "SET", None)
    m = tmp_path / ".goodpush"
    monkeypatch.setattr(Athletes, "HOME", m)
    (m / "garmin_tokens").mkdir(parents=True)
    (m / "garmin_personal_api.json").write_text("{}")
    (m / "privacy_terms.txt").write_text("x")
    return tmp_path


def test_athletes_have_their_own_folder_and_credentials(home, monkeypatch):
    from goodpush.cli import main
    from goodpush.sync import secrets_dir
    athlete_name, other_athlete = home / "athlete_name", home / "other_athlete"
    main(["athlete", "add", "athlete_name", str(athlete_name), "--credentials"])             # your folder + your sign-ins
    main(["athlete", "add", "other_athlete", str(other_athlete)])                              # a new one: a fresh folder
    reg = Athletes()
    assert reg.default == "athlete_name" and reg.folder("other_athlete") == other_athlete.resolve() and (other_athlete / "goodpush.toml").exists()
    mine = reg.secrets("athlete_name")
    assert sorted(p.name for p in mine.iterdir()) == ["garmin_personal_api.json", "garmin_tokens"]   # not the machine's
    assert (home / ".goodpush" / "garmin_personal_api.json").exists()          # copied, nothing removed
    assert load_project(athlete="other_athlete").root == other_athlete.resolve() and secrets_dir() == reg.secrets("other_athlete")
    assert load_project(root=athlete_name).root == athlete_name.resolve() and secrets_dir() == mine   # found by folder too
    monkeypatch.chdir(home)
    assert load_project().root == athlete_name.resolve()                             # nowhere in particular: the default
    monkeypatch.setenv("GOODPUSH_SECRETS", str(home / "own"))                  # one you set yourself wins
    load_project(athlete="other_athlete")
    assert os.environ["GOODPUSH_SECRETS"] == str(home / "own")
    with pytest.raises(KeyError, match="No athlete 'bob'"):
        load_project(athlete="bob")
    with pytest.raises(ValueError):
        reg.add("Bob Smith", home / "bob")


class BusyJobs:
    """a job is running"""

    def status(self):
        return {"running": True, "queued": 0}


def test_app_switches_and_adds_athletes(home):
    from goodpush.app.server import AthleteSwitch, DailyRanges, Data, Jobs, PeerHub
    from goodpush.config import init_project
    from goodpush.sync import secrets_dir
    reg = Athletes()
    for name in ("athlete_name", "other_athlete"):
        reg.add(name, home / "athletes" / name)
        init_project(home / "athletes" / name)
    data, jobs = Data(load_project(athlete="athlete_name")), Jobs()
    hub = PeerHub(data, jobs, 0)
    switch = AthleteSwitch(data, hub, DailyRanges(jobs, hub.home_here, lambda: data.proj))
    st = switch.status()
    assert st["current"] == "athlete_name" and [a["name"] for a in st["athletes"]] == ["athlete_name", "other_athlete"]
    switch.switch("other_athlete")
    assert data.proj.root == (home / "athletes" / "other_athlete").resolve()
    assert secrets_dir() == Athletes().secrets("other_athlete") and hub.server.side.root == data.proj.root
    switch.add("new_one", "")                                       # no folder: next to the default athlete's
    assert data.proj.root == (home / "athletes" / "new_one").resolve() and (data.proj.root / "goodpush.toml").exists()
    assert Athletes().default == "athlete_name" and switch.status()["current"] == "new_one"
    with pytest.raises(ValueError, match="already"):
        switch.add("new_one", "")
    with pytest.raises(ValueError, match="no athlete"):
        switch.switch("bob")
    switch.ranges.jobs = BusyJobs()
    with pytest.raises(ValueError, match="current job"):
        switch.switch("athlete_name")


def test_removing_an_athlete_keeps_their_folder(home, capsys):
    from goodpush.app.server import AthleteSwitch, DailyRanges, Data, Jobs, PeerHub
    from goodpush.cli import main
    first, second = home / "athletes" / "first", home / "athletes" / "second"
    main(["athlete", "add", "first", str(first), "--credentials"])
    main(["athlete", "add", "second", str(second), "--credentials"])
    reg = Athletes()
    assert reg.secrets("first").exists() and reg.default == "first"
    main(["athlete", "remove", "first"])                                 # the default one: another takes over
    reg = Athletes()
    assert list(reg.folders) == ["second"] and reg.default == "second"
    assert not reg.secrets("first").exists() and (first / "goodpush.toml").exists()   # sign-ins go, the folder stays
    assert (home / ".goodpush" / "garmin_tokens").exists()                # the machine's own stay too
    assert "folder stays" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="No athlete 'first'"):
        main(["athlete", "remove", "first"])
    reg.add("first", first)
    data, jobs = Data(load_project(athlete="second")), Jobs()
    hub = PeerHub(data, jobs, 0)
    switch = AthleteSwitch(data, hub, DailyRanges(jobs, hub.home_here, lambda: data.proj))
    with pytest.raises(ValueError, match="another athlete first"):
        switch.remove("second")                                         # not the one the app shows
    with pytest.raises(ValueError, match="no athlete"):
        switch.remove("bob")
    switch.remove("first")
    assert [a["name"] for a in switch.status()["athletes"]] == ["second"]
    reg = Athletes()
    reg.remove("second")
    assert Athletes().folders == {} and Athletes().default == ""
