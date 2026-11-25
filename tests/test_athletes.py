import os
from pathlib import Path

import pytest

from goodpush.athletes import Athletes, FolderMove
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


def athlete_folder(root, reg, name="athlete_name"):
    """a registered athlete folder with a few files of old dates (what peer sync compares: size, time)."""
    from goodpush.config import init_project
    init_project(root)
    (root / "data" / "raw" / "rides").mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(("a.fit", "b.fit")):
        f = root / "data" / "raw" / "rides" / n
        f.write_bytes(bytes(range(256)) * (i + 1))
        os.utime(f, (1_700_000_000 + i, 1_700_000_000 + i))
    (root / "data" / "empty").mkdir()
    reg.add(name, root)
    return root


@pytest.mark.parametrize("drive", ["same", "other"])
def test_moving_a_folder_keeps_what_sync_compares(tmp_path, monkeypatch, drive):
    """same drive: a rename; another: a copy keeping file times, checked, then the old folder deleted. Either way
    peer sync's view (FolderSide.files: size, time) is unchanged, so the next sync has nothing to send."""
    from goodpush.sync.peer import FolderSide
    reg = Athletes()
    old = athlete_folder(tmp_path / "old", reg)
    seen = FolderSide(old).files()
    monkeypatch.setattr(FolderMove, "same_drive", lambda self: drive == "same")
    (tmp_path / "new").mkdir()                                                   # empty: fine
    new = FolderMove(reg, "athlete_name", tmp_path / "new" / "athlete_name", progress=lambda *a: None).run()
    assert not old.exists() and Athletes().folder("athlete_name") == new
    assert FolderSide(new).files() == seen and (new / "data" / "empty").is_dir()


class ShortCopy(FolderMove):
    """a move to another drive where one file arrives cut short."""

    def same_drive(self) -> bool:
        return False

    def copy(self, src: str, dst: str) -> str:
        out = super().copy(src, dst)
        if dst.endswith("b.fit"):
            with open(dst, "r+b") as f:
                f.truncate(10)
        return out


def test_a_copy_that_differs_is_removed_and_the_folder_stays(tmp_path):
    reg = Athletes()
    old = athlete_folder(tmp_path / "old", reg)
    with pytest.raises(ValueError, match="differs"):
        ShortCopy(reg, "athlete_name", tmp_path / "new", progress=lambda *a: None).run()
    assert not (tmp_path / "new").exists() and (old / "data" / "raw" / "rides" / "b.fit").stat().st_size == 512
    assert Athletes().folder("athlete_name") == old


def test_a_move_refuses_before_changing_anything(tmp_path):
    reg = Athletes()
    old = athlete_folder(tmp_path / "old", reg)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x.txt").write_text("x")
    for dest, why in ((tmp_path / "full", "isn't empty"), (old / "inside", "outside their folder"),
                      (tmp_path, "outside their folder"), (Path("relative"), "whole folder path"),
                      (old, "already")):
        with pytest.raises(ValueError, match=why):
            FolderMove(reg, "athlete_name", dest)
    with pytest.raises(ValueError, match="No athlete 'bob'"):
        FolderMove(reg, "bob", tmp_path / "new")
    assert Athletes().folder("athlete_name") == old and (old / "goodpush.toml").exists()


def test_a_folder_you_moved_yourself_is_pointed_to(tmp_path, capsys):
    from goodpush.cli import main
    reg = Athletes()
    old = athlete_folder(tmp_path / "old", reg)
    os.replace(old, tmp_path / "moved")
    with pytest.raises(SystemExit, match="has no goodpush.toml"):
        main(["athlete", "move", "athlete_name", str(tmp_path / "elsewhere")])
    main(["athlete", "move", "athlete_name", str(tmp_path / "moved")])
    assert Athletes().folder("athlete_name") == tmp_path / "moved"
    with pytest.raises(SystemExit, match="already, at"):                         # add no longer re-points
        main(["athlete", "add", "athlete_name", str(tmp_path / "other")])
    main(["athlete", "add", "athlete_name", str(tmp_path / "moved"), "--default"])   # the same folder: fine


def test_the_app_moves_the_folder_it_shows(home):
    """as a job; the app then serves the new place (its paired computers' server restarted on it)."""
    import time
    from goodpush.app.server import AthleteSwitch, DailyRanges, Data, Jobs, PeerHub
    reg = Athletes()
    old = athlete_folder(home / "athletes" / "first", reg, "first")
    data, jobs = Data(load_project(athlete="first")), Jobs()
    hub = PeerHub(data, jobs, 0)
    switch = AthleteSwitch(data, hub, DailyRanges(jobs, hub.home_here, lambda: data.proj))
    with pytest.raises(ValueError, match="isn't empty"):
        switch.move("first", str(home / ".goodpush"))
    assert switch.move("first", str(home / "elsewhere" / "first")) is True
    for _ in range(100):
        if not jobs.status()["running"]:
            break
        time.sleep(0.05)
    assert jobs.status()["ok"], jobs.status()["log"]
    assert data.proj.root == (home / "elsewhere" / "first").resolve() and not old.exists()
    assert switch.status()["current"] == "first"
