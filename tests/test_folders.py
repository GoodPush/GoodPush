import os
import sys
from pathlib import Path

import platformdirs
import pytest

from goodpush.athletes import Athletes
from goodpush.folders import Folders


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """a computer from before: ~/.goodpush with sign-ins and the registry, and the installed app's project in the
    app's data folder itself. Folders.migrate may run (conftest stops it everywhere else)."""
    legacy, data = tmp_path / "home" / ".goodpush", tmp_path / "data"
    (legacy / "garmin_tokens").mkdir(parents=True)
    (legacy / "garmin_personal_api.json").write_text("{}")
    (data / "data" / "raw").mkdir(parents=True)
    (data / "goodpush.toml").write_text("# the app's project\n")
    (data / "data" / "raw" / "ride.txt").write_text("a ride")
    Athletes(legacy).add("athlete_name", data)
    monkeypatch.setattr(Folders, "LEGACY", legacy)
    monkeypatch.setattr(Folders, "data_dir", classmethod(lambda cls: data))
    monkeypatch.setattr(Folders, "migrated", False)
    monkeypatch.delenv(Folders.HOME_ENV, raising=False)
    return legacy, data


def test_folders_are_where_they_were_and_can_move(tmp_path, monkeypatch):
    """the default project's parent stays the place it had before platformdirs (installs keep their data); the
    machine folder is the platform's; both follow their environment variable."""
    old = {"darwin": Path.home() / "Library" / "Application Support",
           "win32": Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")}.get(
        sys.platform, Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"))
    assert Folders.data_dir() == old / "goodpush"
    monkeypatch.setattr(Folders, "data_dir", classmethod(lambda cls: tmp_path))   # not this computer's own
    monkeypatch.delenv(Folders.HOME_ENV, raising=False)
    assert Folders.default_project() == tmp_path / "athlete"
    monkeypatch.setenv(Folders.HOME_ENV, str(tmp_path / "elsewhere"))
    assert Folders.default_project() == tmp_path / "elsewhere"
    assert Folders.machine() == Path(os.environ[Folders.CONFIG_ENV])          # conftest's: the tests' own
    if sys.platform == "win32":                                                    # Incyclist: unchanged too
        from goodpush.sync.incyclist import default_folder
        assert default_folder() == Path(os.environ["APPDATA"]) / "Incyclist"
        assert platformdirs.user_config_dir("goodpush", appauthor=False, roaming=False) == \
            str(Path(os.environ["LOCALAPPDATA"]) / "goodpush")


@pytest.mark.parametrize("shared", [False, True], ids=["apart", "macos"])
def test_old_folders_move_once(machine, tmp_path, monkeypatch, shared):
    """~/.goodpush -> the machine folder, the project down into athlete/, its registry entry follows. On macOS the
    machine folder is the app's data folder itself (`shared`): the project's subfolder sits next to the sign-ins."""
    legacy, data = machine
    config = data if shared else tmp_path / "config"
    monkeypatch.setenv(Folders.CONFIG_ENV, str(config))
    assert (legacy / "garmin_tokens").is_dir() and Folders.machine() == config     # asking moves nothing
    Folders.migrate()                                                            # the entry points do
    project = data / "athlete"
    assert (project / "goodpush.toml").exists() and (project / "data" / "raw" / "ride.txt").read_text() == "a ride"
    assert not (data / "goodpush.toml").exists() and not legacy.exists()
    assert (config / "garmin_personal_api.json").exists() and (config / "garmin_tokens").is_dir()
    assert Athletes(config).folder("athlete_name") == project
    assert Athletes(config).copy_credentials("athlete_name") == ["garmin_personal_api.json", "garmin_tokens"]
    monkeypatch.setattr(Folders, "migrated", False)                              # the next start: nothing to do
    Folders.migrate()
    assert sorted(p.name for p in project.iterdir()) == ["data", "goodpush.toml"]


def test_a_clash_overwrites_nothing_and_keeps_the_old_project_in_use(machine, tmp_path, monkeypatch):
    """a name at both places: both kept. The project's goodpush.toml stays until all else moved, so the app keeps
    using the old place (no second, empty project) and the move goes on once you've sorted it out."""
    legacy, data = machine
    config = tmp_path / "config"
    monkeypatch.setenv(Folders.CONFIG_ENV, str(config))
    (config / "garmin_tokens").mkdir(parents=True)                               # already at the new place
    (data / "athlete" / "data").mkdir(parents=True)
    (data / "logs").mkdir()
    Folders.migrate()
    assert Folders.default_project() == data                                     # still the old one
    assert (legacy / "garmin_tokens").is_dir() and (config / "garmin_personal_api.json").exists()
    assert (data / "data" / "raw" / "ride.txt").exists() and (data / "athlete" / "logs").is_dir()
    assert (data / "goodpush.toml").exists() and not (data / "athlete" / "goodpush.toml").exists()
    os.replace(data / "data", tmp_path / "kept")                                 # you sorted it out
    monkeypatch.setattr(Folders, "migrated", False)
    Folders.migrate()
    assert Folders.default_project() == data / "athlete" and (data / "athlete" / "goodpush.toml").exists()
    assert Athletes(config).folder("athlete_name") == data / "athlete"


def test_privacy_hook_finds_its_terms_in_the_machine_folder(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    try:
        from privacy_check import PrivacyCheck
    finally:
        sys.path.pop(0)
    monkeypatch.delenv("GOODPUSH_SECRETS", raising=False)
    (tmp_path / "privacy_terms.txt").write_text("athlete_name\n", encoding="utf-8")
    monkeypatch.setenv("GOODPUSH_CONFIG", str(tmp_path))
    assert PrivacyCheck().terms == ["athlete_name"]


def test_a_goodpush_command_moves_them(machine, tmp_path, monkeypatch, capsys):
    from goodpush.cli import main
    legacy, data = machine
    monkeypatch.setenv(Folders.CONFIG_ENV, str(tmp_path / "config"))
    monkeypatch.setattr(Athletes, "HOME", None)
    main(["athlete", "list"])
    assert not legacy.exists() and "athlete_name" in capsys.readouterr().out
    assert Athletes().folder("athlete_name") == data / "athlete"
