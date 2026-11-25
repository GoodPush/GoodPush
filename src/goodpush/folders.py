"""Where goodpush keeps things on this computer, in the platform's own places (platformdirs):

* the machine folder: this computer's credentials, the athlete registry, ride devices (never inside a project):
  %LOCALAPPDATA%\\goodpush, ~/Library/Application Support/goodpush, ~/.config/goodpush;
* the installed app's default project (when goodpush isn't started from a project folder and no athlete is
  registered): <the app's data folder>/athlete (%APPDATA%\\goodpush\\athlete, ...).

Either can go elsewhere, e.g. another drive: GOODPUSH_CONFIG (the machine folder) and GOODPUSH_HOME (the default
project). The folders from before (~/.goodpush, and the default project one level up) are moved over once by
Folders.migrate, which only the entry points call (cli.main, launcher.main): code that merely asks where a folder is
(a library import, a server thread, a test) never moves anything.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

import platformdirs


class Folders:
    APP = "goodpush"
    CONFIG_ENV = "GOODPUSH_CONFIG"
    HOME_ENV = "GOODPUSH_HOME"
    PROJECT = "athlete"                      # the default project's folder in the app's data folder
    PROJECT_FILE = "goodpush.toml"           # what makes a folder a project (config.CONFIG_NAME)
    OLD_CREDENTIALS = "athletes"             # Athletes.CREDENTIALS until 2026-11-25 (it read like athlete folders)
    LEGACY = Path.home() / ".goodpush"       # the machine folder until 2026-11 (None: nothing to move; the tests)
    lock = threading.Lock()
    migrated = False                         # this process has looked (the tests: True for the whole session)

    @classmethod
    def machine(cls) -> Path:
        """this computer's own folder (credentials, athletes.toml, ride_devices.json, ...)."""
        return cls.config_dir()

    @classmethod
    def default_project(cls) -> Path:
        """the installed app's project when it isn't started from one and no athlete is registered (the old place
        while a move of it is unfinished: Folders.move_project)."""
        if os.environ.get(cls.HOME_ENV):
            return Path(os.environ[cls.HOME_ENV]).expanduser()
        old = cls.data_dir()
        return old if (old / cls.PROJECT_FILE).is_file() else old / cls.PROJECT

    @classmethod
    def config_dir(cls) -> Path:
        if os.environ.get(cls.CONFIG_ENV):
            return Path(os.environ[cls.CONFIG_ENV]).expanduser()
        return Path(platformdirs.user_config_dir(cls.APP, appauthor=False, roaming=False))

    @classmethod
    def data_dir(cls) -> Path:
        """the app's data folder: the default project's parent (and, before, the default project itself)."""
        return Path(platformdirs.user_data_dir(cls.APP, appauthor=False, roaming=True))

    @classmethod
    def migrate(cls) -> None:
        """once per process: the default project down into its own folder, then ~/.goodpush's files into the
        machine folder, its athletes/ renamed credentials/, then the registry's entry for that project. Each step moves only what isn't at the new
        place yet, so a step cut short (a file in use on Windows) goes on next time; nothing is overwritten."""
        with cls.lock:
            if cls.migrated:
                return
            cls.migrated = True
            try:
                moved = cls.move_project()
                cls.move_machine()
                cls.rename_credentials()
                if moved:
                    cls.reregister(*moved)
            except OSError as e:
                print(f"goodpush: couldn't move its folders yet ({e}); it tries again next start", file=sys.stderr)

    @classmethod
    def move_project(cls) -> tuple[Path, Path] | None:
        """the default project from the app's data folder itself into data_dir()/athlete (the platform's machine
        folder can be that same folder: macOS). Its goodpush.toml goes last: while it's there, a move is unfinished."""
        old = cls.data_dir()
        if os.environ.get(cls.HOME_ENV) or not (old / cls.PROJECT_FILE).is_file():
            return None
        new = old / cls.PROJECT
        new.mkdir(exist_ok=True)
        for p in sorted(old.iterdir()):
            if p.name not in (cls.PROJECT, cls.PROJECT_FILE) and not (new / p.name).exists():
                os.replace(p, new / p.name)
        left = sorted(p.name for p in old.iterdir() if p.name != cls.PROJECT and
                      (p.name != cls.PROJECT_FILE or (new / p.name).exists()))
        if left:                                 # also in the new place: you choose; until then the old one is used
            print(f"goodpush: {', '.join(left)} in {old} are also in {new}: move what you want to keep into {new}"
                  f" and delete the rest; until then goodpush uses {old}", file=sys.stderr)
            return None
        os.replace(old / cls.PROJECT_FILE, new / cls.PROJECT_FILE)
        print(f"goodpush: moved the app's project to {new}", file=sys.stderr)
        return old, new

    @classmethod
    def rename_credentials(cls) -> None:
        """the machine folder's athletes/<name> (sign-ins) -> credentials/<name>."""
        from .athletes import Athletes
        old, new = cls.config_dir() / cls.OLD_CREDENTIALS, cls.config_dir() / Athletes.CREDENTIALS
        if old.is_dir() and not new.exists():
            os.replace(old, new)
            print(f"goodpush: renamed {old} to {new}", file=sys.stderr)

    @classmethod
    def move_machine(cls) -> None:
        """~/.goodpush's files into the machine folder (those already there stay as they are, and so does
        ~/.goodpush then); ~/.goodpush goes once it's empty."""
        old, new = cls.LEGACY, cls.config_dir()
        if old is None or not old.is_dir() or old.resolve() == new.resolve():
            return
        new.mkdir(parents=True, exist_ok=True)
        for p in sorted(old.iterdir()):
            if not (new / p.name).exists():
                shutil.move(str(p), str(new / p.name))      # a rename on one drive; another drive: copy, delete
        left = sorted(p.name for p in old.iterdir())
        if left:
            print(f"goodpush: {old} still has {', '.join(left)} (also in {new}, which is used)", file=sys.stderr)
        else:
            old.rmdir()
            print(f"goodpush: moved {old} to {new}", file=sys.stderr)

    @classmethod
    def reregister(cls, old: Path, new: Path) -> None:
        """an athlete registered at the default project's old place: at its new one."""
        from .athletes import Athletes
        reg = Athletes(cls.config_dir())
        hits = [k for k, v in reg.folders.items() if v.resolve() == old.resolve()]
        for k in hits:
            reg.folders[k] = new
        if hits:
            reg.save()
