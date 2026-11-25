"""Several athletes on one machine: a registry of their folders (athletes.toml in the machine folder,
folders.Folders) and each one's own credentials (athletes/<name>/ there).

An athlete folder holds everything of that person (goodpush.toml, data, plan, model fits, reports): copy or sync it
to move them to another machine. Credentials stay on each machine, per athlete, and are never in the folder.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from pathlib import Path

from .folders import Folders

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class Athletes:
    """The athletes of this machine: name -> folder, a default, and where each one's credentials live."""
    REGISTRY = "athletes.toml"
    # not credentials of a person (Folders.PROJECT: on macOS the installed app's project is in this folder too)
    MACHINE_FILES = ("athletes.toml", "athletes", "privacy_terms.txt", "app_key", Folders.PROJECT)
    SET = None                     # the GOODPUSH_SECRETS value `use` set (one you set yourself is left alone)
    HOME = None                    # this machine's folder when not Folders.machine() (the tests: never the real one)

    def __init__(self, home: Path | None = None):
        self.home = Path(home or self.HOME or Folders.machine())
        f = self.home / self.REGISTRY
        data = tomllib.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        self.default = str(data.get("default") or "")
        self.folders = {k: Path(v) for k, v in (data.get("athletes") or {}).items()}

    def folder(self, name: str) -> Path:
        if name not in self.folders:
            known = ", ".join(sorted(self.folders)) or "none yet"
            raise KeyError(f"No athlete '{name}' on this machine ({known}): `goodpush athlete add {name} FOLDER`")
        return self.folders[name]

    def name_of(self, root: Path) -> str | None:
        """the athlete whose folder `root` is, if any."""
        r = Path(root).resolve()
        return next((k for k, v in self.folders.items() if v.resolve() == r), None)

    def secrets(self, name: str) -> Path:
        return self.home / "athletes" / name

    def add(self, name: str, folder, default: bool = False, credentials: bool = False) -> Path:
        """Register `folder` as athlete `name` (the first one becomes the default). `credentials`: copy this
        machine's current credentials (the machine folder's, from before there were athletes) to theirs; nothing is
        removed or overwritten."""
        if not NAME.match(name):
            raise ValueError("an athlete name is lower-case letters, digits, - or _ (e.g. 'athlete_name')")
        self.folders[name] = Path(folder).expanduser().resolve()
        if default or not self.default:
            self.default = name
        self.save()
        if credentials:
            self.copy_credentials(name)
        return self.folders[name]

    def remove(self, name: str) -> Path:
        """Take athlete `name` off this machine: out of the registry, their credentials here deleted. Their folder
        (data, plan, fits) stays on disk and is returned; another athlete becomes the default if they were it."""
        folder = self.folder(name)
        del self.folders[name]
        if self.default == name:
            self.default = min(self.folders, default="")
        self.save()
        shutil.rmtree(self.secrets(name), ignore_errors=True)
        return folder

    def copy_credentials(self, name: str) -> list[str]:
        dest = self.secrets(name)
        dest.mkdir(parents=True, exist_ok=True)
        done = []
        for p in sorted(self.home.iterdir()):           # a handful of files
            if p.name in self.MACHINE_FILES or (dest / p.name).exists():
                continue
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, dest / p.name)
            done.append(p.name)
        return done

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        lines = [f"default = {json.dumps(self.default)}", "", "[athletes]"] + \
            [f"{k} = {json.dumps(str(v))}" for k, v in sorted(self.folders.items())]
        (self.home / self.REGISTRY).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def use(self, name: str) -> None:
        """this process works for athlete `name`: their credentials (sync.secrets_dir follows GOODPUSH_SECRETS)."""
        cur = os.environ.get("GOODPUSH_SECRETS")
        if cur is None or cur == Athletes.SET:
            Athletes.SET = os.environ["GOODPUSH_SECRETS"] = str(self.secrets(name))
