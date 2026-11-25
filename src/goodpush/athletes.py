"""Several athletes on one machine: a registry of their folders (athletes.toml in the machine folder,
folders.Folders) and each one's own credentials (credentials/<name>/ there).

An athlete folder holds everything of that person (goodpush.toml, data, plan, model fits, reports): copy or sync it
to move them to another machine, or give them another folder on this one (FolderMove). Credentials stay on each
machine, per athlete, and are never in the folder.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from pathlib import Path

from .folders import Folders
from .progress import Progress

NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class Athletes:
    """The athletes of this machine: name -> folder, a default, and where each one's credentials live."""
    REGISTRY = "athletes.toml"
    CREDENTIALS = "credentials"    # each athlete's sign-ins: <machine folder>/credentials/<name>
    # not credentials of a person (Folders.PROJECT: on macOS the installed app's project is in this folder too;
    # Folders.OLD_CREDENTIALS: CREDENTIALS before 2026-11-25)
    MACHINE_FILES = ("athletes.toml", CREDENTIALS, Folders.OLD_CREDENTIALS, "privacy_terms.txt", "app_key",
                     Folders.PROJECT)
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
        return self.home / self.CREDENTIALS / name

    def add(self, name: str, folder, default: bool = False, credentials: bool = False) -> Path:
        """Register `folder` as athlete `name` (the first one becomes the default). `credentials`: copy this
        machine's current credentials (the machine folder's, from before there were athletes) to theirs; nothing is
        removed or overwritten. An athlete there already keeps their folder (FolderMove gives them another)."""
        if not NAME.match(name):
            raise ValueError("an athlete name is lower-case letters, digits, - or _ (e.g. 'athlete_name')")
        folder = Path(folder).expanduser().resolve()
        if name in self.folders and self.folders[name].resolve() != folder:
            raise ValueError(f"There is an athlete '{name}' already, at {self.folders[name]}: "
                             f"`goodpush athlete move {name} FOLDER` gives them another folder.")
        self.folders[name] = folder
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


class FolderMove:
    """Athlete `name`'s folder to `dest`, the registry following. On the same drive a rename; else a copy that keeps
    every file's time (peer sync compares size and time: a file it saw stays unchanged), checked file by file, and
    only then the old folder deleted. Their old folder gone already (you moved it yourself): only the registry
    follows, to a `dest` holding their goodpush.toml. ValueError before anything changed: why not."""
    MB = 1 << 20
    TIME_S = 2                       # a file's time on another kind of drive may round to 2 s (FAT)

    def __init__(self, reg: Athletes, name: str, dest, progress=print):
        from .config import CONFIG_NAME
        self.reg, self.name, self.progress, self.config = reg, name, progress, CONFIG_NAME
        try:
            self.src = reg.folder(name).resolve()
        except KeyError as e:
            raise ValueError(e.args[0]) from None
        dest = Path(str(dest).strip()).expanduser()
        if not dest.is_absolute():
            raise ValueError("Give the whole folder path.")
        self.dest = dest.resolve()
        self.done = self.total = 0
        self.check()

    def check(self) -> None:
        if not self.found():
            if not (self.dest / self.config).is_file():
                raise ValueError(f"{self.name}'s folder isn't at {self.src} any more, and {self.dest} has no "
                                 f"{self.config}: give the folder it's in now.")
            return
        if self.dest == self.src:
            raise ValueError(f"That's {self.name}'s folder already.")
        if self.dest.is_relative_to(self.src) or self.src.is_relative_to(self.dest):
            raise ValueError("Pick a folder outside their folder (and not one it's in).")
        if self.dest.exists() and (not self.dest.is_dir() or any(self.dest.iterdir())):
            raise ValueError(f"{self.dest} isn't empty: pick a new or an empty folder.")
        if not self.same_drive():
            self.total = sum(p.stat().st_size for p in self.src.rglob("*") if p.is_file())
            free = shutil.disk_usage(self.existing(self.dest)).free
            if free < self.total * 1.05:
                raise ValueError(f"Not enough space there: {self.total // self.MB} MB needed, {free // self.MB} MB "
                                 f"free.")

    def found(self) -> bool:
        """their folder is still at the old place."""
        return (self.src / self.config).is_file()

    @staticmethod
    def existing(p: Path) -> Path:
        """`p` or the nearest folder above it that exists."""
        while not p.exists():
            p = p.parent
        return p

    def same_drive(self) -> bool:
        return os.stat(self.src).st_dev == os.stat(self.existing(self.dest)).st_dev

    def run(self) -> Path:
        """move (or only re-point); the new folder."""
        if not self.found():
            self.registered()
            self.progress(f"{self.name}'s folder: {self.dest}")
            return self.dest
        if self.same_drive():
            if self.dest.exists():
                self.dest.rmdir()                          # empty (check)
            self.dest.parent.mkdir(parents=True, exist_ok=True)
            os.rename(self.src, self.dest)
            self.registered()
        else:
            self.progress(f"Copying {self.total // self.MB} MB to {self.dest}")
            try:
                shutil.copytree(self.src, self.dest, copy_function=self.copy, dirs_exist_ok=True)
                self.verify()
            except (OSError, ValueError):                   # a copy that failed or differs: gone again
                shutil.rmtree(self.dest, ignore_errors=True)
                raise
            self.registered()
            try:
                shutil.rmtree(self.src)
            except OSError as e:
                self.progress(f"Couldn't delete all of the old folder {self.src} ({e}): delete what's left yourself.")
        self.progress(f"{self.name}'s folder is now {self.dest}")
        return self.dest

    def copy(self, src: str, dst: str) -> str:
        """shutil.copy2 (keeps the file's time), with the bar."""
        out = shutil.copy2(src, dst)
        self.done += os.path.getsize(dst)
        Progress.step(self.progress, "Copying (MB)", self.done // self.MB, max(self.total // self.MB, 1))
        return out

    def verify(self) -> None:
        """every file arrived, same size and time; else ValueError (their folder stays where and as it was)."""
        bad = []
        for p in self.src.rglob("*"):
            if p.is_file():
                q, st = self.dest / p.relative_to(self.src), p.stat()
                if not q.is_file() or q.stat().st_size != st.st_size or \
                        abs(q.stat().st_mtime - st.st_mtime) > self.TIME_S:
                    bad.append(str(p.relative_to(self.src)))
        if bad:
            raise ValueError(f"The copy in {self.dest} differs ({len(bad)} file(s), e.g. {bad[0]}): {self.name}'s "
                             f"folder stays at {self.src} (the copy is removed again).")

    def registered(self) -> None:
        self.reg.folders[self.name] = self.dest
        self.reg.save()
