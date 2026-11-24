"""Progress of long runs: log lines plus a progress bar.

Everything takes a `progress` callable for its log lines (print by default). Loops over many items
also call `Progress.step(progress, label, done, total)`: that moves a bar when `progress` is a
`Progress` (the terminal's tqdm bar, or the app's job chip) and does nothing for a plain function.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm


class Progress:
    """A log (`progress("text")`) with a bar (`on_step(label, done, total)`)."""

    def __init__(self, log=print, on_step=None):
        self.log, self.on_step = log, on_step

    def __call__(self, *msg):
        self.log(*msg)

    @staticmethod
    def step(progress, label: str, done: int, total: int) -> None:
        on_step = getattr(progress, "on_step", None)
        if on_step and total:
            on_step(label, done, total)

    @classmethod
    def terminal(cls) -> Progress:
        bar = TerminalBar()
        return cls(bar.message, bar.draw)


class Console:
    """The Windows console goodpush runs in."""
    QUICK_EDIT, EXTENDED_FLAGS, STD_INPUT = 0x0040, 0x0080, -10

    @classmethod
    def no_quick_edit(cls) -> bool:
        """Turn QuickEdit off: with it, a click in the window starts a selection that freezes every
        program writing to the console (the app and its workers hung on a click, 2026-10-30).
        False when not on Windows or not in a console."""
        import sys
        if sys.platform != "win32":
            return False
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(cls.STD_INPUT)
        mode = ctypes.c_uint32()
        if not h or not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, (mode.value & ~cls.QUICK_EDIT) | cls.EXTENDED_FLAGS))


class TerminalBar:
    """The current stage's tqdm bar (on stderr; tqdm turns it off when that isn't a terminal)."""

    def __init__(self):
        self.bar, self.label = None, None

    @staticmethod
    def message(*msg) -> None:
        tqdm.write(" ".join(str(m) for m in msg))      # above the bar, which stays at the bottom

    def draw(self, label: str, done: int, total: int) -> None:
        if self.bar is None or self.label != label:
            self.close()
            self.bar = tqdm(total=total, desc=label, leave=False, disable=None, dynamic_ncols=True)
            self.label = label
        self.bar.update(done - self.bar.n)
        if done >= total:
            self.close()

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        self.bar, self.label = None, None


class Tee:
    """A text stream that also writes to a log file: a command's output in the terminal and in its log on this
    computer (Project.logs; 2026-11-13: logs redirected into the athlete folder synced to every computer). `into`
    swaps stdout and stderr for the run, only when they are a terminal (a run started by the app, or redirected by
    hand, already goes to a file)."""

    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text: str) -> int:
        if not self.file.closed:                  # after the run the terminal only (2026-11-13: colorama wrapped the
            self.file.write(text)                 # Tee during the run and wrote its colour reset at exit)
            if "\n" in text:                      # a finished line is on the disk at once: the last line before a
                self.file.flush()                 # crash or reset survives (2026-11-13: two resets took the log's
                os.fsync(self.file.fileno())      # last lines with them)
        return self.stream.write(text)

    def flush(self) -> None:
        if not self.file.closed:
            self.file.flush()
        self.stream.flush()

    def __getattr__(self, name):                 # isatty, encoding, ...: the terminal's
        return getattr(self.stream, name)

    @staticmethod
    @contextmanager
    def into(path: Path):
        """output also into `path` (appended) while the block runs; the path is printed first."""
        if not sys.stdout.isatty():
            yield None
            return
        out, err = sys.stdout, sys.stderr
        with open(path, "a", encoding="utf-8", buffering=1) as f:
            print(f"log: {path}")
            sys.stdout, sys.stderr = Tee(out, f), Tee(err, f)
            try:
                yield path
            finally:
                sys.stdout, sys.stderr = out, err
