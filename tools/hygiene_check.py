"""Pre-commit hygiene check: the project's own code rules (CLAUDE.md "Conventions", "App front end") on what a
commit adds.

    python tools/hygiene_check.py FILE...          check the staged files (pre-commit passes them)
    python tools/hygiene_check.py --all FILE...    every line, not only the added ones

Only lines the commit adds count (the diff against HEAD), so older code never blocks a commit; --all checks
whole files. Python: no module-level _helpers (put them on the class), no nested functions.
JavaScript: no inline event handlers (onclick="..."), no window.x = globals, no `import *`.
Secrets, personal data and paths are tools/privacy_check.py's job; lint (unused imports, undefined names) ruff's.
A line with "hygiene: ok" in it is let through.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path


class HygieneCheck:
    OK = "hygiene: ok"
    JS_RULES = {
        "inline event handler (add a listener in wire())": re.compile(r"\son[a-z]+=[\"'$]"),
        "window. global (export it from its module)": re.compile(r"\bwindow\.[A-Za-z_$][\w$]*\s*=(?!=)"),
        "import * (named imports only)": re.compile(r"\bimport\s+\*"),
    }
    HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)
    MOVES: dict[str, str] | None = None             # staged moves: new path -> old (read once)

    def __init__(self):
        self.problems: list[str] = []
        self.lines: list[str] = []
        self.added: set[int] | None = None

    def judged(self, i: int) -> bool:
        """line i (1-based) of the text being checked counts: added (or every line) and not marked ok."""
        return (self.added is None or i in self.added) and self.OK not in self.lines[i - 1]

    @staticmethod
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True, encoding="utf-8").stdout
        except (OSError, subprocess.CalledProcessError):
            return None

    @classmethod
    def added_lines(cls, path: str) -> set[int] | None:
        """line numbers `path` adds against the last commit (in a commit pre-commit stashes what isn't staged, so
        that is the staged change); none for a tracked file without changes; None (every line) for a new file."""
        old = cls.moved_from(path)                   # a moved file: only what changed in it, not every line
        diff = cls.git("diff", "HEAD", "-M", "-U0", "--no-color", "--", *([old] if old else []), path)
        if diff is None:
            return None
        if not diff:
            return set() if cls.git("ls-files", "--", path) else None
        return {n for m in cls.HUNK.finditer(diff)
                for n in range(int(m.group(1)), int(m.group(1)) + int(m.group(2) or 1))}

    @classmethod
    def moved_from(cls, path: str) -> str | None:
        """the path a staged file had in the last commit when the commit moves it (git's rename detection)."""
        if cls.MOVES is None:
            out = cls.git("diff", "--cached", "HEAD", "-M", "--name-status", "--diff-filter=R", "--no-color") or ""
            cls.MOVES = {p[2]: p[1] for p in (line.split("\t") for line in out.splitlines()) if len(p) == 3}
        return cls.MOVES.get(Path(path).as_posix())

    def check_text(self, name: str, text: str, added: set[int] | None = None) -> None:
        """`added`: the line numbers to judge (None: all)."""
        self.lines, self.added = text.splitlines(), added
        if name.endswith(".py"):
            self.check_python(name, text)
        elif name.endswith(".js"):
            for i, line in enumerate(self.lines, 1):
                if self.judged(i):
                    self.problems += [f"{name}:{i}: {what}" for what, rx in self.JS_RULES.items() if rx.search(line)]

    def check_python(self, name: str, text: str) -> None:
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            self.problems.append(f"{name}:{e.lineno}: syntax error: {e.msg}")
            return
        functions = (ast.FunctionDef, ast.AsyncFunctionDef)
        for node in tree.body:
            if isinstance(node, functions) and node.name.startswith("_") and not node.name.startswith("__") \
                    and self.judged(node.lineno):
                self.problems.append(f"{name}:{node.lineno}: module-level helper {node.name}() (put it on its class)")
        nested = {inner: outer for outer in ast.walk(tree) if isinstance(outer, functions)
                  for inner in ast.walk(outer) if inner is not outer and isinstance(inner, functions)}
        for inner, outer in nested.items():          # (a function two deep: named once, with its innermost host)
            if self.judged(inner.lineno):
                self.problems.append(f"{name}:{inner.lineno}: nested function {inner.name}() inside {outer.name}() "
                                     "(a method instead)")

    def check_file(self, path: str, every_line: bool = False) -> None:
        if not path.endswith((".py", ".js")):
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return                                   # deleted / moved / not text
        self.check_text(path, text, None if every_line else self.added_lines(path))

    def run(self, argv: list[str]) -> int:
        every_line = argv[:1] == ["--all"]
        for path in argv[1:] if every_line else argv:
            self.check_file(path, every_line)
        if self.problems:
            print("Hygiene check failed - fix these (or mark a line 'hygiene: ok' if it's really fine):")
            print("\n".join(f"  {p}" for p in self.problems))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(HygieneCheck().run(sys.argv[1:]))
