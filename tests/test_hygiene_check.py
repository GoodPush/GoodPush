"""tools/hygiene_check.py: the project's code rules on the lines a commit adds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hygiene_check import HygieneCheck  # noqa: E402


def problems(name: str, text: str, added=None) -> list[str]:
    c = HygieneCheck()
    c.check_text(name, text, added)
    return c.problems


PY = '''class Plan:
    def build(self):
        def inner():
            return 1
        return inner()


def _helper(x):
    return x


def __getattr__(name):
    return name


def public(x):
    return x
'''


def test_python_rules():
    found = problems("src/x.py", PY)
    assert any("nested function inner() inside build()" in p for p in found)
    assert any("module-level helper _helper()" in p for p in found)
    assert not any("__getattr__" in p or "public" in p for p in found)       # dunders and public names are fine
    assert not problems("src/x.py", "x = 1  # " + "a" * 200)                 # no line-length rule
    assert any("syntax error" in p for p in problems("src/x.py", "def f(:\n"))


def test_only_added_lines_count():
    assert not problems("src/x.py", PY, added={1, 2, 15, 16})               # the old helpers aren't this commit's
    assert len(problems("src/x.py", PY, added={8})) == 1                     # the new _helper is
    two_deep = "def a():\n    def b():\n        def c():\n            pass\n"
    found = problems("src/x.py", two_deep)
    assert len(found) == 2 and any("c() inside b()" in p for p in found)    # each named once, by its host


def test_javascript_rules_and_the_ok_mark():
    assert any("inline event handler" in p for p in problems("a.js", 'html = `<b onclick="go()">x</b>`;'))
    assert any("window. global" in p for p in problems("a.js", "window.S = state;"))
    assert any("import *" in p for p in problems("a.js", 'import * as U from "./util.js";'))
    assert not problems("a.js", 'window.addEventListener("resize", f); if (window.x === 1) {}')
    assert not problems("a.js", "window.S = state;  // hygiene: ok")
    assert not problems("README.md", "def x(): onclick= window.y = 1")     # other files: not this check's


def test_a_moved_file_counts_only_its_changed_lines(tmp_path, monkeypatch):
    import subprocess
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false"]   # privacy: ok (made up)
    monkeypatch.chdir(tmp_path)
    subprocess.run([*git, "init", "-q"], check=True)
    Path("old.py").write_text(PY, encoding="utf-8")
    subprocess.run([*git, "add", "old.py"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    subprocess.run([*git, "mv", "old.py", "new.py"], check=True)
    Path("new.py").write_text(PY.replace("def public", "def _added"), encoding="utf-8")
    subprocess.run([*git, "add", "new.py"], check=True)
    monkeypatch.setattr(HygieneCheck, "MOVES", None)
    c = HygieneCheck()
    c.check_file("new.py")
    assert c.problems == ["new.py:16: module-level helper _added() (put it on its class)"]
