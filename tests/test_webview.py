import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6.QtWebEngineWidgets")


def test_the_app_opens_in_its_own_window(tmp_path):
    """spike (launcher.AppWindow, GOODPUSH_WEBVIEW=1): the page loads in Qt WebEngine, the app's scripts run (its
    tabs), localStorage works and the profile is kept in the machine folder. Its own process, offscreen, on a port
    of its own (another goodpush on 8765 is left alone)."""
    from goodpush.config import init_project
    init_project(tmp_path / "project")
    found = tmp_path / "found.json"
    env = {**os.environ, "GOODPUSH_WEBVIEW": "1", "GOODPUSH_WEBVIEW_CHECK": str(found), "QT_QPA_PLATFORM": "offscreen",
           "GOODPUSH_CONFIG": str(tmp_path / "machine"), "GOODPUSH_HOME": str(tmp_path / "project")}
    env.pop("GOODPUSH_SECRETS", None)
    subprocess.run([sys.executable, "-c", "from goodpush.launcher import App; App(8790).run()"],
                   cwd=tmp_path / "project", env=env, timeout=120, check=True)
    got = json.loads(found.read_text(encoding="utf-8"))
    assert got["loaded"] and got["tabs"] >= 5 and got["storage"] == "1", got
    assert len(got["main_text"]) > 20, got                                       # the page drew (more than Loading…)
    assert (tmp_path / "machine" / "webview").is_dir()
