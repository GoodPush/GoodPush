"""Help & feedback: the app's links to GitHub's issue forms and the guide, and the collaboration files they rely on."""
import re
from pathlib import Path

import yaml

from goodpush import __version__
from goodpush.app import server

REPO = Path(__file__).resolve().parents[1]
FORMS = REPO / ".github" / "ISSUE_TEMPLATE"
HELP_JS = Path(server.__file__).with_name("static") / "js" / "pages" / "help.js"


def form_ids(name: str) -> set[str]:
    return {b["id"] for b in yaml.safe_load((FORMS / f"{name}.yml").read_text(encoding="utf-8"))["body"] if "id" in b}


def test_every_issue_form_the_app_opens_exists_and_has_the_fields_it_fills_in():
    js = HELP_JS.read_text(encoding="utf-8")
    forms = set(re.findall(r'this\.action\("(\w+)"', js))
    assert forms == {"bug_report", "numbers", "idea"}
    for f in forms:
        assert "version" in form_ids(f)
    assert "computer" in form_ids("bug_report")
    assert 'q.set("version"' in js and 'q.set("computer"' in js


def test_the_guide_pages_the_app_links_to_have_sources():
    js = HELP_JS.read_text(encoding="utf-8")
    for page in re.findall(r"\$\{GUIDE\}([\w/]+)\.html", js):
        assert any((REPO / "srcdocs" / f"{page}{ext}").is_file() for ext in (".md", ".rst")), page
    for name in ("CONTRIBUTING.md", "COSTS.md", "DISCLAIMER.md", "CODE_OF_CONDUCT.md", "SECURITY.md"):
        assert (REPO / name).is_file(), name


def test_state_carries_the_version_for_the_issue_forms(tmp_path, monkeypatch):
    from goodpush.config import init_project, load_project
    monkeypatch.setattr(server.DailyRanges, "start", lambda self, proj, day: None)
    init_project(tmp_path)
    about = server.Data(load_project(tmp_path)).state()["about"]
    assert about == {"version": __version__, "installed": False}


def test_the_version_comes_from_the_git_tag():
    """hatch-vcs: X.Y.Z on a tag vX.Y.Z, X.Y.(Z+1).devN+g<commit> between tags; the installed metadata agrees."""
    from importlib.metadata import version
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?(\+[\w.]+)?", __version__), __version__
    assert version("goodpush") == __version__


def test_the_licence_is_agpl_with_the_attribution_terms_and_the_app_shows_its_notice():
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in (REPO / "LICENSE").read_text(encoding="utf-8")
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "section 7(b)" in notice and "Based on GoodPush" in notice and "https://github.com/GoodPush/GoodPush" in notice
    toml = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "AGPL-3.0-or-later"' in toml and '"NOTICE"' in toml
    js = HELP_JS.read_text(encoding="utf-8")
    assert "GNU AGPL v3.0 or later" in js and "/blob/main/NOTICE" in js
