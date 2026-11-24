"""The documentation (srcdocs/) builds without warnings: docstrings it pulls in stay valid reStructuredText."""
from pathlib import Path

import pytest


def test_docs_build(tmp_path):
    sphinx = pytest.importorskip("sphinx.cmd.build")
    src = Path(__file__).resolve().parents[1] / "srcdocs"
    assert sphinx.build_main(["-b", "html", "-W", "-q", "-d", str(tmp_path / "doctrees"), str(src), str(tmp_path / "html")]) == 0
    assert (tmp_path / "html" / "model" / "likelihood.html").exists()
