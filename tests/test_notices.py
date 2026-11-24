"""packaging/notices.py: copyleft parts of the installers point to their source at the exact version bundled."""
import importlib.util
from importlib import metadata
from pathlib import Path

import pytest

NOTICES = Path(__file__).resolve().parents[1] / "packaging" / "notices.py"
SPEC = importlib.util.spec_from_file_location("notices", NOTICES)
notices = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(notices)


class FakeDist:
    def __init__(self, name: str, version: str, licence: str, classifiers: tuple[str, ...] = ()):
        self.metadata = FakeMetadata({"Name": name, "License-Expression": licence}, list(classifiers))
        self.version = version


class FakeMetadata(dict):
    def __init__(self, fields: dict, classifiers: list[str]):
        super().__init__(fields)
        self.classifiers = classifiers

    def get_all(self, field: str):
        return self.classifiers if field == "Classifier" else None


def test_copyleft_packages_point_to_their_exact_source():
    assert notices.source(FakeDist("paramiko", "5.1.0", "LGPL-2.1")) == "https://pypi.org/project/paramiko/5.1.0/#files"
    assert notices.source(FakeDist("tqdm", "4.70.1", "MPL-2.0 AND MIT")).endswith("/tqdm/4.70.1/#files")
    by_classifier = FakeDist("x", "1.0", "", ("License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",))
    assert notices.source(by_classifier) is not None
    assert notices.source(FakeDist("PySide6_Essentials", "6.12.0", "LGPL-3.0-only OR GPL-3.0-only")) == \
        "https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.12.0"


def test_permissive_packages_need_no_source_pointer():
    for lic in ("MIT", "BSD-3-Clause", "Apache-2.0", "Apache-2.0 OR BSD-3-Clause", "PSF-2.0"):
        assert notices.source(FakeDist("x", "1.0", lic)) is None


def test_also_sources_carry_the_bundled_version():
    qt, usb = (src for _, _, _, src, _ in notices.ALSO)
    assert "/archive/qt/6.12/6.12.0/single/" in notices.also_source(qt, "6.12.0")
    assert "/releases/tag/v1.0.30 " in notices.also_source(usb, "1.0.30.1")


def test_notices_file_has_sources(tmp_path):
    try:
        metadata.version("paramiko")
    except metadata.PackageNotFoundError:
        pytest.skip("paramiko ([sync]) not installed")
    out = tmp_path / "THIRD_PARTY_NOTICES.txt"
    notices.main(out)
    text = out.read_text(encoding="utf-8")
    assert f"    source: https://pypi.org/project/paramiko/{metadata.version('paramiko')}/#files" in text
