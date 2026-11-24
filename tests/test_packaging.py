"""The Windows installer script (packaging/goodpush.iss): the files it ships from the repo exist, and the ANT+ USB
setup opens its steps next to Zadig."""
import re
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"
BUILT = {"zadig.exe", "zadig-NOTICE.txt", "THIRD_PARTY_NOTICES.txt"}    # made by the workflow, not in the repo


def test_installer_ships_files_that_exist():
    iss = (PACKAGING / "goodpush.iss").read_text(encoding="utf-8")
    sources = re.findall(r'^Source: "([^"]+)"', iss, flags=re.M)
    assert "ant_usb_setup.txt" in sources and "ant_usb_setup.cmd" in sources
    missing = [s for s in sources if "*" not in s and Path(s).name not in BUILT and not (PACKAGING / s).exists()]
    assert missing == []
    assert re.search(r'ANT\+ USB setup"; Filename: "\{app\}\\tools\\ant_usb_setup\.cmd"', iss)


def test_ant_usb_setup_opens_the_steps_and_zadig():
    cmd = (PACKAGING / "ant_usb_setup.cmd").read_text(encoding="utf-8")
    assert '"%~dp0ant_usb_setup.txt"' in cmd and '"%~dp0zadig.exe"' in cmd
    steps = (PACKAGING / "ant_usb_setup.txt").read_text(encoding="utf-8")
    assert "0FCF" in steps and "WinUSB" in steps and "List All Devices" in steps


def test_app_icon_is_the_bike_and_heartbeat_on_a_dark_tile(monkeypatch, tmp_path):
    """make_icon.py draws icon.svg: the bike's blue and the (see-through) heartbeat's pink on the tile, a clear margin
    around it."""
    pytest.importorskip("PySide6.QtSvg")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.syspath_prepend(str(PACKAGING))
    import make_icon
    from PySide6.QtGui import QColor, QImage
    make_icon.main(tmp_path / "icon.png")
    img = QImage(str(tmp_path / "icon.png"))
    assert img.width() == img.height() == make_icon.N
    pixels = [img.pixelColor(x, y) for x in range(0, img.width(), 4) for y in range(0, img.height(), 4)]
    bike = QColor((PACKAGING / "icon.svg").read_text(encoding="utf-8").split('stroke="')[1][:7])
    assert any(p.rgb() == bike.rgb() for p in pixels)
    assert any(p.red() > p.green() + 30 and p.red() > p.blue() + 10 for p in pixels)      # the pink trace
    assert img.pixelColor(0, 0).alpha() == 0
