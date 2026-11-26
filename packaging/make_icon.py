"""Draw the app icon (packaging/icon.png, 1024 px, and icon.ico for the Windows installer) at build time from
icon.svg: the logo's bike with the heartbeat across it, on a dark tile. PyInstaller turns the PNG into the
platform's icon (.icns / .ico, with Pillow)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

N = 1024
SVG = Path(__file__).with_name("icon.svg")


def main(out: Path) -> None:
    # a QApplication (a QGuiApplication too): later widgets in the same process (the tests) need one
    app = QApplication.instance() or QApplication(sys.argv[:1])
    img = QImage(N, N, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    QSvgRenderer(str(SVG)).render(p)
    p.end()
    img.save(str(out))
    del app
    try:                                              # Windows installer icon (Inno Setup wants an .ico)
        from PIL import Image
        Image.open(out).save(out.with_suffix(".ico"), sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    except ImportError:
        pass


if __name__ == "__main__":
    main(Path(__file__).with_name("icon.png"))
