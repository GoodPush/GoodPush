"""The overlay: a small translucent bar that stays on top of everything (watch a show while you ride),
and the device picker. Qt (PySide6), so it runs on Windows, macOS and Linux.

Drag it anywhere; − / mode / + on the bar: easier, the trainer mode (ERG → RES → SIM), harder. Drag the
marker on the workout profile to move through the workout. Right-click (or the tray icon) for pause, skip,
harder / easier, the mode,
opacity, size, click-through, devices and finish. − / +, the mouse wheel and up / down = 1 % easier / harder,
space = pause, right = skip the step."""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading
import time
from functools import cache, partial
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QFontMetricsF, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QMenu,
                               QMessageBox, QPushButton, QSystemTrayIcon, QVBoxLayout, QWidget)

from ..alpha import LT1_A1, LT2_A1
from .sensors import ROLES, Device
from .session import MODES

SCAN_SECONDS = 10
CUE_S = 15                   # a step's cue shows for its first seconds (then what's next)
ROLE_NAMES = {"trainer": "Trainer", "power": "Power", "cadence": "Cadence", "hr": "Heart rate"}
# the app's dark theme (app/static/app.css): surfaces, ink, status colours and the cividis zone ramp
SURFACE, SURFACE2, LINE = "#1a1a19", "#232321", "#3a3a37"
INK, INK2, MUTED, ACCENT = "#f0efec", "#c3c2b7", "#8f8e86", "#3987e5"
MARK = Path(__file__).parents[1] / "app" / "static" / "mark.svg"     # the logo's bike: the tray icon
GOOD, WARN, BAD, HR = "#3fb06d", "#e0a32a", "#e66767", "#d95926"
ZONES = [(.56, "#3b496c"), (.76, "#5b606e"), (.91, "#777777"), (1.06, "#979177"), (1.21, "#b9ab6d"), (9, "#dcc859")]
FONTS = ["Segoe UI Variable Display", "Segoe UI", "SF Pro Display", ".AppleSystemUIFont", "Helvetica Neue",
         "Roboto", "Cantarell", "sans-serif"]          # the app's system-ui stack
RADIUS = 14
# menus and the device window in the same look
STYLE = f"""
QMenu {{ background:{SURFACE}; color:{INK}; border:1px solid {LINE}; border-radius:10px; padding:6px; }}
QMenu::item {{ padding:6px 22px 6px 14px; border-radius:6px; }}
QMenu::item:selected {{ background:{SURFACE2}; }}
QMenu::item:disabled {{ color:{MUTED}; }}
QMenu::separator {{ height:1px; background:{LINE}; margin:5px 8px; }}
QDialog {{ background:{SURFACE}; color:{INK}; }}
QDialog QLabel {{ color:{INK2}; }}
QComboBox, QPushButton {{ background:{SURFACE2}; color:{INK}; border:1px solid {LINE}; border-radius:8px;
                          padding:5px 10px; min-height:20px; }}
QComboBox QAbstractItemView {{ background:{SURFACE}; color:{INK}; selection-background-color:{SURFACE2};
                               border:1px solid {LINE}; }}
QPushButton:hover, QComboBox:hover {{ border-color:{MUTED}; }}
QPushButton:default {{ background:{ACCENT}; border-color:{ACCENT}; color:#ffffff; }}
"""
W, H = 640, 116              # size at scale 1
NUDGE = 1                    # % harder / easier per click (or up / down key)
# what can be clicked on the bar, at scale 1: the × (finish: save / discard), easier, the mode, harder
BUTTONS = {"close": QRectF(W - 22, 0, 22, 22), "easier": QRectF(136, 58, 26, 18), "mode": QRectF(168, 58, 46, 18),
           "harder": QRectF(220, 58, 26, 18)}
SPLIT, WORK = 362, 376         # x of the line between the readings and the workout, and of the workout's text
CLOSE_BOX = BUTTONS["close"]
MODE_LABELS = dict(zip(MODES, ("ERG", "RES", "SIM")))
MODE_NAMES = dict(zip(MODES, ("ERG: the trainer holds the target", "Resistance: a fixed brake, your effort sets the power",
                              "Sim: a road, so gears and sprints feel like outside")))
PROFILE = QRectF(18, 84, W - 36, 20)     # the workout's steps; the marker on it is where you are: drag it


def objc_send(obj, sel: str, *args):
    """macOS: call an Objective-C method (`[obj sel:args]`) through the runtime, no pyobjc needed.
    obj is a class name (str) or an object pointer; args are ctypes values. Returns a pointer."""
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = objc.sel_registerName.restype = objc.objc_msgSend.restype = ctypes.c_void_p
    objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [type(a) for a in args]
    if isinstance(obj, str):
        obj = objc.objc_getClass(obj.encode())
    return objc.objc_msgSend(obj, objc.sel_registerName(sel.encode()), *args)


BAR_LEVEL = 101       # NSPopUpMenuWindowLevel: over other apps, full-screen ones too; menus (same level, shown
DIALOG_LEVEL = 102    # later) still open over it. Dialogs (save / discard, devices) go one above the bar


def mac_on_top_everywhere(widget: QWidget, level: int = BAR_LEVEL, panel: bool = True) -> None:
    """macOS: Qt's stay-on-top only floats over the current desktop (Space) and a Tool window hides when
    another app is active. Put the window on every Space and over full-screen apps: an accessory app (no Dock
    icon; the tray icon stays), a non-activating panel (showing it doesn't pull you out of a full-screen app),
    joining all Spaces and all apps' Spaces, at a level over full-screen content. `panel`: the bar, not a dialog
    (a dialog has to take the keyboard)."""
    if sys.platform != "darwin" or QApplication.platformName() != "cocoa":    # e.g. offscreen: no NSView
        return
    app = objc_send("NSApplication", "sharedApplication")
    objc_send(app, "setActivationPolicy:", ctypes.c_long(1))              # NSApplicationActivationPolicyAccessory
    window = objc_send(ctypes.c_void_p(int(widget.winId())), "window")    # winId is the NSView
    if not window:
        return
    if panel:
        mask = objc_send(window, "styleMask") or 0
        objc_send(window, "setStyleMask:", ctypes.c_ulong(mask | 1 << 7))   # NSWindowStyleMaskNonactivatingPanel
    objc_send(window, "setLevel:", ctypes.c_long(level))
    # CanJoinAllSpaces | Stationary | IgnoresCycle | FullScreenAuxiliary | CanJoinAllApplications
    objc_send(window, "setCollectionBehavior:", ctypes.c_ulong(1 << 0 | 1 << 4 | 1 << 6 | 1 << 8 | 1 << 18))
    objc_send(window, "setHidesOnDeactivate:", ctypes.c_bool(False))
    objc_send(window, "orderFrontRegardless")


def raise_next_dialog() -> None:
    """Call just before a dialog's or menu's exec(): once it's up, it floats over the bar like the bar does
    (else a save / discard question can open hidden under the bar). The dialog itself is made stay-on-top
    before it opens (Overlay.ask, DevicePicker): that's all Windows needs; macOS needs its window level too."""
    QTimer.singleShot(0, raise_open_dialog)


def raise_open_dialog() -> None:
    w = QApplication.activeModalWidget() or QApplication.activePopupWidget()
    if w is not None:
        mac_on_top_everywhere(w, DIALOG_LEVEL, panel=False)


class DevicePicker(QDialog):
    """Pick the trainer, power, cadence and heart-rate devices from a scan; the saved ones come preselected."""

    seen = Signal(object)            # a Device, from the scan thread
    scanned = Signal()

    def __init__(self, link, choice: dict[str, str]):
        super().__init__()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)          # over a full-screen app (see raise_next_dialog)
        self.setWindowTitle("goodpush - devices")
        self.link, self.saved = link, dict(choice)
        self.stop, self.thread = threading.Event(), None
        self.combos = {r: QComboBox() for r in ROLES}
        form = QFormLayout()
        for r, cb in self.combos.items():
            cb.addItem("from the trainer" if r in ("power", "cadence") else "none", "")
            if self.saved.get(r):
                cb.addItem(f"{Device.from_key(self.saved[r]).label} (saved)", self.saved[r])
                cb.setCurrentIndex(1)
            form.addRow(ROLE_NAMES[r], cb)
        self.status = QLabel()
        again = QPushButton("Scan again")
        again.clicked.connect(self.scan)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Ride")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box = QVBoxLayout(self)
        box.addWidget(QLabel("Wake your sensors (pedal, wear the strap). Devices appear as they're found."))
        box.addLayout(form)
        box.addWidget(self.status)
        box.addWidget(again)
        box.addWidget(buttons)
        self.seen.connect(self.add)
        self.scanned.connect(partial(self.status.setText, "Scan done."))
        self.scan()

    def scan(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop.clear()
        self.status.setText(f"Scanning for {SCAN_SECONDS} s ...")
        self.thread = threading.Thread(target=self.scan_thread, daemon=True)
        self.thread.start()

    def scan_thread(self) -> None:
        try:
            self.link.scan(SCAN_SECONDS, on_found=self.seen.emit, stop=self.stop)
            self.scanned.emit()
        except Exception as e:  # noqa: BLE001 - any stick trouble is shown, the picker stays usable
            self.seen.emit(e if isinstance(e, RuntimeError) else RuntimeError(f"Scan failed: {e}"))

    def add(self, d) -> None:
        if isinstance(d, Exception):
            self.status.setText(str(d))
            return
        for r in d.roles:
            cb = self.combos[r]
            i = cb.findData(d.key)
            if i < 0:
                cb.addItem(d.label, d.key)
                i = cb.count() - 1
            else:
                cb.setItemText(i, d.label)
            if d.key == self.saved.get(r) or (r in ("trainer", "hr") and not cb.currentData()):
                cb.setCurrentIndex(i)

    def choice(self) -> dict[str, str]:
        return {r: cb.currentData() or "" for r, cb in self.combos.items()}

    def done(self, result: int) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(5)       # the scan closes the stick before devices are connected
        super().done(result)


class Overlay(QWidget):
    """The bar. `ride` is a RideSession; `settings` the [ride] table; save(dict) stores changed settings."""

    def __init__(self, ride, settings: dict, ftp: float, save, pick):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.ride, self.ftp, self.save, self.pick = ride, ftp, save, pick
        self.opacity = float(settings.get("opacity", .6))
        self.scale = float(settings.get("scale", 1.0))
        self.drag = None
        self.seeking = False
        self.dim = MUTED
        self.outcome = "save"
        self.hover = ""                        # the button (or "profile") under the pointer
        self.setMouseTracking(True)            # buttons light up under the pointer
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_MacAlwaysShowToolWindow)
        self.setWindowTitle(f"goodpush - {ride.run.title}")
        self.resize(int(W * self.scale), int(H * self.scale))
        x, y = int(settings.get("x", -1)), int(settings.get("y", -1))
        if x < 0 or y < 0:
            g = QApplication.primaryScreen().availableGeometry()
            x, y = g.x() + (g.width() - self.width()) // 2, g.y() + 8
        self.move(x, y)
        self.menu = self.build_menu()
        self.tray = QSystemTrayIcon(self.icon(), self)
        self.tray.setContextMenu(self.menu)
        self.tray.setToolTip("goodpush ride")
        self.tray.show()
        if settings.get("click_through"):
            self.set_click_through(True)
        self.timer = QTimer(self, interval=250)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    # --- menu -------------------------------------------------------------------------
    def build_menu(self) -> QMenu:
        m = QMenu()
        self.act_pause = self.action(m, "Pause", self.toggle_pause, "Space")
        self.action(m, "Skip step", self.ride.run.skip, "Right")
        self.action(m, f"Harder +{NUDGE} %", partial(self.ride.run.nudge, NUDGE), "Up")
        self.action(m, f"Easier -{NUDGE} %", partial(self.ride.run.nudge, -NUDGE), "Down")
        sub = m.addMenu("Trainer mode")
        self.act_mode = {mode: self.action(sub, text, partial(self.set_mode, mode)) for mode, text in MODE_NAMES.items()}
        for a in self.act_mode.values():
            a.setCheckable(True)
        m.addSeparator()
        sub = m.addMenu("Background")
        for v in (0, .3, .6, .85):
            self.action(sub, f"{int(v * 100)} %", partial(self.set_value, "opacity", v))
        sub = m.addMenu("Size")
        for v in (.8, 1.0, 1.25, 1.5):
            self.action(sub, f"{int(v * 100)} %", partial(self.set_value, "scale", v))
        self.act_click = self.action(m, "Click-through (turn off from the tray icon)", self.toggle_click_through)
        self.act_click.setCheckable(True)
        self.act_click.setEnabled(QSystemTrayIcon.isSystemTrayAvailable())
        self.action(m, "Devices ...", self.devices)
        m.addSeparator()
        self.action(m, "Finish and save", self.close)
        self.action(m, "Discard this ride", self.discard)
        m.aboutToShow.connect(self.refresh_menu)
        return m

    def action(self, menu: QMenu, text: str, slot, key: str = "") -> QAction:
        a = QAction(text, self)
        if key:
            a.setShortcut(key)
            self.addAction(a)          # shortcuts work while the bar has focus
        a.triggered.connect(partial(self.fire, slot))
        menu.addAction(a)
        return a

    @staticmethod
    def fire(slot, checked: bool = False) -> None:
        """menu actions take no arguments (Qt passes `checked`)."""
        slot()

    def refresh_menu(self) -> None:
        self.act_pause.setText("Resume" if self.ride.run.paused else "Pause")
        for mode, a in self.act_mode.items():
            a.setChecked(self.ride.run.mode == mode)
        self.act_click.setChecked(self.click_through())

    def toggle_pause(self) -> None:
        self.ride.run.paused = not self.ride.run.paused

    def set_mode(self, mode: str) -> None:
        self.ride.run.choose_mode(mode)

    def set_value(self, name: str, v: float) -> None:
        setattr(self, name, v)
        if name == "scale":
            self.resize(int(W * v), int(H * v))
        self.save({name: v})
        self.update()

    def toggle_click_through(self) -> None:
        self.set_click_through(not self.click_through())

    def click_through(self) -> bool:
        return bool(self.windowFlags() & Qt.WindowTransparentForInput)

    def set_click_through(self, on: bool) -> None:
        self.setWindowFlag(Qt.WindowTransparentForInput, on)
        self.show()                    # changing window flags hides the window
        self.save({"click_through": on})

    def devices(self) -> None:
        choice = self.pick(self.ride.hub.choice, close_first=self.ride.link)
        self.ride.reconnect(self.ride.hub.choice if choice is None else choice)    # cancelled: the ones you had

    def discard(self) -> None:
        if self.ask("Throw this ride away?") == QMessageBox.Yes:
            self.outcome = "discard"
            self.close()

    def ask(self, text: str, buttons=QMessageBox.Yes | QMessageBox.No) -> QMessageBox.StandardButton:
        """A question over the bar, stay-on-top from the start (QMessageBox.question can't do that, and setting
        the flag once it's open would close it)."""
        box = QMessageBox(QMessageBox.Question, "goodpush", text, buttons, self)
        box.setWindowFlag(Qt.WindowStaysOnTopHint)
        raise_next_dialog()
        box.exec()
        return box.standardButton(box.clickedButton())

    # --- events -----------------------------------------------------------------------
    def showEvent(self, e) -> None:
        super().showEvent(e)
        mac_on_top_everywhere(self)    # again after each show: changing window flags makes a new native window

    def tick(self) -> None:
        self.ride.tick(time.time())
        self.update()

    def contextMenuEvent(self, e) -> None:
        raise_next_dialog()
        self.menu.exec(e.globalPos())

    def hit(self, e) -> str:
        """what's under the pointer: a key of BUTTONS, "profile" (the workout: drag the marker) or ""."""
        at = e.position() / self.scale
        name = next((k for k, box in BUTTONS.items() if box.contains(at)), "")
        if not name and self.ride.run.steps and PROFILE.adjusted(-6, -8, 6, 6).contains(at):
            name = "profile"
        return name

    def press(self, name: str) -> None:
        if name == "close":
            self.close()                       # closeEvent asks: save, discard or carry on
        elif name == "mode":
            self.ride.run.next_mode()
        elif name in ("easier", "harder"):
            self.ride.run.nudge(NUDGE if name == "harder" else -NUDGE)
        self.update()

    def seek_to(self, e) -> None:
        """the workout clock to where the pointer is along the profile."""
        f = (e.position().x() / self.scale - PROFILE.x()) / PROFILE.width()
        self.ride.run.seek(f * self.ride.run.total)
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() != Qt.LeftButton:
            return
        name = self.hit(e)
        if name == "profile":
            self.seeking = True
            self.seek_to(e)
        elif name:
            self.press(name)
        else:
            self.drag = e.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, e) -> None:
        if self.seeking:
            self.seek_to(e)
        elif self.drag is not None:
            self.move(e.globalPosition().toPoint() - self.drag)
        elif self.hit(e) != self.hover:
            self.hover = self.hit(e)
            self.setCursor(Qt.SizeHorCursor if self.hover == "profile" else
                           Qt.PointingHandCursor if self.hover else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, e) -> None:
        if self.hover:
            self.hover = ""
            self.unsetCursor()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        self.seeking = False
        if self.drag is not None:
            self.drag = None
            self.save({"x": self.x(), "y": self.y()})

    def wheelEvent(self, e) -> None:
        self.ride.run.nudge(1 if e.angleDelta().y() > 0 else -1)

    def closeEvent(self, e) -> None:
        if self.outcome == "save" and self.ride.recorder.seconds >= 60:
            ans = self.ask("Finish this ride: save it or throw it away?",
                           QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                e.ignore()
                return
            self.outcome = "save" if ans == QMessageBox.Save else "discard"
        elif self.outcome == "save" and self.ride.recorder.seconds > 0:     # under a minute: nothing to keep
            ans = self.ask("This ride is under a minute, too short to keep. Close it?",
                           QMessageBox.Discard | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                e.ignore()
                return
            self.outcome = "discard"
        self.timer.stop()
        self.tray.hide()
        e.accept()
        QApplication.quit()            # only the bar ends the ride (see RideWindow.run)

    # --- drawing ----------------------------------------------------------------------
    def zone(self, watts: float) -> str:
        f = watts / self.ftp if self.ftp else .7
        return next(c for edge, c in ZONES if f < edge)

    def paintEvent(self, e) -> None:
        r, run, now = self.ride, self.ride.run, time.time()
        self.dim = MUTED if self.opacity >= .3 else INK2         # grey text needs a background behind it
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.scale(self.scale, self.scale)
        bg, line = QColor(SURFACE), QColor(LINE)
        bg.setAlphaF(self.opacity)
        line.setAlphaF(max(self.opacity, .35))
        p.setPen(QPen(line, 1))
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(.5, .5, W - 1, H - 1), RADIUS, RADIUS)
        p.drawLine(QPointF(SPLIT, 14), QPointF(SPLIT, 64))      # metrics | workout

        power, target = r.power3(), run.target()
        colour = INK
        if target and power == power:
            colour = GOOD if abs(power - target) <= .05 * target else (WARN if power < target else BAD)
        self.label(p, 18, 10, "POWER")
        self.value(p, 18, 22, self.num(power), "W", 30, colour)
        kind, v = run.command()
        self.text(p, 18, 58, 140, 14, {"power": f"target {v:.0f} W", "grade": f"grade {v:+.1f} %",
                                       "resistance": f"resistance {v:.0f} %"}[kind], 11, self.dim)
        self.label(p, 136, 10, "HEART RATE", HR)
        self.value(p, 136, 24, self.num(r.hub.get("hr", now)), "bpm", 24, INK)
        self.label(p, 230, 10, "CADENCE")
        self.value(p, 230, 24, self.num(r.hub.get("cadence", now)), "rpm", 24, INK)
        if r.recorder.beats:                                    # a strap sending beat-to-beat intervals
            a, err, filled = r.alpha1
            if filled < 1:                                      # the first 2 min: how far to the first value
                self.value(p, 256, 55, "α1", f"{filled:.0%} of 2 min", 15, self.dim)
            else:
                self.value(p, 256, 55, f"α1 {self.num(a, 2)}", f"{err:.0%} err", 15,
                           self.dim if a != a else GOOD if a >= LT1_A1 else WARN if a >= LT2_A1 else BAD)
        self.button(p, "easier", "−")
        self.button(p, "mode", MODE_LABELS[run.mode], on=run.mode != "resistance")
        self.button(p, "harder", "+")

        step, nxt, left = run.step, run.next_step, run.step_left
        if step:
            bpm = run.hr_target()
            name = step["name"] + (f" · {step['cadence']} rpm" if step.get("cadence") else "") + \
                (f" · hold {bpm:.0f} bpm" if bpm else "") + (f"  {run.scale - 1:+.0%}" if run.scale != 1 else "")
            self.text(p, WORK, 10, 150, 20, name, 15, INK, weight=QFont.DemiBold)
            hint = MODE_LABELS.get(step.get("mode"), "")
            cue = step.get("cue", "") + (f" (suits {hint})" if hint and step["mode"] != run.mode else "")
            if cue and step["min"] * 60 - left < CUE_S:          # the step's cue for its first seconds
                self.text(p, WORK, 32, 150, 16, cue.strip(), 11, WARN)
            elif nxt:
                w = run.watts(nxt)
                self.text(p, WORK, 32, 150, 16, f"Next  {nxt['name']}" + (f" · {w:.0f} W" if w else ""), 11, INK2)
            warn = left <= 5 and int(now * 2) % 2 == 0
            self.text(p, 490, 6, 132, 42, self.clock(left), 32, WARN if warn else INK, weight=QFont.DemiBold,
                      right=True)
        else:
            self.text(p, WORK, 10, W - 18 - WORK, 20, "Workout done · ride on or finish" if run.done else run.title, 15, INK,
                      weight=QFont.DemiBold)
        end = run.ends_at(now)
        total = self.clock(r.moving) + (f"  ·  {self.clock(run.elapsed)} / {self.clock(run.total)}"
                                        if run.total else "") + \
            (f"  ·  ends {time.strftime('%H:%M', time.localtime(end))}" if end else "")
        self.text(p, WORK, 50, W - 18 - WORK, 14, total, 11, self.dim, right=True)
        state = self.state(now)
        if state:
            self.text(p, WORK, 50, 180, 14, state, 11, WARN)
        self.profile(p, PROFILE)
        self.close_mark(p)
        p.end()

    def close_mark(self, p: QPainter) -> None:
        """the × in the corner: faint, bright under the pointer."""
        lit = self.hover == "close"
        p.setPen(QPen(QColor(INK if lit else self.dim), 1.6 if lit else 1.2, Qt.SolidLine, Qt.RoundCap))
        m = CLOSE_BOX.adjusted(7, 7, -7, -7)
        p.drawLine(m.topLeft(), m.bottomRight())
        p.drawLine(m.topRight(), m.bottomLeft())

    def button(self, p: QPainter, name: str, s: str, on: bool = False) -> None:
        """a small pill: accent when switched on, a brighter edge under the pointer."""
        box = BUTTONS[name]
        fill = QColor(ACCENT if on else SURFACE2)
        fill.setAlphaF(1 if on else max(self.opacity, .55))
        p.setPen(QPen(QColor(INK2 if self.hover == name else (ACCENT if on else LINE)), 1))
        p.setBrush(fill)
        p.drawRoundedRect(box, 6, 6)
        self.text(p, box.x(), box.y(), box.width(), box.height(), s, 11 if len(s) > 1 else 14,
                  INK if on or self.hover == name else INK2, weight=QFont.DemiBold, centre=True)

    def state(self, now: float) -> str:
        r = self.ride
        trainer = r.hub.choice.get("trainer")
        if trainer and not r.hub.alive(trainer, now) and now - r.start > 15:
            return "No trainer data · right-click › Devices"
        if getattr(r.link, "trouble", ""):
            return f"{r.link.trouble} · right-click › Devices"
        if r.run.paused:
            return "Paused"
        if not r.riding(now) and not r.run.done:
            if r.hub.get("power", now) != r.hub.get("power", now):
                return "No power reading · clock stopped"
            return "Start pedalling" if r.run.elapsed == 0 else "Paused · pedal to go on"
        return ""

    def profile(self, p: QPainter, box: QRectF) -> None:
        run = self.ride.run
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(SURFACE2))
        p.drawRoundedRect(box, 3, 3)
        if not run.steps:
            return
        top = max(max(s["hi"], s.get("hi_end", 0)) for s in run.steps) or 1
        x = box.x()
        for s in run.steps:                      # a few dozen steps: drawing, not data
            w = s["min"] * 60 / run.total * box.width()
            a = (s["lo"] + s["hi"]) / 2
            b = (s["lo_end"] + s["hi_end"]) / 2 if "lo_end" in s else a
            ha, hb = (max(v / top, .15) * box.height() for v in (a, b))
            p.setBrush(QColor(self.zone((a + b) / 2)))
            if "lo_end" in s:                    # a ramp: a slope
                p.drawPolygon(QPolygonF([QPointF(x + .4, box.bottom()), QPointF(x + .4, box.bottom() - ha),
                                         QPointF(x + w - .4, box.bottom() - hb), QPointF(x + w - .4, box.bottom())]))
            else:
                p.drawRoundedRect(QRectF(x + .4, box.bottom() - ha, max(w - .8, .6), ha), 1.5, 1.5)
            x += w
        done = run.elapsed / run.total * box.width()
        p.setBrush(QColor(0, 0, 0, 110))
        p.drawRoundedRect(QRectF(box.x(), box.y(), done, box.height()), 3, 3)
        x = box.x() + done                       # the marker: where you are; drag it to move through the workout
        grab = self.seeking or self.hover == "profile"
        p.setBrush(QColor(ACCENT))
        p.drawRoundedRect(QRectF(x - 1.5, box.y() - 6, 3, box.height() + 9), 1.5, 1.5)
        p.setPen(QPen(QColor(INK), 1.5 if grab else 1))
        p.drawEllipse(QPointF(x, box.y() - 6), 5 if grab else 4, 5 if grab else 4)

    @staticmethod
    @cache
    def families() -> list[str]:
        """FONTS that are installed (Qt warns, slowly, about each missing one), the generic last as a fallback."""
        have = set(QFontDatabase.families())
        return [f for f in FONTS if f in have] + FONTS[-1:]

    @staticmethod
    def font(px: float, weight=QFont.Normal) -> QFont:
        f = QFont()
        f.setFamilies(Overlay.families())
        f.setPixelSize(round(px))
        f.setWeight(weight)
        return f

    def text(self, p: QPainter, x, y, w, h, s: str, px: float, colour: str, weight=QFont.Normal,
             right=False, centre=False) -> None:
        p.setFont(self.font(px, weight))
        align = (Qt.AlignHCenter if centre else Qt.AlignRight if right else Qt.AlignLeft) | Qt.AlignVCenter
        shadow = QColor(0, 0, 0, int(150 * (1 - self.opacity)))   # see-through bar: readable over video
        if shadow.alpha():
            p.setPen(shadow)
            p.drawText(QRectF(x, y + 1, w, h), align, s)
        p.setPen(QColor(colour))
        p.drawText(QRectF(x, y, w, h), align, s)

    def label(self, p: QPainter, x, y, s: str, dot: str = "") -> None:
        """a small spaced-out heading over a number, as on the app's tiles; `dot` = a colour mark after it."""
        f = self.font(9.5, QFont.DemiBold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 1.1)
        p.setFont(f)
        shadow = QColor(0, 0, 0, int(150 * (1 - self.opacity)))
        if shadow.alpha():
            p.setPen(shadow)
            p.drawText(QRectF(x, y + 1, 120, 12), Qt.AlignLeft | Qt.AlignVCenter, s)
        p.setPen(QColor(self.dim))
        p.drawText(QRectF(x, y, 120, 12), Qt.AlignLeft | Qt.AlignVCenter, s)
        if dot:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(dot))
            p.drawEllipse(QPointF(x + QFontMetricsF(f).horizontalAdvance(s) + 6, y + 6), 2.5, 2.5)

    def value(self, p: QPainter, x, y, s: str, unit: str, px: float, colour: str) -> None:
        """a number with its unit after it, small and muted, near the same baseline."""
        self.text(p, x, y, 130, px * 1.25, s, px, colour, weight=QFont.DemiBold)
        adv = QFontMetricsF(self.font(px, QFont.DemiBold)).horizontalAdvance(s)
        self.text(p, x + adv + 4, y + px * .3, 80, px, unit, 11, self.dim)

    @staticmethod
    def num(v: float, digits: int = 0) -> str:
        return "--" if v != v else f"{v:.{digits}f}"

    @staticmethod
    def clock(s: float) -> str:
        s = int(max(s, 0))
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"

    @staticmethod
    def icon() -> QIcon:
        pm = QPixmap(64, 64)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        QSvgRenderer(str(MARK)).render(p)
        p.end()
        return QIcon(pm)


class RideWindow:
    """Runs a ride in Qt: the device picker if needed, then the overlay until you finish.
    Returns the saved FIT path (None if discarded or too short) and the device choice used.
    save(dict) stores the bar's settings, remember(choice) the devices picked."""

    def __init__(self, ride, settings: dict, ftp: float, save, remember):
        self.ride, self.settings, self.ftp, self.save, self.remember = ride, settings, ftp, save, remember

    def pick(self, choice: dict, close_first=None) -> dict | None:
        if close_first is not None:
            close_first.close()
        d = DevicePicker(self.ride.link, choice)
        raise_next_dialog()
        if d.exec() != QDialog.Accepted:
            return None
        c = d.choice()
        self.remember(c)
        return c

    def run(self, pick: bool) -> tuple:
        app = QApplication.instance() or QApplication([])
        app.setStyleSheet(STYLE)
        app.setQuitOnLastWindowClosed(False)     # the bar is a tool window, which Qt doesn't count: closing any
                                                 # dialog (a Cancel) would quit and save the ride
        choice = self.ride.hub.choice
        if pick or not choice.get("trainer"):
            choice = self.pick(choice)
            if choice is None:
                return None, None
        self.ride.reconnect(choice)
        bar = Overlay(self.ride, self.settings, self.ftp, self.save, self.pick)
        bar.show()
        app.exec()
        if bar.outcome == "discard":
            self.ride.discard()
            return None, choice
        return self.ride.finish(), choice
