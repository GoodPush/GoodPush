"""ANT+ through a USB stick (openant): find trainers, power meters, heart-rate straps and cadence sensors,
feed their readings to a SensorHub and steer the trainer over FE-C (ERG target power, resistance, or a
simulated road: gradient with your and the bike's weight).

The stick needs a libusb driver. Windows: libusb-win32 or WinUSB (Zadig; an Incyclist / Zwift install
often set one up already). Linux: a udev rule, e.g. /etc/udev/rules.d/ant-usb.rules with
SUBSYSTEM=="usb", ATTRS{idVendor}=="0fcf", MODE="0666". macOS: no driver; libusb itself comes from the
libusb-package wheel (or Homebrew's libusb) - without it pyusb sees no USB devices at all.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from functools import cache, partial

from .sensors import Device, SensorHub

_log = logging.getLogger(__name__)
NO_STICK = ("No ANT+ USB stick found. Plug it in. On Windows it needs its USB driver once: Start menu > goodpush "
            "ANT+ USB setup (Zadig: WinUSB). On Linux: sh install.sh from the download (or a udev rule for vendor "
            "0fcf). From source on a Mac: pip install libusb-package.")
NO_ANSWER = "The ANT+ stick doesn't answer. Unplug it, plug it back in and try again."
OPEN_TRIES, OPEN_PAUSE_S = 2, 1.5      # a stick that doesn't answer at once gets one more go
ROAD_CRR = 0.004                       # rolling resistance of a road tyre on tarmac (the FE-C default)
WHEEL_M = 0.70                         # 700c wheel diameter: speed from the trainer's flywheel
STATUS_PAGE = [0x46, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x47, 0x01]    # FE-C request: send page 71 (command status) once
RETRY_S = 0.25                         # a command the trainer didn't acknowledge goes again (its next message)
ACK_WAIT_S = 3.0                       # the stick says a sent page was delivered / failed within this, or (an RX
                                       # fail in the same timeslot) never: then the trainer's page 71 tells
TX_DONE, TX_FAILED = 5, 6              # ANT channel events: the trainer acknowledged the page / didn't
MISSES = 8                             # this many failed sends in a row (~ a few s): the trainer isn't listening
HOLDS = {"power": "TargetPower", "resistance": "Basic", "grade": "Track"}   # page 71's mode for each command
FEC = 17                               # ANT+ device type of a smart trainer
MAX_CADENCE = 250                      # rpm: above this a reading is a counter glitch (first message, rollover)
NOT_TAKEN = ("Fail", "NotSupported", "Rejected")                 # page 71 answers: the trainer didn't do it
BATTERY_IDS = 16                       # a battery page (82) numbers a device's batteries 0-15


class Tracking:
    """Mixed into an openant device: its channel tracks the device. openant opens it as a background-scanning
    channel (extended assignment 0x01), which only catches a message now and then and almost never the
    trainer's acknowledgement, so ERG / resistance / gradient commands didn't get through.
    Sensors other than the trainer search at low priority: a channel searching at high priority (a strap
    not worn, a flat battery) takes the trainer's timeslots, and the trainer stops getting commands.
    openant keeps room for batteries 0-14 only: a device reporting battery 15 raised in its message loop and
    every sensor went quiet (a ride lost them all at its first ERG change), so the list gets all 16."""

    def open_channel(self, extended=True, channel_type=None, ext_assign=None):
        from openant.devices.common import BatteryData
        batteries = self.data["batteries"]
        batteries += [BatteryData() for _ in range(BATTERY_IDS - len(batteries))]
        super().open_channel(extended, channel_type, ext_assign)
        if self.device_type != FEC:
            from openant.base.message import Message
            ant, n = self.node.ant, self.channel.id
            ant.write_message(Message(Message.ID.SET_CHANNEL_SEARCH_TIMEOUT, [n, 0]))         # no high priority
            ant.write_message(Message(Message.ID.LOW_PRIORITY_CHANNEL_SEARCH_TIMEOUT, [n, 0xFF]))   # ever


class AntBeats:
    """Beat-to-beat (RR) intervals rebuilt from an ANT+ heart-rate strap (ANT+ heart rate monitor profile). Every
    page carries the last beat's time (1/1024 s, rolling over every 64 s) and a beat count (mod 256); page 4 also
    the time of the beat before it. A count up by one: one interval. Up by two on page 4: both, split at that
    earlier beat. Up by more (messages lost on the radio): one interval over those beats, its length exact and its
    beat count known - a missed-beat artifact for the alpha1 filter, never a guess, and the intervals still add up
    to the ride's time. After silence near the rollover the beats can't be tied together: it starts afresh."""
    ROLL_S = 64.0                       # the beat time's rollover

    def __init__(self):
        self.count = self.time = self.heard_at = None

    def heard(self, page: int, count: int, beat_time: float, previous: float, now: float) -> list[tuple[float, int]]:
        """the intervals this message completes, in order: (seconds, beats it spans: 1 unless beats were missed)."""
        fresh = self.count is None or now - self.heard_at > self.ROLL_S - 4
        n = 0 if fresh else (count - self.count) % 256
        self.heard_at = now
        if fresh or n == 0:
            if fresh:
                self.count, self.time = count, beat_time
            return []
        if n == 2 and page & 0x7F == 4 and previous >= 0:
            rr = [((previous - self.time) % self.ROLL_S, 1), ((beat_time - previous) % self.ROLL_S, 1)]
        else:
            rr = [((beat_time - self.time) % self.ROLL_S, n)]
        self.count, self.time = count, beat_time
        return rr


@cache
def tracking(device_class: type) -> type:
    """`device_class` (an openant device) with a tracking channel."""
    return type(device_class.__name__, (Tracking, device_class), {})


class AntLink:
    """One ANT+ stick. `scan` lists what's around, `connect` opens a channel per chosen device.
    Trainer commands go out on their own thread (an acknowledged send blocks until the trainer answers),
    newest first: a target that's already outdated is never sent."""

    TYPES = {17: "fec", 11: "power", 120: "hr", 122: "cadence", 121: "speed_cadence"}   # ANT+ device types

    def __init__(self, hub: SensorHub, rider_kg: float = 75.0, bike_kg: float = 9.0):
        self.hub = hub
        self.rider_kg, self.bike_kg = rider_kg, bike_kg     # sim mode: the mass you're riding up the hill
        self.node = None
        self.channels = []
        self.trainer = None
        self.configured = False
        self.misses = 0                               # trainer commands not acknowledged in a row
        self.beats: dict[str, AntBeats] = {}          # heart-rate straps: their beat-to-beat intervals
        self._command = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self.lock = threading.RLock()                 # one open / close at a time (the picker scans on a thread)

    def open(self) -> None:
        """the stick, ready for channels. One that doesn't answer is closed again (its USB handle too) and
        tried once more; then a RuntimeError saying why (another app has it, no answer, no stick)."""
        # openant logs every idle USB read timeout as a warning, and harmless channel-close errors
        logging.getLogger("openant").setLevel(logging.CRITICAL)
        from openant.devices import ANTPLUS_NETWORK_KEY
        from openant.easy.node import Node
        self.find_libusb()
        err = None
        with self.lock:
            for _ in range(OPEN_TRIES):                # two tries
                node = None
                try:
                    node = Node()
                    node.set_network_key(0x00, ANTPLUS_NETWORK_KEY)
                    self.node = node
                    break
                except Exception as e:  # noqa: BLE001 - USB / ANT errors of every kind: closed, then said why
                    err = e
                    if node is not None:
                        self.stop_node(node)
                    time.sleep(OPEN_PAUSE_S)
            else:
                raise RuntimeError(self.why_not_open(err)) from err
        threading.Thread(target=self.run_node, args=(self.node,), name="ant-node", daemon=True).start()
        self._stop = threading.Event()             # one per node: a rescan's old sender stops for good
        threading.Thread(target=self.send_loop, args=(self._stop,), name="ant-send", daemon=True).start()

    @staticmethod
    def run_node(node) -> None:
        """openant's message loop, kept going until the node stops: an error handling one message (openant
        misreading a page) would otherwise end it, and every sensor would go quiet for the rest of the ride."""
        while True:
            try:
                node.start()
                return
            except Exception:  # noqa: BLE001 - that message is dropped, the next ones still arrive
                _log.warning("ANT+ message dropped", exc_info=True)

    @staticmethod
    def find_libusb() -> None:
        """pyusb finds libusb by itself when it's installed system-wide (Linux, Homebrew on macOS) and with the
        libusb-win32 driver (it brings its DLL). Otherwise (WinUSB on Windows, a Mac without Homebrew's
        libusb) the libusb-package wheel's copy: pyusb's libusb1 backend is loaded from it once and keeps it
        for every later lookup, openant's too. Only used when nothing else is found: with the libusb-win32
        driver, libusb-1.0 would be refused access."""
        import usb.core
        try:
            usb.core.find(idVendor=0x0FCF)
            return
        except usb.core.NoBackendError:
            pass
        try:
            import libusb_package
        except ImportError:
            return
        import usb.backend.libusb1
        lib = libusb_package.get_library_path()
        if lib:
            os.environ["PATH"] = os.path.dirname(str(lib)) + os.pathsep + os.environ.get("PATH", "")   # Windows DLL search
            usb.backend.libusb1.get_backend(find_library=lambda name: str(lib))       # loaded once, then shared

    def scan(self, seconds: float = 10.0, on_found=None, stop: threading.Event | None = None) -> list[Device]:
        """Devices heard within `seconds` (or until `stop` is set); on_found(device) as each one turns up."""
        from openant.devices.scanner import Scanner
        self.close()
        self.open()
        found = []
        scanner = Scanner(self.node)
        scanner.on_found = partial(self.found, found, on_found)
        (stop or threading.Event()).wait(seconds)
        self.close()             # the stick doesn't free the scan channel: devices get a fresh node
        return found

    def found(self, found: list, on_found, heard: tuple) -> None:
        number, dtype, trans = heard
        if dtype in self.TYPES:
            d = Device("ant", self.TYPES[dtype], number, trans)
            found.append(d)
            if on_found:
                on_found(d)

    def connect(self, devices: list[Device]) -> None:
        from openant.devices.bike_speed_cadence import BikeCadence, BikeSpeedCadence
        from openant.devices.fitness_equipment import FitnessEquipment
        from openant.devices.heart_rate import HeartRate
        from openant.devices.power_meter import PowerMeter
        if self.node is None:
            self.open()
        kinds = {"fec": FitnessEquipment, "power": PowerMeter, "hr": HeartRate, "cadence": BikeCadence,
                 "speed_cadence": BikeSpeedCadence}
        for d in devices:
            ch = tracking(kinds[d.profile])(self.node, device_id=d.number)
            ch.on_device_data = partial(self.on_data, d.key)
            self.channels.append(ch)
            if d.key == self.hub.choice["trainer"]:
                self.trainer = ch
                self.configured, self.misses = False, 0     # sim mode tells a new trainer your weight first

    def on_data(self, key: str, page: int, name: str, data) -> None:
        put = self.hub.put
        if name == "standard_power":
            put(key, "power", data.instantaneous_power)
            if data.cadence != 255:
                put(key, "cadence", data.cadence)
        elif name == "general_fe":
            if data.speed < 65:                       # 65.535 = not sent
                put(key, "speed", data.speed)
        elif name == "heart_rate":
            if data.heart_rate > 0:
                put(key, "hr", data.heart_rate)
            rr = self.beats.setdefault(key, AntBeats()).heard(page, data.beat_count, data.beat_time,
                                                              data.previous_heart_beat_time, time.time())
            if rr:
                self.hub.add_beats(key, rr)
        elif name == "bike_cadence":
            c = data.calculate_cadence()              # (openant's .cadence property drops the value)
            if c is not None and 0 <= c <= MAX_CADENCE:
                put(key, "cadence", c)

    def set_power(self, watts: float) -> None:
        """ERG: the trainer holds this power whatever the cadence (FE-C page 49)."""
        self.command(("power", int(round(min(max(watts, 0), 2000)))))

    def set_resistance(self, percent: float) -> None:
        """Resistance mode (FE-C page 48), 0-100 %: power follows your effort, as for sprints and tests."""
        self.command(("resistance", float(min(max(percent, 0), 100))))

    def set_grade(self, percent: float) -> None:
        """Sim mode (FE-C page 51): a road this steep; the trainer works out the brake from your speed and
        weight, so gears, cadence and sprints feel as they do outside."""
        self.command(("grade", float(min(max(percent, -200), 200))))

    @staticmethod
    def power_page(watts: int) -> list[int]:
        """FE-C page 49 (target power) in 0.25 W."""
        q = int(watts) * 4
        return [0x31, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, q & 0xFF, q >> 8 & 0xFF]

    @staticmethod
    def resistance_page(percent: float) -> list[int]:
        """FE-C page 48 (basic resistance) in 0.5 %."""
        return [0x30, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, int(percent * 2) & 0xFF]

    def pages(self, c: tuple) -> list[list[int]]:
        """the pages for command `c`, the trainer told your weight before its first road gradient."""
        if c[0] == "power":
            return [self.power_page(c[1])]
        if c[0] == "resistance":
            return [self.resistance_page(c[1])]
        return ([] if self.configured else [self.user_page(self.rider_kg, self.bike_kg)]) + [self.track_page(c[1])]

    @property
    def trouble(self) -> str:
        """what's wrong with steering the trainer: commands it keeps not acknowledging, or one its page 71
        reply says it didn't carry out; "" when all's well."""
        status = getattr(getattr(self.trainer, "command_status", None), "name", "")
        if self.misses >= MISSES:
            return "Trainer isn't taking commands"
        return f"Trainer refused the command ({status})" if status in NOT_TAKEN else ""

    @staticmethod
    def track_page(grade: float, crr: float = ROAD_CRR) -> list[int]:
        """FE-C page 51 (track resistance): gradient in 0.01 % from -200 %, rolling resistance in 5e-5."""
        g = int(round((grade + 200) * 100))
        return [0x33, 0xFF, 0xFF, 0xFF, 0xFF, g & 0xFF, g >> 8 & 0xFF, int(round(crr / 5e-5)) & 0xFF]

    @staticmethod
    def user_page(rider_kg: float, bike_kg: float, wheel_m: float = WHEEL_M) -> list[int]:
        """FE-C page 55 (user configuration): rider in 0.01 kg, bike in 0.05 kg (12 bits), wheel in cm."""
        u, b = int(round(rider_kg * 100)), int(round(bike_kg / 0.05))
        return [0x37, u & 0xFF, u >> 8 & 0xFF, 0xFF, (b & 0xF) << 4, b >> 4 & 0xFF, int(round(wheel_m * 100)), 0x00]

    def channel_event(self, channel_id: int, codes: tuple, seconds: float) -> int | None:
        """the first of the events `codes` the stick reports on channel `channel_id` within `seconds` (None:
        none). Only the send thread waits on openant's event queue, so it empties it first (openant never does)."""
        cond, events = self.node._event_cond, self.node._events
        end = time.monotonic() + seconds
        with cond:
            while not (hit := [e[1] for e in events if e[0] == channel_id and e[1] in codes]):
                if time.monotonic() >= end:
                    return None
                cond.wait(max(end - time.monotonic(), 0))
        return hit[0]

    def clear_events(self) -> None:
        with self.node._event_cond:
            self.node._events.clear()

    def deliver(self, page: list[int]) -> int | None:
        """`page` to the trainer as an acknowledged message: TX_DONE once the stick says the trainer
        acknowledged it, TX_FAILED when it says it didn't (radio loss: try again), None when it says nothing in
        ACK_WAIT_S: the stick lost the answer (an RX fail in the same timeslot), though the page may well have
        arrived (`holds` checks). openant's own send misses "transfer failed" and sits out 10 s, so it isn't
        used."""
        ch = self.trainer.channel
        self.clear_events()
        try:
            self.node.ant.send_acknowledged_data(ch.id, page)
        except Exception as e:  # noqa: BLE001 - the stick went away: no answer
            _log.debug("trainer page %s not sent: %s", page[0], e)
            return None
        return self.channel_event(ch.id, (TX_DONE, TX_FAILED), ACK_WAIT_S)

    def holds(self, c: tuple) -> bool:
        """does the trainer's latest page 71 (command status, which openant keeps in data["fe"]) show it
        carrying out command `c`?"""
        fe = self.trainer.data["fe"]
        if getattr(fe.resistance_mode, "name", "") != HOLDS[c[0]]:
            return False
        return c[0] == "grade" or abs(fe.resistance - c[1]) <= 0.5

    def command(self, c: tuple) -> None:
        self._command = c
        self._wake.set()

    def send_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if not self._wake.wait(1.0):
                continue
            self._wake.clear()
            c, self._command = self._command, None
            if c is None or self.trainer is None:
                continue
            answer = TX_DONE
            for page in self.pages(c):
                answer = self.deliver(page)
                if answer != TX_DONE:
                    break
            if answer == TX_DONE:
                self.deliver(STATUS_PAGE)            # the trainer's page 71: what it's doing now (holds, trouble)
            if answer == TX_DONE or self.holds(c):
                self.configured = self.configured or c[0] == "grade"
                self.misses = 0
                continue
            self.misses += 1                         # a missed acknowledgement is ordinary radio loss: again
            if self.misses == MISSES:
                _log.warning("trainer isn't acknowledging commands (%s)", c)
            if self._command is None:                # nothing newer: the same again shortly
                self._command = c
                stop.wait(RETRY_S)
                self._wake.set()

    def close(self) -> None:
        with self.lock:
            self._stop.set()
            self._wake.set()
            for ch in self.channels:
                try:
                    ch.close_channel()
                except Exception:
                    pass
            self.channels, self.trainer = [], None
            if self.node is not None:
                self.stop_node(self.node)
                self.node = None

    @staticmethod
    def stop_node(node) -> None:
        """stop an openant node and let go of the stick: its stop() closes the USB handle only once the node
        has run, so a node that never got going would keep the stick claimed."""
        for step in (node.stop, node.ant._driver.close):
            try:
                step()
            except Exception:  # noqa: BLE001 - already stopped / closed
                pass

    @classmethod
    def why_not_open(cls, err: Exception) -> str:
        """what to tell you when the stick won't open."""
        owner = cls.stick_owner()
        if owner:
            return f"The ANT+ stick is in use by {owner}: only one app can use it at a time. Quit {owner} and try again."
        if isinstance(err, ValueError) or "not found" in str(err).lower() or "NoBackend" in type(err).__name__:
            return f"{NO_STICK} ({err})"
        return f"{NO_ANSWER} ({err})"

    @staticmethod
    def stick_owner() -> str:
        """macOS: the other app holding the ANT+ stick (vendor 0fcf), from the USB registry; "" if none / not
        a Mac."""
        if sys.platform != "darwin":
            return ""
        try:
            out = subprocess.run(["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
        for block in out.split("+-o ")[1:]:            # one per USB device: a handful
            m = re.search(r'"UsbExclusiveOwner" = "pid (\d+), ([^"]+)"', block)
            if '"idVendor" = 4047' in block and m and int(m.group(1)) != os.getpid():
                return m.group(2)
        return ""
