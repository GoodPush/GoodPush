"""Bluetooth (BLE, through bleak and the computer's own Bluetooth): find smart trainers (FTMS), power meters
(Cycling Power), heart-rate straps and speed / cadence sensors (CSC), feed their readings to a SensorHub and steer
the trainer over the FTMS control point (ERG target power, resistance level, or a simulated road gradient).

bleak runs on asyncio: BleLink keeps its own event loop on a thread, so the rest of the ride (the overlay, the
session's tick) calls it like AntLink. Each chosen device has a task that connects and, whenever the device drops
out (out of range, a strap taken off), connects again. A device connected here stops advertising, and most
take one connection at a time: quit other apps using it first.
FTMS has no rider weight: in sim mode the trainer works the brake out with its own default.
"""
from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from functools import partial

from .ant import ACK_WAIT_S, MAX_CADENCE, MISSES, NOT_TAKEN, ROAD_CRR
from .sensors import Device, SensorHub

_log = logging.getLogger(__name__)
BASE_UUID = "0000{:04x}-0000-1000-8000-00805f9b34fb"     # Bluetooth SIG 16-bit ids in full
FTMS, CYCLING_POWER, HEART_RATE, CSC = (BASE_UUID.format(n) for n in (0x1826, 0x1818, 0x180D, 0x1816))
SERVICES = {FTMS: "ftms", CYCLING_POWER: "power", CSC: "speed_cadence", HEART_RATE: "hr"}   # a device: its first
INDOOR_BIKE, CONTROL_POINT, RESISTANCE_RANGE = (BASE_UUID.format(n) for n in (0x2AD2, 0x2AD9, 0x2AD6))
MEASUREMENTS = {"ftms": INDOOR_BIKE, "power": BASE_UUID.format(0x2A63), "speed_cadence": BASE_UUID.format(0x2A5B),
                "hr": BASE_UUID.format(0x2A37)}
# Indoor Bike Data fields after instantaneous speed: (flag bit, bytes); 2 = cadence (0.5 rpm), 6 = power (W)
BIKE_FIELDS = ((1, 2), (2, 2), (3, 2), (4, 3), (5, 2), (6, 2), (7, 2), (8, 5), (9, 1), (10, 1), (11, 2), (12, 2))
REQUEST_CONTROL, START, RESISTANCE, POWER, SIMULATION, ANSWER = 0x00, 0x07, 0x04, 0x05, 0x11, 0x80   # FTMS op codes
SUCCESS, NOT_PERMITTED, INVALID = 0x01, 0x05, 0x03
ANSWERS = {0x02: "NotSupported", INVALID: "Rejected", 0x04: "Fail"}     # the trainer didn't do it (AntLink's names)
CW = 0.51                              # kg/m: wind resistance (drag area x air density) of a rider on the hoods
NO_BLUETOOTH = "Bluetooth isn't available. Switch it on (Windows: Settings > Bluetooth & devices) and try again."
RECONNECT_S = 2.0                      # a device that dropped out (or wasn't found): connect again after this
RETRY_S = 1.0                          # a command the trainer didn't answer goes again


class CrankCadence:
    """Cadence from crank revolution data (Cycling Power, CSC): a revolution count (mod 2^16) and the last crank
    event's time (1/1024 s, mod 64 s). Messages with no new revolution repeat the last one: none for STOPPED_S is
    0 rpm. After longer than the time's rollover the two can't be tied together: it starts afresh."""
    STOPPED_S, ROLL_S = 3.0, 60.0

    def __init__(self):
        self.revs = self.time = None
        self.moved_at = 0.0

    def heard(self, revs: int, event: int, now: float) -> float | None:
        """the cadence this message tells (rpm), None when it tells nothing new."""
        n = 0 if self.revs is None else (revs - self.revs) % 65536
        dt = 0 if self.time is None else (event - self.time) % 65536 / 1024
        if self.revs is None or (n and now - self.moved_at > self.ROLL_S):
            self.revs, self.time, self.moved_at = revs, event, now
            return None
        if n == 0 or dt == 0:
            return 0.0 if now - self.moved_at > self.STOPPED_S else None
        self.revs, self.time, self.moved_at = revs, event, now
        c = 60 * n / dt
        return c if c <= MAX_CADENCE else None


class BleLink:
    """The computer's Bluetooth. `scan` lists what's around, `connect` keeps a connection to each chosen device.
    Trainer commands go out newest first, each waiting for the trainer's answer on the control point."""

    def __init__(self, hub: SensorHub):
        self.hub = hub
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.tasks = []
        self.trainer = None                           # the trainer's BleakClient while it's connected
        self.controlled = False                       # the trainer gave us control (after each connection)
        self.level = (0.0, 100.0)                     # the trainer's resistance levels (resistance range)
        self.level_bytes = 2                          # how it takes a level: sint16, or (FTMS 1.0's text) uint8
        self.wanted = None                            # the latest command
        self.misses = 0                               # commands not answered in a row
        self.refused = ""                             # what the trainer said to the last command it didn't do
        self.cadence: dict[str, CrankCadence] = {}
        self.answer = (None, None)                    # (op code, future) the control point's answer is awaited for
        self._wake = asyncio.Event()
        self.lock = threading.RLock()                 # one open / close at a time (the picker scans on a thread)

    def open(self) -> None:
        """an event loop on its own thread, with the trainer's command sender."""
        with self.lock:
            if self.loop is not None:
                return
            self.loop, self._wake = asyncio.new_event_loop(), asyncio.Event()
            self.thread = threading.Thread(target=self.loop.run_forever, name="ble", daemon=True)
            self.thread.start()
            asyncio.run_coroutine_threadsafe(self.send_loop(), self.loop)

    def scan(self, seconds: float = 10.0, on_found=None, stop: threading.Event | None = None) -> list[Device]:
        """Devices heard within `seconds` (or until `stop` is set); on_found(device) as each one turns up."""
        self.close()             # a device connected here doesn't advertise
        self.open()
        found: dict[str, Device] = {}
        try:
            asyncio.run_coroutine_threadsafe(self.listen(seconds, found, on_found, stop), self.loop).result()
        except Exception as e:  # noqa: BLE001 - no adapter, Bluetooth off, no permission: said why
            raise RuntimeError(f"{NO_BLUETOOTH} ({e})") from e
        finally:
            self.close()
        return list(found.values())

    async def listen(self, seconds: float, found: dict, on_found, stop: threading.Event | None) -> None:
        from bleak import BleakScanner
        scanner = BleakScanner(partial(self.advertised, found, on_found), service_uuids=list(SERVICES))
        await scanner.start()
        try:
            end = time.monotonic() + seconds
            while time.monotonic() < end and not (stop is not None and stop.is_set()):
                await asyncio.sleep(0.2)
        finally:
            await scanner.stop()

    def advertised(self, found: dict, on_found, device, adv) -> None:
        """a device advertising one of SERVICES: listed once, under the first of them it offers."""
        uuids = {u.lower() for u in adv.service_uuids or ()}
        profile = next((p for u, p in SERVICES.items() if u in uuids), None)
        if profile is None or device.address in found:
            return
        d = Device("ble", profile, device.address, name=adv.local_name or device.name or "")
        found[device.address] = d
        if on_found:
            on_found(d)

    def connect(self, devices: list[Device]) -> None:
        self.open()
        for d in devices:
            self.tasks.append(asyncio.run_coroutine_threadsafe(self.keep(d), self.loop))

    async def keep(self, d: Device) -> None:
        """connected to `d` for the rest of the ride: again whenever it drops out."""
        from bleak import BleakClient
        while True:
            gone = asyncio.Event()
            client = BleakClient(d.number, disconnected_callback=partial(self.lost, gone))
            try:
                await client.connect()
                await self.subscribe(client, d)
                await gone.wait()
                _log.info("Bluetooth %s dropped out", d.label)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - not in range yet, busy, refused: tried again shortly
                _log.debug("Bluetooth %s: %s", d.label, e)
            finally:
                if self.trainer is client:
                    self.trainer = None
                try:
                    await client.disconnect()
                except Exception:  # noqa: BLE001 - gone already
                    pass
            await asyncio.sleep(RECONNECT_S)

    @staticmethod
    def lost(gone: asyncio.Event, client) -> None:
        gone.set()

    async def subscribe(self, client, d: Device) -> None:
        handlers = {"ftms": self.on_bike, "power": self.on_power, "speed_cadence": self.on_csc, "hr": self.on_hr}
        await client.start_notify(MEASUREMENTS[d.profile], partial(handlers[d.profile], d.key))
        if d.profile == "ftms" and d.key == self.hub.choice["trainer"]:
            try:
                self.level = self.resistance_range(await client.read_gatt_char(RESISTANCE_RANGE))
            except Exception:  # noqa: BLE001 - optional: then levels 0-100
                self.level = (0.0, 100.0)
            await client.start_notify(CONTROL_POINT, self.on_answer)          # its answers come as indications
            self.trainer, self.controlled, self.misses = client, False, 0
            self._wake.set()                                                  # what it should do, at once

    def on_bike(self, key: str, char, data: bytearray) -> None:
        for q, v in self.indoor_bike(bytes(data)).items():
            if q != "cadence" or v <= MAX_CADENCE:
                self.hub.put(key, q, v)

    def on_power(self, key: str, char, data: bytearray) -> None:
        watts, crank = self.cycling_power(bytes(data))
        self.hub.put(key, "power", watts)
        self.put_cadence(key, crank)

    def on_csc(self, key: str, char, data: bytearray) -> None:
        self.put_cadence(key, self.csc(bytes(data)))

    def put_cadence(self, key: str, crank: tuple[int, int] | None) -> None:
        if crank is not None:
            c = self.cadence.setdefault(key, CrankCadence()).heard(*crank, time.time())
            if c is not None:
                self.hub.put(key, "cadence", c)

    def on_hr(self, key: str, char, data: bytearray) -> None:
        """the strap sends the beat-to-beat intervals themselves: each one as it is (BLE resends what the radio
        lost while connected; beats during a reconnection are simply missing)."""
        bpm, rr = self.heart_rate(bytes(data))
        if bpm > 0:
            self.hub.put(key, "hr", bpm)
        if rr:
            self.hub.add_beats(key, [(x, 1) for x in rr])

    @staticmethod
    def indoor_bike(data: bytes) -> dict[str, float]:
        """FTMS Indoor Bike Data: speed (m/s), cadence (rpm) and power (W), the ones this message carries.
        Flag bit 0 clear: instantaneous speed first (0.01 km/h)."""
        flags, at, out = int.from_bytes(data[:2], "little"), 2, {}
        if not flags & 1 and len(data) >= 4:
            out["speed"] = int.from_bytes(data[2:4], "little") / 360
            at = 4
        for bit, size in BIKE_FIELDS:
            if not flags >> bit & 1:
                continue
            if at + size > len(data):
                break
            if bit == 2:
                out["cadence"] = int.from_bytes(data[at:at + 2], "little") / 2
            elif bit == 6:
                out["power"] = int.from_bytes(data[at:at + 2], "little", signed=True)
            at += size
        return out

    @staticmethod
    def cycling_power(data: bytes) -> tuple[int, tuple[int, int] | None]:
        """Cycling Power Measurement: power (W) and the crank revolution data (count, event time) if it's there.
        Optional fields before it: pedal balance (bit 0, 1 byte), torque (bit 2, 2), wheel revolutions (bit 4, 6)."""
        flags, watts = struct.unpack_from("<Hh", data)
        at = 4 + (flags & 1) + 2 * (flags >> 2 & 1) + 6 * (flags >> 4 & 1)
        crank = struct.unpack_from("<HH", data, at) if flags & 0x20 and len(data) >= at + 4 else None
        return watts, crank

    @staticmethod
    def csc(data: bytes) -> tuple[int, int] | None:
        """CSC Measurement: the crank revolution data (count, event time), after the wheel's (bit 0, 6 bytes)."""
        flags = data[0]
        at = 1 + 6 * (flags & 1)
        return struct.unpack_from("<HH", data, at) if flags & 2 and len(data) >= at + 4 else None

    @staticmethod
    def heart_rate(data: bytes) -> tuple[int, list[float]]:
        """Heart Rate Measurement: bpm (uint8, or uint16 with flag bit 0) and the beat-to-beat intervals (s) since
        the last message (bit 4; after the energy expended, bit 3)."""
        flags = data[0]
        bpm, at = (struct.unpack_from("<H", data, 1)[0], 3) if flags & 1 else (data[1], 2)
        at += 2 * (flags >> 3 & 1)
        n = (len(data) - at) // 2 if flags & 0x10 else 0
        return bpm, [v / 1024 for v in struct.unpack_from(f"<{n}H", data, at)]

    @staticmethod
    def resistance_range(data: bytes) -> tuple[float, float]:
        """FTMS Supported Resistance Level Range: lowest and highest level (0.1)."""
        lo, hi, _ = struct.unpack_from("<hhH", data)
        return lo / 10, hi / 10

    def payload(self, c: tuple) -> bytes:
        """the control point message for command `c`."""
        if c[0] == "power":
            return struct.pack("<Bh", POWER, int(c[1]))
        if c[0] == "resistance":
            lo, hi = self.level
            level = round((lo + c[1] / 100 * (hi - lo)) * 10)            # 0.1
            return struct.pack("<Bh", RESISTANCE, level) if self.level_bytes == 2 else bytes(
                [RESISTANCE, min(max(level, 0), 255)])
        # simulation: wind (0.001 m/s), gradient (0.01 %), rolling resistance (1e-4), wind resistance (0.01 kg/m)
        return struct.pack("<BhhBB", SIMULATION, 0, round(c[1] * 100), round(ROAD_CRR / 1e-4), round(CW / 0.01))

    def set_power(self, watts: float) -> None:
        """ERG: the trainer holds this power whatever the cadence."""
        self.command(("power", int(round(min(max(watts, 0), 2000)))))

    def set_resistance(self, percent: float) -> None:
        """Resistance mode, 0-100 % of the trainer's levels: power follows your effort."""
        self.command(("resistance", float(min(max(percent, 0), 100))))

    def set_grade(self, percent: float) -> None:
        """Sim mode: a road this steep."""
        self.command(("grade", float(min(max(percent, -200), 200))))

    def command(self, c: tuple) -> None:
        self.wanted = c
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._wake.set)

    @property
    def trouble(self) -> str:
        """what's wrong with steering the trainer; "" when all's well."""
        if self.misses >= MISSES:
            return "Trainer isn't taking commands"
        return f"Trainer refused the command ({self.refused})" if self.refused in NOT_TAKEN else ""

    async def send_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            c, client = self.wanted, self.trainer
            if c is None or client is None:
                continue                     # sent once the trainer is connected (subscribe wakes this)
            answer = await self.deliver(client, c)
            if answer == SUCCESS or answer in ANSWERS:
                self.misses, self.refused = 0, ANSWERS.get(answer, "")
                continue
            self.misses += 1                 # no answer, or control taken away: again shortly
            if self.misses == MISSES:
                _log.warning("trainer isn't answering commands (%s)", c)
            self.loop.call_later(RETRY_S, self._wake.set)

    async def deliver(self, client, c: tuple) -> int | None:
        """command `c` to the trainer, asking for control first: its answer (a result code), None for none."""
        if not self.controlled:
            if await self.write(client, bytes([REQUEST_CONTROL])) != SUCCESS:
                return None
            await self.write(client, bytes([START]))          # some trainers need it, some say it's not supported
            self.controlled = True
        answer = await self.write(client, self.payload(c))
        if answer == INVALID and c[0] == "resistance" and self.level_bytes == 2:
            self.level_bytes = 1                              # a trainer that takes FTMS 1.0's one byte
            answer = await self.write(client, self.payload(c))
        if answer == NOT_PERMITTED:
            self.controlled = False
        return answer

    async def write(self, client, message: bytes) -> int | None:
        """`message` to the control point; the trainer's result code, None without one within ACK_WAIT_S."""
        self.answer = (message[0], asyncio.get_running_loop().create_future())
        try:
            await client.write_gatt_char(CONTROL_POINT, message, response=True)
            return await asyncio.wait_for(self.answer[1], ACK_WAIT_S)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - not delivered / not answered: a miss
            _log.debug("trainer command %s: %s", message[0], e)
            return None

    def on_answer(self, char, data: bytearray) -> None:
        op, future = self.answer
        if len(data) >= 3 and data[0] == ANSWER and data[1] == op and future is not None and not future.done():
            future.set_result(data[2])

    def close(self) -> None:
        with self.lock:
            loop, self.loop = self.loop, None
            if loop is None:
                return
            try:
                asyncio.run_coroutine_threadsafe(self.shutdown(), loop).result(10)
            except Exception:  # noqa: BLE001 - a device that won't disconnect: the loop stops anyway
                _log.debug("Bluetooth shutdown", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
            self.thread.join(5)
            loop.close()
            self.tasks, self.trainer = [], None

    @staticmethod
    async def shutdown() -> None:
        """every task on the loop cancelled (connections closed in `keep`)."""
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
