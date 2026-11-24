"""A pretend trainer + heart-rate strap with AntLink's interface: try the overlay without hardware
(`goodpush ride --sim`) and drive the ride logic in tests."""
from __future__ import annotations

import threading
import time

import numpy as np

from .sensors import Device, SensorHub


class SimLink:
    """Power follows the ERG target with a 2-s lag (or 120 W + 3 W per % in resistance mode, 150 W + 25 W per
    % of gradient in sim mode), cadence ~88,
    HR drifts towards 60 + 0.45 x power. `tick(now)` moves it on; `open` runs that on a thread."""

    DEVICES = [Device("sim", "fec", 1), Device("sim", "hr", 2)]

    def __init__(self, hub: SensorHub, seed: int = 0):
        self.hub = hub
        self.rng = np.random.default_rng(seed)
        self.target, self.resistance, self.grade = None, 20.0, None
        self.power, self.hr, self.last = 0.0, 70.0, None
        self.pedalling = True
        self._stop = threading.Event()

    def open(self) -> None:
        self._stop = threading.Event()
        threading.Thread(target=self.run, args=(self._stop,), name="sim", daemon=True).start()

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(0.25):
            self.tick(time.time())

    def scan(self, seconds: float = 10.0, on_found=None, stop=None) -> list[Device]:
        for d in self.DEVICES:
            if on_found:
                on_found(d)
        return list(self.DEVICES)

    def connect(self, devices: list[Device]) -> None:
        self.close()
        self.open()

    def tick(self, now: float) -> None:
        dt = 0.25 if self.last is None else min(now - self.last, 5.0)
        self.last = now
        want = (self.target if self.target is not None else 150 + 25 * self.grade if self.grade is not None
                else 120 + 3 * self.resistance) if self.pedalling else 0.0
        self.power += (want - self.power) * (1 - np.exp(-dt / 2.0))
        self.hr += (60 + 0.45 * self.power - self.hr) * (1 - np.exp(-dt / 30.0))
        noise = self.rng.normal(0, 1, 3)
        fec, hr = (d.key for d in self.DEVICES)
        self.hub.put(fec, "power", max(self.power + 4 * noise[0], 0), now)
        self.hub.put(fec, "cadence", 88 + 2 * noise[1] if self.pedalling else 0, now)
        self.hub.put(fec, "speed", (max(self.power, 0) / 0.25) ** (1 / 3) if self.pedalling else 0, now)
        self.hub.put(hr, "hr", self.hr + noise[2], now)

    def set_power(self, watts: float) -> None:
        self.target, self.grade = float(watts), None

    def set_resistance(self, percent: float) -> None:
        self.target, self.grade, self.resistance = None, None, float(percent)

    def set_grade(self, percent: float) -> None:
        self.target, self.grade = None, float(percent)

    def close(self) -> None:
        self._stop.set()
