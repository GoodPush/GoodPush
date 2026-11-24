"""ANT+ and Bluetooth in one ride: `RideLinks` looks like one link (AntLink's interface) over several."""
from __future__ import annotations

import logging
import threading

from .sensors import Device, SensorHub

_log = logging.getLogger(__name__)


class RideLinks:
    """`links`: transport ("ant", "ble") -> its link. A scan listens on all of them at once; each chosen device is
    connected by its own transport's link; trainer commands go to the link the trainer is on. A transport that
    can't be used (no ANT+ stick, Bluetooth off) doesn't stop the others: its reason is only raised when nothing
    at all was found."""

    def __init__(self, hub: SensorHub, links: dict):
        self.hub, self.links = hub, links

    def scan(self, seconds: float = 10.0, on_found=None, stop: threading.Event | None = None) -> list[Device]:
        found, failed = [], {}
        threads = [threading.Thread(target=self.scan_one, args=(t, seconds, on_found, stop, found, failed),
                                    name=f"scan-{t}", daemon=True) for t in self.links]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if failed and not found:
            raise RuntimeError(" ".join(failed.values()))
        for t, why in failed.items():
            _log.info("%s scan: %s", t, why)
        return found

    def scan_one(self, transport: str, seconds: float, on_found, stop, found: list, failed: dict) -> None:
        try:
            found += self.links[transport].scan(seconds, on_found=on_found, stop=stop)
        except Exception as e:  # noqa: BLE001 - this transport's trouble, said why; the others go on
            failed[transport] = str(e) if isinstance(e, RuntimeError) else f"{transport} scan failed: {e}"

    def connect(self, devices: list[Device]) -> None:
        for t, link in self.links.items():
            mine = [d for d in devices if d.transport == t]
            if mine:
                link.connect(mine)

    @property
    def trainer(self):
        """the link the chosen trainer is on (None: no trainer, or its transport isn't here)."""
        key = self.hub.choice.get("trainer")
        return self.links.get(Device.from_key(key).transport) if key else None

    def set_power(self, watts: float) -> None:
        if self.trainer is not None:
            self.trainer.set_power(watts)

    def set_resistance(self, percent: float) -> None:
        if self.trainer is not None:
            self.trainer.set_resistance(percent)

    def set_grade(self, percent: float) -> None:
        if self.trainer is not None:
            self.trainer.set_grade(percent)

    @property
    def trouble(self) -> str:
        return getattr(self.trainer, "trouble", "")

    def close(self) -> None:
        for link in self.links.values():
            link.close()
