"""The devices on the link and the latest readings from the ones you picked."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

ROLES = ("trainer", "power", "cadence", "hr")
# per device profile: what it's called and which roles it can fill
PROFILES = {
    "fec": ("Smart trainer", ("trainer", "power", "cadence")),
    "ftms": ("Smart trainer", ("trainer", "power", "cadence")),       # Bluetooth (Fitness Machine Service)
    "power": ("Power meter", ("power", "cadence")),
    "hr": ("Heart rate", ("hr",)),
    "cadence": ("Cadence sensor", ("cadence",)),
    "speed_cadence": ("Speed / cadence sensor", ("cadence",)),
}
# where each reading may come from, first choice first: power from a power meter you picked, else the trainer.
# A later choice fills in while the ones before it are silent (one ANT+ channel missing messages for a few s, a
# Bluetooth device reconnecting)
SOURCES = {"power": ("power", "trainer"), "cadence": ("cadence", "power", "trainer"), "hr": ("hr",),
           "speed": ("trainer",)}
STALE = 3.0          # seconds without a reading: no value


@dataclass(frozen=True)
class Device:
    transport: str       # ant | ble | sim
    profile: str         # a key of PROFILES
    number: int | str    # ANT device number; Bluetooth: the device's address
    trans_type: int = 0
    name: str = field(default="", compare=False)      # what a Bluetooth device calls itself (not in the key)

    @property
    def key(self) -> str:
        return f"{self.transport}:{self.profile}:{self.number}"

    @classmethod
    def from_key(cls, key: str) -> Device:
        transport, profile, number = key.split(":", 2)          # a Bluetooth address has colons of its own
        return cls(transport, profile, number if transport == "ble" else int(number))

    @property
    def label(self) -> str:
        kind = PROFILES[self.profile][0]
        if self.transport == "ble":
            return f"{kind} {self.name or str(self.number)[-5:]} (Bluetooth)"
        return f"{kind} #{self.number}"

    @property
    def roles(self) -> tuple:
        return PROFILES[self.profile][1]


class SensorHub:
    """Latest power / cadence / hr / speed, thread-safe. `choice`: role -> device key ("" = none).
    A reading comes from the first of the chosen devices (in SOURCES order) that has a fresh one."""

    def __init__(self, choice: dict[str, str]):
        self._lock = threading.Lock()
        self._values: dict[tuple[str, str], tuple[float, float]] = {}    # (quantity, device) -> (value, time)
        self._beats: list[tuple[float, float, int]] = []   # (time heard, beat-to-beat interval s, beats it spans)
        self.seen: dict[str, float] = {}         # device key -> time of its last reading
        self.choose(choice)

    def choose(self, choice: dict[str, str]) -> None:
        picked = {r: choice.get(r) or "" for r in ROLES}
        with self._lock:
            self.choice = picked
            self.sources = {q: tuple(dict.fromkeys(picked[r] for r in roles if picked[r])) for q, roles in SOURCES.items()}
            self._values.clear()

    def devices(self) -> list[Device]:
        """The chosen devices, each once."""
        return [Device.from_key(k) for k in dict.fromkeys(v for v in self.choice.values() if v)]

    def put(self, key: str, quantity: str, value: float, t: float | None = None) -> None:
        t = time.time() if t is None else t
        with self._lock:
            self.seen[key] = t
            if key in self.sources.get(quantity, ()):
                self._values[quantity, key] = (float(value), t)

    def add_beats(self, key: str, rr: list[tuple[float, int]], t: float | None = None) -> None:
        """beat-to-beat intervals (seconds, beats spanned) from strap `key` (kept when it's the heart-rate source),
        heard at `t`."""
        t = time.time() if t is None else t
        with self._lock:
            if self.sources["hr"][:1] == (key,):
                self._beats += [(t, x, n) for x, n in rr]

    def take_beats(self) -> list[tuple[float, float, int]]:
        """the intervals heard since the last call."""
        with self._lock:
            out, self._beats = self._beats, []
        return out

    def get(self, quantity: str, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            got = [self._values.get((quantity, k), (float("nan"), -1e18)) for k in self.sources.get(quantity, ())]
        return next((v for v, t in got if now - t <= STALE), float("nan"))

    def alive(self, key: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return now - self.seen.get(key, -1e18) <= STALE
