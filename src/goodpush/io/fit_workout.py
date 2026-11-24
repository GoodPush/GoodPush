"""Write structured workouts as FIT workout files (fenix / Edge / Forerunner).

Copy the .fit files to the watch's GARMIN/NewFiles folder over USB (they show up
under Training > Workouts), or import them in Garmin Connect (web: Training >
Workouts > Import) and send them to the device.

Targets are absolute power ranges (FIT convention: watts + 1000).
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

FIT_EPOCH_OFFSET = 631065600

_CRC_TABLE = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
              0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]


def fit_crc(data: bytes, crc: int = 0) -> int:
    for byte in data:
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[byte & 0xF]
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[(byte >> 4) & 0xF]
    return crc


# (field number, base type, struct char, size)
_T = {"enum": (0x00, "B", 1), "uint8": (0x02, "B", 1), "uint16": (0x84, "H", 2),
      "uint32": (0x86, "I", 4), "uint32z": (0x8C, "I", 4), "string": (0x07, "s", None)}

INTENSITY = {"active": 0, "rest": 1, "warmup": 2, "cooldown": 3}


class _Writer:
    def __init__(self):
        self.buf = bytearray()
        self.local = 0

    def message(self, gmn: int, fields: list[tuple[int, str, object, int | None]]):
        """fields: (field_num, type, value, string_size). Writes definition + data."""
        lt = self.local % 16
        self.local += 1
        defs, fmt, vals = [], "<", []
        for num, typ, val, size in fields:
            base, ch, sz = _T[typ]
            if typ == "string":
                raw = str(val).encode("utf-8")[: size - 1]
                raw = raw + b"\x00" * (size - len(raw))
                defs.append((num, size, base))
                fmt += f"{size}s"
                vals.append(raw)
            else:
                defs.append((num, sz, base))
                fmt += ch
                vals.append(int(val))
        self.buf += struct.pack("<BBBHB", 0x40 | lt, 0, 0, gmn, len(defs))
        for num, sz, base in defs:
            self.buf += struct.pack("<BBB", num, sz, base)
        self.buf += struct.pack("<B", lt) + struct.pack(fmt, *vals)

    def messages(self, gmn: int, fields: list[tuple[int, str]], columns: list) -> None:
        """Many messages of one kind: one definition, then all rows at once. fields: (field_num, type), the type
        a (type, n) pair for an array field; columns: one integer array per field ((rows, n) for an array),
        invalid values already filled in."""
        lt = self.local % 16
        self.local += 1
        self.buf += struct.pack("<BBBHB", 0x40 | lt, 0, 0, gmn, len(fields))
        dtype = [("header", "u1")]
        for i, (num, typ) in enumerate(fields):
            typ, n = typ if isinstance(typ, tuple) else (typ, 1)
            base, ch, sz = _T[typ]
            self.buf += struct.pack("<BBB", num, sz * n, base)
            dtype.append((f"f{i}", "<" + {"B": "u1", "H": "u2", "I": "u4"}[ch], (n,) if n > 1 else ()))
        rows = np.zeros(len(columns[0]), dtype=dtype)
        rows["header"] = lt
        for i, c in enumerate(columns):
            rows[f"f{i}"] = c
        self.buf += rows.tobytes()

    def bytes(self) -> bytes:
        hdr = struct.pack("<BBHI4s", 14, 0x10, 2132, len(self.buf), b".FIT")
        hdr += struct.pack("<H", fit_crc(hdr))
        body = hdr + bytes(self.buf)
        return body + struct.pack("<H", fit_crc(body))


def workout_fit(name: str, items: list[dict], created: float | None = None) -> bytes:
    """items: steps {name, min, lo, hi (W), kind (warmup|active|rest|cooldown), cadence?, cue?} and repeats
    {reps, steps}, nested as deep as you like (FIT repeat steps nest). A cadence is a secondary target
    (± 5 rpm), a cue the step's notes."""
    w = _Writer()
    t = int((created or time.time()) - FIT_EPOCH_OFFSET)
    w.message(0, [(0, "enum", 5, None), (1, "uint16", 255, None), (2, "uint16", 0, None),
                  (3, "uint32z", (t & 0xFFFFFFF) or 1, None), (4, "uint32", t, None)])
    rows = fit_rows(items, [])
    w.message(26, [(4, "enum", 2, None), (6, "uint16", len(rows), None),
                   (8, "string", name, 32)])
    for idx, st in enumerate(rows):                  # a workout's few dozen steps
        if "repeat_from" in st:
            w.message(27, [(254, "uint16", idx, None), (1, "enum", 6, None),
                           (2, "uint32", st["repeat_from"], None),
                           (4, "uint32", st["count"], None)])
            continue
        lo, hi = int(round(st["lo"])), int(round(st["hi"]))
        fields = [(254, "uint16", idx, None), (0, "string", st["name"][:15], 16),
                  (1, "enum", 0, None), (2, "uint32", int(st["min"] * 60000), None),
                  (3, "enum", 4, None), (4, "uint32", 0, None),
                  (5, "uint32", lo + 1000, None), (6, "uint32", hi + 1000, None),
                  (7, "enum", INTENSITY.get(st.get("kind", "active"), 0), None)]
        if st.get("cue"):
            fields.append((8, "string", st["cue"][:50], 51))
        if st.get("cadence"):
            c = int(st["cadence"])
            fields += [(19, "enum", 3, None), (20, "uint32", 0, None),
                       (21, "uint32", c - 5, None), (22, "uint32", c + 5, None)]
        w.message(27, fields)
    return w.bytes()


def fit_rows(items: list[dict], out: list) -> list:
    """the nested steps in FIT's order: each repeat's steps, then a repeat step pointing back to its first."""
    for x in items:
        if "reps" in x:
            first = len(out)
            fit_rows(x["steps"], out)
            if x["reps"] > 1:
                out.append({"repeat_from": first, "count": x["reps"]})
        else:
            out.append(x)
    return out


def write_workout(path: Path, name: str, items: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(workout_fit(name, items))
    return path
