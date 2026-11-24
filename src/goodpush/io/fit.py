"""Minimal, dependency-free reader for Garmin FIT files.

Implements the FIT binary protocol (headers, definition/data messages,
compressed-timestamp headers, developer fields, chained files). It returns raw
field values keyed by field-definition number; scaling is applied by the
higher-level helpers below for the handful of messages we care about.

Reference: Garmin FIT SDK "FIT Protocol" document.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

FIT_EPOCH_OFFSET = 631065600  # seconds between 1970-01-01 and 1989-12-31

# base type number -> (struct char, size, invalid value)
_BASE_TYPES = {
    0x00: ("B", 1, 0xFF),  # enum
    0x01: ("b", 1, 0x7F),  # sint8
    0x02: ("B", 1, 0xFF),  # uint8
    0x03: ("h", 2, 0x7FFF),  # sint16
    0x04: ("H", 2, 0xFFFF),  # uint16
    0x05: ("i", 4, 0x7FFFFFFF),  # sint32
    0x06: ("I", 4, 0xFFFFFFFF),  # uint32
    0x07: ("s", 1, None),  # string
    0x08: ("f", 4, None),  # float32
    0x09: ("d", 8, None),  # float64
    0x0A: ("B", 1, 0x00),  # uint8z
    0x0B: ("H", 2, 0x0000),  # uint16z
    0x0C: ("I", 4, 0x00000000),  # uint32z
    0x0D: ("B", 1, 0xFF),  # byte
    0x0E: ("q", 8, 0x7FFFFFFFFFFFFFFF),  # sint64
    0x0F: ("Q", 8, 0xFFFFFFFFFFFFFFFF),  # uint64
    0x10: ("Q", 8, 0x0000000000000000),  # uint64z
}

# Global message numbers we use
MESG_FILE_ID = 0
MESG_SESSION = 18
MESG_LAP = 19
MESG_RECORD = 20
MESG_EVENT = 21
MESG_SPORT = 12


class FitError(Exception):
    pass


class _Def:
    __slots__ = ("gmn", "struct", "slices", "total")

    def __init__(self, gmn, struct_obj, slices, total):
        self.gmn = gmn
        self.struct = struct_obj
        self.slices = slices  # list of (field_num, start, count, kind, invalid)
        self.total = total


def _read_bytes(src) -> bytes:
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        p = Path(src)
        with open(p, "rb") as fh:
            data = fh.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def _build_def(arch_big: bool, gmn: int, fields, dev_fields, dev_types=None) -> _Def:
    fmt = [">" if arch_big else "<"]
    slices = []
    idx = 0
    for fnum, size, btype in fields:
        bnum = btype & 0x1F
        char, bsize, invalid = _BASE_TYPES.get(bnum, ("s", 1, None))
        if char == "s" or size % bsize != 0:
            fmt.append(f"{size}s")
            slices.append((fnum, idx, 1, "str" if char == "s" else "raw", None))
            idx += 1
        else:
            n = size // bsize
            fmt.append(f"{n}{char}")
            slices.append((fnum, idx, n, "num", invalid))
            idx += n
    for fnum, size, ddi in dev_fields:
        bt = (dev_types or {}).get((ddi, fnum))
        char, bsize, invalid = _BASE_TYPES.get((bt or 0) & 0x1F, ("s", 1, None))
        if bt is None or char == "s" or size % bsize != 0:
            fmt.append(f"{size}x")  # unknown / text developer field: skip
            continue
        n = size // bsize
        fmt.append(f"{n}{char}")
        slices.append((f"dev_{ddi}_{fnum}", idx, n, "num", invalid))
        idx += n
    st = struct.Struct("".join(fmt))
    return _Def(gmn, st, slices, st.size)


def iter_messages(src, wanted: Iterable[int] | None = None):
    """Yield (global_msg_num, {field_num: value}) for every data message.

    `wanted` restricts decoding to those global message numbers (others are
    skipped cheaply). Timestamps (field 253) are raw FIT seconds; compressed
    timestamps are expanded into field 253.
    """
    data = _read_bytes(src)
    wanted = set(wanted) if wanted is not None else None
    pos = 0
    n = len(data)
    while pos + 12 <= n:
        hsize = data[pos]
        if hsize not in (12, 14) or data[pos + 8 : pos + 12] != b".FIT":
            if pos == 0:
                raise FitError("not a FIT file")
            break
        data_size = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + hsize
        end = min(start + data_size, n)
        defs: dict[int, _Def] = {}
        dev_types: dict = {}     # (developer index, field number) -> base type, from field descriptions
        last_ts = None
        p = start
        while p < end:
            h = data[p]
            p += 1
            if h & 0x80:  # compressed timestamp header
                local = (h >> 5) & 0x03
                offset = h & 0x1F
                d = defs.get(local)
                if d is None:
                    raise FitError("data before definition")
                if last_ts is not None:
                    ts = (last_ts & ~0x1F) + offset
                    if offset < (last_ts & 0x1F):
                        ts += 0x20
                    last_ts = ts
                else:
                    ts = None
                if wanted is None or d.gmn in wanted:
                    msg = _decode(d, data, p)
                    if ts is not None:
                        msg[253] = ts
                    yield d.gmn, msg
                p += d.total
            elif h & 0x40:  # definition message
                has_dev = bool(h & 0x20)
                local = h & 0x0F
                arch_big = data[p + 1] == 1
                gmn = struct.unpack_from(">H" if arch_big else "<H", data, p + 2)[0]
                nf = data[p + 4]
                p += 5
                fields = []
                for _ in range(nf):
                    fields.append((data[p], data[p + 1], data[p + 2]))
                    p += 3
                dev = []
                if has_dev:
                    nd = data[p]
                    p += 1
                    for _ in range(nd):
                        dev.append((data[p], data[p + 1], data[p + 2]))
                        p += 3
                defs[local] = _build_def(arch_big, gmn, fields, dev, dev_types)
            else:  # normal data message
                local = h & 0x0F
                d = defs.get(local)
                if d is None:
                    raise FitError("data before definition")
                if p + d.total > n:
                    break
                need = wanted is None or d.gmn in wanted
                ts_needed = True
                if need or ts_needed:
                    msg = _decode(d, data, p)
                    if d.gmn == 206 and msg.get(0) is not None and msg.get(1) is not None:
                        dev_types[(msg[0], msg[1])] = msg.get(2)
                    t = msg.get(253)
                    if t is not None:
                        last_ts = t
                    if need:
                        yield d.gmn, msg
                p += d.total
        pos = end + 2  # skip file CRC; continue with chained FIT file if any


def _decode(d: _Def, data: bytes, p: int) -> dict:
    vals = d.struct.unpack_from(data, p)
    out = {}
    for fnum, i, cnt, kind, invalid in d.slices:
        if kind == "num":
            if cnt == 1:
                v = vals[i]
                if v == invalid or (isinstance(v, float) and v != v):
                    v = None
            else:
                v = [x for x in vals[i : i + cnt]]
                if invalid is not None and all(x == invalid for x in v):
                    v = None
        elif kind == "str":
            v = vals[i].split(b"\x00", 1)[0].decode("utf-8", "replace")
        else:
            v = vals[i]
        out[fnum] = v
    return out


def read_messages(src, wanted: Iterable[int] | None = None) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for gmn, msg in iter_messages(src, wanted):
        out.setdefault(gmn, []).append(msg)
    return out


# ---------------------------------------------------------------------------
# Activity helpers
# ---------------------------------------------------------------------------
MESG_HRV = 78


def read_rr(src) -> np.ndarray:
    """An activity's beat-to-beat intervals (s) from its hrv messages, in order. Invalid entries are padding (each
    message has room for 5), not gaps: missed beats show as a long interval. Empty without any."""
    v = np.array([x for m in read_messages(src, [MESG_HRV]).get(MESG_HRV, []) for x in (m.get(0) or ())], float)
    return v[v != 0xFFFF] / 1000

SPORTS = {0: "generic", 1: "running", 2: "cycling", 5: "swimming", 10: "training",
          11: "walking", 17: "hiking", 4: "fitness_equipment"}
SUB_SPORTS = {58: "virtual_activity", 6: "indoor_cycling", 7: "road", 8: "mountain",
              46: "gravel_cycling", 28: "e_bike_fitness", 2: "trail"}


def read_activity(src) -> dict:
    """Return {'records': dict of column lists, 'sport': str, 'sub_sport': str}.

    Record columns: time (unix s), power, heart_rate, cadence, speed (m/s),
    altitude (m), distance (m), temperature (C).
    """
    cols = {k: [] for k in ("time", "power", "heart_rate", "cadence", "speed",
                            "altitude", "distance", "temperature")}
    sport = sub = None
    for gmn, m in iter_messages(src, (MESG_RECORD, MESG_SESSION, MESG_SPORT)):
        if gmn == MESG_RECORD:
            ts = m.get(253)
            if ts is None:
                continue
            cols["time"].append(ts + FIT_EPOCH_OFFSET)
            cols["power"].append(_first(m.get(7)))
            cols["heart_rate"].append(_first(m.get(3)))
            cols["cadence"].append(_first(m.get(4)))
            sp = m.get(73)
            if sp is None:
                sp = m.get(6)
            cols["speed"].append(sp / 1000.0 if isinstance(sp, (int, float)) else None)
            alt = m.get(78)
            if alt is None:
                alt = m.get(2)
            cols["altitude"].append(alt / 5.0 - 500.0 if isinstance(alt, (int, float)) else None)
            dist = m.get(5)
            cols["distance"].append(dist / 100.0 if isinstance(dist, (int, float)) else None)
            cols["temperature"].append(_first(m.get(13)))
        else:
            s = m.get(5 if gmn == MESG_SESSION else 0)
            ss = m.get(6 if gmn == MESG_SESSION else 1)
            if s is not None and sport is None:
                sport = SPORTS.get(s, str(s))
            if ss is not None and sub is None:
                sub = SUB_SPORTS.get(ss, str(ss))
    return {"records": cols, "sport": sport, "sub_sport": sub}


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def read_developer_fields(src) -> dict:
    """Connect IQ data fields recorded in an activity (e.g. alphaHRV's 'Alpha1').

    Returns {'fields': {name: {'units': str, 'scale': float}}, 'time': [unix s],
             'values': {name: [value per record, None when missing]}}.
    """
    data = _read_bytes(src)
    names, info = {}, {}
    rows = []
    for g, m in iter_messages(data, (206, MESG_RECORD)):
        if g == 206:
            nm = m.get(3)
            if isinstance(nm, str) and nm:
                key = f"dev_{m.get(0)}_{m.get(1)}"
                names[key] = nm
                info[nm] = {"units": m.get(8) if isinstance(m.get(8), str) else "",
                            "scale": float(m.get(6)) if isinstance(m.get(6), (int, float)) and m.get(6) else 1.0}
        elif m.get(253) is not None:
            rows.append(m)
    vals = {nm: [] for nm in info}
    t = []
    for m in rows:
        t.append(m[253] + FIT_EPOCH_OFFSET)
        for key, nm in names.items():
            v = _first(m.get(key))
            vals[nm].append(None if v is None else v / info[nm]["scale"])
    return {"fields": info, "time": t, "values": vals}
