"""Read activity files (FIT / TCX / GPX / Strava JSON, optionally .gz) into a
common column format: dict of equal-length lists with keys
time (unix seconds), power, heart_rate, cadence, speed, altitude, distance.
"""
from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from . import fit

COLS = ("time", "power", "heart_rate", "cadence", "speed", "altitude", "distance")


def _empty():
    return {k: [] for k in COLS}


def _open_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def _parse_time(s: str) -> float:
    s = s.strip().replace("Z", "+00:00")
    # trim fractional seconds beyond microseconds
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return datetime.fromisoformat(s).timestamp()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_tcx(data: bytes) -> dict:
    cols = _empty()
    root = ET.fromstring(data.strip())
    sport = None
    for el in root.iter():
        if _local(el.tag) == "Activity":
            sport = el.attrib.get("Sport")
            break
    for tp in root.iter():
        if _local(tp.tag) != "Trackpoint":
            continue
        row = dict.fromkeys(COLS)
        for ch in tp.iter():
            name = _local(ch.tag)
            if name == "Time":
                row["time"] = _parse_time(ch.text)
            elif name == "AltitudeMeters":
                row["altitude"] = _num(ch.text)
            elif name == "DistanceMeters":
                row["distance"] = _num(ch.text)
            elif name == "HeartRateBpm":
                for v in ch:
                    if _local(v.tag) == "Value":
                        row["heart_rate"] = _num(v.text)
            elif name == "Cadence":
                row["cadence"] = _num(ch.text)
            elif name == "Watts":
                row["power"] = _num(ch.text)
            elif name == "Speed":
                row["speed"] = _num(ch.text)
        if row["time"] is None:
            continue
        for k in COLS:
            cols[k].append(row[k])
    return {"records": cols, "sport": sport, "sub_sport": None}


def read_gpx(data: bytes) -> dict:
    cols = _empty()
    root = ET.fromstring(data.strip())
    sport = None
    for el in root.iter():
        if _local(el.tag) == "type":
            sport = (el.text or "").strip()
            break
    for tp in root.iter():
        if _local(tp.tag) != "trkpt":
            continue
        row = dict.fromkeys(COLS)
        for ch in tp.iter():
            name = _local(ch.tag).lower()
            if name == "time":
                row["time"] = _parse_time(ch.text)
            elif name == "ele":
                row["altitude"] = _num(ch.text)
            elif name in ("hr", "heartrate"):
                row["heart_rate"] = _num(ch.text)
            elif name in ("cad", "cadence"):
                row["cadence"] = _num(ch.text)
            elif name in ("power", "powerinwatts", "watts"):
                row["power"] = _num(ch.text)
            elif name == "speed":
                row["speed"] = _num(ch.text)
        if row["time"] is None:
            continue
        for k in COLS:
            cols[k].append(row[k])
    return {"records": cols, "sport": sport, "sub_sport": None}


_JSON_MAP = {"time": "time", "heartrate": "heart_rate", "heart_rate": "heart_rate",
             "watts": "power", "power": "power", "cadence": "cadence",
             "elevation": "altitude", "altitude": "altitude", "speed": "speed",
             "distance": "distance"}


def read_strava_json(data: bytes) -> dict:
    cols = _empty()
    obj = json.loads(data)
    meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    for block in obj.get("data", []) if isinstance(obj, dict) else []:
        fields = block.get("fields", [])
        idx = {f: i for i, f in enumerate(fields)}
        for vals in block.get("values", []):
            row = dict.fromkeys(COLS)
            for f, i in idx.items():
                k = _JSON_MAP.get(f)
                if k and i < len(vals):
                    v = vals[i]
                    row[k] = v if isinstance(v, (int, float)) else None
            if row["time"] is None:
                continue
            if row["speed"] is not None and row["speed"] < 0:
                row["speed"] = None
            for k in COLS:
                cols[k].append(row[k])
    return {"records": cols, "sport": meta.get("activity_type"), "sub_sport": None}


def read_any(path: str | Path) -> dict:
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".fit") or name.endswith(".fit.gz"):
        res = fit.read_activity(path)
        res["records"].pop("temperature", None)
        return res
    data = _open_bytes(path)
    if name.endswith((".tcx", ".tcx.gz")):
        return read_tcx(data)
    if name.endswith((".gpx", ".gpx.gz")):
        return read_gpx(data)
    if name.endswith((".json", ".json.gz")):
        return read_strava_json(data)
    raise ValueError(f"unsupported activity file: {path}")


def read_track(path: str | Path, max_points: int = 1500) -> list[list[float]]:
    """GPS track [[lat, lon], ...] (thinned to max_points) for a FIT / GPX / TCX / Strava JSON
    file; [] for indoor rides or files without positions."""
    from .fit import MESG_RECORD, iter_messages
    path = Path(path)
    data = _open_bytes(path)
    name = path.name.lower().replace(".gz", "")
    pts = []
    if name.endswith(".fit"):
        k = 180.0 / 2 ** 31
        for _, m in iter_messages(data, (MESG_RECORD,)):
            la, lo = m.get(0), m.get(1)
            if isinstance(la, int) and isinstance(lo, int):
                pts.append((la * k, lo * k))
    elif name.endswith(".gpx"):
        root = ET.fromstring(data.strip())
        for tp in root.iter():
            if _local(tp.tag) == "trkpt" and "lat" in tp.attrib:
                pts.append((float(tp.attrib["lat"]), float(tp.attrib["lon"])))
    elif name.endswith(".tcx"):
        root = ET.fromstring(data.strip())
        for el in root.iter():
            if _local(el.tag) == "Position":
                v = {_local(c.tag): _num(c.text) for c in el}
                if v.get("LatitudeDegrees") is not None and v.get("LongitudeDegrees") is not None:
                    pts.append((v["LatitudeDegrees"], v["LongitudeDegrees"]))
    elif name.endswith(".json"):
        import json
        js = json.loads(data)
        for blk in js.get("data", []):
            f = blk.get("fields", [])
            if "latlng" in f:
                i = f.index("latlng")
                pts += [tuple(r[i]) for r in blk.get("values", []) if r[i]]
    pts = [p for p in pts if p[0] is not None and abs(p[0]) <= 90 and abs(p[1]) <= 180 and (p[0], p[1]) != (0, 0)]
    step = max(1, -(-len(pts) // max_points))
    return [[round(a, 5), round(b, 5)] for a, b in pts[::step]]
