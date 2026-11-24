"""All per-ride sample streams in one HDF5 file: data/processed/streams.h5.

Layout (one group per kind of stream, every channel is ONE long 1-D dataset with all
rides back to back - a "ragged" array):

    /ride/power, heart_rate, cadence, speed, altitude   float32   1 Hz samples
    /ride/moving                                        uint8
    /ride/ids     S64   ride id                 \
    /ride/offset  int64 first sample            |  index, one row per ride
    /ride/length  int64 number of samples       |
    /ride/start   f8    unix start time         |
    /ride/src     S40   size-mtime of the .npz it was packed from
    /alpha/t, a1, p, hr, art                            5-s bins (alphaHRV)
    /alpha/ids, offset, length, src
    /hrv/t, hrv            int64 unix s, float32 ms   overnight 5-min HRV readings, one entry per night
    /hrv/ids, offset, length, src                     (id = the night's wake-up date, YYYY-MM-DD)

One ride = a hyperslab  ds[offset:offset+length]; every ride = one read of each channel,
so numpy can work on the whole history at once (see alpha.features).

Writing: workers still drop one <id>.npz (<id>.a1.npz) per parsed ride into
processed/streams/ (crash-safe staging, parallel-friendly). `pack()` then appends them to
the HDF5 file. It writes a copy and swaps it in with os.replace, so a crash or the app
reading a ride page mid-update never sees a half-written file. Replaced and removed rides
leave dead samples behind; when they exceed 25 % the file is rewritten compactly.

Without h5py installed everything falls back to the .npz files (slower, same results).
`prune()` deletes .npz files that are safely packed.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np

from .stamps import in_seconds, stamp

H5_NAME = "streams.h5"
GROUPS = {
    "ride": {"suffix": ".npz", "channels": {"power": "f4", "heart_rate": "f4", "cadence": "f4", "speed": "f4",
                                           "altitude": "f4", "moving": "u1"}, "scalars": ("start",)},
    "alpha": {"suffix": ".a1.npz", "channels": {"t": "i4", "a1": "f4", "p": "f4", "hr": "f4", "art": "f4"},
              "scalars": ()},
    "hrv": {"suffix": ".hrv.npz", "channels": {"t": "i8", "hrv": "f4"}, "scalars": ()},
}
CHUNK = 1 << 14          # 16k samples: one ride page reads ~3 ms of chunks (64k: ~6 ms), same size
COMPRESSION = {"compression": "gzip", "compression_opts": 1, "shuffle": True}   # lzf: 25 % bigger, ~10 % faster
COMPACT_AT = 0.25


def available() -> bool:
    try:
        import h5py  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def h5_path(processed: Path) -> Path:
    return Path(processed) / H5_NAME


def stream_dir(processed: Path) -> Path:
    return Path(processed) / "streams"


# --- reading ---------------------------------------------------------------------------------
def _open(path: Path, tries: int = 20):
    import h5py
    for i in range(tries):
        try:
            return h5py.File(path, "r")
        except OSError:
            if not path.exists() or i == tries - 1:
                raise
            time.sleep(0.1)               # being swapped in by pack() right now


def _index(f, group: str) -> dict:
    """{id: (offset, length, row)} for one group."""
    if group not in f:
        return {}
    g = f[group]
    ids = [x.decode() for x in g["ids"][:]]
    return {i: (int(o), int(n), r) for r, (i, o, n) in enumerate(zip(ids, g["offset"][:], g["length"][:]))}


def _npz_ids(processed: Path, group: str) -> dict:
    """{id: path} for staged .npz files of a group (one directory listing)."""
    d = stream_dir(processed)
    suf = GROUPS[group]["suffix"]
    out = {}
    if not d.exists():
        return out
    with os.scandir(d) as it:
        for e in it:
            n = e.name
            if not n.endswith(suf) or n.startswith("_"):
                continue
            if any(len(o["suffix"]) > len(suf) and n.endswith(o["suffix"]) for o in GROUPS.values()):
                continue                  # x.a1.npz / x.hrv.npz belong to the group with the longer suffix
            out[n[: -len(suf)]] = e.path
    return out


def _from_npz(path, group):
    z = np.load(path)
    out = {c: z[c] for c in GROUPS[group]["channels"] if c in z}
    for s in GROUPS[group]["scalars"]:
        if s in z:
            out[s] = float(np.asarray(z[s]).ravel()[0])
    return out


def ids(processed: Path, group: str = "ride") -> set:
    """Ids with a stream (packed or staged)."""
    out = set(_npz_ids(processed, group))
    p = h5_path(processed)
    if available() and p.exists():
        with _open(p) as f:
            out |= set(_index(f, group))
    return out


def read(processed: Path, aid: str, group: str = "ride") -> dict | None:
    """One ride's channels (+ scalars), or None. A hyperslab read from the HDF5 file."""
    return read_many(processed, [aid], group).get(str(aid))


def read_many(processed: Path, aids, group: str = "ride", channels=None) -> dict:
    """{id: {channel: array, ...}} for the requested rides; one file open for all of them.
    Staged .npz files that are newer than the packed copy win (pack() hasn't run yet)."""
    aids = [str(a) for a in aids]
    want = list(channels or GROUPS[group]["channels"])
    out = {}
    p = h5_path(processed)
    staged = _npz_ids(processed, group)
    if available() and p.exists():
        with _open(p) as f:
            idx = _index(f, group)
            hit = [a for a in aids if a in idx and not _staged_newer(staged.get(a), f[group], idx[a][2])]
            if hit:
                g = f[group]
                whole = len(hit) > 40           # many rides: read each channel once, slice in memory
                data = {c: g[c][:] for c in want if c in g} if whole else None
                scal = {s: g[s][:] for s in GROUPS[group]["scalars"] if s in g}
                for a in hit:
                    o, n, r = idx[a]
                    d = {c: (data[c][o:o + n] if whole else g[c][o:o + n]) for c in want if c in g}
                    if "moving" in d:
                        d["moving"] = d["moving"].astype(bool)
                    for s, v in scal.items():
                        d[s] = float(v[r])
                    out[a] = d
    for a in aids:
        if a not in out and a in staged:
            try:
                d = _from_npz(staged[a], group)
                out[a] = {k: v for k, v in d.items() if k in want or k in GROUPS[group]["scalars"]}
            except Exception:  # noqa: BLE001 - a broken staging file is just skipped
                pass
    return out


def _staged_newer(npz_path, g, row) -> bool:
    if not npz_path:
        return False
    try:
        st = os.stat(npz_path)
    except OSError:
        return False
    return in_seconds(g["src"][row].decode()) != stamp(st)


def read_all(processed: Path, group: str = "ride", channels=None):
    """Every ride's channels concatenated: (ids, lengths, {channel: flat array}).
    When everything is packed this is one read per channel - no per-ride work at all."""
    want = list(channels or GROUPS[group]["channels"])
    p = h5_path(processed)
    if available() and p.exists():
        staged = _npz_ids(processed, group)
        with _open(p) as f:
            idx = _index(f, group)
            if idx and not any(a not in idx or _staged_newer(path, f[group], idx[a][2])
                               for a, path in staged.items()):
                g = f[group]
                keys = sorted(idx, key=lambda a: idx[a][0])
                o = np.array([idx[a][0] for a in keys], np.int64)
                n = np.array([idx[a][1] for a in keys], np.int64)
                data = {c: g[c][:] for c in want if c in g}
                if not _is_dense(o, n, data):              # skip samples of replaced rides
                    take = np.repeat(o - np.cumsum(n) + n, n) + np.arange(n.sum())
                    data = {c: v[take] for c, v in data.items()}
                if "moving" in data:
                    data["moving"] = data["moving"].astype(bool)
                return keys, n, data
    many = read_many(processed, sorted(ids(processed, group)), group, want)   # fallback: per ride
    keys = [k for k in many if all(c in many[k] for c in want)]
    n = np.array([many[k][want[0]].size for k in keys], np.int64)
    flat = {c: (np.concatenate([many[k][c] for k in keys]) if keys else np.zeros(0)) for c in want}
    return keys, n, flat


def _is_dense(o, n, data) -> bool:
    if not len(o):
        return True
    total = next(iter(data.values())).size if data else 0
    return o[0] == 0 and np.all(o[1:] == (o + n)[:-1]) and (o[-1] + n[-1]) == total


# --- writing ---------------------------------------------------------------------------------
def pack(processed: Path, progress=print, compact: bool | None = None) -> dict:
    """Append staged .npz streams that are new or changed to streams.h5 (all groups).
    Returns {group: number packed}. No-op without h5py."""
    if not available():
        return {}
    import h5py
    processed = Path(processed)
    p = h5_path(processed)
    todo = {}
    with (_open(p) if p.exists() else _nullfile()) as f:
        for group in GROUPS:
            staged = _npz_ids(processed, group)
            idx = _index(f, group) if f is not None else {}
            new = {}
            for a, path in staged.items():
                key = stamp(os.stat(path))
                if a in idx and in_seconds(f[group]["src"][idx[a][2]].decode()) == key:
                    continue
                new[a] = (path, key)
            if new:
                todo[group] = new
        dead = _dead_fraction(f) if f is not None else 0.0
    if not todo and not compact:
        return {}
    t0 = time.time()
    tmp = p.with_suffix(".h5.tmp")
    if p.exists() and not (compact or dead > COMPACT_AT):
        shutil.copyfile(p, tmp)
        rewrite = False
    else:
        rewrite = True
    counts = {}
    with h5py.File(tmp, "w" if rewrite else "a") as w:
        if rewrite and p.exists():
            with _open(p) as old:
                for group in GROUPS:
                    _copy_live(old, w, group)
        for group, new in todo.items():
            counts[group] = _append(w, group, new)
        w.attrs["format"] = "goodpush-streams-1"
    _swap(tmp, p)
    if counts:
        progress(f"streams.h5: packed {sum(counts.values())} new/changed streams "
                 f"({', '.join(f'{v} {k}' for k, v in counts.items())}) in {time.time() - t0:.1f}s")
    return counts


class _nullfile:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _dead_fraction(f) -> float:
    tot = live = 0
    for group in GROUPS:
        if group in f:
            g = f[group]
            first = next(iter(GROUPS[group]["channels"]))
            tot += g[first].shape[0]
            live += int(g["length"][:].sum())
    return 0.0 if not tot else 1 - live / tot


def _ensure(w, group):
    g = w.require_group(group)
    spec = GROUPS[group]
    for c, dt in spec["channels"].items():
        if c not in g:
            g.create_dataset(c, shape=(0,), maxshape=(None,), dtype=dt, chunks=(CHUNK,), **COMPRESSION)
    for name, dt in (("ids", "S64"), ("src", "S40"), ("offset", "i8"), ("length", "i8"),
                     *((s, "f8") for s in spec["scalars"])):
        if name not in g:
            g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt, chunks=(1024,))
    return g


def _append_rows(g, arrays: dict):
    """Append equal-length index columns."""
    n0 = g["ids"].shape[0]
    k = len(arrays["ids"])
    for name, vals in arrays.items():
        ds = g[name]
        ds.resize((n0 + k,))
        ds[n0:] = vals


def _write_index(g, cols: dict):
    for name, vals in cols.items():
        ds = g[name]
        ds.resize((len(vals),))
        if len(vals):
            ds[:] = vals


def _append(w, group: str, new: dict) -> int:
    """new: {id: (npz path, src key)}. Replaced ids are re-pointed to their new samples."""
    g = _ensure(w, group)
    spec = GROUPS[group]
    first = next(iter(spec["channels"]))
    loaded = []
    for a, (path, key) in sorted(new.items()):
        try:
            d = _from_npz(path, group)
        except Exception:  # noqa: BLE001
            continue
        if first not in d:
            continue
        loaded.append((a, key, d))
    if not loaded:
        return 0
    lens = np.array([d[first].size for _, _, d in loaded], np.int64)
    base = g[first].shape[0]
    offs = base + np.cumsum(lens) - lens
    for c, dt in spec["channels"].items():
        ds = g[c]
        cat = np.concatenate([np.asarray(d.get(c, np.full(d[first].size, np.nan)), dtype=dt)
                              for _, _, d in loaded])
        ds.resize((base + cat.size,))
        ds[base:] = cat
    # index: drop old rows for re-packed ids, add the new ones
    old_ids = [x.decode() for x in g["ids"][:]]
    replace = {a for a, _, _ in loaded}
    keep = np.array([i not in replace for i in old_ids], bool)
    cols = {name: g[name][:][keep] for name in ("ids", "src", "offset", "length", *spec["scalars"])}
    add = {"ids": np.array([a.encode() for a, _, _ in loaded], "S64"),
           "src": np.array([k.encode() for _, k, _ in loaded], "S40"),
           "offset": offs, "length": lens}
    for s in spec["scalars"]:
        add[s] = np.array([d.get(s, np.nan) for _, _, d in loaded], "f8")
    _write_index(g, {k: np.concatenate([cols[k], add[k]]) for k in cols})
    return len(loaded)


def _copy_live(old, w, group):
    """Rewrite a group compactly (only samples that the index still points to)."""
    if group not in old:
        return
    src = old[group]
    g = _ensure(w, group)
    spec = GROUPS[group]
    o = src["offset"][:]
    n = src["length"][:]
    take = np.repeat(o - np.cumsum(n) + n, n) + np.arange(n.sum())
    for c in spec["channels"]:
        if c in src:
            v = src[c][:][take]
            g[c].resize((v.size,))
            g[c][:] = v
    cols = {name: src[name][:] for name in ("ids", "src", *spec["scalars"]) if name in src}
    cols["length"] = n
    cols["offset"] = np.cumsum(n) - n
    _write_index(g, cols)


def remove(processed: Path, aids, progress=print) -> int:
    """Delete activities' streams: their staged .npz files and their rows in streams.h5's index. Their samples
    stay behind as dead space, like a replaced ride's (HDF5 doesn't give that space back), until there's
    enough of it (COMPACT_AT) to be worth rewriting the file. Returns how many of `aids` had a stream."""
    processed, drop = Path(processed), set(aids)
    gone = set()
    for group in GROUPS:
        for a, path in _npz_ids(processed, group).items():
            if a in drop:
                os.remove(path)
                gone.add(a)
    p = h5_path(processed)
    if not (available() and p.exists()):
        return len(gone)
    import h5py
    want = np.array([a.encode() for a in drop], "S64")
    with _open(p) as f:
        packed = {x.decode() for g in GROUPS if g in f for x in np.intersect1d(f[g]["ids"][:], want)}
    if packed:
        tmp = p.with_suffix(".h5.tmp")
        shutil.copyfile(p, tmp)
        with h5py.File(tmp, "a") as w:
            for group in GROUPS:
                if group in w:
                    _unindex(w[group], group, want)
            dead = _dead_fraction(w)
        _swap(tmp, p)
        progress(f"streams.h5: removed {len(packed)} activities' streams")
        if dead > COMPACT_AT:
            pack(processed, progress, compact=True)
    return len(gone | packed)


def _unindex(g, group: str, ids: np.ndarray) -> None:
    """Drop rows from a group's index; their samples become dead space."""
    live = ~np.isin(g["ids"][:], ids)
    _write_index(g, {name: g[name][:][live] for name in ("ids", "src", "offset", "length", *GROUPS[group]["scalars"])})


def _swap(tmp: Path, dest: Path, tries: int = 50):
    """os.replace with retries: on Windows it fails while another handle has the file open."""
    for i in range(tries):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.1)


def prune(processed: Path, progress=print) -> int:
    """Delete staged .npz files whose exact contents are already packed in streams.h5."""
    if not available() or not h5_path(processed).exists():
        return 0
    n = 0
    with _open(h5_path(processed)) as f:
        for group in GROUPS:
            if group not in f:
                continue
            idx = _index(f, group)
            src = f[group]["src"][:]
            for a, path in _npz_ids(processed, group).items():
                if a not in idx:
                    continue
                if in_seconds(src[idx[a][2]].decode()) == stamp(os.stat(path)):
                    try:
                        os.remove(path)
                        n += 1
                    except OSError:
                        pass
    progress(f"removed {n} .npz files that are packed in {H5_NAME}")
    return n
