"""A file's stamp: its size and modification time, to tell whether a cached result still describes the file.

Whole seconds: peer sync (sync/peer.py) carries a file's mtime to the other computer in whole seconds, so a finer
stamp makes every synced file look changed there and be read again. Stamps were in nanoseconds before; `in_seconds`
writes those in this form, so caches made before (or on the other computer) still count.
"""
from __future__ import annotations

import re

NS = re.compile(r"^(\d+-\d{9,10})\d{9}(?=-|$)")     # size-<mtime ns>[-...]: its last 9 digits are the fraction


def stamp(st) -> str:
    """"size-mtime" of an os.stat / os.DirEntry.stat result, mtime in whole seconds."""
    return f"{st.st_size}-{st.st_mtime_ns // 1_000_000_000}"


def in_seconds(key):
    """`key` (a str, or a pandas Series of them) with a nanosecond stamp written in whole seconds; others unchanged."""
    if isinstance(key, str):
        return NS.sub(r"\1", key)
    return key.astype(str).str.replace(NS.pattern, r"\1", regex=True)
