"""Minimal in-place editing of goodpush.toml: set scalar / list / inline-table values in one [section],
replace a whole [section] or the [[name]] tables, keeping comments and everything else untouched."""
from __future__ import annotations

import json
import re
from pathlib import Path


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(round(v, 3)) if isinstance(v, float) else str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_fmt(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v))


def _key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) else json.dumps(k)


def set_table(path: Path, section: str, values: dict) -> None:
    """[section] holds exactly `values`: keys not in it are removed (e.g. a date taken out of [ftp.history])."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == f"[{section}]"), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        keep = [l for l in lines[start + 1:end]
                if not (m := re.match(r'^\s*("?[^"=\s]+"?)\s*=', l)) or m.group(1).strip('"') in values]
        lines[start + 1:end] = keep
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    set_values(path, section, values)


def set_tables(path: Path, name: str, rows: list[dict]) -> None:
    """All the [[name]] tables become `rows` (in place of the first one; at the end if there were none)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    out, at, skipping = [], None, False
    for line in lines:
        head = line.strip()
        if head.startswith("["):
            skipping = head == f"[[{name}]]"
            if skipping and at is None:
                at = len(out)
        if not skipping or not head or head.startswith("#"):    # comments and gaps stay where they are
            out.append(line)
    block = [x for r in rows for x in ([f"[[{name}]]"] + [f"{_key(k)} = {_fmt(v)}" for k, v in r.items()] + [""])]
    if at is None:
        at = len(out)
        block = [""] + block
    out[at:at] = block
    path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")


def set_values(path: Path, section: str, values: dict) -> None:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    head = f"[{section}]"
    start = next((i for i, l in enumerate(lines) if l.strip() == head), None)
    if start is None:
        lines += ["", head]
        start = len(lines) - 1
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
               len(lines))
    todo = dict(values)
    for i in range(start + 1, end):
        m = re.match(r'^(\s*)("?[A-Za-z0-9_-]+"?)(\s*=\s*)(.*?)(\s+#.*)?$', lines[i])
        if m and m.group(2).strip('"') in todo:
            v = todo.pop(m.group(2).strip('"'))
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{_fmt(v)}{m.group(5) or ''}"
    # keys like dates ("2026-10-29" in [ftp.history]) are written quoted
    insert = [f"{_key(k)} = {_fmt(v)}" for k, v in todo.items()]
    # insert before trailing blank lines of the section
    j = end
    while j - 1 > start and not lines[j - 1].strip():
        j -= 1
    lines[j:j] = insert
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
