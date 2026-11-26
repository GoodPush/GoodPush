"""In-place editing of goodpush.toml (tomlkit): set values in one [section], replace a whole [section] or the
[[name]] tables, keeping comments and everything else untouched."""
from __future__ import annotations

import numbers
from pathlib import Path

import tomlkit
from tomlkit.items import AoT, KeyType, SingleKey, Table


class TomlEdit:
    """A TOML file opened for editing; `save()` writes it back as UTF-8."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.doc = tomlkit.parse(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.write_text(tomlkit.dumps(self.doc), encoding="utf-8")

    def section(self, name: str) -> Table:
        """[name] (dotted: [ftp.history]), made at the end if missing."""
        node = self.doc
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part not in node:
                node.add(part, tomlkit.table(is_super_table=i < len(parts) - 1))
            node = node[part]
        return node

    @classmethod
    def value(cls, v):
        """`v` as TOML: floats to 3 decimals, dicts as inline tables, numpy numbers as plain ones, others as text."""
        if isinstance(v, (bool, str)):
            return v
        if isinstance(v, numbers.Integral):
            return int(v)
        if isinstance(v, numbers.Real):
            return round(float(v), 3)
        if isinstance(v, dict):
            t = tomlkit.inline_table()
            t.update({str(k): cls.value(x) for k, x in v.items()})
            return t
        if isinstance(v, (list, tuple)):
            a = tomlkit.array()
            a.extend(cls.value(x) for x in v)
            return a
        return str(v)

    @staticmethod
    def key(k: str) -> str | SingleKey:
        """keys like dates ("2026-10-29" in [ftp.history]) stay quoted, as goodpush has always written them"""
        return k if k[:1].isalpha() or k[:1] == "_" else SingleKey(k, KeyType.Basic)

    def set_values(self, section: str, values: dict) -> None:
        t = self.section(section)
        for k, v in values.items():
            t[self.key(k)] = self.value(v)

    def set_table(self, section: str, values: dict) -> None:
        t = self.section(section)
        for k in [k for k, v in t.items() if k not in values and not isinstance(v, (Table, AoT))]:
            del t[k]                              # its sub-tables ([ftp.history] under [ftp]) stay
        self.set_values(section, values)

    def set_tables(self, name: str, rows: list[dict]) -> None:
        if not rows:
            self.doc.pop(name, None)
            return
        aot = tomlkit.aot()
        for r in rows:
            t = tomlkit.table()
            for k, v in r.items():
                t[k] = self.value(v)
            aot.append(t)
        self.doc[name] = aot


def set_values(path: Path, section: str, values: dict) -> None:
    """set `values` in [section] (made if missing); the other keys and comments stay."""
    f = TomlEdit(path)
    f.set_values(section, values)
    f.save()


def set_table(path: Path, section: str, values: dict) -> None:
    """[section] holds exactly `values`: keys not in it are removed (e.g. a date taken out of [ftp.history])."""
    f = TomlEdit(path)
    f.set_table(section, values)
    f.save()


def set_tables(path: Path, name: str, rows: list[dict]) -> None:
    """All the [[name]] tables become `rows` (in place of the old ones; at the end if there were none)."""
    f = TomlEdit(path)
    f.set_tables(name, rows)
    f.save()
