"""Workouts you write yourself: a short text format (both ways, so the drag-and-drop builder and the text stay in
sync) and your own library of them (data/my_workouts.json).

The text, one step per line (a "#" starts a comment):

    Warm up 10m 50%>70%           a ramp from 50 to 70 % of FTP
    3x                            repeat the indented lines (or, not indented, the lines up to a blank one)
      Threshold 8m 95-100%        a band
      Recover 4m 55%
    5x 30s 120%, 30s 50%          a repeat on one line
    Tempo 20m Z3 90rpm            a zone (Z1-Z7) and a cadence
    Steady 1h 200-210W            watts (turned into % of your FTP)
    Cool down 10m 50%

Durations: 90s, 10m, 1h, 1h30m, 1m30s or 1:30 (min:s). A step's name is the words left over; "warm" / "cool"
make it a warm-up / cool-down, and an easy step (below the work line) inside a repeat is a recovery.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .steps import WORK, Repeat, Step, flat, items_from_dicts

ZONES = {1: (.45, .55), 2: (.56, .75), 3: (.76, .90), 4: (.91, 1.05), 5: (1.06, 1.20), 6: (1.21, 1.50), 7: (1.51, 2.00)}
DURATION = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?:in)?)?(?:(\d+(?:\.\d+)?)s(?:ec)?)?$", re.I)
CLOCK = re.compile(r"^(\d+):([0-5]\d)$")
PERCENT = re.compile(r"^(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?%$")
WATTS = re.compile(r"^(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?w$", re.I)
RAMP = re.compile(r"^(\d+(?:\.\d+)?)%?(?:>|->|to)(\d+(?:\.\d+)?)%$", re.I)
ZONE = re.compile(r"^z([1-7])$", re.I)
RPM = re.compile(r"^(\d+)rpm$", re.I)
REPEAT = re.compile(r"^(\d+)\s*x\s*:?\s*(.*)$", re.I)
TYPES = ("recovery", "endurance", "long", "tempo", "sweetspot", "threshold", "vo2max", "anaerobic", "openers")


@dataclass
class ParseError:
    line: int
    message: str


@dataclass
class WorkoutText:
    """One workout's text <-> its steps (Steps and Repeats, power as a fraction of FTP)."""
    ftp: float = 0.0                       # for watts in the text
    errors: list = field(default_factory=list)

    # --- text -> steps ----------------------------------------------------------------------------
    def parse(self, text: str) -> list:
        """the steps; problems go to `errors` (that line is left out)."""
        self.errors = []
        lines = []
        for i, raw in enumerate(text.splitlines(), 1):
            body = raw.split("#", 1)[0].rstrip()
            lines.append((i, len(body) - len(body.lstrip()), body.strip().lstrip("-• ").strip()))
        items, _ = self.block(lines, 0, -1, False)
        return items

    def block(self, lines, at: int, parent_indent: int, until_blank: bool):
        """items from lines[at:] while they belong to the block: indented deeper than its header, or (a header
        whose lines aren't indented) up to a blank line. Returns (items, next index)."""
        items = []
        while at < len(lines):
            no, indent, body = lines[at]
            if not body:
                at += 1
                if until_blank:
                    break
                continue
            if indent <= parent_indent and not until_blank:
                break
            m = REPEAT.match(body)
            if m and not DURATION.match(body.split()[0]):
                reps, rest = int(m.group(1)), m.group(2).strip()
                if rest:                                   # 5x 30s 120%, 30s 50%
                    inner = [self.step(no, part.strip(), True) for part in rest.split(",") if part.strip()]
                    at += 1
                else:
                    nxt = next(((ind, b) for _, ind, b in lines[at + 1:] if b), None)
                    deeper = nxt is not None and nxt[0] > indent
                    inner, at = self.block(lines, at + 1, indent, not deeper)
                    inner = [self.as_recovery(x) for x in inner]
                inner = [x for x in inner if x is not None]
                if reps < 1 or not inner:
                    self.errors.append(ParseError(no, "a repeat needs a count of at least 1 and steps under it"))
                else:
                    items.append(Repeat(reps, inner))
                continue
            s = self.step(no, body, False)
            if s is not None:
                items.append(s)
            at += 1
        return items, at

    @classmethod
    def as_recovery(cls, x):
        """an easy step inside a repeat is the recovery between efforts (named so, unless you named it)."""
        if isinstance(x, Step) and x.kind == "active" and x.hi < WORK:
            if x.name == cls.default_name(x.lo, x.hi, "active"):
                x.name = "Recover"
            x.kind = "rest"
        return x

    def step(self, no: int, body: str, in_repeat: bool) -> Step | None:
        minutes, band, ramp_end, cadence, name = None, None, None, None, []
        toks = [t.strip(",") for t in body.split()]
        explicit = any(RAMP.match(t) or PERCENT.match(t) or WATTS.match(t) for t in toks)   # then "Z2" is a name
        for tok in toks:
            t = tok.strip(",")
            if minutes is None and (m := self.minutes(t)) is not None:
                minutes = m
            elif (r := RAMP.match(t)) and band is None:
                band = (float(r.group(1)) / 100,) * 2
                ramp_end = (float(r.group(2)) / 100,) * 2
            elif (p := PERCENT.match(t)) and band is None:
                lo = float(p.group(1)) / 100
                band = (lo, float(p.group(2)) / 100 if p.group(2) else lo)
            elif (w := WATTS.match(t)) and band is None:
                if not self.ftp:
                    self.errors.append(ParseError(no, "watts need your FTP"))
                    return None
                lo = float(w.group(1)) / self.ftp
                band = (lo, float(w.group(2)) / self.ftp if w.group(2) else lo)
            elif (z := ZONE.match(t)) and band is None and not explicit:
                band = ZONES[int(z.group(1))]
            elif (c := RPM.match(t)) and cadence is None:
                cadence = int(c.group(1))
            else:
                name.append(t)
        if minutes is None or minutes <= 0:
            self.errors.append(ParseError(no, f"no duration in \"{body}\" (e.g. 10m, 30s, 1h30m)"))
            return None
        if band is None:
            self.errors.append(ParseError(no, f"no power in \"{body}\" (e.g. 90%, 85-95%, 50%>70%, Z2, 200W)"))
            return None
        lo, hi = sorted(band)
        if hi > 3.0:
            self.errors.append(ParseError(no, "more than 300 % of FTP"))
            return None
        title = " ".join(name)
        low = title.lower()
        kind = ("warmup" if low.startswith("warm") else "cooldown" if low.startswith("cool")
                else "rest" if low.startswith(("rec", "rest", "easy")) or (in_repeat and hi < WORK) else "active")
        return Step(title or self.default_name(lo, hi, kind), minutes, lo, hi, kind,
                    lo_end=ramp_end[0] if ramp_end else None, hi_end=ramp_end[1] if ramp_end else None, cadence=cadence)

    @staticmethod
    def minutes(tok: str) -> float | None:
        if m := CLOCK.match(tok):
            return int(m.group(1)) + int(m.group(2)) / 60
        m = DURATION.match(tok)
        if not m or not any(m.groups()):
            return None
        h, mi, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 60 + mi + s / 60

    @staticmethod
    def default_name(lo: float, hi: float, kind: str) -> str:
        if kind != "active":
            return {"warmup": "Warm up", "cooldown": "Cool down", "rest": "Recover"}[kind]
        mid = (lo + hi) / 2
        return next(n for top, n in ((.55, "Easy"), (.75, "Endurance"), (.87, "Tempo"), (.95, "Sweet spot"),
                                     (1.05, "Threshold"), (1.2, "VO2max"), (9, "Hard")) if mid <= top)

    # --- steps -> text ----------------------------------------------------------------------------
    @classmethod
    def format(cls, items: list, indent: int = 0) -> str:
        out = []
        for x in items:                                    # a workout's few dozen items
            if isinstance(x, Repeat):
                out.append(" " * indent + f"{x.reps}x")
                out.append(cls.format(x.steps, indent + 2))
            else:
                out.append(" " * indent + cls.step_text(x))
        return "\n".join(out)

    @classmethod
    def step_text(cls, s: Step) -> str:
        pct = lambda v: f"{round(v * 100, 1):g}"           # noqa: E731
        if s.ramp:
            power = f"{pct(s.mid)}%>{pct(s.mid_end)}%"
        else:
            power = f"{pct(s.lo)}%" if abs(s.hi - s.lo) < 1e-9 else f"{pct(s.lo)}-{pct(s.hi)}%"
        parts = [s.name, cls.duration_text(s.minutes), power] + ([f"{s.cadence}rpm"] if s.cadence else [])
        return " ".join(p for p in parts if p)

    @staticmethod
    def duration_text(minutes: float) -> str:
        sec = int(round(minutes * 60))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return "".join(f"{v}{u}" for v, u in ((h, "h"), (m, "m"), (s, "s")) if v) or "0s"

    # --- what it is -------------------------------------------------------------------------------
    @staticmethod
    def guess_type(items: list) -> str:
        """the plan's session type that fits it best: by its hardest sustained work, else by length."""
        st = flat(items)
        total = sum(s.minutes for s in st)
        work = [s for s in st if s.mid >= WORK]
        for top, need, kind in ((1.25, 2, "anaerobic"), (1.06, 6, "vo2max"), (.96, 10, "threshold"),
                                (.86, 15, "sweetspot"), (WORK, 15, "tempo")):
            if sum(s.minutes for s in work if s.mid >= top) >= need:
                return kind
        steady = max((s.mid for s in st if s.kind == "active"), default=0)
        return "long" if total >= 150 else "recovery" if steady and steady < .56 else "endurance"


class MyWorkouts:
    """Your own workouts, kept in data/my_workouts.json (your input, not a cache): {id: {title, type, text,
    steps, minutes}}."""
    FILE = "my_workouts.json"

    def __init__(self, root):
        self.path = Path(root) / "data" / self.FILE

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save(self, title: str, kind: str, steps: list, wid: str | None = None) -> str:
        """add (or with `wid` replace) one; returns its id."""
        items = items_from_dicts(steps)
        if not items:
            raise ValueError("an empty workout")
        if kind not in TYPES:
            raise ValueError(f"unknown session type {kind!r}")
        all_ = self.load()
        wid = wid or self.new_id(title, all_)
        all_[wid] = {"title": title.strip() or "My workout", "type": kind, "text": WorkoutText.format(items),
                     "steps": [x.to_dict() for x in items], "minutes": round(sum(s.minutes for s in flat(items)), 1)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(all_, indent=1), encoding="utf-8")
        return wid

    def delete(self, wid: str) -> None:
        all_ = self.load()
        if all_.pop(wid, None) is not None:
            self.path.write_text(json.dumps(all_, indent=1), encoding="utf-8")

    @staticmethod
    def new_id(title: str, existing: dict) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "workout"
        wid, n = base, 2
        while wid in existing:
            wid, n = f"{base}-{n}", n + 1
        return wid
