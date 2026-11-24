"""intervals.icu workout text for plan sessions: paste it into a workout's description (or upload the .txt) and
intervals.icu sends the session on to Garmin, Wahoo or Zwift.

One step per line: `- <text> <duration> <target> [<cadence>rpm]`, targets in % of FTP (`55%`, a range `88-94%`,
a ramp `ramp 50-75%`) or % of max heart rate (`80% HR`). A line `<n>x` opens a repeat that runs to the next blank
line. intervals.icu repeats don't nest, so a repeat inside a repeat is written out inside its set.
"""
from __future__ import annotations

import re

from ..planning.steps import Repeat, flat


class IcuText:
    """A session's Steps and Repeats as intervals.icu workout text."""
    NUMBERS = re.compile(r"\S*\d\S*|%|\brpm\b|\bW\b", re.IGNORECASE)   # the parser reads these as targets

    @classmethod
    def workout(cls, items: list, description: str = "") -> str:
        out = [description.strip(), ""] if description.strip() else []
        for x in items:                      # a workout's few dozen items
            if isinstance(x, Repeat):
                if out and out[-1]:
                    out.append("")
                out += [f"{x.reps}x", *[cls.step(s) for s in flat(x.steps)], ""]
            else:
                out.append(cls.step(x))
        return "\n".join(out).strip() + "\n"

    @classmethod
    def step(cls, st) -> str:
        text = " ".join(cls.NUMBERS.sub(" ", st.cue or st.name).split())
        parts = ["-", text, cls.duration(st.minutes), cls.target(st)]
        if st.cadence:
            parts.append(f"{st.cadence}rpm")
        return " ".join(p for p in parts if p)

    @staticmethod
    def duration(minutes: float) -> str:
        h, rest = divmod(int(round(minutes * 60)), 3600)
        m, s = divmod(rest, 60)
        return "".join(f"{v}{u}" for v, u in ((h, "h"), (m, "m"), (s, "s")) if v) or "0s"

    @staticmethod
    def target(st) -> str:
        if st.hr is not None:
            return f"{round(st.hr * 100)}% HR"
        if st.ramp:
            return f"ramp {round(st.mid * 100)}-{round(st.mid_end * 100)}%"
        lo, hi = sorted((round(st.lo * 100), round(st.hi * 100)))
        return f"{lo}%" if lo == hi else f"{lo}-{hi}%"
