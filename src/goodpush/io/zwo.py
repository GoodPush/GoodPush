"""Zwift workout files (.zwo) for plan sessions: read by Incyclist, Zwift and TrainingPeaks.
Power is a fraction of FTP, so the app riding it scales targets with its own FTP setting
(or `ftp_override` pins them to the watts goodpush planned).

Tags used (https://github.com/h4l/zwift-workout-file-reference): Warmup (ramp up), Ramp (a ramp step),
SteadyState (target = middle of the step's range, Cadence when set), IntervalsT (a repeated on / off pair),
textevent (a step's cue). Other repeats are written out (ZWO has no general repeat).
"""
from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr


def _p(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _step(st) -> str:
    sec = int(round(st.minutes * 60))
    cad = f' Cadence="{st.cadence}"' if st.cadence else ""
    if st.ramp:
        tag = f'<Ramp Duration="{sec}" PowerLow="{_p(st.mid)}" PowerHigh="{_p(st.mid_end)}"{cad}'
    elif st.kind == "warmup" and st.hi > st.lo:
        tag = f'<Warmup Duration="{sec}" PowerLow="{_p(st.lo)}" PowerHigh="{_p(st.hi)}"{cad}'
    else:
        tag = f'<SteadyState Duration="{sec}" Power="{_p(st.mid)}"{cad}'
    if not st.cue:
        return tag + "/>"
    return tag + f'><textevent timeoffset="0" message={quoteattr(st.cue)}/></' + tag[1:tag.index(" ")] + ">"


def _items(items) -> list[str]:
    body = []
    for x in items:                          # a workout's few dozen items; a repeat is what has reps
        if not hasattr(x, "reps"):
            body.append(_step(x))
        elif len(x.steps) == 2 and not any(hasattr(y, "reps") or y.ramp or y.cue or y.cadence for y in x.steps):
            on, off = x.steps                # on / off: one IntervalsT
            body.append(f'<IntervalsT Repeat="{x.reps}" OnDuration="{int(round(on.minutes * 60))}" '
                        f'OffDuration="{int(round(off.minutes * 60))}" OnPower="{_p(on.mid)}" '
                        f'OffPower="{_p(off.mid)}"/>')
        else:
            body += _items(x.steps) * x.reps
    return body


def workout_zwo(name: str, items: list, description: str = "", ftp_override: float | None = None) -> str:
    """items: Steps (name, minutes, lo, hi as fractions of FTP, kind, ramp / cadence / cue) and Repeats."""
    body = _items(items)
    over = f"\n  <ftpOverride>{int(round(ftp_override))}</ftpOverride>" if ftp_override else ""
    return ("<workout_file>\n"
            "  <author>goodpush</author>\n"
            f"  <name>{escape(name)}</name>\n"
            f"  <description>{escape(description)}</description>\n"
            "  <sportType>bike</sportType>" + over + "\n"
            "  <workout>\n    " + "\n    ".join(body) + "\n  </workout>\n"
            "</workout_file>\n")
