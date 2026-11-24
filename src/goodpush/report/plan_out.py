"""Plan outputs: plan.md, plan.ics (calendar), plan.json, workouts/*.fit and a dashboard section."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..io.fit_workout import write_workout
from ..io.icu import IcuText
from ..io.zwo import workout_zwo
from ..planning.workouts import LEVELS, Repeat, Workout

LABEL = {"rest": "Rest", "recovery": "Recovery", "endurance": "Endurance", "long": "Long ride",
         "threshold": "Threshold", "vo2max": "VO2max", "sweetspot": "Sweet spot", "tempo": "Tempo",
         "anaerobic": "Anaerobic", "test": "FTP test", "ramp": "Ramp test", "race": "Race", "openers": "Openers",
         "lsct": "LSCT"}
WHY = {
    "threshold": "Main FTP builder: long efforts just under FTP.",
    "vo2max": "Raises the ceiling FTP can grow into. Hit the range, don't sprint the first rep.",
    "sweetspot": "High aerobic stimulus at manageable fatigue.",
    "tempo": "Steady aerobic work just under LT2. A hard day for recovery, so used sparingly.",
    "lsct": "Tracks fitness without a hard effort: the power it takes to hold set heart rates, and how fast your "
            "heart rate falls afterwards. Same time of day and conditions each time.",
    "anaerobic": "Race-specific punch: short, very hard efforts with full recoveries. Fresh legs only.",
    "endurance": "Aerobic base: most of a good year's riding is easy riding. Keep it truly easy.",
    "long": "Durability and fat oxidation. Fuel 60+ g carbs/hour.",
    "recovery": "Very easy spin or take it off if you feel flat.",
    "rest": "Rest. Sleep is the training today.",
    "test": "Re-anchors FTP so every target stays right.",
    "openers": "Short efforts to wake the legs up for tomorrow's race.",
    "race": "Race day.",
    "ramp": "Finds LT1 and LT2 from alpha1 the way the method was validated, plus your heart rate at each.",
}


def _w(x, ftp):
    return int(round(x * ftp))


def step_watts(st, ftp) -> dict:
    """a Step in watts for the page, the ride bar and the FIT file: name, min, lo, hi, kind, and when set
    lo_end / hi_end (a ramp's end), cadence, cue, mode, hr (a heart-rate target, fraction of max HR: the watts
    are then only a guide)."""
    d = {"name": st.name, "min": st.minutes, "lo": _w(st.lo, ftp), "hi": _w(st.hi, ftp), "kind": st.kind}
    if st.ramp:
        d.update(lo_end=_w(st.lo_end, ftp), hi_end=_w(st.hi_end, ftp))
    d.update({k: getattr(st, k) for k in ("cadence", "cue", "mode", "hr") if getattr(st, k)})
    return d


def blocks(items, ftp, split_ramps=False) -> list[dict]:
    """Steps and Repeats in watts, nested as written: steps, and {reps, steps} for a repeat."""
    out = []
    for x in items:                        # a workout's few dozen items
        if isinstance(x, Repeat):
            out.append({"reps": x.reps, "steps": blocks(x.steps, ftp, split_ramps)})
        else:
            out += [step_watts(p, ftp) for p in (x.pieces() if split_ramps else [x])]
    return out


def steps_text(items, ftp) -> str:
    """10' Warm up 120-160 W; 3 x (5' VO2max 260-275 W, 5' Recover 105-130 W); ..."""
    parts = []
    for x in items:
        if isinstance(x, Repeat):
            parts.append(f"{x.reps} x (" + ", ".join(steps_text([y], ftp) for y in x.steps) + ")")
            continue
        ramp = f" → {_w(x.lo_end, ftp)}-{_w(x.hi_end, ftp)} W" if x.ramp else ""
        parts.append(f"{x.minutes:g}' {x.name} {_w(x.lo, ftp)}-{_w(x.hi, ftp)} W{ramp}"
                     + (f" @ {x.cadence} rpm" if x.cadence else ""))
    return "; ".join(parts)


def session_steps(s, ftp) -> dict:
    """what the page and the ride need of a session: its steps written out, as nested blocks, and as text."""
    return {"steps": [step_watts(st, ftp) for st in s.expanded()], "blocks": blocks(s.steps, ftp),
            "workout": steps_text(s.steps, ftp)}


BRAND = "GoodPush"     # starts every session title sent out (intervals.icu, Incyclist, .fit, .zwo, calendar)


def outbound_title(title: str) -> str:
    return f"{BRAND} - {title}"


def has_file(s) -> bool:
    """Sessions you ride from a file (not rest days, not races)."""
    return s.type not in ("rest", "race") and bool(s.steps)


def fit_name(s, ext="fit") -> str:
    n = f"{s.key.split('#')[1]}_" if s.key else ""        # a second session that day
    return f"gp_{s.date:%Y-%m-%d}_{n}{_slug(s.title)}.{ext}"


def session_zwo(s) -> str:
    """The session as a Zwift workout (% of FTP) for Incyclist / Zwift."""
    desc = " ".join(x for x in (WHY.get(s.type, ""), s.note, s.adjusted and f"Adjusted: {s.adjusted}") if x)
    return workout_zwo(outbound_title(f"{s.date:%a %d %b} - {s.title}"), s.steps, desc)


def session_icu(s) -> str:
    """The session as intervals.icu workout text (% of FTP)."""
    return IcuText.workout(s.steps, " ".join(x for x in (WHY.get(s.type, ""), s.note) if x))


def rows(out) -> list[dict]:
    ftp = out["ftp"]
    planned = out.get("planned") or {}
    res = []
    for s in out["sessions"]:
        ds = str(s.date.date())
        res.append({"date": ds, "key": s.key or ds, "day": s.date.strftime("%a"), "type": s.type,
                    "title": s.title, "minutes": int(s.minutes), "tss": round(s.tss()),
                    "why": WHY.get(s.type, ""),
                    "note": s.note, "adjusted": s.adjusted,
                    "status": (planned.get(s.key or ds) or {}).get("status", ""), "why_change": s.why_change,
                    "fit": fit_name(s) if has_file(s) else "",
                    "zwo": fit_name(s, "zwo") if has_file(s) else "",
                    "icu": fit_name(s, "txt") if has_file(s) else "", **session_steps(s, ftp)})
    return res


def proposal_rows(out) -> list[dict]:
    """Proposed changes waiting for accept / reject (sessions: what you have -> what's proposed)."""
    res = []
    for c in out.get("proposal") or []:
        if c.get("kind", "session") != "session":
            res.append(dict(c))
            continue
        s = Workout.from_dict(c["date"], c["session"])
        res.append({"kind": "session", "date": c["date"], "why": c["why"], "from": c["from"],
                    "to": {"type": s.type, "title": s.title, "minutes": int(s.minutes), "tss": round(s.tss()),
                           "workout": steps_text(s.steps, out["ftp"])}})
    return res


def _slug(t):
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")[:40]


def write_fit_files(d: Path, out) -> list[Path]:
    """Workouts (.fit), the same sessions as .zwo (Incyclist, Zwift) and as intervals.icu text (.txt)."""
    d.mkdir(parents=True, exist_ok=True)
    for old in [p for p in d.glob("gp_*") if p.suffix in (".fit", ".zwo", ".txt")]:   # only ours
        try:
            old.unlink()
        except OSError:
            pass
    ftp, paths = out["ftp"], []
    for s in out["sessions"]:
        if not has_file(s):
            continue
        name = outbound_title(f"{s.date:%m%d} {s.title}")[:31]                           # FIT: 31 characters
        p = d / fit_name(s)
        paths.append(write_workout(p, name, blocks(s.steps, ftp, split_ramps=True)))       # FIT can't ramp
        (d / fit_name(s, "zwo")).write_text(session_zwo(s), encoding="utf-8")
        (d / fit_name(s, "txt")).write_text(session_icu(s), encoding="utf-8")
    return paths


def _ics_text(t: str) -> str:
    return t.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> str:
    b = line.encode("utf-8")
    if len(b) <= 74:
        return line
    out, cur = [], ""
    for ch in line:
        if len((cur + ch).encode("utf-8")) > 73:
            out.append(cur)
            cur = " " + ch
        else:
            cur += ch
    out.append(cur)
    return "\r\n".join(out)


def ics(out) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seq = int(datetime.now(timezone.utc).timestamp()) // 60
    L = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//goodpush//training plan//EN",
         "CALSCALE:GREGORIAN", "METHOD:PUBLISH", "X-WR-CALNAME:goodpush plan"]
    for r, s in zip(rows(out), out["sessions"]):
        d0 = s.date.strftime("%Y%m%d")
        d1 = (s.date + pd.Timedelta(days=1)).strftime("%Y%m%d")
        summ = outbound_title(s.title) if s.type != "rest" else "Rest day"
        if s.type not in ("rest",):
            summ += f" · {r['tss']} TSS"
        desc = "\n".join(x for x in [r["workout"], r["why"], r["note"],
                                      ("Adjusted: " + r["adjusted"]) if r["adjusted"] else ""] if x)
        uid = d0 + (f"-{s.key.split('#')[1]}" if s.key else "")
        L += ["BEGIN:VEVENT", f"UID:goodpush-plan-{uid}@goodpush", f"DTSTAMP:{now}", f"SEQUENCE:{seq}",
              f"DTSTART;VALUE=DATE:{d0}", f"DTEND;VALUE=DATE:{d1}", "TRANSP:TRANSPARENT",
              f"SUMMARY:{_ics_text(summ)}", f"DESCRIPTION:{_ics_text(desc)}", "END:VEVENT"]
    L.append("END:VCALENDAR")
    return "\r\n".join(_fold(x) for x in L) + "\r\n"


def markdown(out, name="") -> str:
    rd = out["readiness"]
    proj = out["projection"]
    L = [f"# Training plan{(' – ' + name) if name else ''}", "",
         f"Generated {out['today']:%Y-%m-%d} · FTP for targets **{out['ftp']:.0f} W** ({out['ftp_source']})", "",
         f"## Today: {rd['status'].upper()}", ""]
    L += [f"- {x}" for x in rd["reasons"]] or ["- Nothing flagged. Train as planned."]
    L += [f"- Note: {x}" for x in rd.get("notes", [])]
    L += ["", "## Next days", "", "| Date | Day | Session | Min | TSS | Workout |", "|---|---|---|---|---|---|"]
    for r in rows(out):
        t = r["title"] + (f" ⚠ {r['adjusted']}" if r["adjusted"] else "")
        L.append(f"| {r['date']} | {r['day']} | {t} | {r['minutes']} | {r['tss']} | {r['workout']} |")
    L += ["", "## Where you are in the progression", ""]
    phases = (out.get("season") or {}).get("phases") or []
    L += [f"- {p['name']}: {p['start']}, {p['weeks']} weeks" for p in phases]
    L += [f"- Build weeks aim for a CTL ramp of {out['cfg']['min_ramp']} → {out['cfg']['max_ramp']}/week. "
          f"Load (CTL) now {out['ctl_now']:.0f}" + (f", projected {proj[-1][1]:.0f} at the end." if proj else "."),
          "", "| Week of | Phase | Block/week | Ramp target | TSS target | TSS planned | Hours planned |",
          "|---|---|---|---|---|---|---|"]
    for w in out["weeks"]:
        tag = " (recovery)" if w["recovery"] else " (taper)" if w.get("taper") else ""
        cap = " — capped by your hours" if w["capped"] else ""
        L.append(f"| {w['week']} | {w.get('phase', '')} | {w['block']}/{w['week_in_block']}{tag} | "
                 f"{w['ramp_target']:+.1f} | {w['tss_target']} | {w['tss_planned']} | {w['hours_planned']}{cap} |")
    if any(w["capped"] for w in out["weeks"]):
        L.append("\nWith the hours you have, easy volume alone can't lift CTL that fast; progression then "
                 "comes from the interval ladders (more time at threshold each week).")
    L += ["",
          "- Ladder levels (go up one step each time you complete that session): " +
          ", ".join(f"{k} {out['levels'].get(k, 0) + 1}/{v}" for k, v in LEVELS.items()) + "."]
    if out["need_test"]:
        L.append("- An FTP test is scheduled: no 20-min test in the last "
                 f"{out['cfg']['test_every_weeks']} weeks, so targets may be stale.")
    if out.get("need_ramp"):
        L.append("- A ramp test is scheduled (with alphaHRV): it gives LT1 and LT2 from alpha1, no ramp in the last "
                 f"{out['cfg'].get('ramp_every_weeks', 6)} weeks.")
    if out["log"]:
        L += ["", "## Since last run", ""] + [f"- {d}: {k} – {st}" for d, k, st in out["log"][-10:]]
    L += ["", "## How it adapts", "",
          "- **Red** (resting HR +3 bpm two mornings running, HRV z ≤ −2 or < 5.5 h sleep): today and "
          "tomorrow become easy spins; the missed session is not crammed in, and the next quality day "
          "only goes ahead if you've bounced back.",
          "- **Amber** (low HRV, short sleep, 3-night sleep debt, TSB < −30, recent sickness note): "
          "threshold/VO2 are trimmed (one rep fewer, 2 % lower); a test moves to the next free easy day.",
          "- Levels only advance after a session of that type was actually done (from your Strava/Garmin rides).",
          "- Run `goodpush sync garmin` then `goodpush plan` each morning (or `goodpush update`).", ""]
    return "\n".join(L)


def html_section(out) -> str:
    if not out:
        return ""
    rd = out["readiness"]
    col = {"green": "var(--good)", "amber": "var(--s4)", "red": "var(--bad)"}[rd["status"]]
    reasons = "".join(f"<li>{html.escape(x)}</li>" for x in rd["reasons"]) or "<li>Nothing flagged.</li>"
    reasons += "".join(f"<li>Note: {html.escape(x)}</li>" for x in rd.get("notes", []))
    trs = []
    for r in rows(out):
        adj = f"<div class='muted'>⚠ {html.escape(r['adjusted'])}</div>" if r["adjusted"] else ""
        trs.append(f"<tr><td>{r['date']} {r['day']}</td><td><b>{html.escape(r['title'])}</b>{adj}</td>"
                   f"<td>{r['minutes']}</td><td>{r['tss']}</td><td>{html.escape(r['workout'])}</td></tr>")
    return (f"<h2>Training plan</h2><p class='expl'>Targets from FTP {out['ftp']:.0f} W "
            f"({html.escape(out['ftp_source'])}). Workout files (.fit, .zwo, intervals.icu .txt) are in "
            f"<code>reports/plan/workouts</code>; calendar in <code>reports/plan/plan.ics</code>.</p>"
            f"<div class='card'><div style='font-weight:650;color:{col}'>Today: {rd['status'].upper()}</div>"
            f"<ul class='muted'>{reasons}</ul><div class='tw'><table><thead><tr><th>Date</th><th>Session</th>"
            f"<th>Min</th><th>TSS</th><th>Workout</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div></div>")


def write_all(d: Path, out, name="") -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.md").write_text(markdown(out, name), encoding="utf-8")
    (d / "plan.ics").write_text(ics(out), encoding="utf-8", newline="")
    js = {"generated": str(out["today"].date()), "ftp": out["ftp"], "ftp_source": out["ftp_source"],
          "readiness": out["readiness"], "levels": out["levels"], "need_test": out["need_test"],
          "need_ramp": out.get("need_ramp", False),
          "ramp_per_week": round(out["ramp_per_week"], 2), "ramp_range": out.get("ramp_range"),
          "ctl_now": round(out["ctl_now"], 1), "ctl_end": round(out["projection"][-1][1] if out["projection"] else out["ctl_now"], 1),
          "weeks": out["weeks"], "override": out.get("override") or "", "sessions": rows(out),
          "intensity": out.get("intensity"), "code": out.get("code", ""), "session_mix": out.get("session_mix"),
          "proposal": proposal_rows(out), "season": out.get("season"),
          "projection": {k: [str(p[0].date()) if i == 0 else round(float(p[i]), 1) for p in out["projection"]]
                         for i, k in enumerate(("date", "ctl", "atl", "tsb"))}}
    (d / "plan.json").write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    write_fit_files(d / "workouts", out)
    return d
