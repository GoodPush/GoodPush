"""Human-readable athlete profile (Markdown), regenerated on every run."""
from __future__ import annotations

FEATURE_LABEL = None


def _ok(x):
    return x is not None and x == x


def _w(x, unit="W"):
    return f"{x:.0f} {unit}" if _ok(x) else "n/a"


def _pct(x):
    return f"{x:.0%}" if _ok(x) else "-"


def render(res: dict) -> str:
    a, p, ld = res["athlete"], res["power"], res["load"]
    b90, ball = p["bests_90d"], p["bests_all"]
    L = []
    L.append(f"# Athlete profile - {a.get('name') or 'rider'}")
    L.append(f"_Generated {res['generated']} by goodpush from {res['data']['activities']} activities "
             f"({res['data']['first']} to {res['data']['last']}). Re-run `goodpush run` to refresh._\n")

    L.append("## Snapshot")
    L.append("| | Now (last 90 days) | Career best |")
    L.append("|---|---|---|")
    L.append(f"| FTP estimate | {_w(p['ftp_now'])} ({p['wkg_now'] or 'n/a'} W/kg) | "
             f"{_w(p['ftp_peak'])} ({p['wkg_peak'] or 'n/a'} W/kg) on {p['ftp_peak_date']} |")
    for d, name in (("5", "5 s"), ("60", "1 min"), ("300", "5 min"), ("1200", "20 min"),
                    ("3600", "60 min")):
        L.append(f"| Best {name} | {_w(b90[d]['w'])} | {_w(ball[d]['w'])} ({ball[d]['date']}) |")
    L.append(f"\nCritical power (90 d): **{_w(p['cp_90d'])}**, W' **{p['wprime_90d_kj'] or 'n/a'} kJ**. "
             f"Weight used: {a['weight_kg']} kg ({a['weight_source']}). "
             f"Max HR {a['max_hr']:.0f}, LTHR {a['lthr']:.0f} ({a['lthr_source']}), "
             f"resting HR {a['resting_hr']:.0f} ({a['resting_hr_source']}).\n")

    L.append("### Rider type")
    sc = p["profile_scores_peak"]
    cat = p["profile_category_peak"]
    L.append(f"**{p['phenotype']}** - {p['phenotype_reason']}.\n")
    L.append("| Duration | Career-best score (0-100) | Category | Score now |")
    L.append("|---|---|---|---|")
    for k, lab in (("5s", "5 s sprint"), ("1min", "1 min anaerobic"), ("5min", "5 min VO2max"),
                   ("ftp", "FTP")):
        L.append(f"| {lab} | {sc.get(k)} | {cat.get(k)} | {p['profile_scores_now'].get(k)} |")
    L.append("\n_Scores place W/kg on Coggan's power-profile scale (0 = untrained, 100 = world class)._\n")

    L.append("### Training zones (current FTP / LTHR)")
    pz = " | ".join(f"{z['zone']} <= {z['upper_w']} W" if z["upper_w"] else f"{z['zone']} above"
                    for z in res["zones"]["power"])
    hz = " | ".join(f"{z['zone']} <= {z['upper_bpm']}" if z["upper_bpm"] else f"{z['zone']} above"
                    for z in res["zones"]["hr"])
    L.append(f"- Power: {pz}\n- Heart rate: {hz}\n")

    L.append("## Training history")
    L.append(f"Current load CTL {ld['ctl_now']}, fatigue ATL {ld['atl_now']}, form TSB {ld['tsb_now']}. "
             f"Peak CTL {ld['ctl_peak']} on {ld['ctl_peak_date']}. Typical week: "
             f"{ld['typical_week_hours']} h / {ld['typical_week_tss']} TSS.\n")
    L.append("| Year | Hours | TSS | Avg CTL | Peak CTL | Hard/wk | Z1-2 (power) | Z4 | Z5+ | Z1-2 (HR) | Indoor | Best tested 20-min eq. |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for y in res["years"]:
        L.append(f"| {y['year']} | {y['hours']:.0f} | {y['tss']:.0f} | {y['avg_ctl']:.0f} | "
                 f"{y['peak_ctl']:.0f} | {y['hard_per_wk']:.1f} | {_pct(y['z12_share'])} | "
                 f"{_pct(y['z4_share'])} | {_pct(y['z5p_share'])} | {_pct(y['hr_z12_share'])} | "
                 f"{_pct(y['indoor_share'])} | "
                 f"{_w(y['best_perf'])} |")

    L.append("\n_Power-zone shares use the FTP estimate of the day: when the estimated FTP is low "
             "(e.g. few hard efforts), the same rides land in higher power zones. The HR-based "
             "share is independent of FTP - compare both._")
    L.append("\n## What drove progress")
    L.append("Performance marker = best *tested* 20-min-equivalent power over the previous 6 weeks "
             "(from hard rides only, so weeks without hard efforts are 'untested', not 'detrained').\n")
    if res["progress_periods"]:
        L.append("### Biggest gains")
        for q in res["progress_periods"]:
            L.append(f"- **{q['start']} -> {q['end']}** ({q['weeks']} wk): {q['perf_start_w']} -> "
                     f"{q['perf_end_w']} W (+{q['change_pct']}%). {q['hours_wk']:.1f} h/wk, "
                     f"{q['hard_wk']:.1f} hard/wk, Z1-2 {q['f_z12']:.0%}, Z4 {q['f_z4']:.0%}, "
                     f"Z5+ {q['f_z5p']:.0%}. _{q['what_was_different']}_")
    if res["decline_periods"]:
        L.append("\n### Biggest declines")
        for q in res["decline_periods"]:
            L.append(f"- **{q['start']} -> {q['end']}** ({q['weeks']} wk): {q['perf_start_w']} -> "
                     f"{q['perf_end_w']} W ({q['change_pct']}%). {q['hours_wk']:.1f} h/wk, "
                     f"{q['hard_wk']:.1f} hard/wk. _{q['what_was_different']}_")
    d = res["drivers"]
    rows = [r for r in d["table"] if r.get("strong")]
    if rows:
        L.append(f"\n### Statistically notable associations ({d['window_weeks']}-week windows)")
        for t, name in (("d_perf_pct", "tested power"), ("d_ef_pct", "aerobic efficiency (EF)")):
            sel = sorted([r for r in rows if r["target"] == t], key=lambda r: -abs(r["rho"]))
            if sel:
                L.append(f"- Change in **{name}**: " + "; ".join(
                    f"{r['label']} (rho {r['rho']:+.2f})" for r in sel))
    m = d.get("model", {})
    if m.get("ok"):
        L.append(f"- Cross-validated model R^2 = {m['cv_r2']:.2f} "
                 f"({'some' if m['cv_r2'] > 0.15 else 'little'} out-of-sample predictive power).")
    L.append("\n_Correlations use overlapping windows; 'notable' uses an effective sample size of "
             "weeks/window. 'Performance level at start' being negative is regression to the mean "
             "(you gain more from a lower starting point)._")

    L.append("\n## When it was too much")
    eps = res["episodes"]
    if eps:
        L.append("| Start | Weeks | Peak ATL | Min TSB | Max ACWR | h/wk | Flags | Recovery flags | "
                 "Perf after | Verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in eps:
            ch = e.get("change_after_pct")
            L.append(f"| {e['start']} | {e['weeks']} | {e['peak_atl']:.0f} | {e['min_tsb']:.0f} | "
                     f"{e['max_acwr']:.1f} | {e['hours_wk']:.1f} | {e['load_flags']} | "
                     f"{e['recovery_flags'] or '-'} | {'n/a' if ch is None else f'{ch:+.1f}%'} | "
                     f"**{e['verdict']}** |")
        n_bad = sum(e["verdict"] == "too much" for e in eps)
        n_good = sum(e["verdict"] == "productive overload" for e in eps)
        L.append(f"\n{n_good} overload blocks were followed by higher tested power within 6 weeks; "
                 f"{n_bad} look like too much (performance fell, a forced break followed, or "
                 "several recovery markers went red).")
    L.extend(_recovery_section(res))
    L.extend(_illness_section(res))
    L.extend(_buildup_section(res))
    ban = res["banister"]
    if ban.get("ok"):
        L.append("\n### Personal fitness-fatigue response (Banister model)")
        L.append(f"Fitness decays with tau ~ {ban['tau_fitness']:.0f} days, fatigue with tau ~ "
                 f"{ban['tau_fatigue']:.0f} days. A training stimulus turns net-positive after "
                 f"~{ban.get('days_to_positive_effect', float('nan')):.0f} days and its benefit peaks "
                 f"after ~{ban.get('days_to_peak_after_load', float('nan')):.0f} days - a guide to "
                 f"taper length. Fit R^2 {ban['r2']:.2f} on {ban['n']} tested days (low R^2 = treat "
                 "as rough).")

    if res["notes"]:
        L.append("\n## Data-quality notes")
        for n in res["notes"]:
            L.append(f"- {n}")
    return "\n".join(L) + "\n"


def _f(x, fmt="{:.2f}"):
    return fmt.format(x) if _ok(x) else "-"


def _hm(h):
    if not _ok(h):
        return "-"
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def _recovery_section(res) -> list:
    rec = res.get("recovery") or {}
    if not rec.get("available"):
        return []
    L = ["\n## Sleep and recovery (Garmin)"]
    cov = res.get("wellness", {}).get("coverage", {})
    if cov:
        L.append("Data: " + "; ".join(f"{k} {v}" for k, v in cov.items()
                                       if k in ("sleep_h", "rhr", "hrv", "readiness")) + ".\n")
    yr = rec.get("yearly") or []
    if yr:
        L.append("| Year | Sleep (h) | Bedtime | Resting HR | HRV (ms) | Stress | Sleep score | Avg CTL |")
        L.append("|---|---|---|---|---|---|---|---|")
        for y in yr:
            L.append(f"| {y['year']} | {_f(y.get('sleep_h'))} | {_hm(y.get('bedtime_h'))} | "
                     f"{_f(y.get('rhr'), '{:.0f}')} | {_f(y.get('hrv'), '{:.0f}')} | "
                     f"{_f(y.get('stress'), '{:.0f}')} | {_f(y.get('sleep_score'), '{:.0f}')} | "
                     f"{_f(y.get('ctl'), '{:.0f}')} |")
    fm = {r["marker"]: r for r in rec.get("fitness_vs_markers") or []}
    if "rhr" in fm:
        L.append(f"\nResting HR tracks fitness: weekly RHR vs CTL rho {fm['rhr']['rho_with_ctl']:+.2f}"
                 + (f"; HRV vs CTL rho {fm['hrv']['rho_with_ctl']:+.2f}" if "hrv" in fm else "") + ".")
    dose = rec.get("dose_next_morning") or []
    if dose:
        L.append("\n### How a training day shows up the next morning")
        L.append("| Day's load (TSS) | Days | HRV z | Resting HR z | Sleep (h) | Sleep score | Readiness |")
        L.append("|---|---|---|---|---|---|---|")
        for d in dose:
            L.append(f"| {d['bin']} | {d['days']} | {_f(d.get('hrv_z'))} | {_f(d.get('rhr_z'))} | "
                     f"{_f(d.get('sleep_h'))} | {_f(d.get('sleep_score'), '{:.0f}')} | "
                     f"{_f(d.get('readiness'), '{:.0f}')} |")
        L.append("_z = standard deviations from your own 28-day baseline (HRV: negative = suppressed; "
                 "RHR: positive = elevated)._")
    cur = rec.get("after_hard_day") or []
    if cur:
        L.append(f"\n**Recovery curve after a 150+ TSS day** (n = {rec.get('n_hard_days')}): " + ", ".join(
            f"day {c['days_after']}: HRV {_f(c.get('hrv_z'), '{:+.2f}')} / RHR {_f(c.get('rhr_z'), '{:+.2f}')}"
            for c in cur[:4]))
    for key, title in (("readiness_vs_performance", "Morning readiness"),
                       ("hrv_vs_performance", "Morning HRV"),
                       ("sleep_vs_performance", "Previous night's sleep")):
        t = rec.get(key) or []
        if not t:
            continue
        L.append(f"\n### {title} vs performance that day")
        L.append("| State | Hard-effort days | Test power vs 90-day best | Steady rides | EF vs 6-wk baseline |")
        L.append("|---|---|---|---|---|")
        for r in t:
            L.append(f"| {r['bin']} | {r['test_vs_90d_best_n']} | {_f(r.get('test_vs_90d_best'), '{:.0%}')} | "
                     f"{r['ef_vs_6wk_n']} | {_f(r.get('ef_vs_6wk'), '{:.0%}')} |")
    return L


def _illness_section(res) -> list:
    il = res.get("illness") or {}
    eps = il.get("episodes") or []
    if not eps:
        return []
    L = ["\n## Illness (Strava notes + Garmin)"]
    conf = [e for e in eps if not str(e.get("severity", "")).startswith("unconfirmed")]
    L.append(f"{il.get('notes_found')} ride notes mention sickness; they group into {len(eps)} illnesses "
             f"({len(conf)} confirmed by the data). Notes are usually written on the comeback ride, so "
             "onset is reconstructed from the training gap and the resting-HR rise before the note. "
             "Run `goodpush illness` to check the list; add or correct with `[[events]] type = \"illness\"`.\n")
    L.append("| Onset | Last note | Days off | RHR peak | HRV low (z) | Severity | Test power before -> after | Days to regain |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e in eps:
        L.append(f"| {e['onset']} | {e['comeback_note']} | {e['days_off']} | "
                 f"{_f(e.get('rhr_peak_delta'), '{:+.0f}')} bpm | {_f(e.get('hrv_min_z'), '{:+.1f}')} | "
                 f"{e.get('severity')} | {_f(e.get('perf_before'), '{:.0f}')} -> "
                 f"{_f(e.get('perf_after_6wk'), '{:.0f}')} W | {_f(e.get('days_to_regain_97pct'), '{:.0f}')} |")
    prof = {p["day"]: p for p in il.get("profile") or []}
    if prof:
        L.append("\n### What getting sick looks like (average over all illnesses, day 0 = onset)")
        L.append("| Day | Resting HR vs baseline | HRV (z) | Sleep vs baseline (h) | Stress vs baseline | TSS |")
        L.append("|---|---|---|---|---|---|")
        for k in (-7, -3, -1, 0, 1, 2, 3, 5, 7, 10, 14, 21):
            p = prof.get(k)
            if p:
                L.append(f"| {k:+d} | {_f(p.get('rhr_d'), '{:+.1f}')} | {_f(p.get('hrv_z'), '{:+.2f}')} | "
                         f"{_f(p.get('sleep_d'), '{:+.2f}')} | {_f(p.get('stress_d'), '{:+.1f}')} | "
                         f"{_f(p.get('tss'), '{:.0f}')} |")
    pre = il.get("pre_signals") or []
    if pre:
        L.append("\n### Early warning signs? (the 2 weeks before onset vs normal 2-week windows)")
        L.append("| Measure | Before illness | Normal | Avg percentile (50 = normal) | Verdict |")
        L.append("|---|---|---|---|---|")
        for p in pre:
            L.append(f"| {p['label']} | {_f(p['before_illness'])} | {_f(p['normal'])} | "
                     f"{p['avg_percentile']:.0f} (+/-{p['margin']:.0f}) | {p['signal']} |")
    w = il.get("warning") or {}
    if w:
        L.append(f"\n**Rule check - {w['rule']}:** fires {w['alarms_per_year']:.0f}x a year; "
                 f"{w['alarms_near_illness']} of {w['alarms']} alarms were at an illness, and it caught "
                 f"{w['illnesses_caught']} of {w['illnesses']} illnesses within 3 days of onset.")
    dec = il.get("declines") or []
    if dec:
        n = sum(1 for x in dec if x.get("explained_by_illness"))
        L.append(f"\n**Big power drops vs illness:** {n} of {len(dec)} decline periods had an illness "
                 "starting within 6 weeks before or during them:")
        for x in dec:
            L.append(f"- {x['start']} -> {x['end']} ({x['change_pct']:+.0f}%): "
                     + (f"illness {x['illness_onsets']}" if x["illness_onsets"] else "no illness recorded"))
    return L


_TL = [("recovery", "Rec"), ("endurance", "End"), ("long_endurance", "Long"), ("tempo", "Tempo"),
       ("sweetspot", "SS"), ("threshold", "Thr"), ("vo2max", "VO2"), ("anaerobic", "Ana"),
       ("race", "Race"), ("test", "Test")]


def _buildup_section(res) -> list:
    bu = res.get("buildup") or {}
    if not bu.get("peak_blocks"):
        return []
    L = [f"\n## Your peak build (tested peak {bu['peak_value']:.0f} W, week of {bu['peak_week']})"]
    L.append("Rides are classified from the power stream (sustained efforts vs FTP, confirmed by "
             "heart rate) with app workout names as a check. Numbers are sessions per week.\n")
    head = "| 4 weeks from | h/wk | CTL end | Ramp/wk | Best test | HR Z1-2 | Rest days/wk | " + \
        " | ".join(l for _, l in _TL) + " |"
    L.append(head)
    L.append("|" + "---|" * (7 + len(_TL)))
    for b in bu["peak_blocks"]:
        L.append(f"| {b['start']} | {_f(b['hours_wk'], '{:.1f}')} | {_f(b['ctl_end'], '{:.0f}')} | "
                 f"{_f(b['ramp_wk'], '{:+.1f}')} | {_f(b['best_test'], '{:.0f}')} | "
                 f"{_f(b['hr_z12'], '{:.0%}')} | {_f(b['rest_days_wk'], '{:.1f}')} | "
                 + " | ".join(_f(b.get(t), "{:.2f}") for t, _ in _TL) + " |")
    builds = bu.get("builds") or []
    if builds:
        L.append("\n### The 16 weeks before each big gain vs the last 16 weeks")
        L.append("| Build | h/wk | CTL | Max 4-wk ramp | HR Z1-2 | Rest days | Long/wk | Quality/wk | "
                 "Thr | VO2 | SS+Tempo | Tests | Sleep |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for b in builds:
            L.append(f"| {b['label']} | {_f(b['hours_wk'], '{:.1f}')} | {_f(b['ctl_end'], '{:.0f}')} | "
                     f"{_f(b['max_ramp_4wk'], '{:.1f}')} | {_f(b['hr_z12'], '{:.0%}')} | "
                     f"{_f(b['rest_days_wk'], '{:.1f}')} | {_f(b['long_wk'], '{:.2f}')} | "
                     f"{_f(b['quality_wk'], '{:.2f}')} | {_f(b['threshold'], '{:.2f}')} | "
                     f"{_f(b['vo2max'], '{:.2f}')} | {_f((b['sweetspot'] or 0) + (b['tempo'] or 0), '{:.2f}')} | "
                     f"{_f(b['test'] * 16 if b.get('test') is not None else None, '{:.0f}')} | "
                     f"{_f(b['sleep_h'], '{:.1f}')} |")
    rt = bu.get("ramp_table") or []
    if rt:
        L.append("\n### How fast can you ramp? (CTL change per week, averaged over 4 weeks)")
        L.append(f"Illness base rate: {bu['illness_base_rate']:.0%} of weeks are followed by an "
                 "illness within 6 weeks.\n")
        L.append("| CTL ramp/wk | Weeks | Illness next 6 wk | Power change | Efficiency change | RHR after (z) |")
        L.append("|---|---|---|---|---|---|")
        for r in rt:
            L.append(f"| {r['ctl_ramp_per_week']} | {r['weeks']} | {_f(r['illness_next_6wk'], '{:.0%}')} | "
                     f"{_f(r['power_change_pct'], '{:+.1f}%')} | {_f(r['ef_change_pct'], '{:+.1f}%')} | "
                     f"{_f(r['rhr_z_after'], '{:+.2f}')} |")
    runs = bu.get("ramp_runs") or []
    if runs:
        L.append("\nLongest sustained ramps (4-week ramp >= 3 CTL/week):")
        for r in runs:
            L.append(f"- {r['start']} -> {r['end']}: CTL {r['ctl_from']:.0f} -> {r['ctl_to']:.0f} "
                     f"({r['avg_ramp']:.1f}/wk), illness within 3 wk: {'yes' if r['illness_within_3wk'] else 'no'}")
    td = [t for t in bu.get("type_drivers") or [] if t.get("notable")]
    if td:
        L.append("\nNotable session-type associations (8-week windows): " + "; ".join(
            f"{t['feature']} -> {t['target']} (rho {t['rho']:+.2f})" for t in td))
    return L
