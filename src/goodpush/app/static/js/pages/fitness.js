// Fitness (what you did): load (PMC) with form under it, FTP over time, power curve, gains / builds (FitnessPage).
import {Pref, Segmented} from "../controls.js";
import {Page} from "./page.js";
import {api, S} from "../api.js";
import {$, css, esc, fmt, hhmm} from "../util.js";
import {hint, niceDate, tile, TYPE_LABEL} from "../widgets.js";
import {CAUSE, dips, PERIOD} from "../dips.js";
import {typeColor} from "../calendar/calendar.js";
import {EChart, SelectionPanel, TimeWindow} from "../charts.js";

export const ZOOM_HELP = "• Zoom: Ctrl + wheel or pinch, or drag the ends of the bar under the chart\n• Zoomed in: drag, Shift + wheel or the bar moves sideways\n• Double-click: show it all";
// form (TSB) in words, the usual cut-offs: [lowest TSB, word], highest first
const FORM_WORDS = [[5, "fresh"], [-10, "neutral"], [-30, "building"], [-Infinity, "very tired"]];

class FitnessPage extends Page {
  showGains = new Pref("pmc_gains", false);   // the load chart's gains bands (FTP up 4 % or more)
  // "Mark periods": a drag on the load chart picks days and you say what they were (models.dips BreakLabels
  // periods); a period of yours clicked: its edges move, its label changes. Kept beside the detected ones
  marking = false;
  // a drag on the load chart: marking, what the days were; else that stretch's numbers (drawStats)
  pmcSel = {range: null, onChange: (r) => (this.marking ? this.periodPicked(r) : this.drawStats())};
  editing = null;                              // the period of yours being changed ({start, end, cause})
  marked = null;                               // saved while marking, not learned from yet: {gain: a gain among them}
  selPanel = null;                             // SelectionPanel, made on first draw (charts.js is no leaf module)
  days = null;                                 // the load chart's days, ridden then planned: {x (ms), ctl, atl, tsb, tss, hours, last}

  formWord(tsb) { return tsb == null ? "" : FORM_WORDS.find(([lo]) => tsb >= lo)[1]; }

  msOf(ds) { return new Date(ds + "T12:00:00").getTime(); }

  dayOf(v) { const d = new Date(v); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`; }

  span(p) { return [this.msOf(p.start) - 432e5, this.msOf(p.end) + 432e5]; }   // a period's days on the chart: midnight to midnight

  // a stretch picked (or a period's edges moved): what was it? `cause` = the one to suggest
  periodPicked(r, cause) {
    if (!r) { this.editing = null; return; }
    const e = this.editing, start = this.dayOf(r[0] + 432e5), end = this.dayOf(r[1] - 432e5);
    if (end < start) return;
    if (e && !(r[0] < this.span(e)[1] && this.span(e)[0] < r[1])) this.editing = null;   // a new stretch elsewhere
    const on = this.editing?.cause || cause, n = Math.round((this.msOf(end) - this.msOf(start)) / 864e5) + 1;
    const d = (x) => esc(niceDate(x, {day: "numeric", month: "short", year: "numeric"}));
    dips.menu(`<div class="th">${d(start)} – ${d(end)}</div>
    <div class="small muted">${n} days${this.editing ? " · drag its edges to move them" : ""}. What was it?</div>
    ${Object.entries(PERIOD).map(([k, [lab, col]]) => `<button data-v="${k}" class="${on === k ? "on" : ""}"><i style="background:var(${col})"></i>${esc(lab)}</button>`).join("")}
    ${this.editing ? `<button data-v=""><i style="background:var(--line)"></i>Remove this period</button>` : ""}`, (b) => this.savePeriod(start, end, b.dataset.v));
  }

  // saved at once, learned from when marking ends (setMarking). A detected break's own days: you confirm (or
  // relabel) that break (✓) rather than draw a copy over it (server: label_period)
  savePeriod(start, end, value) {
    const e = this.editing, data = dips.drops.data;
    if (!data) return;
    data.periods = (data.periods || []).filter((p) => !(e && p.start === e.start && p.end === e.end));   // shown straight away
    const q = value !== "gain" && (data.dips || []).find((d) => d.start === start && d.end === end);
    if (q) { q.label = value || null; q.cause = q.label || q.auto_cause; }
    else if (value) data.periods.push({start, end, cause: value});
    api.action({action: "label_period", start, end, value, was: e ? [e.start, e.end] : null}, "Saved – learned from when you finish marking");
    this.marked = {gain: !!this.marked?.gain || [value, e?.cause].includes("gain")};
    this.editing = this.pmcSel.range = null;
    this.chartPMC(null);
  }

  // select a stretch on the load chart (a period of yours: to change it) and ask what it was
  pickPeriod(range, mine, cause) {
    this.editing = mine || null;
    this.pmcSel.range = range;
    this.setMarking(true);
    this.periodPicked(range, cause);
  }

  // the days in [r0, r1] (each day midnight to midnight, as periods): CTL ramp (per week, from the day before the
  // first to the last, and its steepest 7 days), averages of load / fatigue / form, training a week on ridden days
  stretch(r) {
    const D = this.days, ix = D.x.flatMap((v, i) => (v >= r[0] && v <= r[1] && D.ctl[i] != null ? [i] : []));
    if (!ix.length) return null;
    const a = Math.max(ix[0] - 1, 0), b = ix[ix.length - 1], n = ix.length;
    const vals = (k, I = ix) => I.map((i) => D[k]?.[i]).filter((v) => v != null);
    const mean = (v) => (v.length ? v.reduce((p, q) => p + q, 0) / v.length : null);
    const ridden = ix.filter((i) => D.x[i] <= D.last), sum = (k) => vals(k, ridden).reduce((p, q) => p + q, 0);
    let steep = null;
    for (const i of ix) if (i - 7 >= a && D.ctl[i - 7] != null) steep = Math.max(steep ?? -Infinity, D.ctl[i] - D.ctl[i - 7]);
    return {start: D.x[ix[0]], end: D.x[b], n, planned: D.x[b] > D.last, ctl0: D.ctl[a], ctl1: D.ctl[b],
      ramp: b > a ? (D.ctl[b] - D.ctl[a]) / (b - a) * 7 : null, steep, ctl: mean(vals("ctl")), ctlMax: Math.max(...vals("ctl")),
      atl: mean(vals("atl")), atlMax: Math.max(...vals("atl")), tsb: mean(vals("tsb")), tsbMin: Math.min(...vals("tsb")),
      ridden: ridden.length, tss: sum("tss"), hours: sum("hours")};
  }

  drawStats() {   // the selected stretch's numbers beside the charts (as a ride's)
    const r = this.pmcSel.range;
    this.selPanel ??= new SelectionPanel("pmcsel", "pmc", this.pmcSel, "Drag across a chart to see the numbers for that stretch, with its ramp rate.");
    if (this.marking) { this.selPanel.empty = "Marking periods: a drag marks days. Switch it off to see a stretch’s numbers."; this.selPanel.draw(null); return; }
    this.selPanel.empty = "Drag across a chart to see the numbers for that stretch, with its ramp rate.";
    const s = r && this.days ? this.stretch(r) : null;
    if (!s) { this.selPanel.draw(null); return; }
    const d = (x) => esc(niceDate(this.dayOf(x), {day: "numeric", month: "short", year: "numeric"}));
    const sign = (v, k = 1) => (v > 0 ? "+" : "") + fmt(v, k), wk = s.ridden / 7;
    const rr = S.plan?.ramp_range, word = s.ramp == null || !rr ? "" : s.ramp > rr.hi ? " · faster" : s.ramp < rr.lo && s.ramp > 0 ? " · slower" : "";
    this.selPanel.draw(`<b>${d(s.start)} – ${d(s.end)}</b> <span class="muted">(${s.n} days${s.planned ? ", partly planned" : ""})</span>`, `<div class="tiles">
      ${tile("Ramp rate", s.ramp == null ? "–" : sign(s.ramp), "CTL/wk", `load ${fmt(s.ctl0)} → ${fmt(s.ctl1)}`, "",
        "How fast your load (CTL) rose or fell over the stretch, per week.\n• Your range (Plan page): ramps that brought gains without more illness\n• Else the usual 1.5–5 a week")}
      ${tile("Steepest week", s.steep == null ? "–" : sign(s.steep), "CTL", rr ? `your range ${fmt(rr.lo, 1)}–${fmt(rr.hi, 1)}/wk${word}` : "", "",
        "The biggest rise in load over any 7 days of the stretch.")}
      ${tile("Load (CTL)", fmt(s.ctl), "avg", `peak ${fmt(s.ctlMax)}`)}
      ${tile("Fatigue (ATL)", fmt(s.atl), "avg", `peak ${fmt(s.atlMax)}`)}
      ${tile("Form (TSB)", sign(s.tsb, 0), "avg", `lowest ${sign(s.tsbMin, 0)}`)}
      ${s.ridden && this.days.tss ? tile("Training", fmt(s.tss / wk), "TSS/wk", `${fmt(s.hours / wk, 1)} h/wk · ${fmt(s.tss)} TSS${s.planned ? " ridden" : ""}`) : ""}
    </div>`);
  }

  setMarking(on) {
    if (!on && this.marked) {                     // done marking: learn once from it all
      api.action({action: "periods_done", gain: this.marked.gain},
        this.marked.gain ? "Re-analysing with your periods…" : "Re-learning your markers from your labels…");
      this.marked = null;
    }
    if (on && !this.marking)                      // leaving the page ends marking too
      addEventListener("hashchange", () => { if (this.marking) this.setMarking(false); }, {once: true});
    this.marking = on;
    if (!on) this.editing = this.pmcSel.range = null;
    $("#pmcmark")?.classList.toggle("on", on);
    this.chartPMC(null);
  }

  html() {
    const p = S.power || {}, t = S.today || {}, ins = S.insights || {};
    const per = (rows) => rows.map((q) => `<tr><td>${esc(q.start)} → ${esc(q.end)}${q.yours ? ' <span class="chip">✎ yours</span>' : ""}</td><td class="n">${q.change_pct > 0 ? "+" : ""}${fmt(q.change_pct, 1)} %</td><td class="n">${fmt(q.hours_wk, 1)}</td><td>${esc(q.what_was_different || "")}</td></tr>`).join("");
    const builds = (ins.builds || []).map((b) => `<tr><td>${esc(b.label)}</td><td class="n">${fmt(b.hours_wk, 1)}</td><td class="n">${fmt(b.ctl_end)}</td><td class="n">${fmt(b.max_ramp_4wk, 1)}</td><td class="n">${fmt(b.quality_wk, 1)}</td><td class="n">${fmt(b.sleep_h, 1)}</td></tr>`).join("");
    return `
  <h1>Fitness &amp; power</h1>
  <p class="sub">What you did: training load, FTP over the years and your power curve. What you can do now: <a href="#rider">Rider</a>.</p>
  <div class="tiles">
    ${tile("FTP now", fmt(p.ftp_now), "W", `${fmt(p.wkg_now, 2)} W/kg`)}
    ${tile("Load (CTL)", fmt(t.ctl), "", `peak ${fmt(S.load?.ctl_peak)}`)}
    ${tile("Fatigue (ATL)", fmt(t.atl), "", t.atl != null ? `about ${fmt(t.atl)} TSS a day lately` : "")}
    ${tile("Form (TSB)", (t.tsb > 0 ? "+" : "") + fmt(t.tsb), "", this.formWord(t.tsb))}
  </div>
  <h2>Load, fatigue and form${hint("• Load (CTL): training load over ~6 weeks · Fatigue (ATL): over ~1 week\n• Form (TSB) = load − fatigue: negative while building, positive when rested\n• A ride's load is TSS (an hour all-out = 100)\n• Shaded: breaks (illness, travel, time off) · dashed: the plan\n• Click a day: its ride · a break: say what it was · hover: titles and notes\n• Drag across a chart: that stretch's numbers and ramp rate; double-click clears it\nMark periods:\n• Drag across the load chart to mark days (illness, travel, a gain …)\n• Click one of yours (✎) to change it; drag its edges to move it\n• Picking a found break as it is confirms it (✓)\n• Switching off (or leaving) learns from them once: illness / overreaching markers, Biggest gains\nZoom:\n• Ctrl + wheel, pinch, Zoom to it on a stretch, or the ends of the bar under the charts\n• Zoomed in: Shift + drag, Shift + wheel or the bar moves sideways; All shows it all")}</h2>
  <div class="card">
    <div class="row"><div class="legend"><span><i style="background:var(--s1)"></i>Load (CTL)</span><span><i style="background:var(--s2)"></i>Fatigue (ATL)</span><span><i class="dash" style="border-color:var(--muted)"></i>Planned (if you ride the plan)</span></div><span class="spacer"></span>
    <div class="seg"><button id="pmcgains" class="${this.showGains.value ? "on" : ""}" title="Shade the stretches where your FTP rose 4 % or more">Gains</button><button id="pmcmark" class="${this.marking ? "on" : ""}" title="Drag across the load chart to mark a period and say what it was">Mark periods</button></div>
    <div class="seg" id="pmcrange"><button data-d="90">3 m</button><button data-d="180" class="on">6 m</button><button data-d="365">1 y</button><button data-d="0">All</button></div></div>
    <div class="selgrid"><div>
    <div id="pmc" class="chart"></div>
    <div class="legend"><span><i class="band" style="background:var(--s3)"></i>Form (TSB): above 0 fresh, below 0 tired</span></div>
    <div id="pmcForm" class="chart xs"></div>
    </div><aside id="pmcsel" class="selpanel"></aside></div>
    <div class="legend" style="margin-top:6px"><span id="pmcGainsKey" ${this.showGains.value ? "" : "hidden"}><i class="band" style="background:var(--good)"></i>Gains (FTP up 4 %+)</span>
      <span><i class="band" style="background:var(--s2)"></i>Training breaks found, by cause (✓ = you confirmed it)</span><span><i class="band" style="background:var(--s1)"></i>✎ Periods you marked (stronger)</span></div>
  </div>
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><h3>FTP over the years${hint("• FTP: about your one-hour power (your setting, else from rides and tests)\n• Dots: your best test-like effort each week\n" + ZOOM_HELP)}</h3><div class="legend"><span><i style="background:var(--s1)"></i>FTP</span><span><i class="dot" style="background:var(--s2)"></i>Best test that week</span></div><div id="ftph" class="chart sm"></div></div>
    <div class="card"><h3>Power-duration curve${hint("Your best power for every duration, 5 s to hours.\n• Blue: last 90 days · grey: all-time best\n• Blue far below grey: the most to regain")}</h3><div class="legend"><span><i style="background:var(--s1)"></i>Last 90 days</span><span><i style="background:var(--muted)"></i>All-time best</span></div><div id="pdc" class="chart sm"></div></div>
  </div>
  ${(ins.progress || []).length ? `<details class="fold" id="fitGains"><summary>Biggest gains</summary>
  <div class="card"><div class="tw"><table><thead><tr><th>Period</th><th class="n">Change</th><th class="n">h/wk</th><th>What was different</th></tr></thead><tbody>${per(ins.progress)}</tbody></table></div></div></details>` : ""}
  ${builds ? `<details class="fold" id="fitBuilds"><summary>Your builds compared</summary>
  <div class="card"><div class="tw"><table><thead><tr><th>Build</th><th class="n">h/wk</th><th class="n">CTL</th><th class="n">Max ramp</th><th class="n">Quality/wk</th><th class="n">Sleep h</th></tr></thead><tbody>${builds}</tbody></table></div></div></details>` : ""}
  <p class="small muted" style="margin-top:16px">Your rides are on the <a href="#calendar">Calendar</a>, or search them all in <a href="#rides">All rides</a>.
    The full analysis dashboard is still here: <a href="/dashboard" target="_blank">open dashboard.html</a>. Where the methods come from: <a href="#sources">Sources</a>.</p>`;
  }

  chartPMC(days) {   // your whole history, then the plan's forecast (dashed); shows the last `days` (0 = all;
                              // none given: where you were, else the last 6 months). Form in its own chart under it, same axis
    const s = S.pmc || S.series; if (!s) return;
    const p = S.plan?.projection || {date: []}, d = s.date, last = d[d.length - 1];
    const fut = p.date.map((x, i) => [x, i]).filter(([x]) => x > last);
    const pad = (arr) => arr.concat(fut.map(() => null));
    const ahead = (k) => d.map((_, i) => (i === d.length - 1 ? s[k][i] : null)).concat(fut.map(([, i]) => p[k][i]));
    const sign = (v) => (v > 0 ? "+" : "") + fmt(v);
    const ms = (x) => new Date(x + "T12:00:00").getTime(), x = d.concat(fut.map(([q]) => q));
    TimeWindow.of("pmc").showLast(days, ms(last), ms(x[x.length - 1]), 180);
    if (!dips.drops.data && S.markers) dips.drops.load();               // the breaks (with their causes) shade the chart once loaded
    const day = 864e5, drops = dips.drops.data?.dips || [], mine = dips.drops.data?.periods || [];
    const gains = this.showGains.value ? S.insights?.gains || [] : [];
    const inside = (p, xv) => this.span(p)[0] <= xv && xv <= this.span(p)[1];
    const bands = [
      ...gains.map((g) => [ms(g.start), ms(g.end), css("--good"), `+${fmt(g.change_pct, 1)} %`]),
      ...drops.map((q) => [ms(q.start) - day / 2, ms(q.end) + day / 2, css((CAUSE[q.cause] || ["", "--muted"])[1]),
        (CAUSE[q.cause] || [q.cause])[0].split(" /")[0] + (q.label ? " ✓" : "")]),        // ✓: you confirmed it
      ...mine.map((p) => [...this.span(p), css((PERIOD[p.cause] || ["", "--muted"])[1]), "✎ " + (PERIOD[p.cause] || [p.cause])[0].split(" /")[0], 0.32, true])];
    const pick = (xv) => {                            // a click in a period of yours: change it; in a break: what was it
      const p = mine.find((q) => inside(q, xv));      // (marking: copy it as yours); on a day with a ride: open it
      if (p) { this.pickPeriod(this.span(p), p); return; }
      const q = drops.find((r) => inside(r, xv)), g = gains.find((r) => ms(r.start) <= xv && xv <= ms(r.end));
      if (this.marking && (q || g)) { this.pickPeriod(q ? this.span(q) : [ms(g.start) - day / 2, ms(g.end) + day / 2], null, q ? q.cause : "gain"); return; }
      if (q) { dips.dropMenu(q); return; }
      let k = 0;
      d.forEach((q, i) => { if (Math.abs(ms(q) - xv) < Math.abs(ms(d[k]) - xv)) k = i; });
      const id = s.ride?.[k];
      if (id && Math.abs(ms(d[k]) - xv) < 864e5) location.hash = "#ride/" + encodeURIComponent(id);
    };
    const kinds = new Map(d.map((q, i) => [ms(q), s.kind?.[i]]));     // day -> its main ride's workout type
    const dayIx = new Map(d.map((q, i) => [ms(q), i]));               // day -> its row (titles, notes)
    const plan = {};                                                    // a planned day -> its biggest session's type
    for (const q of S.plan?.sessions || []) {
      if (q.type === "rest" || q.date <= last) continue;
      if (!plan[q.date] || (q.tss || 0) > plan[q.date].tss) plan[q.date] = {type: q.type, tss: q.tss || 0};
    }
    for (const [q] of fut) if (plan[q]) kinds.set(ms(q), "planned:" + plan[q].type);
    const vline = fut.length ? ms(last) : null, planFrom = ms(last);
    EChart.draw("pmc", {x, xType: "date", group: "pmc", xAxis: false, yFmt: (v) => v, vline, onPick: pick, planFrom, bands, bandOpacity: 0.13,
      select: this.pmcSel,
      tipNote: (xv) => {                              // the day's workout type, its activities' titles and notes
        const k = kinds.get(xv), i = dayIx.get(xv), note = s.note?.[i];
        const [later, t] = (k || "").startsWith("planned:") ? [true, k.slice(8)] : [false, k];
        return (k ? `<div><span class="kchip${later ? " plan" : ""}" style="--c:${typeColor(t)}">${esc(TYPE_LABEL[t] || t)}</span></div>` : "")
          + (s.title?.[i] ? `<div class="tipnote"><b>${esc(s.title[i])}</b></div>` : "")
          + (note ? `<div class="tipnote muted">${esc(note.length > 280 ? note.slice(0, 280) + "…" : note)}</div>` : "");
      },
      series: [
      {kind: "line", y: pad(s.ctl), color: css("--s1"), name: "Load (CTL)"},
      {kind: "line", y: pad(s.atl), color: css("--s2"), name: "Fatigue (ATL)"},
      {kind: "line", y: ahead("ctl"), color: css("--s1"), dash: true, name: "Load, planned", tipName: "Load (CTL)", planned: true},
      {kind: "line", y: ahead("atl"), color: css("--s2"), dash: true, name: "Fatigue, planned", tipName: "Fatigue (ATL)", planned: true}]});
    EChart.draw("pmcForm", {x, xType: "date", group: "pmc", scrollbar: true, select: this.pmcSel, yFmt: (v) => v, vline, onPick: pick, planFrom, series: [
      {kind: "bar", y: pad(s.tsb), color: css("--s3"), opacity: 0.6, name: "Form (TSB)", fmt: sign},
      {kind: "bar", y: ahead("tsb"), color: css("--s3"), opacity: 0.25, name: "Form, planned", tipName: "Form (TSB)", planned: true, fmt: sign}]});
    const both = (k) => { const f = ahead(k); return pad(s[k]).map((v, i) => v ?? f[i]); };
    this.days = {x: x.map(ms), ctl: both("ctl"), atl: both("atl"), tsb: both("tsb"), tss: s.tss, hours: s.hours, last: ms(last)};
    this.drawStats();
  }

  chartFTP() {
    const h = S.history; if (!h) return;
    const W = (v) => fmt(v) + " W";
    EChart.draw("ftph", {x: h.date, xType: "date", yFmt: (v) => v + " W", series: [
      {kind: "line", y: h.ftp, color: css("--s1"), name: "FTP", fmt: W},
      {kind: "dots", y: h.test, color: css("--s2"), name: "Best test", fmt: W}]});
  }

  chartPDC() {
    const p = S.power || {}; if (!p.bests_90d) return;
    const pts = (o) => Object.entries(o || {}).map(([k, v]) => [Number(k), v && v.w]).filter(([k, w]) => w && k >= 5).sort((a, b) => a[0] - b[0]);
    const a = pts(p.bests_90d), b = pts(p.bests_all);
    const lab = (s) => s < 60 ? `${s} s` : s < 3600 ? `${Math.round(s / 60)} min` : hhmm(s / 60);
    EChart.draw("pdc", {xType: "log", yFmt: (v) => v + " W", xLabel: lab,
      xTicks: [[5, "5s"], [15, "15s"], [60, "1m"], [300, "5m"], [1200, "20m"], [3600, "1h"], [7200, "2h"]],
      series: [
        {kind: "line", x: b.map((q) => q[0]), y: b.map((q) => q[1]), color: css("--muted"), dash: true, name: "All-time", fmt: (v) => fmt(v) + " W"},
        {kind: "line", x: a.map((q) => q[0]), y: a.map((q) => q[1]), color: css("--s1"), name: "90 days", fmt: (v) => fmt(v) + " W"}]});
  }

  wire() {
    requestAnimationFrame(() => { this.chartPMC(); this.chartFTP(); this.chartPDC(); });
    new Segmented("#pmcrange", (b) => this.chartPMC(Number(b.dataset.d)));
    $("#pmcgains")?.addEventListener("click", (e) => {
      this.showGains.set(!this.showGains.value);
      e.target.classList.toggle("on", this.showGains.value);
      $("#pmcGainsKey").hidden = !this.showGains.value;
      this.chartPMC(null);
    });
    $("#pmcmark")?.addEventListener("click", () => { this.pmcSel.range = null; this.setMarking(!this.marking); });
  }
}
export const fitnessPage = new FitnessPage();
