// Training breaks (DipsSection): the list, causes, profiles; shown on Recovery (folded) and shaded on Fitness.
import {Lazy} from "./controls.js";
import {api, S} from "./api.js";
import {boxSigma, css, esc, expSmooth, fmt} from "./util.js";
import {niceDate} from "./widgets.js";
import {router} from "./main.js";
import {fitnessPage} from "./pages/fitness.js";
import {EChart} from "./charts.js";

// training breaks: cause -> [label, colour] (models.dips CAUSES)
export const CAUSE = {illness: ["Illness", "--s2"], overreaching: ["Overreaching / burnout", "--s7"], travel: ["Travel", "--s1"],
  life: ["Life / holiday", "--muted"], injury: ["Injury", "--s4"], planned: ["Planned recovery", "--ink2"], unclear: ["Unclear", "--line"]};
// what a period you mark on the load chart can be (models.dips BreakLabels.KINDS, "unclear" left out)
export const PERIOD = {...Object.fromEntries(Object.entries(CAUSE).filter(([k]) => k !== "unclear")), gain: ["Gain", "--good"]};
// illness markers
export const LEVEL = {clear: ["green", "Clear"], watch: ["amber", "Watch"], warning: ["red", "Illness warning"]};

class DipsSection {
  drops = new Lazy("/api/dips", () => { if (["recovery", "fitness"].includes(router.route())) router.render(); });
  dipFilter = "";
  dipsOpen = false;   // the list of breaks starts folded

  // a training break clicked on the load chart: pick what it was (your label wins and is what the markers learn from)
  lastPointer = {x: 0, y: 0};

  // a small menu at the last pointer: `pick(button)` on a click; a click outside or Escape closes it
  menu(html, pick) {
    document.querySelector(".popmenu")?.remove();
    const box = document.body.appendChild(document.createElement("div"));
    box.className = "popmenu";
    box.innerHTML = html;
    box.style.left = Math.min(this.lastPointer.x + 8, innerWidth - box.offsetWidth - 8) + "px";
    box.style.top = Math.min(this.lastPointer.y + 8, innerHeight - box.offsetHeight - 8) + "px";
    const close = (e) => { if (!e || e.key === "Escape" || (e.type === "pointerdown" && !box.contains(e.target))) {
      box.remove(); document.removeEventListener("pointerdown", close, true); document.removeEventListener("keydown", close); } };
    setTimeout(() => { document.addEventListener("pointerdown", close, true); document.addEventListener("keydown", close); });
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { close(); pick(b); }));
  }

  dropMenu(q) {
    const auto = (CAUSE[q.auto_cause] || [q.auto_cause])[0];
    this.menu(`<div class="th">Training break · ${esc(niceDate(q.start, {day: "numeric", month: "short", year: "numeric"}))} – ${esc(niceDate(q.end, {day: "numeric", month: "short", year: "numeric"}))}</div>
    <div class="small muted">${q.days} days off · load −${fmt(q.ctl_drop_pct)} %. What was it?</div>
    ${Object.entries(CAUSE).map(([k, [lab, col]]) => `<button data-v="${k}" class="${q.label === k ? "on" : ""}"><i style="background:var(${col})"></i>${esc(lab)}</button>`).join("")}
    <button data-v="" class="${q.label ? "" : "on"}"><i style="background:var(--line)"></i>Automatic guess: ${esc(auto)}</button>`, (b) => {
      q.label = b.dataset.v || null; q.cause = q.label || q.auto_cause;      // shown straight away; the server re-learns
      api.action({action: "label_dip", start: q.start, value: b.dataset.v}, "Saved – re-learning your markers…");
      fitnessPage.chartPMC(null);
    });
  }

  causeChip(c, conf) {
    const [lab, col] = CAUSE[c] || [c, "--line"];
    return `<span class="cchip"><i style="background:var(${col})"></i>${esc(lab)}${conf ? ` <span class="muted">· ${esc(conf)}</span>` : ""}</span>`;
  }

  html() {
    if (!S.markers) return "";
    if (!this.drops.data) { this.drops.load(); return `<div class="card muted small">Loading…</div>`; }
    const list = (this.drops.data.dips || []).slice().reverse();
    if (!list.length) return `<div class="card small">No breaks long enough to cost fitness were found. Press "Analyze again" under "How your markers were chosen" after new data.</div>`;
    const P = this.drops.data.profile || {}, causes = Object.keys(P.causes || {});
    const counts = list.reduce((o, d) => ((o[d.cause] = (o[d.cause] || 0) + 1), o), {});
    const legend = causes.map((c) => `<span><i style="background:var(${(CAUSE[c] || ["", "--line"])[1]})"></i>${esc((CAUSE[c] || [c])[0])} (${P.causes[c].n})</span>`).join("");
    const opts = (d) => `<select data-dip="${esc(d.start)}"><option value="">${d.label ? "Use automatic guess" : "Automatic: " + esc((CAUSE[d.auto_cause] || [d.auto_cause])[0])}</option>${
      Object.entries(CAUSE).map(([k, [lab]]) => `<option value="${k}" ${d.label === k ? "selected" : ""}>${esc(lab)}</option>`).join("")}</select>`;
    const shown = list.filter((d) => !this.dipFilter || d.cause === this.dipFilter);
    const rows = shown.map((d) => `
    <details class="day dip" data-start="${esc(d.start)}">
      <summary>
        <div class="dd"><span>${new Date(d.start + "T12:00:00").toLocaleDateString(undefined, {month: "short"})}</span><b>${new Date(d.start + "T12:00:00").getDate()}</b><span>${d.start.slice(0, 4)}</span></div>
        <div class="tt"><div>${this.causeChip(d.cause, d.label ? "your label" : d.confidence + " confidence")}</div>
          <div class="meta">${d.days} days off · load ${fmt(d.ctl_before)} → ${fmt(d.ctl_min)} (−${fmt(d.ctl_drop_pct)} %)</div></div>
        <span></span>
      </summary>
      <div class="body">
        <ul class="reasons" style="margin-top:0">${(d.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
        <div class="grid g2" style="margin-top:10px">
          <div><div class="legend"><span><i style="background:var(--s2)"></i>Resting HR vs baseline</span></div><div class="chart xs" id="dr_${esc(d.start)}"></div></div>
          <div><div class="legend"><span><i style="background:var(--s1)"></i>HRV vs baseline</span></div><div class="chart xs" id="dh_${esc(d.start)}"></div></div>
          <div><div class="legend"><span><i style="background:var(--s7)"></i>Sleep vs usual</span></div><div class="chart xs" id="ds_${esc(d.start)}"></div></div>
          <div><div class="legend"><span><i style="background:var(--s3)"></i>Training load, last 7 days</span></div><div class="chart xs" id="dl_${esc(d.start)}"></div></div>
        </div>
        <p class="small muted" style="margin:6px 0 0">Four weeks either side of the start (day 0); the shaded part is the break.</p>
        <div class="row" style="margin-top:10px"><span class="small">What was it?</span>${opts(d)}</div>
      </div>
    </details>`).join("");
    const filt = `<div class="seg small" id="dipfilt"><button data-c="" class="${this.dipFilter ? "" : "on"}">All ${list.length}</button>${Object.keys(counts).map((c) => `<button data-c="${c}" class="${this.dipFilter === c ? "on" : ""}">${esc((CAUSE[c] || [c])[0].split(" /")[0])} ${counts[c]}</button>`).join("")}</div>`;
    return `<p class="small muted" id="drops">Every break long enough to cost fitness, with the 4 weeks before and after it. What the app already knows comes first (days away, a planned recovery phase, a day you marked yourself sick); otherwise each gets an automatic guess from what your body and training did around it (resting HR / HRV / breathing rate as training stopped, heavy load with body signs before, sleep on another clock and lots of steps for travel), or "unclear" when there are no body readings. Correct any that are wrong: your labels are what the markers learn from.</p>
  <div class="card"><h3>What your breaks look like, by cause</h3>
    <div class="legend">${legend}</div>
    <div class="grid g2" style="margin-top:6px">
      <div><div class="small muted">Resting HR vs baseline (bpm)</div><div class="chart xs" id="pf_rhr_d"></div></div>
      <div><div class="small muted">HRV vs baseline (SD)</div><div class="chart xs" id="pf_hrv_z"></div></div>
      <div><div class="small muted">Sleep vs usual (h)</div><div class="chart xs" id="pf_sleep_d"></div></div>
      <div><div class="small muted">Training load, last 7 days (TSS)</div><div class="chart xs" id="pf_load7"></div></div>
    </div>
    <p class="small muted" style="margin:6px 0 0">Average of all breaks of each cause (5-day smoothed), lined up on the day training stopped (0). Days before 0 are the warning signs; after 0, how you came back.</p>
  </div>
  <details class="card dipslist" id="dipslist" ${this.dipsOpen ? "open" : ""}>
    <summary><b>All ${list.length} training breaks</b> <span class="small muted">${Object.entries(counts).map(([c, k]) => `${esc((CAUSE[c] || [c])[0].split(" /")[0])} ${k}`).join(" · ")}</span></summary>
    <div class="row" style="margin:12px 0 8px">${filt}</div>
    <div class="daylist">${rows}</div>
  </details>`;
  }

  drawDipProfiles() {
    const P = this.drops.data && this.drops.data.profile; if (!P || !P.causes) return;
    const lab = (o) => (o === 0 ? "day 0" : `${o > 0 ? "+" : ""}${o} d`);
    for (const m of ["rhr_d", "hrv_z", "sleep_d", "load7"]) {
      EChart.draw("pf_" + m, {x: P.offset, xType: "num", xTickFmt: (v) => `${v > 0 ? "+" : ""}${v}`, xLabel: lab, vline: 0, hline: m === "load7" ? null : 0,
        group: "pf", yFmt: (v) => v, series: Object.entries(P.causes).map(([c, v]) => ({kind: "line", y: m === "load7" ? v[m] : expSmooth(v[m], boxSigma(5)), color: css((CAUSE[c] || ["", "--line"])[1]),
          name: (CAUSE[c] || [c])[0], fmt: (q) => fmt(q, m === "load7" ? 0 : 1), width: 2, connect: true}))});
    }
  }

  drawDip(start) {
    const d = (this.drops.data.dips || []).find((x) => x.start === start); if (!d || !d.around) return;
    const off = this.drops.data.profile.offset, lab = (o) => (o === 0 ? "day 0" : `${o > 0 ? "+" : ""}${o} d`);
    const common = {x: off, xType: "num", xTickFmt: (v) => `${v > 0 ? "+" : ""}${v}`, xLabel: lab, vline: 0, shade: [0, d.days - 1], group: "dip_" + start};
    EChart.draw("dr_" + start, {...common, hline: 0, series: [{kind: "line", y: d.around.rhr_d, color: css("--s2"), name: "Resting HR", fmt: (v) => (v > 0 ? "+" : "") + fmt(v, 1) + " bpm", markers: true}]});
    EChart.draw("dh_" + start, {...common, hline: 0, series: [{kind: "line", y: d.around.hrv_z, color: css("--s1"), name: "HRV", fmt: (v) => fmt(v, 1) + " SD", markers: true}]});
    EChart.draw("ds_" + start, {...common, hline: 0, series: [{kind: "bar", y: d.around.sleep_d, color: css("--s7"), name: "Sleep", fmt: (v) => (v > 0 ? "+" : "") + fmt(v, 1) + " h"}]});
    EChart.draw("dl_" + start, {...common, zero: true, series: [{kind: "line", y: d.around.load7, color: css("--s3"), name: "7-day load", fmt: (v) => fmt(v) + " TSS", area: true}]});
  }
}
export const dips = new DipsSection();

for (const t of ["pointerdown", "pointerup"])          // a menu opens where you clicked or ended a drag
  document.addEventListener(t, (e) => { dips.lastPointer = {x: e.clientX, y: e.clientY}; }, true);
