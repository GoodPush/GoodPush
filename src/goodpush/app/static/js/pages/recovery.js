// Recovery: how your body is doing - illness / overreaching watch, HRV, resting HR, sleep and warm-up alpha1 on one
// time axis, then the history folded: training breaks and how the markers were chosen (RecoveryPage).
import {Segmented} from "../controls.js";
import {Page} from "./page.js";
import {api, S} from "../api.js";
import {$, css, fmt, hhmm} from "../util.js";
import {hint} from "../widgets.js";
import {router} from "../main.js";
import {dips} from "../dips.js";
import {markers} from "../markers.js";
import {ZOOM_HELP} from "./fitness.js";
import {drawChart, tail} from "../charts.js";

const NIGHTS = 7;                 // the HRV chart's bold line: the mean of this many nights' medians

class RecoveryPage extends Page {
  breaksOpen = false;             // the history folds keep their state while the page is redrawn
  statsOpen = false;

  html() {
    const alpha = S.alpha && S.alpha.rides?.length;
    const stats = markers.stats() + markers.stats("overreaching");
    return `
  <h1>Recovery</h1>
  <p class="sub">How your body is doing: the illness and overreaching watch, overnight HRV, resting heart rate, sleep${alpha ? " and warm-up alpha1" : ""}, and your training breaks.</p>
  ${markers.status()}
  ${markers.status("overreaching")}
  <h2>Body signs${hint("The charts share one time axis: zoom or scroll one, the others follow.\n" + ZOOM_HELP)}</h2>
  <div class="card"><h3>Overnight HRV</h3><div class="legend"><span><i style="background:var(--s1);height:3px"></i>${NIGHTS}-night mean</span><span><i style="background:var(--s1)"></i>Each night (median of its 5-min readings)</span><span><i class="band" style="background:var(--s1)"></i>Middle half of the night's readings</span><span><i class="band" style="background:var(--muted)"></i>Your normal range</span></div><div id="hrv" class="chart sm"></div></div>
  <div class="grid g2" style="margin-top:12px">
    <div class="card"><h3>Resting heart rate</h3><div class="legend"><span><i style="background:var(--s2)"></i>Resting HR</span><span><i style="background:var(--muted)"></i>6-week baseline</span></div><div id="rhr" class="chart sm"></div></div>
    <div class="card"><h3>Sleep</h3><div class="legend"><span><i style="background:var(--s7)"></i>Hours asleep</span><span><i style="background:var(--muted)"></i>Your usual</span></div><div id="sleep" class="chart sm"></div></div>
  </div>
  ${alpha ? `<div class="card" style="margin-top:12px"><h3>Warm-up alpha1 vs your normal${hint("DFA alpha1 (chest strap, e.g. the alphaHRV field) in each warm-up, against your normal at that power (in SD).\n• Below the band: low (fatigue, illness coming on)\n• Above: unusually high\n• Whether it warns for you: How your markers were chosen")}</h3><div class="legend"><span><i class="dot" style="background:var(--s7)"></i>Each ride (SD from normal at that power)</span><span><i class="band" style="background:var(--muted)"></i>±2 SD</span></div><div id="a1z" class="chart sm"></div></div>` : ""}
  ${S.markers ? `<h2>History</h2>
  <details class="fold" id="recBreaks" ${this.breaksOpen ? "open" : ""}><summary>Training breaks</summary>${dips.html()}</details>
  ${stats ? `<details class="fold" id="recStats" ${this.statsOpen ? "open" : ""}><summary>How your markers were chosen</summary>${stats}</details>` : ""}` : ""}`;
  }

  // the trailing mean of the last n non-empty values (nights without a reading don't count), per day
  trailingMean(arr, n) {
    return arr.map((v, i) => {
      const w = arr.slice(Math.max(0, i - n + 1), i + 1).filter((x) => x != null);
      return v == null || !w.length ? null : w.reduce((a, b) => a + b, 0) / w.length;
    });
  }

  chartRecovery() {
    const s = S.series; if (!s) return;
    const n = s.date.length, d = s.date, group = "recovery";   // all of it, opening on 6 months
    const view = n ? [new Date(d[Math.max(0, n - 182)] + "T12:00:00").getTime(), new Date(d[n - 1] + "T12:00:00").getTime()] : null;
    const q = (k) => tail(s[k] || [], n), ms = (v) => fmt(v) + " ms", c = css("--s1");
    drawChart("hrv", {x: d, xType: "date", group, view, yFmt: (v) => v, series: [
      {kind: "band", y: q("hrv_base_low"), y2: q("hrv_base_high"), color: css("--muted")},
      {kind: "band", y: q("hrv_q25"), y2: q("hrv_q75"), color: c},
      {kind: "line", y: this.trailingMean(q("hrv_median"), NIGHTS), color: c, width: 2.6, name: `${NIGHTS}-night mean`, fmt: ms},
      {kind: "line", y: q("hrv_median"), color: c, width: 1, name: "That night", fmt: ms}]});
    drawChart("rhr", {x: d, xType: "date", group, view, yFmt: (v) => v, series: [
      {kind: "line", y: tail(s.rhr_base, n), color: css("--muted"), dash: true, name: "Baseline", fmt: (v) => fmt(v) + " bpm"},
      {kind: "line", y: tail(s.rhr, n), color: css("--s2"), name: "Resting HR", markers: true, fmt: (v) => fmt(v) + " bpm"}]});
    const R = (S.alpha && S.alpha.rides) || [];
    drawChart("sleep", {x: d, xType: "date", group, view, scrollbar: !R.length, zero: true, yFmt: (v) => v + " h", series: [
      {kind: "bar", y: tail(s.sleep_h, n), color: css("--s7"), opacity: 0.75, name: "Asleep", fmt: (v) => hhmm(v * 60)},
      {kind: "line", y: tail(s.sleep_h_base, n), color: css("--muted"), dash: true, name: "Usual", fmt: (v) => hhmm(v * 60)}]});
    if (R.length && $("#a1z")) {
      const a = R.map((r) => r.date);
      drawChart("a1z", {x: a, xType: "date", group, view, scrollbar: true, yFmt: (v) => v, hline: 0, series: [
        {kind: "band", y: a.map(() => -2), y2: a.map(() => 2), color: css("--muted")},
        {kind: "dots", y: R.map((r) => r.a1_warm_z), color: css("--s7"), name: "Warm-up alpha1", fmt: (v) => (v > 0 ? "+" : "") + fmt(v, 1) + " SD"}]});
    }
  }

  wire() {
    requestAnimationFrame(() => { this.chartRecovery(); if (this.breaksOpen) dips.drawDipProfiles(); });
    $("#recBreaks")?.addEventListener("toggle", (e) => { this.breaksOpen = e.target.open; if (e.target.open) requestAnimationFrame(() => dips.drawDipProfiles()); });
    $("#recStats")?.addEventListener("toggle", (e) => { this.statsOpen = e.target.open; });
    document.querySelectorAll("details.dip").forEach((det) => det.addEventListener("toggle", () => { if (det.open) requestAnimationFrame(() => dips.drawDip(det.dataset.start)); }));
    document.querySelectorAll("select[data-dip]").forEach((sel) => sel.addEventListener("change", () => {
      api.action({action: "label_dip", start: sel.dataset.dip, value: sel.value}, "Saved – re-learning your markers…");
      dips.drops.forget();
    }));
    new Segmented("#dipfilt", (b) => { dips.dipFilter = b.dataset.c; router.render(); });
    $("#dipslist")?.addEventListener("toggle", (e) => { dips.dipsOpen = e.target.open; });
  }
}
export const recoveryPage = new RecoveryPage();
