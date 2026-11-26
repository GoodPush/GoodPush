// The SVG chart engine (drawChart, hover groups, selection) and TimeWindow (zoom / pan for every time axis).
import {css, esc, fmt} from "./util.js";
import {niceDate} from "./widgets.js";
import {router} from "./main.js";
import {rideMap} from "./ride_map.js";
import {calendarPage} from "./calendar/calendar.js";
import {init} from "../vendor/echarts.esm.min.js";

// charts (small SVG library, no CDN)
const SVGNS = "http://www.w3.org/2000/svg";
export const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const charts = {};   // id -> spec, redrawn on resize
const hoverGroups = {};   // charts that share a crosshair
export const selectGroups = {};  // charts that share a selection band
export function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  const t = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
  return t;
}
function dateTicks(t0, t1) {
  const days = (t1 - t0) / 864e5, out = [];
  const d = new Date(t0); d.setHours(12, 0, 0, 0);
  if (days > 900) {            // years
    for (let y = d.getFullYear() + 1; ; y++) { const t = new Date(y, 0, 1).getTime(); if (t > t1) break; out.push([t, String(y)]); }
    const step = Math.ceil(out.length / 7);
    return out.filter((_, i) => i % step === 0);
  }
  if (days > 75) {             // months
    d.setDate(1); d.setMonth(d.getMonth() + 1);
    const step = days > 400 ? 3 : days > 200 ? 2 : 1;
    while (d.getTime() <= t1) { out.push([d.getTime(), MON[d.getMonth()] + (d.getMonth() === 0 ? " " + String(d.getFullYear()).slice(2) : "")]); d.setMonth(d.getMonth() + step); }
    return out;
  }
  const step = days > 35 ? 14 : 7;   // weeks
  d.setDate(d.getDate() + ((8 - d.getDay()) % 7));
  while (d.getTime() <= t1) { out.push([d.getTime(), `${d.getDate()} ${MON[d.getMonth()]}`]); d.setDate(d.getDate() + step); }
  return out;
}
export function el(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
// A zoomable x range: one per chart, or one per group of charts on the same axis (they zoom together); all
// forgotten on a page change. extent = all the data, view = what's shown (null = all of it). Every time chart
// uses it - drawChart, the power-curve heatmap, the ride page (which fetches finer data for what it shows) - so
// zoom, pan, wheel, grab-and-drag and the scrollbar behave the same everywhere.
export class TimeWindow {
  static windows = {};           // key (chart id or group) -> window
  static dragging = null;        // a window being dragged (its scrollbar thumb, or the chart grabbed)

  static of(key) { return (TimeWindow.windows[key] ||= new TimeWindow(key)); }
  static forget(...keys) { for (const k of keys) delete TimeWindow.windows[k]; }
  static forgetAll() { TimeWindow.windows = {}; }

  constructor(key) {
    this.key = key;
    this.extent = null;          // [x0, x1] of all the data
    this.view = null;            // [x0, x1] shown, null = everything
    this.opened = false;         // its opening window (a chart's spec.view) was applied: "Reset zoom" then shows all
    this.redraw = null;          // for a chart drawn outside drawChart (the heatmap)
  }

  shown() { return this.view || this.extent; }

  // a range button (6 months / 1 y / All): the last `days` before `from` up to `to` (ms; 0 = all); undefined keeps
  // a zoom you made, else opens on `fallback` days. The extent is measured again by the next draw
  showLast(days, from, to, fallback) {
    this.extent = null;
    if (days === undefined) days = this.view ? null : fallback;
    if (days) this.view = [from - days * 864e5, to];
    else if (days === 0) this.view = null;
  }

  widen([a, b]) {                // the group's full extent (its charts may differ)
    this.extent = this.extent ? [Math.min(this.extent[0], a), Math.max(this.extent[1], b)] : [a, b];
  }

  zoomTo(r) {                    // r = [a, b] in x units, or null for everything
    const [e0, e1] = this.extent, span = r ? r[1] - r[0] : 0;
    if (!r || span >= e1 - e0) this.view = null;
    else if (span > (e1 - e0) / 1000) { const a = Math.max(e0, Math.min(e1 - span, r[0])); this.view = [a, a + span]; }
    this.changed();
  }

  zoomAt(at, f) {                // around x = at; f > 1 zooms out
    const [a, b] = this.shown();
    this.zoomTo([at - (at - a) * f, at + (b - at) * f]);
  }

  pan(frac) {                    // move a zoomed window by a fraction of its width
    const z = this.view;
    if (z) this.zoomTo([z[0] + frac * (z[1] - z[0]), z[1] + frac * (z[1] - z[0])]);
  }

  changed() {                    // redraw what shows it (or let it redraw itself, e.g. the ride page with finer data)
    const mine = Object.entries(charts).filter(([cid, sp]) => (sp.group || cid) === this.key);
    const hook = mine.find(([, sp]) => sp.onView);
    if (this.redraw) this.redraw();
    else if (hook) hook[1].onView(this.view);
    else mine.forEach(([cid, sp]) => { if (document.getElementById(cid)) drawChart(cid, sp); });
    Object.entries(EChart.live).filter(([cid, c]) => (c.spec.group || cid) === this.key).forEach(([cid, c]) => EChart.draw(cid, c.spec));
  }

  wheel(ev, at) {                // Ctrl + wheel / pinch zooms around x = at, Shift + wheel / a sideways swipe pans
    const pan = ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY);
    if (ev.ctrlKey || ev.metaKey) this.zoomAt(at, ev.deltaY > 0 ? 1.5 : 1 / 1.5);
    else if (pan && this.view) this.pan(0.15 * Math.sign(ev.shiftKey ? ev.deltaY : ev.deltaX));
    else return;                 // a plain wheel keeps scrolling the page
    ev.preventDefault();
  }

  // drag the window: `span` x units over `w` pixels (negative: grabbing the chart, it moves with the pointer).
  // The drag is followed on the window, so it survives the chart redrawing underneath it.
  grab(ev, span, w) {
    const [a, b] = this.shown();
    TimeWindow.dragging = {win: this, x: ev.clientX, a, b, span, w, d: 0, timer: 0};
  }

  static follow(ev) {            // redraw at most every 30 ms while dragging
    const g = TimeWindow.dragging;
    if (!g) return;
    g.d = (ev.clientX - g.x) / g.w * g.span;
    if (!g.timer) g.timer = setTimeout(() => { g.timer = 0; if (TimeWindow.dragging === g) g.win.zoomTo([g.a + g.d, g.b + g.d]); }, 30);
  }

  static drop() {                // let go: exactly where you dropped it
    const g = TimeWindow.dragging;
    TimeWindow.dragging = null;
    if (g) { clearTimeout(g.timer); if (g.d) g.win.zoomTo([g.a + g.d, g.b + g.d]); }
  }

  // a scrollbar under chart `id`: the thumb is the window shown; drag it, or click the track to jump there
  scrollbar(id, box, m, W) {
    box.parentNode.querySelector(`.cscroll[data-for="${id}"]`)?.remove();
    const e = this.extent, [a, b] = this.shown(), span = e[1] - e[0];
    const track = document.createElement("div"), thumb = document.createElement("div");
    track.className = "cscroll"; track.dataset.for = id;
    track.style.marginLeft = m.l + "px"; track.style.width = (W - m.l - m.r) + "px";
    thumb.className = "cthumb";
    thumb.style.left = ((a - e[0]) / span * 100) + "%"; thumb.style.width = Math.max((b - a) / span * 100, 2) + "%";
    track.appendChild(thumb);
    box.after(track);
    thumb.addEventListener("pointerdown", (ev) => { ev.preventDefault(); this.grab(ev, span, track.clientWidth); });
    track.addEventListener("pointerdown", (ev) => {
      if (ev.target !== track) return;
      const at = e[0] + (ev.clientX - track.getBoundingClientRect().left) / track.clientWidth * span;
      this.zoomTo([at - (b - a) / 2, at + (b - a) / 2]);
    });
  }
}
// A stretch selected on a chart group (spec.select = `sel` {range, onChange}): its numbers in an aside beside the
// charts (.selgrid > aside.selpanel), with Zoom to it / Clear. The page works the numbers out; draw(head, body)
// shows them (head null: the `empty` line)
export class SelectionPanel {
  constructor(id, group, sel, empty) { Object.assign(this, {id, group, sel, empty}); }

  draw(head, body = "") {
    const box = document.getElementById(this.id);
    if (!box) return;
    if (head == null || !this.sel.range) { box.innerHTML = `<div class="small muted">${this.empty}</div>`; return; }
    box.innerHTML = `<div class="selstats"><div>${head}</div>
    <div class="row" style="margin-top:6px"><button class="btn sm" data-sel="zoom">Zoom to it</button><button class="btn sm" data-sel="clear">Clear</button></div>${body}</div>`;
    box.querySelector('[data-sel="zoom"]').addEventListener("click", () => this.zoom());
    box.querySelector('[data-sel="clear"]').addEventListener("click", () => this.clear());
  }

  zoom() {
    const r = this.sel.range, pad = (r[1] - r[0]) * 0.15;
    TimeWindow.of(this.group).zoomTo([r[0] - pad, r[1] + pad]);
  }

  clear() {
    this.sel.range = null;
    (selectGroups[this.group] || []).forEach((h) => h.show(null));
    this.sel.onChange(null);
  }
}
// spec: {x:[...], xType:"date"|"log", series:[{y, kind:"line"|"bar"|"dots"|"band", y2 (band top), color, name, fmt, dash, x,
//   range (false: doesn't set the y axis), opacity (bands), hollow / alpha [per point] / lo, hi [per point bar] (dots)}], yUnit, zero}
// Date / number charts zoom (per chart, or together with their group): drag = zoom to that stretch,
// Ctrl + wheel / pinch = zoom, Shift + wheel / sideways swipe = pan, double-click or "Reset zoom" = all.
// Zoomed in, a drag pans and Shift + drag zooms; with select, drag selects and Shift + drag pans.
// Every zoomable chart: a scrollbar once zoomed (a group: on the chart with scrollbar: true), and a strip at the
// top for band labels and "Reset zoom". view [a, b]: the window it opens on (e.g. the last months of a long history).
// optional: yExtra [arrays on x] (also set the y axis, not drawn: e.g. series not shown, so it holds still), xAxis false (no x labels: for charts stacked above one that has them), xDomain [a, b] (the full x range, when it isn't just the data's), onView(range | null) (called
// instead of the redraw after each zoom), select {range, onChange(range)} (drag selects a stretch instead of zooming; double-click clears).
export function drawChart(id, spec) {
  const box = document.getElementById(id);
  if (!box) return;
  charts[id] = spec;
  box.innerHTML = "";
  box.style.position = "relative";
  const W = box.clientWidth || 600, H = box.clientHeight || 260;
  const labelled = (spec.bands || []).some((b) => b[3]);
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img"}, box);
  const xv = (spec.x || []).map((v) => spec.xType === "date" ? new Date(v + "T12:00:00").getTime() : v);
  const xs = (s) => s.x ? s.x : xv;
  const allX = spec.series.flatMap((s) => xs(s)).filter((v) => v != null);
  const zoomable = (spec.xType === "date" || spec.xType === "num") && allX.length > 1;
  const m = {l: 44, r: 10, t: zoomable || labelled ? 20 : 6, b: spec.xAxis === false ? 4 : 22};
  const zkey = spec.group || id;
  const tw = zoomable ? TimeWindow.of(zkey) : null;
  if (tw && spec.view && !tw.view && !tw.opened) { tw.view = spec.view; tw.opened = true; }
  if (tw) tw.widen(spec.xDomain || [Math.min(...allX), Math.max(...allX)]);
  const win = tw?.view || spec.xDomain || null;   // fixed x window, if any
  const seen = (x) => !win || (x != null && x >= win[0] && x <= win[1]);
  let all = [];
  for (const s of spec.series) {                // y range from what is visible (not from series with range: false)
    if (s.range === false) continue;
    const X = xs(s);
    all = all.concat((s.y || []).filter((v, i) => v != null && seen(X[i])));
    if (s.y2) all = all.concat(s.y2.filter((v, i) => v != null && seen(X[i])));
  }
  for (const y of spec.yExtra || []) all = all.concat(y.filter((v, i) => v != null && seen(xv[i])));
  if (!all.length) { box.innerHTML = '<div class="muted small" style="padding:30px 0;text-align:center">No data yet</div>'; return; }
  let y0 = Math.min(...all), y1 = Math.max(...all);
  if (spec.zero || spec.series.some((s) => s.kind === "bar")) { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
  const pad = (y1 - y0) * 0.06 || 1; if (!(spec.zero && y0 === 0)) y0 -= pad; y1 += pad;
  const yt = niceTicks(y0, y1, H < 260 ? 4 : 5); y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
  let x0 = Math.min(...allX), x1 = Math.max(...allX);
  if (win) [x0, x1] = win;
  const log = spec.xType === "log";
  const fx = log ? (v) => m.l + (Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0) || 1) * (W - m.l - m.r)
                 : (v) => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
  const fy = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  spec.yScale = {y0, y1, top: m.t, bottom: H - m.b, H};       // for a histogram beside it (sideHist)
  const ink2 = css("--ink2"), grid = css("--grid");
  // grid + y labels
  for (const t of yt) {
    el("line", {x1: m.l, x2: W - m.r, y1: fy(t), y2: fy(t), stroke: grid, "stroke-width": 1}, svg);
    const tx = el("text", {x: m.l - 6, y: fy(t) + 4, "text-anchor": "end", "font-size": 11, fill: ink2}, svg);
    tx.textContent = (spec.yFmt || ((v) => v))(t);
  }
  // x labels
  const xt = spec.xType === "date" ? dateTicks(x0, x1)
    : spec.xType === "num" && spec.xStep ? stepTicks(x0, x1, spec.xStep(x1 - x0, Math.max(4, Math.floor((W - m.l - m.r) / 90))), spec.xTickFmt)
    : spec.xType === "num" ? niceTicks(x0, x1, Math.max(4, Math.floor((W - m.l - m.r) / 90))).map((t) => [t, (spec.xTickFmt || ((v) => v))(t)])
    : (spec.xTicks || []);
  for (const [t, lab] of spec.xAxis === false ? [] : xt) {
    const x = fx(t); if (x < m.l - 1 || x > W - m.r + 1) continue;
    const tx = el("text", {x, y: H - 6, "text-anchor": "middle", "font-size": 11, fill: ink2}, svg);
    tx.textContent = lab;
  }
  // shaded x ranges: `shade` [from, to] (faint), `bands` [[from, to, colour, label?, opacity?, inside?], ...]
  // (intervals; gains and breaks), at bandOpacity unless their own; a label sits at the band's top left when it
  // fits, above the plot (inside: just inside its top, so two overlapping bands' labels don't overprint)
  const bandOp = spec.bandOpacity ?? 0.22;
  for (const [b0, b1, c, op, label, inside] of [...(spec.shade ? [[...spec.shade, css("--ink"), 0.06]] : []),
    ...(spec.bands || []).map(([a, b, col, lab, o, ins]) => [a, b, col, o ?? bandOp, lab, ins])]) {
    if (b1 < x0 || b0 > x1) continue;
    const [a0, a1] = [b0, b1].map((v) => fx(Math.max(x0, Math.min(x1, v))));
    el("rect", {x: Math.min(a0, a1), y: m.t, width: Math.max(Math.abs(a1 - a0), 2), height: H - m.t - m.b, fill: c, "fill-opacity": op}, svg);
    if (label && Math.abs(a1 - a0) >= 7 * label.length) {
      const t = el("text", {x: Math.min(a0, a1) + 4, y: inside ? m.t + 13 : m.t - 6, "font-size": 10.5, "font-weight": 600, fill: inside ? css("--ink") : c}, svg);
      t.textContent = label;
    }
  }
  if (spec.vline != null) el("line", {x1: fx(spec.vline), x2: fx(spec.vline), y1: m.t, y2: H - m.b, stroke: css("--ink2"), "stroke-width": 1, "stroke-dasharray": "2 3"}, svg);
  if (spec.hline != null && spec.hline >= y0 && spec.hline <= y1) el("line", {x1: m.l, x2: W - m.r, y1: fy(spec.hline), y2: fy(spec.hline), stroke: css("--ink2"), "stroke-width": 1}, svg);
  // marks (clipped to the plot area when the x window is fixed)
  const n = win ? xv.filter(seen).length : xv.length, bw = Math.max(1, (W - m.l - m.r) / Math.max(n, 1) * 0.8);
  let plot = svg;
  if (win) {
    el("rect", {x: m.l, y: 0, width: W - m.l - m.r, height: H}, el("clipPath", {id: `clip-${id}`}, el("defs", {}, svg)));
    plot = el("g", {"clip-path": `url(#clip-${id})`}, svg);
  }
  for (const s of spec.series) {
    const X = xs(s), col = s.color;
    if (s.kind === "band") {
      let d = "", seg = [];
      const fyc = (v) => fy(Math.max(y0, Math.min(y1, v)));          // cut at the plot's edges
      const flush = () => {
        if (seg.length > 1) d += "M" + seg.map((i) => `${fx(X[i]).toFixed(1)},${fyc(s.y[i]).toFixed(1)}`).join("L") + "L" +
          seg.slice().reverse().map((i) => `${fx(X[i]).toFixed(1)},${fyc(s.y2[i]).toFixed(1)}`).join("L") + "Z";
        seg = [];
      };
      X.forEach((_, i) => { if (s.y[i] != null && s.y2[i] != null) seg.push(i); else flush(); }); flush();
      el("path", {d, fill: col, "fill-opacity": s.opacity ?? 0.16, stroke: "none"}, plot);
    } else if (s.kind === "bar") {
      const z = fy(Math.max(0, y0));
      X.forEach((xx, i) => {
        const v = s.y[i]; if (v == null || !seen(xx)) return;
        const yv = fy(v), h = Math.abs(yv - z);
        el("rect", {x: fx(xx) - bw / 2, y: Math.min(yv, z), width: bw, height: Math.max(h, 0.5), fill: col, "fill-opacity": s.opacity ?? 0.85, rx: Math.min(2, bw / 3)}, plot);
      });
    } else if (s.kind === "dots") {             // r: a smaller dot for many points (no outline); hollow: rings;
      X.forEach((xx, i) => { const v = s.y[i]; if (v == null || !seen(xx)) return;   // alpha: each point's opacity; lo / hi: a bar each
        const op = s.alpha ? s.alpha[i] : s.r ? 0.75 : 1;
        if (s.lo && s.lo[i] != null) el("line", {x1: fx(xx), x2: fx(xx), y1: fy(s.lo[i]), y2: fy(s.hi[i]), stroke: col, "stroke-width": 1.2, "stroke-opacity": op}, plot);
        el("circle", s.hollow ? {cx: fx(xx), cy: fy(v), r: s.r || 4, fill: "none", stroke: col, "stroke-opacity": op, "stroke-width": 1.2}
          : {cx: fx(xx), cy: fy(v), r: s.r || 4, fill: col, "fill-opacity": op, stroke: s.r ? "none" : css("--surface"), "stroke-width": 1.5}, plot); });
    } else {
      let d = "", pen = false;
      X.forEach((xx, i) => { const v = s.y[i]; if (v == null) { if (!s.connect) pen = false; return; }
        d += (pen ? "L" : "M") + fx(xx).toFixed(1) + "," + fy(v).toFixed(1); pen = true; });
      if (s.area && d) {
        const base = fy(Math.max(y0, 0)).toFixed(1);
        const area = d.split("M").filter((q) => q.includes("L")).map((q) => {
          const pts = q.split("L"), first = pts[0].split(",")[0], last = pts[pts.length - 1].split(",")[0];
          return `M${first},${base}L${q}L${last},${base}Z`;
        }).join("");
        if (area) el("path", {d: area, fill: col, "fill-opacity": 0.14, stroke: "none"}, plot);
      }
      el("path", {d, fill: "none", stroke: col, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": s.dash ? "5 4" : "none"}, plot);
      if (s.markers && (W - m.l - m.r) / Math.max(X.length, 1) > 4) X.forEach((xx, i) => { const v = s.y[i]; if (v == null) return;
        el("circle", {cx: fx(xx), cy: fy(v), r: 2.6, fill: col}, plot); });
    }
  }
  // hover: crosshair + tooltip (mouse and touch)
  const cross = el("line", {y1: m.t, y2: H - m.b, stroke: css("--muted"), "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden"}, svg);
  const tip = document.createElement("div");
  tip.className = "tip"; tip.hidden = true; box.appendChild(tip);
  const nearest = (X, px) => { let bi = -1, bd = 1e9; X.forEach((xx, i) => { if (xx == null) return; const d = Math.abs(fx(xx) - px); if (d < bd) { bd = d; bi = i; } }); return bi; };
  const base = spec.series.find((q) => q.kind !== "band") || spec.series[0];
  const showAt = (px) => {
    if (px < m.l || px > W - m.r) { hide(); return; }
    const i = nearest(xs(base), px); if (i < 0) return;
    const xx = xs(base)[i], cx = fx(xx);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    const head = spec.xType === "date" ? niceDate(spec.x[i], {weekday: "short", day: "numeric", month: "short", year: "numeric"}) : (spec.xLabel ? spec.xLabel(xx) : xx);
    // planFrom: the last day of data; after it the day is "planned" (once, by the date) and the planned series
    // (planned: true; legend "X, planned") show as plain "X" - before it they're left out (they only join on)
    const ahead = spec.planFrom != null && xx > spec.planFrom;
    const shown = spec.series.filter((q) => q.name && q.kind !== "band" && (!q.planned || ahead));
    const lines = (spec.tipReverse ? shown.reverse() : shown).map((q) => {
      const j = q.x ? nearest(q.x, cx) : i; const v = q.y[j];
      const name = q.tipName || (q.planned ? q.name.replace(/, planned$/, "") : q.name);
      return v == null ? "" : `<div><i style="background:${q.color}"></i>${esc(name)} <b>${esc((q.fmt || ((z) => fmt(z)))(v))}</b></div>`;
    }).join("");
    if (!lines) { hide(); return; }
    tip.innerHTML = `<div class="th">${esc(head)}${ahead ? ' <span class="muted">· planned</span>' : ""}</div>${lines}${spec.tipNote ? spec.tipNote(xx) : ""}`;   // tipNote: extra html line
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(cx - tw / 2, 0), W - tw) + "px";
    tip.style.top = "0px";
    if (spec.onHover) spec.onHover(xx);
  };
  const toPx = (ev) => (ev.touches ? ev.touches[0].clientX : ev.clientX) - svg.getBoundingClientRect().left;
  const xAtPx = (px) => log ? Math.exp(Math.log(x0) + (px - m.l) / (W - m.l - m.r) * (Math.log(x1) - Math.log(x0)))
                            : x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
  const move = (ev) => {
    const px = toPx(ev);
    if (spec.group) { const xval = xAtPx(px); (hoverGroups[spec.group] || []).forEach((h) => h.at(xval)); }
    else showAt(px);
  };
  const hide = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); };
  if (spec.group) {
    const g = (hoverGroups[spec.group] = (hoverGroups[spec.group] || []).filter((h) => h.id !== id && document.getElementById(h.id)));
    g.push({id, at: (xval) => showAt(fx(xval)), hide});
  }
  const leave = () => { if (spec.group) (hoverGroups[spec.group] || []).forEach((h) => h.hide()); else hide(); if (spec.onHover) spec.onHover(null); };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerdown", move);
  svg.addEventListener("pointerleave", leave);
  svg.addEventListener("touchmove", move, {passive: true});
  const geo = {m, W, H, x0, x1, fx, xAtPx, toPx};
  if (tw && (spec.scrollbar || (!spec.group && tw.view))) tw.scrollbar(id, box, m, W);
  else box.parentNode?.querySelector(`.cscroll[data-for="${id}"]`)?.remove();
  if (spec.select) chartSelect(id, spec, svg, geo);
  if (zoomable) {
    if (!spec.select) chartSelect(id, {...spec, zoomDrag: true, select: {range: null, onChange: (r) => tw.zoomTo(r)}}, svg, geo);
    svg.addEventListener("wheel", (ev) => tw.wheel(ev, xAtPx(toPx(ev))), {passive: false});
    if (tw.view) {
      const b = document.createElement("button");
      b.className = "zoomreset"; b.textContent = "Reset zoom";
      b.addEventListener("click", () => tw.zoomTo(null));
      box.appendChild(b);
    }
  }
}
function stepTicks(x0, x1, step, label) {   // every whole `step` in [x0, x1]
  const out = [];
  for (let t = Math.ceil(x0 / step - 1e-9) * step; t <= x1 + 1e-9; t += step) out.push([t, label(t, step)]);
  return out;
}
const DBLCLICK_MS = 300;   // a click waits this long in case it is the first of a double-click
function chartSelect(id, spec, svg, {m, W, H, x0, x1, fx, xAtPx, toPx}) {
  // drag across the chart to select; drag an edge of the selection to move it
  const sel = spec.select, key = spec.group || id, EDGE = 6;
  const band = el("rect", {y: m.t, height: H - m.t - m.b, fill: css("--accent"), "fill-opacity": 0.14,
    stroke: css("--accent"), "stroke-width": 1, visibility: "hidden", "pointer-events": "none"}, svg);
  const show = (r) => {
    if (!r || r[1] < x0 || r[0] > x1) { band.setAttribute("visibility", "hidden"); return; }
    const a = fx(Math.max(x0, r[0])), b = fx(Math.min(x1, r[1]));
    band.setAttribute("x", a); band.setAttribute("width", Math.max(b - a, 1)); band.setAttribute("visibility", "visible");
  };
  show(sel.range);
  selectGroups[key] = (selectGroups[key] || []).filter((h) => h.id !== id && document.getElementById(h.id));
  selectGroups[key].push({id, show});
  const edgeAt = (px) => sel.range && [0, 1].find((k) => Math.abs(fx(sel.range[k]) - px) <= EDGE);
  let drag = null, pickT = 0;
  svg.style.touchAction = "pan-y";
  // zoomed in, a drag pans (Shift + drag zooms further); on a chart whose drag selects, Shift + drag pans
  const pans = (ev) => !!TimeWindow.of(key).view && (spec.zoomDrag ? !ev.shiftKey : ev.shiftKey);
  svg.addEventListener("pointerdown", (ev) => {
    if (ev.button > 0) return;
    clearTimeout(pickT);
    if (ev.pointerType === "mouse") ev.preventDefault();   // no text selection flashing while dragging (macOS)
    const px = toPx(ev), k = edgeAt(px);
    if (k == null && pans(ev)) {   // grab the chart: the window follows the pointer (the scrollbar's drag, reversed)
      const tw = TimeWindow.of(key);
      tw.grab(ev, tw.view[1] - tw.view[0], -(W - m.l - m.r));
      svg.style.cursor = "grabbing";
      return;
    }
    drag = {fixed: k != null ? sel.range[1 - k] : xAtPx(px), px0: px, edge: k != null, moved: false};
    try { svg.setPointerCapture(ev.pointerId); } catch (e) { /* synthetic pointer */ }
  });
  svg.addEventListener("pointermove", (ev) => {
    const px = toPx(ev);
    if (!drag) { svg.style.cursor = edgeAt(px) != null ? "ew-resize" : TimeWindow.dragging ? "grabbing" : pans(ev) ? "grab" : "crosshair"; return; }
    if (!drag.edge && Math.abs(px - drag.px0) < 4) return;   // a click, not a drag
    const v = Math.max(x0, Math.min(x1, xAtPx(px)));
    sel.range = [Math.min(drag.fixed, v), Math.max(drag.fixed, v)];
    drag.moved = true;
    selectGroups[key].forEach((h) => h.show(sel.range));
  });
  const stop = () => { drag = null; };
  svg.addEventListener("pointerup", () => {
    if (drag?.moved) sel.onChange(sel.range);
    else if (drag && !drag.edge && spec.onPick) {   // a click: what's at that x, unless it becomes a double-click
      const at = drag.fixed;
      pickT = setTimeout(() => spec.onPick(at), DBLCLICK_MS);
    }
    stop();
  });
  svg.addEventListener("pointercancel", stop);
  svg.addEventListener("dblclick", () => {
    clearTimeout(pickT);
    sel.range = null;
    selectGroups[key].forEach((h) => h.show(null));
    sel.onChange(null);
  });
}
// drawChart's spec drawn by Apache ECharts (static/vendor): pages move to EChart.draw one at a time, then drawChart
// goes. ECharts does the axes, ticks, marks, crosshair, tooltip box, zoom / pan (dataZoom: Ctrl + wheel / pinch zooms,
// drag or Shift + wheel pans, the slider is the scrollbar, double-click shows all). A group's charts (spec.group) share
// their x range, zoom and crosshair: passed on here (pointer, zoomed), not by ECharts' connect, which shows a pointer only
// in a chart the mouse is in. The zoom is kept in the TimeWindow of the chart (or its group), so range buttons and
// redraws keep it.
// Not yet: select (stretch selection), xStep, xTicks, yScale (the ride page's side histogram).
export class EChart {
  static live = {};              // id -> EChart drawn (disposed once its box has left the page)
  static PICK_MS = 300;          // a click waits this long in case it is the first of a double-click
  static echo = false;           // a group's charts are following one of them (pointer, zoom): not passed on again

  static draw(id, spec) {
    const box = document.getElementById(id);
    if (!box) return;
    for (const [k, c] of Object.entries(EChart.live)) if (!c.chart.getDom().isConnected) { c.chart.dispose(); delete EChart.live[k]; }
    const c = EChart.live[id];
    if (c && c.chart.getDom() === box) c.update(spec);
    else { c?.chart.dispose(); EChart.live[id] = new EChart(box, spec); }
  }

  static resizeAll() { Object.values(EChart.live).forEach((c) => c.chart.resize()); }

  constructor(box, spec) {
    this.box = box;
    this.downX = null;
    this.pickT = 0;
    box.innerHTML = "";
    box.style.position = "relative";
    this.chart = init(box, null, {renderer: "svg"});
    box.addEventListener("wheel", (ev) => this.wheel(ev), {capture: true});
    this.chart.on("datazoom", () => this.zoomed());
    const zr = this.chart.getZr();
    zr.on("mousedown", (e) => { this.downX = e.offsetX; });
    zr.on("click", (e) => this.click(e));
    zr.on("dblclick", () => this.showAll());
    this.chart.on("updateAxisPointer", (e) => this.pointer(e.axesInfo?.[0]?.value ?? null));
    zr.on("globalout", () => this.pointer(null));
    this.update(spec);
  }

  update(spec) {
    this.spec = spec;
    this.xv = (spec.x || []).map((v) => spec.xType === "date" ? new Date(v + "T12:00:00").getTime() : v);
    const allX = spec.series.flatMap((s) => this.xs(s)).filter((v) => v != null);
    const has = spec.series.some((s) => (s.y || []).some((v) => v != null));
    if (!has) { this.chart.clear(); this.empty(true); return; }
    this.empty(false);
    this.zoomable = (spec.xType === "date" || spec.xType === "num") && allX.length > 1;
    this.tw = this.zoomable ? TimeWindow.of(spec.group || this.box.id) : null;
    if (this.tw && spec.view && !this.tw.view && !this.tw.opened) { this.tw.view = spec.view; this.tw.opened = true; }
    const before = this.tw?.extent?.join();
    if (this.tw) this.tw.widen(spec.xDomain || [Math.min(...allX), Math.max(...allX)]);
    this.chart.setOption(this.option(allX), {notMerge: true});
    if (spec.group && before !== this.tw?.extent?.join()) this.widenGroup();
  }

  // a group's charts share one x range (the TimeWindow's extent), so stacked charts line up: one drawn later with
  // more data widens the ones already drawn
  widenGroup() {
    const [min, max] = this.tw.extent;
    this.others().filter((c) => c.tw).forEach((c) => c.chart.setOption({xAxis: {min, max}}));
  }

  others() {                     // the other charts of this one's group
    return this.spec.group ? Object.values(EChart.live).filter((c) => c !== this && c.spec.group === this.spec.group) : [];
  }

  // run fn on the group's other charts once: what they do in turn (their own events) isn't passed on again
  toGroup(fn) {
    if (EChart.echo) return;
    EChart.echo = true;
    try {
      this.others().forEach(fn);
    } finally {
      EChart.echo = false;
    }
  }

  empty(on) {
    this.box.querySelector(".nodata")?.remove();
    if (on) this.box.insertAdjacentHTML("beforeend", '<div class="nodata muted small" style="position:absolute;inset:0;padding:30px 0;text-align:center">No data yet</div>');
  }

  xs(s) { return s.x || this.xv; }

  option(allX) {
    const spec = this.spec, ink2 = css("--ink2"), grid = css("--grid");
    const slider = this.hasSlider();
    const labelled = (spec.bands || []).some((b) => b[3]);
    const dom = this.tw?.extent || spec.xDomain || [Math.min(...allX), Math.max(...allX)];
    const series = spec.series.flatMap((s) => this.series(s));
    (spec.yExtra || []).forEach((y) => series.push({type: "line", data: this.pairs(this.xv, y), symbol: "none", silent: true,
      lineStyle: {opacity: 0}, tooltip: {show: false}}));
    if (series.length) Object.assign(series[0], this.marks());
    const view = this.tw?.view;
    return {
      animation: false,
      grid: {left: 44, right: 10, top: this.zoomable || labelled ? 20 : 6, bottom: (spec.xAxis === false ? 4 : 22) + (slider ? 24 : 0)},
      xAxis: {type: spec.xType === "date" ? "time" : spec.xType === "log" ? "log" : "value", min: dom[0], max: dom[1],
        axisLine: {show: false}, axisTick: {show: false}, splitLine: {show: false},
        axisLabel: {show: spec.xAxis !== false, color: ink2, fontSize: 11, hideOverlap: true,
          ...(spec.xTickFmt ? {formatter: (v) => spec.xTickFmt(v)} : {})}},
      yAxis: {type: "value", scale: !(spec.zero || spec.series.some((s) => s.kind === "bar")), ...this.yRange(view),
        axisLine: {show: false}, axisTick: {show: false}, splitLine: {lineStyle: {color: grid}}, axisLabel: {color: ink2, fontSize: 11, formatter: (v) => (spec.yFmt || String)(v)}},
      tooltip: {trigger: "axis", confine: true, className: "etip", transitionDuration: 0, padding: [6, 9],
        backgroundColor: css("--surface"), borderColor: css("--line"), borderWidth: 1, textStyle: {color: css("--ink"), fontSize: 12.5},
        extraCssText: "border-radius:9px;box-shadow:none;white-space:nowrap",
        axisPointer: {type: "line", lineStyle: {color: css("--muted"), type: [3, 3]}}, formatter: (p) => this.tip(p)},
      dataZoom: this.zoomable ? [
        {type: "inside", xAxisIndex: 0, zoomOnMouseWheel: "ctrl", moveOnMouseWheel: "shift", moveOnMouseMove: true,
          startValue: view?.[0], endValue: view?.[1]},
        ...(slider ? [{type: "slider", xAxisIndex: 0, height: 12, bottom: 4, showDetail: false, brushSelect: false,
          borderColor: "transparent", backgroundColor: css("--surface2"), fillerColor: css("--line"),
          dataBackground: {lineStyle: {opacity: 0}, areaStyle: {opacity: 0}}, startValue: view?.[0], endValue: view?.[1]}] : [])] : [],
      series,
    };
  }

  // the scrollbar (slider): always with spec.scrollbar, else on a chart of its own once it is zoomed
  hasSlider() {
    return this.zoomable && (!!this.spec.scrollbar || (!this.spec.group && !!this.tw?.view));
  }

  pairs(X, y) { return X.map((x, i) => [x, y?.[i] ?? null]); }

  // {min, max} of the y axis when a series has range: false (ECharts can't leave one out): from the other series
  // and yExtra in the x window shown, padded and rounded out to whole ticks as drawChart did; {} = ECharts' own
  yRange(view) {
    const spec = this.spec;
    if (!spec.series.some((s) => s.range === false)) return {};
    const seen = (x) => !view || (x != null && x >= view[0] && x <= view[1]);
    const ys = [];
    for (const s of spec.series) {
      if (s.range === false) continue;
      const X = this.xs(s);
      for (const y of [s.y, s.y2]) (y || []).forEach((v, i) => { if (v != null && seen(X[i])) ys.push(v); });
    }
    (spec.yExtra || []).forEach((y) => y.forEach((v, i) => { if (v != null && seen(this.xv[i])) ys.push(v); }));
    if (!ys.length) return {};
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (spec.zero || spec.series.some((s) => s.kind === "bar")) { y0 = Math.min(0, y0); y1 = Math.max(0, y1); }
    const pad = (y1 - y0) * 0.06 || 1;
    if (!(spec.zero && y0 === 0)) y0 -= pad;
    y1 += pad;
    const t = niceTicks(y0, y1, 5), step = t.length > 1 ? t[1] - t[0] : 1;
    return {min: Math.floor(y0 / step) * step, max: Math.ceil(y1 / step) * step};   // whole ticks at both ends
  }

  // one spec series as ECharts series: line (area, dash, markers, connect), bar, dots (hollow, alpha, lo / hi bars),
  // band (y .. y2 filled: a transparent lower line with the gap stacked on it)
  series(s) {
    const X = this.xs(s), col = s.color;
    if (s.kind === "band") {
      const stack = "band" + Math.random();
      const lo = X.map((x, i) => [x, s.y[i] != null && s.y2[i] != null ? s.y[i] : null]);
      const gap = X.map((x, i) => [x, s.y[i] != null && s.y2[i] != null ? s.y2[i] - s.y[i] : null]);
      const quiet = {type: "line", symbol: "none", silent: true, stack, stackStrategy: "all", tooltip: {show: false}, lineStyle: {opacity: 0}};
      return [{...quiet, data: lo}, {...quiet, data: gap, areaStyle: {color: col, opacity: s.opacity ?? 0.16}}];
    }
    if (s.kind === "bar") {
      return [{type: "bar", data: this.pairs(X, s.y), barMaxWidth: 24,
        itemStyle: {color: col, opacity: s.opacity ?? 0.85, borderRadius: 2}}];
    }
    if (s.kind === "dots") {
      const data = X.flatMap((x, i) => s.y[i] == null ? [] : [{value: [x, s.y[i]],
        itemStyle: {opacity: s.alpha ? s.alpha[i] : s.r ? 0.75 : 1}}]);
      const out = [{type: "scatter", data, symbolSize: 2 * (s.r || 4), itemStyle: s.hollow
        ? {color: "transparent", borderColor: col, borderWidth: 1.2}
        : {color: col, borderColor: s.r ? "transparent" : css("--surface"), borderWidth: s.r ? 0 : 1.5}}];
      if (s.lo) {
        const bars = X.flatMap((x, i) => s.y[i] == null || s.lo[i] == null ? [] : [[x, s.lo[i], s.hi[i], s.alpha ? s.alpha[i] : 1]]);
        out.push({type: "custom", data: bars, encode: {x: 0, y: [1, 2]}, silent: true, tooltip: {show: false},
          renderItem: (_, api) => this.errorBar(api, col)});
      }
      return out;
    }
    return [{type: "line", data: this.pairs(X, s.y), connectNulls: !!s.connect, showSymbol: !!s.markers, symbolSize: 5,
      itemStyle: {color: col}, lineStyle: {color: col, width: s.width || 2, type: s.dash ? [5, 4] : "solid"},
      ...(s.area ? {areaStyle: {color: col, opacity: 0.14}} : {})}];
  }

  errorBar(api, col) {
    const x = api.value(0), a = api.coord([x, api.value(1)]), b = api.coord([x, api.value(2)]);
    return {type: "line", shape: {x1: a[0], y1: a[1], x2: b[0], y2: b[1]}, style: {stroke: col, lineWidth: 1.2, opacity: api.value(3)}};
  }

  // vline / hline, shade [from, to] and bands [[from, to, colour, label?, opacity?, inside?], ...] on the first series
  marks() {
    const spec = this.spec, ink2 = css("--ink2"), bandOp = spec.bandOpacity ?? 0.22;
    const lines = [];
    if (spec.vline != null) lines.push({xAxis: spec.vline, lineStyle: {color: ink2, width: 1, type: [2, 3]}});
    if (spec.hline != null) lines.push({yAxis: spec.hline, lineStyle: {color: ink2, width: 1, type: "solid"}});
    const areas = [...(spec.shade ? [[...spec.shade, css("--ink"), null, 0.06]] : []), ...(spec.bands || [])].map(([a, b, col, label, op, inside]) => [
      {xAxis: a, itemStyle: {color: col, opacity: op ?? bandOp},
        label: {show: !!label, formatter: label || "", position: inside ? "insideTopLeft" : "top", align: "left", fontSize: 10.5,
          fontWeight: 600, color: inside ? css("--ink") : col}},
      {xAxis: b}]);
    return {
      ...(lines.length ? {markLine: {silent: true, symbol: "none", label: {show: false}, animation: false, data: lines}} : {}),
      ...(areas.length ? {markArea: {silent: true, animation: false, data: areas}} : {}),
    };
  }

  nearest(X, x) {
    const f = this.spec.xType === "log" ? Math.log : (v) => v;
    let bi = -1, bd = Infinity;
    X.forEach((v, i) => { if (v == null) return; const d = Math.abs(f(v) - f(x)); if (d < bd) { bd = d; bi = i; } });
    return bi;
  }

  // the tooltip: the date (or x), then each named series at that x (planFrom: after the last day of data the planned
  // series show as plain "X" - before it they're left out), then tipNote(x)
  tip(params) {
    const spec = this.spec, at = params?.[0]?.axisValue;
    const base = spec.series.find((q) => q.kind !== "band") || spec.series[0], BX = this.xs(base);
    const i = at == null ? -1 : this.nearest(BX, at);
    if (i < 0) return "";
    const xx = BX[i];
    const head = spec.xType === "date" ? niceDate(spec.x[i], {weekday: "short", day: "numeric", month: "short", year: "numeric"}) : (spec.xLabel ? spec.xLabel(xx) : xx);
    const ahead = spec.planFrom != null && xx > spec.planFrom;
    const shown = spec.series.filter((q) => q.name && q.kind !== "band" && (!q.planned || ahead));
    const lines = (spec.tipReverse ? shown.reverse() : shown).map((q) => {
      const j = q.x ? this.nearest(q.x, xx) : i, v = q.y[j];
      const name = q.tipName || (q.planned ? q.name.replace(/, planned$/, "") : q.name);
      return v == null ? "" : `<div><i style="background:${q.color}"></i>${esc(name)} <b>${esc((q.fmt || ((z) => fmt(z)))(v))}</b></div>`;
    }).join("");
    const note = spec.tipNote ? spec.tipNote(xx) : "";
    if (!lines && !note) return "";
    return `<div class="th">${esc(head)}${ahead ? ' <span class="muted">· planned</span>' : ""}</div>${lines}${note}`;
  }

  zoomed() {                     // the zoom ECharts shows -> the TimeWindow (null: all of it)
    if (!this.tw) return;
    const dz = this.chart.getOption().dataZoom?.[0], e = this.tw.extent;
    if (!dz || !e) return;
    const had = this.hasSlider();
    this.tw.view = dz.startValue <= e[0] && dz.endValue >= e[1] ? null : [dz.startValue, dz.endValue];
    if (had !== this.hasSlider()) { setTimeout(() => this.update(this.spec)); return; }   // the scrollbar comes or goes
    const y = this.yRange(this.tw.view);
    if (y.min != null) this.chart.setOption({yAxis: y});
    this.toGroup((c) => c.chart.dispatchAction({type: "dataZoom", startValue: dz.startValue, endValue: dz.endValue}));
  }

  // the crosshair at x (null: gone): onHover(x), and the group's other charts show theirs and their tooltip there
  pointer(x) {
    this.spec.onHover?.(x);
    this.toGroup((c) => c.showAt(x));
  }

  showAt(x) {
    if (x == null) { this.chart.dispatchAction({type: "hideTip"}); return; }
    this.chart.dispatchAction({type: "showTip", x: this.chart.convertToPixel({xAxisIndex: 0}, x), y: this.chart.getHeight() / 2});
  }

  // a plain wheel scrolls the page: ECharts' inside zoom takes every wheel over the plot (also with zoomOnMouseWheel
  // "ctrl"), so it never sees one without Ctrl (pinch) or Shift
  wheel(ev) {
    if (!(ev.ctrlKey || ev.shiftKey)) ev.stopPropagation();
  }

  showAll() {
    clearTimeout(this.pickT);
    if (this.zoomable) this.chart.dispatchAction({type: "dataZoom", start: 0, end: 100});
  }

  click(e) {                     // a click (not a drag) in the plot: onPick(x), unless it becomes a double-click
    if (!this.spec.onPick || this.downX == null || Math.abs(e.offsetX - this.downX) > 4) return;
    if (!this.chart.containPixel({gridIndex: 0}, [e.offsetX, e.offsetY])) return;
    const x = this.chart.convertFromPixel({xAxisIndex: 0}, e.offsetX);
    clearTimeout(this.pickT);
    this.pickT = setTimeout(() => this.spec.onPick(x), EChart.PICK_MS);
  }
}
let rsT;
export const tail = (arr, n) => (arr || []).slice(Math.max(0, (arr || []).length - n));

window.addEventListener("pointermove", (ev) => TimeWindow.follow(ev));
window.addEventListener("pointerup", () => TimeWindow.drop());

window.addEventListener("resize", () => { clearTimeout(rsT); rsT = setTimeout(() => {
  Object.entries(charts).forEach(([id, sp]) => document.getElementById(id) && drawChart(id, sp));
  EChart.resizeAll();
  if (router.route() === "ride" && document.getElementById("rmap")) rideMap.drawMap();
  if (calendarPage.calResize) calendarPage.calResize();
}, 150); });
