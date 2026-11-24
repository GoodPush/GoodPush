// Pure helpers: DOM lookup, escaping, number / time formats, dates.
export const $ = (sel, el = document) => el.querySelector(sel);
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
export const fmt = (v, nd = 0) => (v == null || Number.isNaN(Number(v)) ? "–" : (Math.abs(v) < 0.5 * 10 ** -nd ? 0 : Number(v)).toFixed(nd));
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const LOCAL = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
// durations: seconds as m:ss, from an hour as h:mm ("1:05 h"); minutes under an hour as "45 min", else h:mm
export const mmss = (sec) => { sec = Math.round(sec); const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s2 = sec % 60;
  return h ? `${h}:${String(m).padStart(2, "0")} h` : `${m}:${String(s2).padStart(2, "0")}`; };
export const hhmm = (min, unit = true) => { if (min == null || min !== min) return "–"; min = Math.round(min);
  return min < 60 && unit ? `${min} min` : `${Math.floor(min / 60)}:${String(min % 60).padStart(2, "0")}${unit ? " h" : ""}`; };
// Symmetric box filter (no time shift): the mean over i - h .. i + h, h a real half-width in samples - the samples
// h - floor(h) past the integer half-width count with that weight, so the width varies smoothly (total 2h + 1).
// O(n) at any width via running sums. Gaps (null) stay gaps; windows touching a gap or an end average what is there.
export function boxSmooth(arr, h) {
  const n = arr.length; if (!(h > 0) || n < 2) return arr;
  const m = Math.floor(h), f = h - m, val = new Float64Array(n + 1), has = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { const ok = arr[i] != null; val[i + 1] = val[i] + (ok ? arr[i] : 0); has[i + 1] = has[i] + ok; }
  const lo = (i) => Math.max(i, 0), hi = (i) => Math.min(i, n);          // prefix-sum indices, clamped to the ends
  return Array.from(arr, (v, i) => {
    if (v == null) return null;
    const a = lo(i - m), b = hi(i + m + 1), a2 = lo(i - m - 1), b2 = hi(i + m + 2);
    const s = val[b] - val[a] + f * (val[a] - val[a2] + val[b2] - val[b]);
    const w = has[b] - has[a] + f * (has[a] - has[a2] + has[b2] - has[b]);
    return s / w;
  });
}
export function addDays(ds, n) {
  const d = new Date(ds + "T12:00:00"); d.setDate(d.getDate() + n);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}
