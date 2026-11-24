// Illness / overreaching markers (MarkersSection) on the Recovery page.
import {S} from "./api.js";
import {esc, fmt} from "./util.js";
import {ICON, niceDate} from "./widgets.js";
import {LEVEL} from "./dips.js";

class MarkersSection {
  a1Verdict(z) {
    if (z == null) return ["", "no baseline yet"];
    if (z <= -2) return ["missed", "low for the power"];
    if (z >= 2) return ["amber", "high for the power"];
    if (Math.abs(z) >= 1.5) return ["amber", z < 0 ? "a bit low" : "a bit high"];
    return ["done", "normal"];
  }

  chipFor(cls, text) {
    return cls === "amber" ? `<span class="chip" style="background:var(--warn-bg);color:var(--warn)">${esc(text)}</span>` : `<span class="chip ${cls}">${esc(text)}</span>`;
  }

  alphaLine() {
    const L = S.alpha && S.alpha.latest; if (!L) return "";
    const [cls, txt] = this.a1Verdict(L.a1_warm_z);
    const hr = L.a1_warm_hr_z != null && L.a1_warm_hr_z >= 2 ? " · heart rate high for the power" : "";
    return `<div class="small" style="margin-top:6px">Last warm-up alpha1 (${niceDate(L.date, {day: "numeric", month: "short"})}): ${this.chipFor(cls, txt)} ${fmt(L.a1_warm, 2)} at ${fmt(L.a1_warm_p)} W${L.a1_warm_exp != null ? `, your normal there ≈ ${fmt(L.a1_warm_exp, 2)}` : ""}${hr} · <a href="#ride/${encodeURIComponent(L.id)}">ride</a></div>`;
  }

  overWatchLine() {
    const os = S.over_status;
    if (!os || !os.level) return "";
    const [cls, lab] = os.level === "clear" ? ["green", "Clear"] : ["amber", "Watch"];
    return `<div class="small" style="margin-top:6px">Overreaching watch: <span class="chip ${cls === "green" ? "done" : ""}" style="${cls === "amber" ? "background:var(--warn-bg);color:var(--warn)" : ""}">${lab}</span> ${os.firing?.length ? esc(os.firing.join(", ")) : ""} · <a href="#recovery">details</a></div>`;
  }

  illWatchLine() {
    const ms = S.marker_status;
    if (!S.markers) return `<div class="small" style="margin-top:12px">Illness watch: <a href="#recovery">set up your personal markers →</a></div>`;
    if (!ms || !ms.level) return "";
    const [cls, lab] = LEVEL[ms.level];
    return `<div class="small" style="margin-top:12px">Illness watch: <span class="chip ${cls === "green" ? "done" : cls === "red" ? "missed" : ""}" style="${cls === "amber" ? "background:var(--warn-bg);color:var(--warn)" : ""}">${lab}</span>
    ${ms.firing?.length ? esc(ms.firing.join(", ")) : `${S.markers.fallback ? "general resting-HR rule" : "none of your markers firing"}`} · <a href="#recovery">details</a></div>`;
  }

  // the watch now: the level chip with its 14-day marker strip, or what is needed to set the markers up
  status(target = "illness") {
    const over = target === "overreaching";
    const M = over ? S.markers_over : S.markers, ms = (over ? S.over_status : S.marker_status) || {};
    const title = `<h2 id="${over ? "omarkers" : "markers"}">${over ? "Overreaching watch" : "Illness watch"}</h2>`;
    if (over) {
      if (!M) return "";
      const marked = M.confirmed_episodes || 0;
      if (!(M.chosen || []).length && marked >= (M.min_episodes || 3)) return `${title}<div class="card small">Nothing in your body signs or load came reliably before the ${marked} periods you marked, so there is no overreaching watch.</div>`;
      if (!(M.chosen || []).length) return `${title}<div class="card small">Not learned yet: confirm at least ${M.min_episodes || 3} training breaks as <b>Overreaching / burnout</b> in the list below (${M.confirmed_episodes || 0} so far). The markers are then learned from the 2 weeks before each one.</div>`;
    }
    if (!M) return `${title}<div class="card"><p class="small muted" style="margin-top:0">Replays your history to find which morning signals showed up in the week before you noticed you were getting sick (from sickness words in your ride notes, and illness events in goodpush.toml), then watches those signals every day. Takes a few seconds.</p><button class="btn primary" data-act="learn_markers">Analyze my history</button></div>`;
    const fb = M.fallback ? `<p class="small" style="margin:0 0 10px">Not enough illness history to personalise yet (${M.confirmed_episodes || 0} confirmed illnesses; at least 4 with Garmin data are needed). Using the general rule meanwhile. Each illness you record teaches it: mark the day <b>Sick</b> on the Today page, label a training break below as illness, add it under Settings → Life events, or write "sick", "cold", "flu"… in a ride note. It learns again by itself after each one.</p>` : "";
    let strip = "";
    if (ms.markers?.length) {
      const old = (i) => (i < ms.dates.length - 7 ? " old" : "");
      const head = ms.dates.map((d, i) => `<th class="dcol${old(i)}">${new Date(d + "T12:00:00").getDate()}</th>`).join("");
      const rows = ms.markers.map((m) => `<tr><td class="small">${esc(m.label)}</td>${m.fired.map((f, i) => {
        const v = m.values[i]; const t = v == null ? "no data" : `${v > 0 && m.unit !== "z" ? "+" : ""}${fmt(v, 1)}${m.unit === "z" ? " SD" : " " + m.unit}`;
        return `<td class="dcol${old(i)}"><span class="mdot ${v == null ? "nodata" : f ? "fired" : ""}" title="${esc(ms.dates[i] + ": " + t)}"></span></td>`; }).join("")}</tr>`).join("");
      const [cls, lab] = LEVEL[ms.level] || ["green", "Clear"];
      strip = `<div class="row" style="margin-top:4px"><span class="status ${cls}">${ICON[cls]}${lab}</span><span class="small muted">as of ${niceDate(ms.as_of)}${ms.firing?.length ? " · " + esc(ms.firing.join(", ")) : ""}</span></div>
      <div class="tw" style="margin-top:10px"><table class="mtab"><thead><tr><th>Last 14 days</th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      <div class="legend" style="margin-top:6px"><span><span class="mdot fired"></span> fired</span><span><span class="mdot"></span> normal</span><span><span class="mdot nodata"></span> no data</span></div>`;
    }
    return fb || strip ? `${title}<div class="card">${fb}${strip}</div>` : "";
  }

  // how the markers were chosen on your history: the chosen ones, the alarm rule, everything tested (Recovery > History)
  stats(target = "illness") {
    const over = target === "overreaching";
    const M = over ? S.markers_over : S.markers;
    if (!M || (over && !(M.chosen || []).length)) return "";
    const pct = (v) => `${Math.round((v || 0) * 100)} %`;
    const chosen = (M.chosen || []).map((c) => `<tr><td><b>${esc(c.label)}</b><div class="small muted">${esc(c.rule.split(": ")[1] || c.rule)}</div></td>
      <td class="n">${c.default ? "–" : `${c.caught} of ${c.n_eval}`}${c.default || c.chance_recall == null ? "" : `<div class="small muted">luck: ~${fmt(c.chance_recall * c.n_eval, 1)}</div>`}</td><td class="n">${c.default ? "–" : fmt(c.alarms_per_year, 0)}</td>
      <td class="n">${c.default ? "–" : `1 in ${c.precision ? Math.round(1 / c.precision) : "–"}`}</td><td class="n">${c.lead_days != null ? fmt(c.lead_days) + " d" : "–"}</td></tr>`).join("");
    const comb = M.combined && !M.fallback ? `<p class="small" style="margin:10px 0 0">Alarm rule: one of your markers firing = <b>go easy</b>, two or more together = <b>red</b>. On your history, ${esc(M.combined.rule)} caught <b>${M.combined.caught} of ${M.combined.n_eval}</b> illnesses, usually about ${fmt(M.combined.lead_days)} days before you noticed, with about ${fmt(M.combined.alarms_per_year)} alarms a year, roughly 1 in ${M.combined.precision ? Math.round(1 / M.combined.precision) : "–"} followed by an actual illness. Treat an alarm as "go easy and watch", not a diagnosis.</p>` : "";
    const all = (M.candidates || []).map((c) => `<tr><td>${esc(c.label)}${c.rule ? `<div class="small muted">${esc(c.rule.split(": ")[1] || "")}</div>` : ""}</td><td>${esc(c.status)}</td>
      <td class="n">${c.n_eval != null ? `${c.caught}/${c.n_eval}` : "–"}</td><td class="n">${c.alarms_per_year != null ? fmt(c.alarms_per_year, 0) : "–"}</td><td class="n">${c.precision != null ? pct(c.precision) : "–"}</td><td class="n">${c.chance_recall != null ? fmt(c.chance_recall * c.n_eval, 1) : "–"}</td></tr>`).join("");
    return `<h3>${over ? "Overreaching markers" : "Illness markers"}</h3>
  ${over ? `<p class="small muted" style="margin-top:-4px">Learned from the 2 weeks before the training breaks you confirmed as overreaching. When they fire, the plan goes amber. The automatic guess for overreaching is itself based on load and body signs before the break, so confirming only the guesses partly re-finds those same signals – your own judgement (how you felt, motivation, legs) makes this more honest.</p>` : ""}
  <div class="card">
    <div class="tw"><table><thead><tr><th>Marker</th><th class="n">Caught</th><th class="n">Alarms / yr</th><th class="n">Real alarms</th><th class="n">Warns</th></tr></thead><tbody>${chosen}</tbody></table></div>
    ${comb}
    <details style="margin-top:12px"><summary class="small">Everything that was tested</summary>
      <p class="small muted">Best version of each signal. "Luck would catch" = how many illnesses a signal firing just as often at random would have caught; a marker is only chosen when it clearly beats that (p &lt; 0.1). With a handful of illnesses these are hints, not proof. ${M.episodes} illnesses found, ${M.confirmed_episodes} confirmed by the data.</p>
      <div class="tw"><table><thead><tr><th>Signal</th><th>Result</th><th class="n">Caught</th><th class="n">Alarms / yr</th><th class="n">Real alarms</th><th class="n">Luck would catch</th></tr></thead><tbody>${all}</tbody></table></div></details>
    <div class="row" style="margin-top:14px">${over ? "" : `<button class="btn sm" data-act="learn_markers">Analyze again</button>`}<span class="small muted">last analysed ${esc(M.analyzed || "")}${over ? "" : " · re-run every few months or after an illness"}</span></div>
  </div>`;
  }
}
export const markers = new MarkersSection();
