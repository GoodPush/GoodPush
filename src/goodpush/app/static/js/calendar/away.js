// Days away: pick days, add a note, remove (AwayDays).
import {api, toast} from "../api.js";
import {$, addDays, esc} from "../util.js";
import {hint, niceDate} from "../widgets.js";
import {router} from "../main.js";
import {calendarPage} from "./calendar.js";
import {el} from "../charts.js";

// days away picked on the calendar (click a ✈ tag, Shift+click for a range) -> Remove takes them all off at once
export const AWAY_SEL = new Set();

class AwayDays {
  // Days away: add a range (with a note) or take one off; the plan re-makes itself around them
  awayRanges() {                     // consecutive days with the same note -> [from, to, note]
    const out = [];
    for (const ds of Object.keys(calendarPage.CAL.away || {}).sort()) {
      const last = out[out.length - 1], note = calendarPage.CAL.away[ds];
      if (last && addDays(last[1], 1) === ds && last[2] === note) last[1] = ds; else out.push([ds, ds, note]);
    }
    return out;
  }

  awayLast = null;

  awayPick(ds, range) {
    const days = range && this.awayLast ? Object.keys(calendarPage.CAL.away || {}).filter((d) => d >= (this.awayLast < ds ? this.awayLast : ds) && d <= (this.awayLast < ds ? ds : this.awayLast)) : [ds];
    const on = range || !AWAY_SEL.has(ds);
    days.forEach((d) => on ? AWAY_SEL.add(d) : AWAY_SEL.delete(d));
    this.awayLast = ds;
    this.awayShow();
  }

  awayShow() {                       // the tags' marks and the bar, without re-drawing the calendar
    [...AWAY_SEL].forEach((d) => calendarPage.CAL.away?.[d] != null || AWAY_SEL.delete(d));
    document.querySelectorAll(".awaytag[data-away]").forEach((t) => t.classList.toggle("sel", AWAY_SEL.has(t.dataset.away)));
    const bar = $("#awaybar"), n = AWAY_SEL.size;
    if (!bar) return;
    bar.hidden = !n;
    $("#awaysel").textContent = `✈ ${n} day${n === 1 ? "" : "s"} away picked · click more (Shift+click: a range) · Delete removes them`;
  }

  awayClear() { AWAY_SEL.clear(); this.awayLast = null; this.awayShow(); }

  awayRemove() {
    const days = [...AWAY_SEL].sort();
    if (!days.length) return;
    days.forEach((d) => delete calendarPage.CAL.away[d]);
    this.awayClear(); calendarPage.CAL._m = null; router.render();
    api.action({action: "away", days}, `${days.length} day${days.length === 1 ? "" : "s"} away removed – updating the plan…`);
  }

  awayModal() {
    const span = (a, b) => a === b ? niceDate(a) : `${niceDate(a)} – ${niceDate(b)}`;
    const list = this.awayRanges().filter((r) => r[1] >= calendarPage.CAL.today);
    const el = document.createElement("div");
    el.className = "modal";
    el.innerHTML = `<div class="modal-box" role="dialog" aria-modal="true" aria-label="Days away">
    <button class="modal-x" aria-label="Close">×</button>
    <div class="sess-head"><div class="sess-title">Days away</div>${hint("Days you can't ride.\n• No session goes on them\n• Key sessions and the long ride move to the days around\n• The week's hours shrink with them")}</div>
    ${list.length ? `<div class="away-list">${list.map(([a, b, n]) => `<div class="row"><span>✈ ${esc(span(a, b))}${n ? ` · <span class="muted">${esc(n)}</span>` : ""}</span><span class="spacer"></span>
      <button class="btn sm" data-from="${a}" data-to="${b}">Remove</button></div>`).join("")}</div>` : '<p class="small muted">None planned.</p>'}
    <div class="edit-row" style="margin-top:12px">
      <label class="f">From<input type="date" id="aw-from" min="${calendarPage.CAL.today}" value="${calendarPage.CAL.today}"></label>
      <label class="f">To<input type="date" id="aw-to" min="${calendarPage.CAL.today}" value="${calendarPage.CAL.today}"></label>
      <label class="f">Note<input type="text" id="aw-note" maxlength="80" placeholder="e.g. work trip"></label>
      <button class="btn sm primary" id="aw-add">Add</button>
    </div></div>`;
    const close = () => { el.remove(); document.removeEventListener("keydown", esc_); };
    const esc_ = (e) => { if (e.key === "Escape") close(); };
    const set = (from, to, note, remove) => {
      for (let d = from; d <= to; d = addDays(d, 1)) { if (remove) delete calendarPage.CAL.away[d]; else calendarPage.CAL.away[d] = note; }
      calendarPage.CAL._m = null; close(); router.render();
      api.action({action: "away", from, to, note, remove}, remove ? "Days away removed – updating the plan…" : "Days away added – updating the plan…");
    };
    el.addEventListener("click", (e) => { if (e.target === el || e.target.closest(".modal-x")) close(); });
    document.addEventListener("keydown", esc_);
    el.querySelector("#aw-from").addEventListener("change", (e) => { const t = el.querySelector("#aw-to"); if (t.value < e.target.value) t.value = e.target.value; });
    el.querySelectorAll("[data-from]").forEach((b) => b.addEventListener("click", () => set(b.dataset.from, b.dataset.to, "", true)));
    el.querySelector("#aw-add").addEventListener("click", () => {
      const from = el.querySelector("#aw-from").value, to = el.querySelector("#aw-to").value || from;
      if (!from || to < from) { toast.show("Pick the first and last day"); return; }
      calendarPage.CAL.away ||= {};
      set(from, to, el.querySelector("#aw-note").value.trim(), false);
    });
    document.body.appendChild(el);
    el.querySelector("#aw-from").focus();
  }
}
export const awayDays = new AwayDays();
