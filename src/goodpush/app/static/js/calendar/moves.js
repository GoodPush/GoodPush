// Moving sessions: copy / paste, drag onto a day, link a ride to a session (SessionMoves).
import {api, toast} from "../api.js";
import {niceDate} from "../widgets.js";
import {router} from "../main.js";
import {calendarPage, calGone} from "./calendar.js";
import {AWAY_SEL, awayDays} from "./away.js";
import {el} from "../charts.js";

const MERGE_HOLD_MS = 800;                                   // held this long over another easy ride: merge
const MERGEABLE = new Set(["recovery", "endurance", "long"]);  // PlanState.MERGEABLE (easy rides)

class SessionMoves {
  // Drag a ride onto a planned session (today or earlier) to link it and mark the session done.
  // Pointer events so it works with a mouse and with touch (hold a ride for a moment, then drag).
  // copy and paste: a copied session goes on the next day you click (or Cmd / Ctrl+V over), once; with Shift held
  // it stays copied for more days, until Done / Esc
  CLIP = null;
  calHover = null;          // the copied session {key, title}; the calendar element under the pointer

  calCopy(key) {
    const p = calendarPage.CAL?.planned[key];
    if (!p || p.type === "rest") return;
    this.CLIP = {key, title: p.title};
    toast.show(`Copied ${p.title} – click a day to paste it (Shift+click: more days)`);
    if (router.route() === "calendar") router.render();
  }

  calPaste(day, more = false) {
    const clip = this.CLIP;
    if (!clip) return;
    if (!more) this.CLIP = null;                       // once; the placeholder's render drops the paste bar
    const to = calendarPage.calPlaceholder(day, clip.key, "pasting");
    api.action({action: "copy_session", from: clip.key, to: day},
      `${clip.title} pasted on ${niceDate(day)} – updating the plan…`)
      .then((ok) => ok || calendarPage.calPending(to, null));
  }

  clipEnd() { this.CLIP = null; if (router.route() === "calendar") router.render(); }

  async moveSession(key, day) {
    const to = calendarPage.calPlaceholder(day, key, "moving here");
    calendarPage.calPending(key, `moving → ${niceDate(day, {weekday: "short"})}`, key.slice(0, 10) >= calendarPage.CAL.today ? calGone : null);
    const from = key.slice(0, 10) >= calendarPage.CAL.today ? ` (the plan re-plans ${niceDate(key.slice(0, 10))})` : "";
    const ok = await api.action({action: "move_session", from: key, to: day},
      `Moved to ${niceDate(day)}${from} – updating the plan…`);
    if (!ok) { calendarPage.calPending(key, null); calendarPage.calPending(to, null); }
  }

  async mergeSessions(key, into) {
    const day = into.slice(0, 10), past = key.slice(0, 10) < calendarPage.CAL.today;
    calendarPage.calPending(key, `merging → ${niceDate(day, {weekday: "short"})}`, past ? null : calGone);
    calendarPage.calPending(into, "merging here");
    const ok = await api.action({action: "merge_session", from: key, to: into},
      `Merged into one ride on ${niceDate(day)} – updating the plan…`);
    if (!ok) { calendarPage.calPending(key, null); calendarPage.calPending(into, null); }
  }

  // dragging an easy ride over another easy ride on another day, without a ride on either: that session's element
  mergeTarget(x, y) {
    const d = this.calDrag, planned = calendarPage.CAL?.planned || {};
    if (d?.kind !== "session" || !MERGEABLE.has(planned[d.id]?.type)) return null;
    const pl = document.elementFromPoint(x, y)?.closest(".pl[data-ds]");
    const p = pl && planned[pl.dataset.ds];
    if (!p || pl.dataset.ds.slice(0, 10) === d.id.slice(0, 10) || !MERGEABLE.has(p.type)) return null;
    return p.ride || p.compliance?.ride || p.pending ? null : pl;
  }

  // held over a mergeable ride for MERGE_HOLD_MS: dropping merges the two; moving off it: a plain move again
  mergeHover(t) {
    const d = this.calDrag;
    if (t === d.mergeEl) return;
    clearTimeout(d.mergeTimer);
    d.mergeEl?.classList.remove("merge");
    d.ghost.textContent = d.title;
    Object.assign(d, {mergeEl: t, merge: null});
    if (!t) return;
    d.mergeTimer = setTimeout(() => {
      if (this.calDrag !== d || d.mergeEl !== t) return;
      d.merge = t.dataset.ds;
      t.classList.add("merge");
      d.ghost.textContent = `Merge with ${calendarPage.CAL.planned[t.dataset.ds].title}`;
    }, MERGE_HOLD_MS);
  }

  async linkRide(ds, rideId) {
    const p = calendarPage.CAL.planned[ds];
    if (rideId) { Object.assign(p, {status: "linked", manual: true, ride: rideId}); delete p.compliance; }
    else { delete p.status; delete p.manual; delete p.ride; delete p.compliance; }
    calendarPage.CAL._m = null;
    router.render();                                          // show it straight away
    await api.action(rideId ? {action: "link", date: ds, ride: rideId} : {action: "unlink", date: ds},
      rideId ? `Linked – scoring the ride against ${p.title}…` : "Unlinked");
    calendarPage.calKey = "";                                       // re-read the saved state
    if (router.route() === "calendar") calendarPage.loadCal();
  }

  calDrag = null;
  calDragRaf = 0;

  // a ride is dropped on a planned session (link it); a planned session without a ride on any other day (add it there)
  dropTarget(x, y) {
    const el = document.elementFromPoint(x, y);
    if (this.calDrag?.kind !== "session") return el?.closest(".pl.drop");
    const cd = el?.closest(".cd[data-day]");
    return cd && cd.dataset.day !== this.calDrag.id.slice(0, 10) ? cd : null;
  }

  calDragEnd() {
    const d = this.calDrag;
    if (!d) return;
    clearTimeout(d.hold); clearTimeout(d.mergeTimer); cancelAnimationFrame(this.calDragRaf);
    d.ghost?.remove(); d.over?.classList.remove("over"); d.mergeEl?.classList.remove("merge");
    d.cal.classList.remove("dragging");
    this.calDrag = null;
  }

  calDragPlace() {
    const d = this.calDrag;
    d.ghost.style.left = d.x + 12 + "px"; d.ghost.style.top = d.y + 8 + "px";
    const t = this.dropTarget(d.x, d.y);
    if (t !== d.over) { d.over?.classList.remove("over"); t?.classList.add("over"); d.over = t; }
    if (d.kind === "session") this.mergeHover(this.mergeTarget(d.x, d.y));
  }

  calDragStart() {
    const d = this.calDrag;
    d.on = true;
    d.cal.classList.add("dragging");
    d.ghost = document.createElement("div");
    d.title = d.el.querySelector(".nm").textContent;
    d.ghost.className = "drag-ghost"; d.ghost.textContent = d.title;
    document.body.appendChild(d.ghost);
    this.calDragPlace();
    const edge = () => {                                // near the top / bottom edge: scroll the calendar
      if (!this.calDrag) return;
      const r = d.cal.getBoundingClientRect(), z = 48;
      if (d.y < r.top + z) d.cal.scrollTop -= (r.top + z - d.y) / 3;
      else if (d.y > r.bottom - z) d.cal.scrollTop += (d.y - r.bottom + z) / 3;
      this.calDragPlace();
      this.calDragRaf = requestAnimationFrame(edge);
    };
    this.calDragRaf = requestAnimationFrame(edge);
  }

  calDrop() {
    const d = this.calDrag;
    if (!d) return;
    const t = d.on && this.dropTarget(d.x, d.y), merge = d.on && d.merge;
    if (d.on) d.cal.dataset.dragged = "1";             // swallow the click that follows
    this.calDragEnd();
    if (merge) this.mergeSessions(d.id, merge);
    else if (t && d.kind === "session") this.moveSession(d.id, t.dataset.day);
    else if (t && t.dataset.date) this.linkRide(t.dataset.date, d.id);
  }

  wireCalDrag(cal) {
    cal.addEventListener("pointerdown", (e) => {
      const ride = e.target.closest(".ac"), pl = e.target.closest(".pl[data-ds]:not(.rest)");
      const p = pl && calendarPage.CAL.planned[pl.dataset.ds];
      if (!ride && p && (p.ride || p.compliance?.ride)) return;     // a ride counts for it: it stays
      const el = ride || pl;
      if (!el || e.button || e.target.closest("button, a:not(.ac)")) return;
      this.calDrag = {cal, el, kind: ride ? "ride" : "session", id: ride ? el.dataset.ride : el.dataset.ds,
        x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, touch: e.pointerType !== "mouse"};
      if (!this.calDrag.touch) e.preventDefault();          // mouse: no text selection while dragging (click still fires)
      if (this.calDrag.touch) this.calDrag.hold = setTimeout(() => { if (this.calDrag && !this.calDrag.on) { this.calDragStart(); navigator.vibrate?.(15); } }, 380);
    });
    cal.addEventListener("click", (e) => {
      if (cal.dataset.dragged) { delete cal.dataset.dragged; if (e.target.closest(".ac, .pl, .cd")) { e.preventDefault(); return; } }
      const u = e.target.closest("[data-unlink]");
      if (u) { e.preventDefault(); this.linkRide(u.dataset.unlink, null); }
    });
    cal.addEventListener("contextmenu", (e) => { if (this.calDrag?.touch) e.preventDefault(); });   // long-press menu
  }
}
export const sessionMoves = new SessionMoves();

document.addEventListener("keydown", (e) => {
  if (router.route() !== "calendar" || document.querySelector(".modal") || e.target.closest?.("input, textarea, select")) return;
  if (e.key === "Escape" && sessionMoves.CLIP) { sessionMoves.clipEnd(); return; }
  if (e.key === "Escape" && AWAY_SEL.size) { awayDays.awayClear(); return; }
  if ((e.key === "Delete" || e.key === "Backspace") && AWAY_SEL.size) { e.preventDefault(); awayDays.awayRemove(); return; }
  if (!(e.metaKey || e.ctrlKey)) return;
  const k = e.key.toLowerCase();
  if (k === "c" && !String(getSelection())) {
    const pl = sessionMoves.calHover?.closest?.(".pl[data-ds]");
    if (pl) { e.preventDefault(); sessionMoves.calCopy(pl.dataset.ds); }
  } else if (k === "v" && sessionMoves.CLIP) {
    const cd = sessionMoves.calHover?.closest?.(".cd[data-day]");
    if (cd) { e.preventDefault(); sessionMoves.calPaste(cd.dataset.day, e.shiftKey); }
  }
});

// once: these follow the pointer anywhere on the page
window.addEventListener("pointermove", (e) => {
  const d = sessionMoves.calDrag;
  if (!d) return;
  d.x = e.clientX; d.y = e.clientY;
  if (!d.on) {
    const moved = Math.hypot(d.x - d.x0, d.y - d.y0) > 6;
    if (moved && d.touch) sessionMoves.calDragEnd();               // it's a scroll, not a drag
    else if (moved) sessionMoves.calDragStart();
    return;
  }
  sessionMoves.calDragPlace();
});

window.addEventListener("pointerup", () => sessionMoves.calDrop());

window.addEventListener("pointercancel", () => { if (!sessionMoves.calDrag?.on) sessionMoves.calDragEnd(); });

// touch: once a drag is on, keep the page still and follow the finger even if the browser cancelled the pointer
document.addEventListener("touchmove", (e) => {
  if (!sessionMoves.calDrag?.on) return;
  e.preventDefault();
  sessionMoves.calDrag.x = e.touches[0].clientX; sessionMoves.calDrag.y = e.touches[0].clientY; sessionMoves.calDragPlace();
}, {passive: false});

document.addEventListener("touchend", () => sessionMoves.calDrop());
