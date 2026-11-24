// Rides: the searchable list (RidesPage).
import {Page} from "./page.js";
import {$, esc, fmt} from "../util.js";

export const ZONE_NAMES = ["Z1 Recovery", "Z2 Endurance", "Z3 Tempo", "Z4 Threshold", "Z5 VO2max", "Z6 Anaerobic", "Z7 Sprint"];
export const HR_ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5a", "Z5b", "Z5c"];

class RidesPage extends Page {
  // rides list + ride detail
  ridesQ = "";
  ridesRows = [];
  ridesTotal = 0;

  html() {
    return `
  <p><a href="#fitness">← Fitness</a></p>
  <h1>Rides</h1>
  <div class="row" style="margin:10px 0 14px"><input type="search" id="rq" placeholder="Search name, type (threshold, vo2max…) or date" value="${esc(this.ridesQ)}" style="max-width:420px"></div>
  <div class="card"><div class="tw"><table><thead><tr><th>Date</th><th>Ride</th><th>Type</th><th class="n">Min</th><th class="n">km</th><th class="n">TSS</th><th class="n">NP</th><th class="n">IF</th><th class="n">HR</th></tr></thead>
  <tbody id="rbody"><tr><td colspan="9" class="muted">Loading…</td></tr></tbody></table></div>
  <div class="row" style="margin-top:10px"><span class="small muted" id="rcount"></span><span class="spacer"></span><button class="btn sm" id="rmore" hidden>Show more</button></div></div>`;
  }

  rideRow(r) {
    const h = `#ride/${encodeURIComponent(r.id)}`;
    return `<tr class="click" data-href="${h}"><td>${esc(r.date)}</td><td><a href="${h}">${esc(r.name || "Ride")}</a>${r.indoor ? ' <span class="chip">indoor</span>' : ""}</td><td>${esc(r.workout)}</td>
    <td class="n">${fmt(r.min)}</td><td class="n">${r.km ? fmt(r.km, 1) : "–"}</td><td class="n">${fmt(r.tss)}</td><td class="n">${fmt(r.np)}</td><td class="n">${fmt(r.if, 2)}</td><td class="n">${fmt(r.hr)}</td></tr>`;
  }

  async loadRides(reset) {
    if (reset) this.ridesRows = [];
    const r = await fetch(`/api/activities?q=${encodeURIComponent(this.ridesQ)}&offset=${this.ridesRows.length}&limit=50`).then((x) => x.json());
    this.ridesRows = this.ridesRows.concat(r.rows || []); this.ridesTotal = r.total || 0;
    const b = $("#rbody"); if (!b) return;
    b.innerHTML = this.ridesRows.map((r) => this.rideRow(r)).join("") || '<tr><td colspan="9" class="muted">No rides match</td></tr>';
    $("#rcount").textContent = `${this.ridesRows.length} of ${this.ridesTotal}`;
    $("#rmore").hidden = this.ridesRows.length >= this.ridesTotal;
  }

  wire() {
    let t;
    $("#rq").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { this.ridesQ = e.target.value; this.loadRides(true); }, 250); });
    $("#rmore").addEventListener("click", () => this.loadRides(false));
    this.loadRides(true);
  }
}
export const ridesPage = new RidesPage();
