// The ride's route (RideMap): Leaflet + OSM tiles when online, a plain SVG route offline.
import {css} from "./util.js";
import {ridePage} from "./pages/ride.js";
import {el} from "./charts.js";

// the route on a map: Leaflet and the map tiles come from the internet when the ride opens (nothing is
// stored); offline, the route is drawn on its own
const LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/leaflet";

class RideMap {
  mapGeo = null;

  leafletLoad = null;

  loadLeaflet() {
    if (window.L) return Promise.resolve(window.L);
    if (!this.leafletLoad) this.leafletLoad = new Promise((ok, fail) => {
      const css = document.createElement("link");
      css.rel = "stylesheet"; css.href = LEAFLET + ".css";
      const js = document.createElement("script");
      js.src = LEAFLET + ".js";
      js.onload = () => ok(window.L);
      js.onerror = () => fail(new Error("offline"));          // offline: no retry until the page reloads
      document.head.append(css, js);
    });
    return this.leafletLoad;
  }

  drawMap() {
    const box = document.getElementById("rmap"); const tr = ridePage.RIDE.track || [];
    if (!box || tr.length < 2) { this.mapGeo = null; return; }
    if (this.mapGeo?.map && this.mapGeo.box === box) { this.mapGeo.map.invalidateSize(); return; }   // chart redraws keep the map
    const id = ridePage.RIDE.id;
    this.loadLeaflet().then((L) => { if (ridePage.RIDE?.id === id && document.getElementById("rmap") === box) this.drawMapTiles(L, box, tr); })
      .catch(() => this.drawRoute(box, tr));
  }

  drawMapTiles(L, box, tr) {
    if (this.mapGeo?.map) this.mapGeo.map.remove();
    box.innerHTML = "";
    box.parentNode.querySelectorAll(".maplink").forEach((n) => n.remove());
    const map = L.map(box, {scrollWheelZoom: false});
    const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      {maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'});
    const cyc = L.tileLayer("https://{s}.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
      {maxZoom: 19, attribution: '<a href="https://www.cyclosm.org">CyclOSM</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'});
    osm.addTo(map);
    L.control.layers({"Map": osm, "Cycling (CyclOSM)": cyc}).addTo(map);
    const line = L.polyline(tr, {color: css("--s1"), weight: 4, opacity: 0.9}).addTo(map);
    const dot = (q, c, t) => L.circleMarker(q, {radius: 6, color: "#fff", weight: 2, fillColor: c, fillOpacity: 1}).bindTooltip(t).addTo(map);
    dot(tr[tr.length - 1], css("--s2"), "Finish"); dot(tr[0], css("--s3"), "Start");
    const mk = L.circleMarker(tr[0], {radius: 7, color: "#fff", weight: 2, fillColor: css("--ink"), fillOpacity: 1});
    map.fitBounds(line.getBounds(), {padding: [16, 16]});
    this.mapGeo = {n: tr.length, map, box,
      show: (i) => { mk.setLatLng(tr[i]); if (!map.hasLayer(mk)) mk.addTo(map); },
      hide: () => mk.remove()};
  }

  drawRoute(box, tr) {   // offline: the route alone
    box.innerHTML = "";
    box.parentNode.querySelectorAll(".maplink").forEach((n) => n.remove());
    const W = box.clientWidth || 500, H = box.clientHeight || 300, pad = 14;
    const lat0 = tr.reduce((p, q) => p + q[0], 0) / tr.length, k = Math.cos(lat0 * Math.PI / 180);
    const xsv = tr.map((q) => q[1] * k), ysv = tr.map((q) => -q[0]);
    const xmin = Math.min(...xsv), xmax = Math.max(...xsv), ymin = Math.min(...ysv), ymax = Math.max(...ysv);
    const sc = Math.min((W - 2 * pad) / (xmax - xmin || 1e-6), (H - 2 * pad) / (ymax - ymin || 1e-6));
    const ox = (W - (xmax - xmin) * sc) / 2, oy = (H - (ymax - ymin) * sc) / 2;
    const P = tr.map((_, i) => [ox + (xsv[i] - xmin) * sc, oy + (ysv[i] - ymin) * sc]);
    const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Route"}, box);
    el("path", {d: "M" + P.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L"), fill: "none", stroke: css("--surface"), "stroke-width": 6, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg);
    el("path", {d: "M" + P.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L"), fill: "none", stroke: css("--s1"), "stroke-width": 3, "stroke-linejoin": "round", "stroke-linecap": "round"}, svg);
    const dot = (p, c, t) => { const d = el("circle", {cx: p[0], cy: p[1], r: 6, fill: c, stroke: css("--surface"), "stroke-width": 2}, svg); const ti = el("title", {}, d); ti.textContent = t; };
    dot(P[P.length - 1], css("--s2"), "Finish"); dot(P[0], css("--s3"), "Start");
    const mk = el("circle", {r: 7, fill: css("--ink"), stroke: css("--surface"), "stroke-width": 2, visibility: "hidden"}, svg);
    this.mapGeo = {n: P.length,
      show: (i) => { mk.setAttribute("cx", P[i][0]); mk.setAttribute("cy", P[i][1]); mk.setAttribute("visibility", "visible"); },
      hide: () => mk.setAttribute("visibility", "hidden")};
    const mid = tr[Math.floor(tr.length / 2)];
    const link = document.createElement("div");
    link.className = "small maplink"; link.style.marginTop = "6px";
    link.innerHTML = `<span class="legend" style="display:inline-flex"><span><i class="dot" style="background:var(--s3)"></i>Start</span><span><i class="dot" style="background:var(--s2)"></i>Finish</span></span> · <a href="https://www.openstreetmap.org/?mlat=${mid[0]}&mlon=${mid[1]}#map=13/${mid[0]}/${mid[1]}" target="_blank" rel="noopener">Open in OpenStreetMap</a>`;
    box.after(link);
  }

  moveMapMarker(t) {   // where you were at minute t
    if (!this.mapGeo || !ridePage.RIDE.streams) return;
    if (t == null) { this.mapGeo.hide(); return; }
    const tot = ridePage.RIDE.streams.t_min[ridePage.RIDE.streams.t_min.length - 1] || 1;
    this.mapGeo.show(Math.min(this.mapGeo.n - 1, Math.max(0, Math.round(t / tot * (this.mapGeo.n - 1)))));
  }
}
export const rideMap = new RideMap();
