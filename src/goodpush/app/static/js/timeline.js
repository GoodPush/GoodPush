// Blocks on a timeline (BlockTimeline): the drag-and-drop the workout builder and the Calendar's season editor share.
//
// Items sit back to back along x, each `size(item)` units wide at `scale()` px a unit. A palette chip (`palette`)
// dragged in inserts a new item where it is dropped, double-clicked one at the end; an item dragged along moves to
// where it is dropped; its right edge (an element with data-h="right") resizes it; other handles (data-h="name") are the owner's (`handles`). Items are
// elements with data-p = their path ("2", or "2.1" inside a set); only top-level items move.
//
// The owner passes:
//   items()        the top-level list (changed in place)
//   at(path)       the item at a path
//   size(item)     its length in units; scale() px a unit; left() the client x of unit 0; canvas() the drop area
//   resize(item, units)   optional: a new length while its right edge is dragged
//   handles        optional {name: (item, target element) => ({move(dx, dy)})} for its own handles
//   dropped(item, px)     optional: a chip dropped at px from unit 0; true when the owner placed it itself
//   select(path)   the item picked (null: none); preview() redraw while dragging; done(changed) after a release
export class BlockTimeline {
  constructor(o) { this.o = o; }

  indexAt(px) {                   // where an item dropped at px (from unit 0) goes
    const items = this.o.items(), scale = this.o.scale();
    let x = 0;
    for (let i = 0; i < items.length; i++) {
      const w = this.o.size(items[i]) * scale;
      if (px < x + w / 2) return i;
      x += w;
    }
    return items.length;
  }

  palette(chip, make) {           // a palette chip: drag it in, or double-click it: a new item at the end
    chip.addEventListener("pointerdown", (e) => this.chip(e, chip, make));
    chip.addEventListener("dblclick", () => this.append(make()));
  }

  append(item) {
    const items = this.o.items();
    items.push(item); this.o.select([items.length - 1]); this.o.done(true);
  }

  chip(e, chip, make) {           // a chip dragged in (a ghost follows the pointer); a plain click adds nothing
    e.preventDefault();
    const ghost = chip.cloneNode(true); ghost.classList.add("b-ghost"); document.body.appendChild(ghost);
    const x0 = e.clientX, y0 = e.clientY;
    const move = (ev) => { ghost.style.left = ev.clientX + 8 + "px"; ghost.style.top = ev.clientY + 8 + "px"; };
    const up = (ev) => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); ghost.remove();
      const r = this.o.canvas().getBoundingClientRect(), item = make();
      const dragged = Math.hypot(ev.clientX - x0, ev.clientY - y0) > 6;
      const inside = ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
      if (!dragged || !inside) return;
      const px = ev.clientX - this.o.left();
      if (this.o.dropped?.(item, px)) { this.o.done(true); return; }
      const items = this.o.items(), at = this.indexAt(px);
      items.splice(at, 0, item); this.o.select([at]); this.o.done(true);
    };
    move(e);
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  }

  pointer(e) {                    // on the timeline: select, and drag an item / its right edge / the owner's handles
    const hit = e.target.closest("[data-p]");
    if (!hit) { this.o.select(null); this.o.done(false); return; }
    const path = String(hit.dataset.p).split(".").map(Number), item = this.o.at(path), handle = e.target.dataset?.h;
    this.o.select(path);
    e.preventDefault(); e.stopPropagation();
    const custom = handle && handle !== "right" ? this.o.handles?.[handle]?.(item, e.target) : null;
    const sx = e.clientX, sy = e.clientY, size0 = this.o.size(item);
    let moved = false;
    const move = (ev) => {
      const dx = ev.clientX - sx, dy = ev.clientY - sy;
      if (!moved && Math.hypot(dx, dy) < 4) return;
      moved = true;
      if (handle === "right" && this.o.resize) this.o.resize(item, size0 + dx / this.o.scale());
      else if (custom) custom.move(dx, dy);
      else return;                                   // an item dragged along: placed on release
      this.o.preview();
    };
    const up = (ev) => {
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up);
      if (moved && !handle && path.length === 1) {   // moved along: to where it was dropped
        const items = this.o.items(), [it] = items.splice(path[0], 1);
        const at = Math.min(this.indexAt(ev.clientX - this.o.left()), items.length);
        items.splice(at, 0, it); this.o.select([at]);
      }
      this.o.done(moved);
    };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  }
}
