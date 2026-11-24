// Small controls every page reuses: look here before writing one by hand.

// a row of buttons with one on (a view or range choice): a click marks it, then onPick(button); multi: each button
// turns on and off by itself (the last one on stays on), onPick(button) after each change
export class Segmented {
  constructor(selector, onPick, multi = false) {
    this.buttons = [...document.querySelectorAll(`${selector} button`)];
    this.multi = multi;
    this.buttons.forEach((b) => b.addEventListener("click", () => { if (this.multi ? this.toggle(b) : this.mark(b)) onPick(b); }));
  }

  mark(b) { this.buttons.forEach((x) => x.classList.toggle("on", x === b)); return true; }

  toggle(b) {                    // false: it was the last one on
    if (b.classList.contains("on") && this.buttons.filter((x) => x.classList.contains("on")).length === 1) return false;
    b.classList.toggle("on");
    return true;
  }
}

// a setting remembered in this browser (localStorage), still working where storage is off; booleans are kept as
// "1" / "0", null removes it
export class Pref {
  constructor(key, fallback) {
    this.key = key;
    let v = null;
    try { v = localStorage.getItem(key); } catch (e) { /* storage off */ }
    this.value = v === null ? fallback : typeof fallback === "boolean" ? v !== "0" : v;
  }

  set(v) {
    this.value = v;
    try {
      if (v === null) localStorage.removeItem(this.key);
      else localStorage.setItem(this.key, typeof v === "boolean" ? (v ? "1" : "0") : v);
    } catch (e) { /* storage off */ }
  }
}

// server data fetched on first use, once (asked again while it's on its way: that one answers); ready() runs when
// it's back, e.g. a redraw of the page showing it. forget() = fetch it again next time (a job changed it)
export class Lazy {
  constructor(url, ready) {
    this.url = url;              // or a function: the address can depend on a choice (the model rung)
    this.ready = ready;
    this.data = null;
    this.loading = false;
  }

  load() {
    if (this.loading || this.data) return;
    this.loading = true;
    fetch(typeof this.url === "function" ? this.url() : this.url).then((r) => r.json())
      .then((j) => { this.data = j; this.loading = false; this.ready(); })
      .catch(() => { this.loading = false; });
  }

  forget() { this.data = null; }
}
