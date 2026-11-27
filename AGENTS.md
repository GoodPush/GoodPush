# AGENTS.md

## Styles: Tailwind CSS + daisyUI

The app's stylesheet is Tailwind CSS v4 with daisyUI 5. Details: `styles/README.md`.

* Edit `styles/app.css`, never `src/goodpush/app/static/app.css` (compiled output, committed so the app, pip
  installs and `packaging/` need no build step).
* After changing `styles/app.css`, or adding a Tailwind / daisyUI class to `static/index.html` or `static/js/`, run
  `python styles/build.py` and commit both files. No Node, no npm: `build.py` downloads Tailwind's standalone CLI.
* Versions are pinned: Tailwind CLI 4.3.5 in `styles/build.py` (with release SHA-256s), daisyUI 5.8.3 vendored in
  `styles/vendor/`. Don't upgrade them as a side effect; an upgrade updates the hashes in
  `tests/test_app_static.py` and the versions in `packaging/notices.py` too.
* Themes are the daisyUI themes `light` and `dark` (`data-theme` on `<html>`; none follows the system). Colours the
  app and the charts use (`--ink`, `--muted`, `--s1`...) are defined in those theme blocks, in hex (ECharts reads
  them and can't parse `oklch`). Keep text colours at WCAG AA (4.5:1) in both themes.
* Tailwind's preflight is off, and daisyUI's `card`, `loading`, `stack`, `status`, `steps`, `toast`, `tab` and
  `modal` are excluded because the app uses those class names itself. `.btn` is daisyUI's button.
* Keep the global `:focus-visible` ring; don't add `outline:none`.
* Check a style change in light and dark mode at laptop and phone widths (the phone bottom bar has Today, Calendar,
  Fitness, Plan and More).
* No GitHub Actions builds: the compiled CSS is built locally and committed.
