"""`goodpush import`: put a downloaded Strava or Garmin export in the right place.

Accepts the .zip exactly as downloaded (or an already-unzipped folder) and
detects which service it came from:

* Strava export   - contains activities.csv (+ activities/ with .fit.gz/.gpx/.tcx)
                    -> data/raw/strava   (media/ photos are skipped)
* Garmin export   - contains DI_CONNECT/  -> data/raw/garmin
* Loose watch FIT files (sleep/HRV/monitoring) -> data/raw/garmin/fit/

A newer export simply overwrites the older one's files; anything downloaded by
`goodpush sync` (the api/ sub-folders) is left alone.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PurePosixPath

SKIP_PREFIXES = ("media/", "__macosx/")


def _detect(names: list[str]) -> tuple[str | None, str]:
    """Return (kind, root_prefix) from a list of archive-relative posix paths."""
    low = [n.lower() for n in names]
    for n, l in zip(names, low):
        p = PurePosixPath(l)
        if p.name == "activities.csv" and len(p.parts) <= 2:
            root = str(PurePosixPath(n).parent)
            return "strava", "" if root == "." else root + "/"
    for n, l in zip(names, low):
        if "di_connect/" in l:
            i = l.index("di_connect/")
            return "garmin", n[:i]
    if names and all(l.endswith((".fit", "/")) for l in low):
        return "garmin_fit", ""
    return None, ""


def export_kind(path: str | Path) -> str | None:
    """"strava" / "garmin" for an account export .zip, else None."""
    try:
        with zipfile.ZipFile(path) as z:
            kind, _ = _detect([n for n in z.namelist() if not n.endswith("/")])
    except (OSError, zipfile.BadZipFile):
        return None
    return kind if kind in ("strava", "garmin") else None


def _safe_target(base: Path, rel: str) -> Path | None:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        return None
    return base.joinpath(*p.parts)


def import_path(proj, src: str | Path, progress=print) -> str:
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(src)
    from .sync import garmin as _g
    if _g.is_browser_file(src):
        _g.import_browser(proj.path("garmin"), src, progress=progress)
        return "garmin_browser"
    if src.is_dir():
        names = [str(p.relative_to(src).as_posix()) for p in src.rglob("*") if p.is_file()]
    elif zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
    elif src.suffix.lower() == ".fit":
        names = [src.name]
        src = src.parent
    else:
        raise ValueError(f"{src}: expected a Strava/Garmin export .zip, a folder, or .fit files")
    kind, root = _detect(names)
    if kind is None:
        raise ValueError(f"{src}: doesn't look like a Strava export (no activities.csv) or a "
                         "Garmin export (no DI_CONNECT folder)")
    dest = {"strava": proj.path("strava"), "garmin": proj.path("garmin"),
            "garmin_fit": proj.path("garmin") / "fit"}[kind]
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    zf = None if src.is_dir() else zipfile.ZipFile(src)
    for name in names:
        if not name.startswith(root):
            continue
        rel = name[len(root):]
        if kind == "strava" and rel.lower().startswith(SKIP_PREFIXES):
            continue
        tgt = _safe_target(dest, rel)
        if tgt is None:
            continue
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if zf is None:
            shutil.copy2(src / name, tgt)
        else:
            with zf.open(name) as fi, open(tgt, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        n += 1
    if zf is not None:
        zf.close()
    label = {"strava": "Strava export", "garmin": "Garmin export",
             "garmin_fit": "Garmin FIT files"}[kind]
    progress(f"{label}: {n} files -> {dest}")
    return kind
