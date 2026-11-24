"""DFA alpha1 (from the alphaHRV Connect IQ field) as a readiness signal.

Alpha1 measures how "organised" your beat-to-beat heart rhythm is. At easy intensity it sits
around 1.0; it falls through ~0.75 near the aerobic threshold (VT1) and ~0.5 near the second
threshold. On a given day, at a given easy power, an alpha1 that is clearly *lower* than your
normal (or, less often, clearly *higher*) tends to go with fatigue, illness or a flat day.

What goodpush does
1. ingest:   rides whose .fit file contains alphaHRV data get a compact 5-second series
             (alpha1, power, heart rate, artifacts) saved next to the other streams.
2. features: for each ride, alpha1 / HR / power in the warm-up (minutes 5-20) and as the
             session gets going (minutes 20-40), using only steady easy riding (50-75 % FTP)
             and clean data (artifacts < 5 %); plus an estimate of the power where alpha1
             crosses 0.75 (aerobic threshold, HRVT1) when the ride covers a range of power.
3. baseline: each ride is compared with your previous 6 weeks of rides in the same setting
             (indoor / outdoor), as a z-score. Too high and too low both count.
"""
from __future__ import annotations

import gzip
import os
import re
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hashlib
import numpy as np
import pandas as pd

from .io.stamps import in_seconds
from .metrics.power import ride_seconds
from .progress import Progress

ALPHA_VERSION = 3       # 2: alpha1 times on the ride timeline (long stops cut); 3: indoors goodpush's own alpha1
INDOOR = {"indoor_cycling", "virtual_activity"}    # FIT sub-sports ridden indoors: goodpush's own alpha1 there
INDEX = "alpha_index.csv"
BIN_S = 5
WARM = (5, 20)        # minutes
SESSION = (20, 40)
EASY = (0.50, 0.75)   # fraction of FTP
MIN_SECONDS = 180
RIDING_W = 20.0       # a bin at this power or more: the ride has begun (WARM / SESSION minutes count from there)
MAX_ARTIFACTS = 5.0
COMPANION_S = 300         # a Garmin recording starting this close to a goodpush one is the same ride


def _raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


HRV_DEF = re.compile(rb"[\x40-\x4f\x60-\x6f]\x00[\x00\x01](?:\x4e\x00|\x00\x4e)")   # FIT hrv (78) definition
MIN_BEATS = 300                 # beat intervals a ride needs for goodpush's own alpha1


def extract(job: tuple) -> dict:
    """Worker: one activity file -> 5 s series saved as <id>.a1.npz (if it has alpha1). Indoors goodpush's own
    alpha1 from the beat intervals (metrics.dfa) where the file has them, else alphaHRV's; outdoors alphaHRV's, else
    goodpush's (`source` says which). With a sixth item, the path of the ride the series is for (a goodpush recording
    with a Garmin copy, `companions`): its own beat intervals first, and the series goes on that ride's timeline by
    clock time, with its power."""
    act_id, path, key, out_dir = job[:4]
    own = job[5] if len(job) > 5 else ""
    row = {"id": act_id, "source_key": key, "has_alpha": False, "error": "", "source": "",
           "stat_key": job[4] if len(job) > 4 else ""}
    try:
        data = _raw(Path(path))
        mine = _raw(Path(own)) if own else b""
        if not path.lower().replace(".gz", "").endswith(".fit") or not (
                b"Alpha1" in data or HRV_DEF.search(data) or HRV_DEF.search(mine)):
            return row
        from .io.fit import read_activity, read_developer_fields, read_rr
        act = read_activity(data)
        rec = act["records"]
        own_rec = read_activity(mine)["records"] if mine else None
        rr = read_rr(mine) if mine else np.zeros(0)
        rr_rec = own_rec if rr.size >= MIN_BEATS else rec                 # the beats' own clock
        if rr.size < MIN_BEATS:
            rr = read_rr(data)
        dev = read_developer_fields(data)["values"] if b"Alpha1" in data else {}
        a1 = dev.get("Alpha1") if dev.get("Alpha1") and len(dev["Alpha1"]) == len(rec["time"]) else None
        indoor = bool(own) or act["sub_sport"] in INDOOR
        if rr.size >= MIN_BEATS and (indoor or a1 is None):          # goodpush's own, from the beats
            from .metrics import dfa
            t = np.asarray(rr_rec["time"], float)
            ours, share = dfa.series(rr, t[0], t)
            df = pd.DataFrame({"t": t, "a1": ours, "p": rr_rec["power"], "hr": rr_rec["heart_rate"],
                               "art": share * 100}).apply(pd.to_numeric, errors="coerce")
            row["source"] = "own"      # our own alpha1
        elif a1 is not None:                                          # alphaHRV's
            art = dev.get("Artifacts") or [None] * len(a1)
            df = pd.DataFrame({"t": rec["time"], "a1": a1, "p": rec["power"], "hr": rec["heart_rate"],
                               "art": art}).apply(pd.to_numeric, errors="coerce")
            row["source"] = "alphaHRV"
        else:
            return row
        df = df.dropna(subset=["t"]).sort_values("t")
        if own:
            df = on_ride(df, own_rec)
        else:
            df["t"] = ride_seconds(df["t"].to_numpy())   # the ride's timeline (long stops cut, as the 1 Hz streams)
        df = df[df["t"] < 12 * 3600]
        df["a1"] = df["a1"].where((df["a1"] > 0.05) & (df["a1"] < 2.5))
        b = (df["t"] // BIN_S).astype(int)
        g = df.groupby(b).mean()
        np.savez_compressed(Path(out_dir) / f"{act_id}.a1.npz", t=(g.index.to_numpy() * BIN_S).astype(np.int32),
                            a1=g["a1"].to_numpy(np.float32), p=g["p"].to_numpy(np.float32),
                            hr=g["hr"].to_numpy(np.float32), art=g["art"].to_numpy(np.float32))
        row["has_alpha"] = True
        row["n_bins"] = int(len(g))
    except Exception as e:  # noqa: BLE001 - one odd file must not stop the rest
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def on_ride(df: pd.DataFrame, rec: dict) -> pd.DataFrame:
    """alpha1 rows (t: unix s) of a companion recording on ride `rec`'s timeline (its records, as read_activity):
    t is that ride's seconds, power its power at the same clock second; rows outside the ride left out."""
    rt = np.asarray(rec["time"], float)
    order = np.argsort(rt)
    rt = rt[order]
    t = df["t"].to_numpy(float)
    i = np.clip(np.searchsorted(rt, t), 0, rt.size - 1)
    pw = pd.to_numeric(pd.Series(rec["power"]).iloc[order], errors="coerce").to_numpy(float)
    return df.assign(t=ride_seconds(rt)[i], p=pw[i])[(t >= rt[0]) & (t <= rt[-1])]


def companions(proj, rides: pd.DataFrame) -> dict[str, Path]:
    """goodpush recordings with a Garmin recording of the same ride (started within COMPANION_S: load_activities_csv
    keeps only the goodpush one): {ride id: the Garmin file}, whose alphaHRV the ride takes."""
    from .io.strava import load_garmin_table
    g = load_garmin_table(proj.path("garmin"), proj.path("strava"))
    mine = rides[rides["id"].astype(str).str.startswith("goodpush_")]
    if g is None or not len(g) or not len(mine):
        return {}
    g = g[g["filename"].astype(bool) & g["start_utc"].notna()].sort_values("start_utc")
    gt = g["start_utc"].to_numpy("datetime64[ns]")
    mt = mine["start_utc"].to_numpy("datetime64[ns]")
    j = np.clip(np.searchsorted(gt, mt), 1, max(len(gt) - 1, 1)) if len(gt) > 1 else np.zeros(len(mt), int)
    j = np.where(np.abs(gt[np.maximum(j - 1, 0)] - mt) < np.abs(gt[j] - mt), np.maximum(j - 1, 0), j)
    near = np.abs(gt[j] - mt) <= np.timedelta64(COMPANION_S, "s")
    return {str(a): proj.path("strava") / f for a, f, ok in zip(mine["id"], g["filename"].to_numpy()[j], near) if ok}


def _key(p: Path) -> str:
    st = p.stat()
    with open(p, "rb") as fh:
        head = fh.read(65536)
    return f"{st.st_size}-{zlib.crc32(head):08x}-a{ALPHA_VERSION}"


def ingest_alpha(proj, acts: pd.DataFrame, workers: int | None = None, progress=print,
                 stats: dict | None = None) -> pd.DataFrame:
    """Extract alpha1 series for rides that have them (cached; only new files are read)."""
    strava_dir = proj.path("strava")
    out_dir = proj.processed / "streams"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = proj.processed / INDEX
    cache = pd.read_csv(idx_path, dtype={"id": str, "stat_key": str}) if idx_path.exists() \
        else pd.DataFrame(columns=["id", "source_key", "stat_key"])
    if "stat_key" not in cache:
        cache["stat_key"] = ""
    known = dict(zip(cache["id"].astype(str), cache["source_key"]))
    known_stat = dict(zip(cache["id"].astype(str), in_seconds(cache["stat_key"])))
    from .ingest import stat_index
    jobs, migrated = [], {}
    rides = acts[acts["type"].astype(str).str.contains("Ride|Bike|Cycl|ride", regex=True)] if "type" in acts else acts
    files = {str(r.id): strava_dir / r.filename for r in rides.itertuples()
             if isinstance(getattr(r, "filename", ""), str) and r.filename}
    own = {}
    if "start_utc" in rides:
        alt = companions(proj, rides)                    # goodpush rides: alpha1 from their Garmin copy
        own = {aid: files[aid] for aid in alt if aid in files}
        files.update({aid: alt[aid] for aid in own})
    if stats is None:
        stats = stat_index(files.values())
    elif own:
        stats = {**stats, **stat_index(files[a] for a in own)}
    for aid, p in files.items():
        sk = stats.get(str(p))
        sk = sk and f"{sk}-a{ALPHA_VERSION}"          # a new ALPHA_VERSION re-reads every file
        if sk is None or known_stat.get(aid) == sk:
            continue
        k = _key(p)
        if known.get(aid) == k:
            migrated[aid] = sk
            continue
        jobs.append((aid, str(p), k, str(out_dir), sk, str(own.get(aid, ""))))
    if migrated:
        cache["stat_key"] = cache["id"].astype(str).map(migrated).fillna(cache["stat_key"].astype(str).replace("nan", ""))
        if not jobs:
            cache.to_csv(idx_path, index=False)
    rows = []
    if jobs:
        progress(f"alphaHRV: checking {len(jobs)} ride files")
        n = workers or max(1, (os.cpu_count() or 2) - 1)
        if n == 1 or len(jobs) < 8:
            for i, j in enumerate(jobs, 1):
                rows.append(extract(j))
                Progress.step(progress, "Reading alpha1", i, len(jobs))
        else:
            with ProcessPoolExecutor(max_workers=n) as ex:
                for i, f in enumerate(as_completed([ex.submit(extract, j) for j in jobs]), 1):
                    rows.append(f.result())
                    Progress.step(progress, "Reading alpha1", i, len(jobs))
        new = pd.DataFrame(rows)
        cache = pd.concat([cache[~cache["id"].astype(str).isin(new["id"])], new], ignore_index=True)
        cache.to_csv(idx_path, index=False)
        progress(f"alphaHRV: {int(new['has_alpha'].fillna(False).astype(bool).sum())} new rides with alpha1")
    return cache


# --- per-ride features ---------------------------------------------------------------------
STEADY_BINS = 24          # alpha1 uses a ~2-minute window: power must have been steady that long
STEADY_SD = 0.10          # ... within 10 % of FTP
P_RANGE = (0.35, 0.85)    # power range used for the warm-up / session comparison (x FTP)
TH_RANGE = (0.40, 1.15)   # power range used for the aerobic-threshold estimate (x FTP)
TH2_RANGE = (0.60, 1.40)  # ... and for the second threshold (alpha1 = 0.5), which sits higher
TH_BIN_W = 10             # aerobic threshold: alpha1 averaged in 10 W power bins
TH_MAX_W = 700
LT1_A1, LT2_A1 = 0.75, 0.5   # HRVT1 / HRVT2 (Rogers et al. 2021; cycling: Frontiers Physiol 2024)
RAMP_WIN = 10             # ramp test: at least 10 minutes of ...
RAMP_SLOPE = (4.0, 40.0)  # ... power climbing 4-40 W per minute ...
RAMP_R2 = 0.9             # ... in a near straight line, nearly every minute at least as high as the
RAMP_UP = 0.8             # last (within 3 %: steps can be 2 min long),
RAMP_RISE = 0.6           # rising by >= 60 % of FTP to
RAMP_TOP = 1.1            # >= 110 % of FTP, then stopping (power < 70 % of the top within 3 min)
RAMP_STOP = 0.7
BASE_DAYS = 42


# All of this is vectorised with numpy over every ride at once (no per-ride pandas work):
# the rides' 5-s bins are concatenated into flat arrays with a ride index, and per-ride
# statistics are group medians / sums; the 6-week baselines use prefix sums over rides
# sorted by date.

def _group_median(v, g, n):
    """Median of v within each group g (0..n-1), ignoring NaN; NaN for empty groups."""
    m = np.isfinite(v)
    v, g = v[m], g[m]
    out = np.full(n, np.nan)
    if not v.size:
        return out, np.zeros(n, int)
    o = np.lexsort((v, g))
    v, g = v[o], g[o]
    cnt = np.bincount(g, minlength=n)
    start = np.cumsum(cnt) - cnt
    has = cnt > 0
    lo = start + (cnt - 1) // 2
    hi = start + cnt // 2
    out[has] = 0.5 * (v[lo[has]] + v[hi[has]])
    return out, cnt


def _window_sums(x, lo, hi):
    """Sums of rows lo..hi-1 of x (prefix sums), for each row's own [lo, hi)."""
    c = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    return c[hi] - c[lo]


def _load_rides(proj, acts):
    sdir = proj.processed / "streams"
    ids = acts["id"].astype(str).str.replace(r"\.0$", "", regex=True).to_numpy()
    ftp = pd.to_numeric(acts["ftp"], errors="coerce").to_numpy(float) if "ftp" in acts else np.full(len(acts), np.nan)
    dates = pd.to_datetime(acts["date"]).to_numpy("datetime64[D]")
    indoor = acts["indoor"].fillna(False).astype(bool).to_numpy() if "indoor" in acts else np.zeros(len(acts), bool)
    from .io import streamstore
    if streamstore.available() and streamstore.h5_path(proj.processed).exists():
        # everything in one read per channel from streams.h5, then a vectorised gather
        keys, lens, data = streamstore.read_all(proj.processed, "alpha")
        pos = {k: i for i, k in enumerate(keys)}
        seg = np.array([pos.get(a, -1) for a in ids], np.int64)
        keep = np.flatnonzero(np.isfinite(ftp) & (ftp > 0) & (seg >= 0))
        if not keep.size:
            return None
        keep = keep[np.argsort(dates[keep], kind="stable")]
        sg = seg[keep]
        starts = np.cumsum(lens) - lens
        n = lens[sg]
        take = np.repeat(starts[sg] - (np.cumsum(n) - n), n) + np.arange(n.sum())
        return {"ids": ids[keep], "dates": dates[keep], "ftp": ftp[keep], "indoor": indoor[keep],
                "rid": np.repeat(np.arange(len(keep)), n),
                **{c: data[c][take].astype(float) for c in ("t", "a1", "p", "hr", "art")}, "n": len(keep)}
    # without h5py: the per-ride .npz files (one directory listing instead of a stat per ride)
    present = {}
    if sdir.exists():
        with os.scandir(sdir) as it:
            for e in it:
                if e.name.endswith(".a1.npz"):
                    st = e.stat()
                    present[e.name[:-7]] = (e.path, st.st_size, st.st_mtime_ns)
    ok = np.isfinite(ftp) & (ftp > 0) & np.isin(ids, list(present)) if present else np.zeros(len(ids), bool)
    keep = list(np.flatnonzero(ok))
    sig = hashlib.md5(repr([(ids[i],) + present[ids[i]][1:] for i in keep]).encode()).hexdigest()
    cache = sdir / "_alpha_all.npz"
    arrs = None
    if keep and cache.exists():
        try:
            z = np.load(cache)
            if str(z["sig"]) == sig:
                lens0, flat0 = z["lens"], z["flat"].astype(float)
                arrs = np.split(flat0, np.cumsum(lens0)[:-1], axis=1)
        except Exception:
            arrs = None
    if arrs is None:
        arrs = []
        for i in keep:
            z = np.load(present[ids[i]][0])
            arrs.append(np.vstack([z["t"].astype(float), z["a1"].astype(float), z["p"].astype(float),
                                   z["hr"].astype(float), z["art"].astype(float)]))
        if keep:
            try:
                np.savez(cache, sig=np.array(sig), lens=np.array([a.shape[1] for a in arrs]),
                         flat=np.hstack(arrs).astype(np.float32))
            except OSError:
                pass
    if not keep:
        return None
    keep = np.array(keep)
    order = np.argsort(dates[keep], kind="stable")
    keep = keep[order]
    arrs = [arrs[k] for k in order]
    lens = np.array([a.shape[1] for a in arrs])
    flat = np.hstack(arrs)
    return {"ids": ids[keep], "dates": dates[keep], "ftp": ftp[keep], "indoor": indoor[keep],
            "rid": np.repeat(np.arange(len(keep)), lens), "t": flat[0], "a1": flat[1], "p": flat[2],
            "hr": flat[3], "art": flat[4], "n": len(keep)}


def _steady(R):
    """Per 5-s bin: mean / SD of power over the previous 2 minutes (same ride only)."""
    p, rid = R["p"], R["rid"]
    w = STEADY_BINS
    fin = np.isfinite(p)
    p0 = np.where(fin, p, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(p0)])
    c2 = np.concatenate([[0.0], np.cumsum(p0 * p0)])
    cn = np.concatenate([[0.0], np.cumsum(fin)])
    i = np.arange(p.size)
    j = np.maximum(i - w + 1, 0)
    same = rid[j] == rid              # the 2-minute window must not reach into the previous ride
    n = cn[i + 1] - cn[j]
    s1 = c1[i + 1] - c1[j]
    s2 = c2[i + 1] - c2[j]
    with np.errstate(all="ignore"):
        mean = s1 / n
        var = (s2 - s1 * s1 / n) / (n - 1)
    ok = same & (n >= w - 4)
    return np.where(ok, mean, np.nan), np.where(ok, np.sqrt(np.maximum(var, 0)), np.nan)


def _threshold_from_bins(sums, cnts, total):
    """Vectorised 0.75-crossing for each row of 10 W bin sums / counts."""
    nb = sums.shape[1]
    # tame single odd bins: pool each bin with its neighbours
    pad = lambda x: np.pad(x, ((0, 0), (1, 1)))  # noqa: E731
    s3 = pad(sums)[:, :-2] + sums + pad(sums)[:, 2:]
    c3 = pad(cnts)[:, :-2] + cnts + pad(cnts)[:, 2:]
    with np.errstate(all="ignore"):
        m = s3 / c3
    valid = cnts >= 6
    above = valid & (m >= 0.75)
    below = valid & (m < 0.75)
    has_lo = above.any(1)
    lo = nb - 1 - np.argmax(above[:, ::-1], axis=1)
    after = below & (np.arange(nb)[None, :] > lo[:, None])
    has_hi = after.any(1)
    hi = np.argmax(after, axis=1)
    r = np.arange(sums.shape[0])
    m_lo, m_hi = m[r, lo], m[r, hi]
    with np.errstate(all="ignore"):
        p75 = (lo + (m_lo - 0.75) / (m_lo - m_hi) * (hi - lo)) * TH_BIN_W + TH_BIN_W / 2
    ok = has_lo & has_hi & (valid.sum(1) >= 4) & (total >= 60)
    return np.where(ok, p75, np.nan)


def _pool3(x):
    """Each 10 W bin pooled with its neighbours (rows = rides / windows)."""
    pad = np.pad(x, ((0, 0), (1, 1)))
    return pad[:, :-2] + x + pad[:, 2:]


def _sustained_crossing(m, valid, level, after_w=None):
    """Power where a binned value drops below `level` for good: the first valid bin below the
    level whose next valid bin is below it too (alpha1 levels off near 0.5 at high power, so a
    single dip mustn't count), interpolated from the last valid bin above the level before it.
    `after_w`: only look above this power (e.g. LT1). NaN when it never happens."""
    rows, nb = m.shape
    idx = np.arange(nb)[None, :]
    r = np.arange(rows)[:, None]
    below = valid & (m < level)
    first_after = np.minimum.accumulate(np.where(valid, idx, nb)[:, ::-1], axis=1)[:, ::-1]
    nxt = np.concatenate([first_after[:, 1:], np.full((rows, 1), nb)], axis=1)   # next valid bin
    m_next = np.where(nxt < nb, m[r, np.minimum(nxt, nb - 1)], np.nan)
    hit = below & (m_next < level)
    if after_w is not None:
        hit &= (idx * TH_BIN_W + TH_BIN_W / 2) > np.nan_to_num(after_w, nan=-1)[:, None]
    has = hit.any(1)
    b = np.argmax(hit, axis=1)
    last_above = np.maximum.accumulate(np.where(valid & (m >= level), idx, -1), axis=1)
    pa = np.where(b > 0, last_above[np.arange(rows), np.maximum(b - 1, 0)], -1)
    m_b, m_a = m[np.arange(rows), b], m[np.arange(rows), np.maximum(pa, 0)]
    with np.errstate(all="ignore"):
        pos = pa + (m_a - level) / (m_a - m_b) * (b - pa)
    return np.where(has & (pa >= 0), pos * TH_BIN_W + TH_BIN_W / 2, np.nan)


def _at_power(w, mean):
    """Linear interpolation of a per-bin mean (e.g. HR) at power w (one value per row)."""
    pos = (w - TH_BIN_W / 2) / TH_BIN_W
    ok = np.isfinite(pos)
    lo = np.clip(np.floor(np.nan_to_num(pos)).astype(int), 0, mean.shape[1] - 1)
    hi = np.minimum(lo + 1, mean.shape[1] - 1)
    r = np.arange(mean.shape[0])
    f = np.nan_to_num(pos) - lo
    v = mean[r, lo] * (1 - f) + mean[r, hi] * f
    return np.where(ok, v, np.nan)


def _ramps(R, p2):
    """Ramp tests inside rides: >= RAMP_WIN minutes of power climbing in a near straight line,
    rising by RAMP_RISE x FTP to at least RAMP_TOP x FTP (not asked of a planned ramp session you rode: R["planned"]
    per ride), then stopping. Returns per-ride rows with HRVT1 / HRVT2
    taken from the ramp itself (power = the 2-min average alpha1 is computed over, HR at the time),
    the setting the alpha1 thresholds were validated in."""
    n, rid, t = R["n"], R["rid"], R["t"].astype(np.int64)
    cols = ["ramp_start_min", "ramp_min", "ramp_top_w", "ramp_lt1_w", "ramp_lt1_hr", "ramp_lt2_w", "ramp_lt2_hr",
            "ramp_max_hr"]
    out = {c: np.full(n, np.nan) for c in cols}
    if not t.size:
        return out
    # 1-min block means of power, rides back to back
    key = rid.astype(np.int64) * 100_000 + t // 60
    uk, inv = np.unique(key, return_inverse=True)
    fin = np.isfinite(R["p"])
    cnt = np.bincount(inv, fin, uk.size)
    with np.errstate(all="ignore"):
        bp = np.where(cnt >= 6, np.bincount(inv, np.where(fin, R["p"], 0.0), uk.size) / cnt, np.nan)
    brid, bmin = uk // 100_000, (uk % 100_000).astype(float)
    # straight-line fit of power on minute over every RAMP_WIN consecutive blocks (prefix sums)
    w = RAMP_WIN
    good = np.isfinite(bp)
    x0 = np.where(good, bmin, 0.0)
    y0 = np.where(good, bp, 0.0)
    c = lambda v: np.concatenate([[0.0], np.cumsum(v)])  # noqa: E731
    S1, Sx, Sy, Sxx, Sxy, Syy = c(good * 1.0), c(x0), c(y0), c(x0 * x0), c(x0 * y0), c(y0 * y0)
    k = np.arange(w - 1, uk.size)
    j = k - w + 1
    N, X, Y, XX, XY, YY = (S[k + 1] - S[j] for S in (S1, Sx, Sy, Sxx, Sxy, Syy))
    with np.errstate(all="ignore"):
        sxx, sxy, syy = XX - X * X / N, XY - X * Y / N, YY - Y * Y / N
        slope = sxy / sxx
        r2 = sxy * sxy / (sxx * syy)
    same = (brid[j] == brid[k]) & (bmin[k] - bmin[j] == w - 1) & (N >= w - 1)
    q = same & (slope >= RAMP_SLOPE[0]) & (slope <= RAMP_SLOPE[1]) & (r2 >= RAMP_R2)
    # blocks covered by a qualifying window -> runs
    cov = np.zeros(uk.size + 1)
    np.add.at(cov, j[q], 1)
    np.add.at(cov, k[q] + 1, -1)
    covered = np.cumsum(cov)[:-1] > 0
    start = covered & ~np.r_[False, covered[:-1] & (brid[1:] == brid[:-1])]
    run = np.cumsum(start) - 1
    if not covered.any():
        return out
    rr = run[covered]
    b_idx = np.flatnonzero(covered)
    first = np.full(rr.max() + 1, uk.size)
    np.minimum.at(first, rr, b_idx)
    last = np.zeros(rr.max() + 1, int)
    np.maximum.at(last, rr, b_idx)
    ftp = R["ftp"][brid[first]]
    lo_p = np.nanmean(np.c_[bp[first], bp[np.minimum(first + 1, last)]], axis=1)
    hi_p = np.nanmean(np.c_[bp[last], bp[np.maximum(last - 1, first)]], axis=1)
    up = np.r_[False, (bp[1:] >= 0.97 * bp[:-1]) & (brid[1:] == brid[:-1])]   # not lower than the minute before
    n_up = np.bincount(rr[1:], up[b_idx][1:] & (run[b_idx][1:] == run[b_idx][:-1]), rr.max() + 1)
    steps = (last - first).astype(float)
    # ramp to failure: within 3 minutes after the top, power falls well below it (or the ride ends)
    nxt = last[None, :] + np.arange(1, 4)[:, None]
    after = np.where((nxt < uk.size) & (brid[np.minimum(nxt, uk.size - 1)] == brid[last][None, :]),
                     bp[np.minimum(nxt, uk.size - 1)], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stopped = (np.nanmin(after, axis=0) < RAMP_STOP * hi_p) | np.isnan(after).all(0)
    # a planned ramp session you rode is a ramp test whatever FTP says (FTP only set its watts; the test is what
    # tells whether FTP moved): a higher FTP set for the day after the fact had dropped one below RAMP_TOP)
    planned = R.get("planned", np.zeros(R["n"], bool))[brid[first]]
    sized = planned | ((hi_p - lo_p >= RAMP_RISE * ftp) & (hi_p >= RAMP_TOP * ftp))
    ok = sized & (n_up >= RAMP_UP * steps) & stopped
    if not ok.any():
        return out
    runs = pd.DataFrame({"ride": brid[first], "t0": bmin[first] * 60, "t1": bmin[last] * 60 + 60,
                         "len": last - first + 1, "top": hi_p})[ok]
    runs = runs.sort_values("len").groupby("ride").tail(1)      # the longest ramp of a ride
    a1 = np.where(R["art"] > MAX_ARTIFACTS, np.nan, R["a1"])
    for x in runs.itertuples():      # a handful of tests: per-test crossing on its 5-s bins
        m = (rid == x.ride) & (t >= x.t0) & (t <= x.t1 + 120)     # + 2 min: alpha1 lags the ramp
        a, pw, hr = a1[m], p2[m], R["hr"][m]
        a = pd.Series(a).rolling(3, min_periods=2).mean().to_numpy()     # 15 s
        g = int(x.ride)
        out["ramp_start_min"][g], out["ramp_min"][g], out["ramp_top_w"][g] = x.t0 / 60, x.len, x.top
        out["ramp_max_hr"][g] = np.nanmax(hr) if np.isfinite(hr).any() else np.nan
        for lvl, kw, kh in ((LT1_A1, "ramp_lt1_w", "ramp_lt1_hr"), (LT2_A1, "ramp_lt2_w", "ramp_lt2_hr")):
            below = np.isfinite(a) & (a < lvl)
            if below.size < 7:
                continue
            stay = np.lib.stride_tricks.sliding_window_view(np.r_[below, np.ones(5, bool)], 6).all(1)
            i = np.flatnonzero(stay & np.r_[False, np.isfinite(a[:-1]) & (a[:-1] >= lvl)])
            if not i.size:
                continue
            i = i[0]
            f = (a[i - 1] - lvl) / (a[i - 1] - a[i])
            out[kw][g] = pw[i - 1] + f * (pw[i] - pw[i - 1])
            out[kh][g] = hr[i - 1] + f * (hr[i] - hr[i - 1])
    return out


def features(proj, acts: pd.DataFrame) -> pd.DataFrame:
    """Alpha1 and HR in the warm-up (min 5-20) and as the session gets going (min 20-40) - minutes from the
    first pedal stroke -
    compared with what *your* last 6 weeks of rides say they should be at the same power.

    For each ride the expected alpha1 / HR at each steady power comes from a straight-line fit
    of all steady easy-to-moderate riding in the previous 42 days (indoor and outdoor together).
    The ride's deviation (observed - expected) is then expressed as a z-score against the
    deviations of those previous rides. Too high and too low are both unusual."""
    from .models.ramp import RampTest
    from .planning import PlanState
    R = _load_rides(proj, acts)
    if R is None:
        return pd.DataFrame()
    R["planned"] = np.isin(R["ids"], list(RampTest.planned_rides(PlanState.load(proj.processed / "plan_state.json"))))
    n, rid = R["n"], R["rid"]
    ftp_b = R["ftp"][rid]
    p2, sd2 = _steady(R)
    ok = np.isfinite(R["a1"]) & np.isfinite(p2) & (sd2 < STEADY_SD * ftp_b) & ~(R["art"] > MAX_ARTIFACTS)
    go = np.nan_to_num(R["p"]) >= RIDING_W        # a recording started before the riding (standing by the trainer)
    t0 = np.zeros(n)
    t0[np.unique(rid[go])] = np.inf
    np.minimum.at(t0, rid[go], R["t"][go])
    mins = (R["t"] - t0[rid]) / 60.0                # minutes from the first pedal stroke
    ok &= mins >= 4
    rel = p2 / ftp_b
    ez = ok & (rel >= P_RANGE[0]) & (rel <= P_RANGE[1])
    day = R["dates"].astype("int64")
    lo_r = np.searchsorted(day, day - BASE_DAYS, side="left")   # first ride inside the 6 weeks
    hi_r = np.arange(n)                                           # rides strictly before this one
    out = {"id": R["ids"], "date": pd.to_datetime(R["dates"]), "indoor": R["indoor"]}
    # per-ride sums for the straight-line fits (all steady easy riding of the ride)
    ez_cnt = np.bincount(rid[ez], minlength=n).astype(float)
    eligible = (ez_cnt >= 24).astype(float)
    n_prev = _window_sums(eligible[:, None], lo_r, hi_r)[:, 0]
    fits = {}
    for col, yv in (("a1", R["a1"]), ("hr", R["hr"])):
        m = ez & np.isfinite(yv) & (eligible[rid] > 0)
        x, y, g = p2[m], yv[m], rid[m]
        S = np.vstack([np.bincount(g, minlength=n), np.bincount(g, x, n), np.bincount(g, y, n),
                       np.bincount(g, x * x, n), np.bincount(g, x * y, n)]).T.astype(float)
        W = _window_sums(S, lo_r, hi_r)
        N, Sx, Sy, Sxx, Sxy = W.T
        with np.errstate(all="ignore"):
            den = N * Sxx - Sx * Sx
            b = (N * Sxy - Sx * Sy) / den
            a = (Sy - b * Sx) / N
        bad = (N < 30) | ~(np.abs(den) > 1e-9)
        fits[col] = (np.where(bad, np.nan, a), np.where(bad, np.nan, b))
    j = np.arange(n)
    inwin = (j[None, :] >= lo_r[:, None]) & (j[None, :] < hi_r[:, None])     # previous 6 weeks
    for name, (m0, m1) in (("warm", WARM), ("session", SESSION)):
        w = ez & (mins >= m0) & (mins < m1)
        a1m, cnt = _group_median(np.where(w, R["a1"], np.nan), rid, n)
        pm, _ = _group_median(np.where(w, p2, np.nan), rid, n)
        hrm, _ = _group_median(np.where(w, R["hr"], np.nan), rid, n)
        okw = cnt * BIN_S >= MIN_SECONDS
        a1m, pm, hrm = (np.where(okw, v, np.nan) for v in (a1m, pm, hrm))
        out[f"a1_{name}"], out[f"a1_{name}_p"], out[f"a1_{name}_hr"] = a1m, pm, hrm
        out[f"a1_{name}_s"] = (cnt * BIN_S).astype(int)
        for col, obs, key in (("a1", a1m, f"a1_{name}"), ("hr", hrm, f"a1_{name}_hr")):
            a, b = fits[col]
            exp = a + b * pm
            # previous rides' own deviations from *this* ride's line -> the normal spread
            dev = obs[None, :] - (a[:, None] + b[:, None] * pm[None, :])
            dev = np.where(inwin & np.isfinite(dev), dev, np.nan)
            ndev = np.isfinite(dev).sum(1)
            has = ndev[:, None] > 0                  # rides with no earlier rides to compare: no median
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(has, dev, 0.0), axis=1)
                mad = np.where(has[:, 0], np.nanmedian(np.where(has, np.abs(dev - med[:, None]), 0.0), axis=1), np.nan)
            scale = np.maximum(1.4826 * mad, 0.04 if col == "a1" else 1.5)
            z = (obs - exp - med) / scale
            good = okw & (n_prev >= 6) & (ndev >= 5) & np.isfinite(exp)
            out[key + "_exp"] = np.where(good, exp, np.nan)
            out[key + "_z"] = np.where(good, z, np.nan)
    # aerobic threshold: alpha1 mean per 10 W bin, per ride and pooled over 4 weeks
    th = ok & (rel >= TH_RANGE[0]) & (rel <= TH_RANGE[1])
    nb = TH_MAX_W // TH_BIN_W
    bi = np.clip((p2[th] // TH_BIN_W).astype(int), 0, nb - 1)
    flat = rid[th] * nb + bi
    sums = np.bincount(flat, R["a1"][th], n * nb).reshape(n, nb)
    cnts = np.bincount(flat, minlength=n * nb).reshape(n, nb).astype(float)
    tot = cnts.sum(1)
    out["hrvt1_w"] = _threshold_from_bins(sums, cnts, tot)
    lo28 = np.searchsorted(day, day - 27, side="left")
    ps, pc = _window_sums(sums, lo28, j + 1), _window_sums(cnts, lo28, j + 1)
    out["hrvt1_28d"] = _threshold_from_bins(ps, pc, pc.sum(1))
    # HR at the aerobic threshold, and the second threshold (alpha1 = 0.5) pooled over 6 weeks
    # with its HR. From ordinary rides alpha1 also falls with time into the ride, so these
    # read lower than a ramp test would.
    hsum = np.bincount(flat, np.nan_to_num(R["hr"][th]), n * nb).reshape(n, nb)
    hcnt = np.bincount(flat, np.isfinite(R["hr"][th]), n * nb).reshape(n, nb).astype(float)
    hs, hc = _window_sums(hsum, lo28, j + 1), _window_sums(hcnt, lo28, j + 1)
    with np.errstate(all="ignore"):
        hr28 = np.where(_pool3(hc) >= 6, _pool3(hs) / _pool3(hc), np.nan)
    out["hrvt1_hr_28d"] = _at_power(out["hrvt1_28d"], hr28)
    th2 = ok & (rel >= TH2_RANGE[0]) & (rel <= TH2_RANGE[1])
    bi2 = np.clip((p2[th2] // TH_BIN_W).astype(int), 0, nb - 1)
    f2 = rid[th2] * nb + bi2
    lo42 = np.searchsorted(day, day - (BASE_DAYS - 1), side="left")
    w42 = lambda v: _window_sums(np.bincount(f2, v, n * nb).reshape(n, nb).astype(float), lo42, j + 1)  # noqa: E731
    c2 = w42(np.ones(f2.size))
    h2, hc2 = w42(np.nan_to_num(R["hr"][th2])), w42(np.isfinite(R["hr"][th2]))
    with np.errstate(all="ignore"):
        hr42 = np.where(_pool3(hc2) >= 6, _pool3(h2) / _pool3(hc2), np.nan)
        # median rather than mean: at high power a few high readings keep the mean above 0.5.
        # The median crosses a level where under half the windows are still at or above it.
        share = lambda lvl: _pool3(w42(R["a1"][th2] >= lvl)) / _pool3(c2)  # noqa: E731
        lt1_42 = _sustained_crossing(share(LT1_A1), c2 >= 6, 0.5)
        out["hrvt2_42d"] = _sustained_crossing(share(LT2_A1), c2 >= 6, 0.5, after_w=lt1_42)
    out["hrvt2_hr_42d"] = _at_power(out["hrvt2_42d"], hr42)
    out["a1_bins"] = np.bincount(rid[ok], minlength=n)
    out.update(_ramps(R, p2))
    return pd.DataFrame(out)


def to_daily(af: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """One value per day (the day's first ride with a usable warm-up)."""
    cols = ["a1_warm", "a1_warm_z", "a1_session_z", "a1_warm_hr_z", "a1_session_hr_z", "hrvt1_w", "hrvt1_28d",
            "hrvt1_hr_28d", "hrvt2_42d", "hrvt2_hr_42d", "ramp_lt1_w", "ramp_lt2_w"]
    if af is None or af.empty:
        return pd.DataFrame(index=days, columns=cols, dtype=float)
    a = af.sort_values("date")
    a = a[a["a1_warm"].notna() | a["a1_session"].notna() | a["hrvt1_w"].notna() | a["ramp_lt1_w"].notna()]
    d = a.groupby("date")[cols].first()
    return d.reindex(days)


def path_table(proj) -> Path:
    return proj.processed / "alpha_rides.csv"
