"""DFA alpha1 of heart-rate variability during exercise, from beat-to-beat (RR) intervals.

Detrended fluctuation analysis (Peng, Havlin, Stanley & Goldberger 1995, Chaos 5:82): integrate the intervals'
deviations from their mean (the profile), split the profile into boxes of n beats, take out each box's
straight-line fit and measure what's left, F(n) = sqrt(mean squared residual). alpha1 is the slope of log F(n)
against log n over the short scales n = 4 ... 16 beats. Uncorrelated beats give 0.5, 1/f-like ("fractal") 1.0,
a random walk 1.5. During exercise alpha1 falls from about 1.0 at easy intensity towards 0.5 and below; 0.75
and 0.5 have been used as the first and second thresholds (Rogers, Giles, Draper, Hoos & Gronwald 2021, Front
Physiol 11:596567; Gronwald, Rogers & Hoos 2020, Front Physiol 11:550572).

Computed over a moving window of beats (WINDOW_S of time, every STEP_S), vectorised over the whole ride: a
box's straight-line residual only depends on the beats inside it, so every box is fitted once from its own
intervals (no long cumulative sums, no rounding trouble) and each window sums the boxes it's split into.
The window and step are settings to test against our own recordings, not validated constants.

Artifacts: DFA reacts strongly to a single missed or extra beat, so the intervals are corrected first
(`corrected`). A beat is odd when it is further from the median of its NEIGHBOURS than SPREAD_K quartile
deviations of how far beats around it (SPREAD_BEATS) usually are - the adaptive threshold of Lipponen &
Tarvainen 2019 (J Med Eng Technol 43:173), so it suits any strap and any intensity: the big breathing swings at
rest aren't flagged, small faults at high heart rate are (with a floor, OFF_MIN). Runs of odd beats whose total
spans a whole number (up to RESOLVES) of normal beats become that many equal beats - a missed beat split in two,
an extra one merged, a misplaced one evened out (Lipponen & Tarvainen's types, simpler): a chest strap's odd
beats are mostly exactly twice the interval around them (a missed beat). Other odd runs (a strap losing
contact for seconds) aren't filled with invented beats: they stay, and only the DFA boxes containing them are
left out - each box's fit depends only on its own beats, so the rest of the window still counts (removing the
segments with missing data keeps DFA's exponent: Ma et al. 2010, Phys Rev E 81:031101). A window's share of
corrected or unresolved beats is its artifact rate, reported with each value; above MAX_ARTIFACTS it has none
(missed beats up to about 5 % changed alpha1 little in Rogers et al. 2021, Sensors 21:821).

Slow trends: heart rate drifting (a ramp, a climb) adds slow variation that pushes alpha1 up, so the intervals
are detrended first with smoothness priors (Tarvainen, Ranta-aho & Karjalainen 2002, IEEE Trans Biomed Eng
49:172), lambda = LAMBDA as commonly used in HRV analysis. On a chest-strap ramp test this about halved the gap
to alphaHRV (a closed tool, compared as an outside reference only).
"""
from __future__ import annotations

import numpy as np

SCALES = np.arange(4, 17)        # box sizes n (beats): alpha1's short-term range
WINDOW_S = 120.0                 # each value from the beats of the last 2 minutes ...
STEP_S = 5.0                     # ... every 5 s
MIN_BEATS = 50                   # a window with fewer beats has no value
SPREAD_K = 5.2                   # odd: further from the neighbours' median than this many quartile deviations
SPREAD_BEATS = 91                # ... of the beats around it (Lipponen & Tarvainen 2019)
OFF_MIN = 0.05                   # ... and at least this share of the median (variability near zero: no flood of flags)
SPANS = 0.10                     # odd beats resolve when their total is this close to whole beats (H10's
                                 # missed beats: within 1-2 % of exactly two)
RESOLVES = 3                     # ... and spans at most this many beats: longer (a strap losing contact)
                                 # would be invented beats - left as they are, their DFA boxes left out
NEIGHBOURS = 11                  # ... the median of this many beats around it
MAX_ARTIFACTS = 0.05             # a window with more corrected / unresolved beats than this has no value
LAMBDA = 500.0                   # smoothness priors detrending: the trend's stiffness


def local_median(rr: np.ndarray) -> np.ndarray:
    """each beat's neighbours' median (NEIGHBOURS around it)."""
    import pandas as pd
    return pd.Series(rr).rolling(NEIGHBOURS, center=True, min_periods=3).median().to_numpy()


def detrended(rr: np.ndarray, bad: np.ndarray | None = None, lam: float = LAMBDA) -> np.ndarray:
    """the intervals less their smoothness-priors trend (plus their mean). The trend solves
    (I + lam^2 D2'D2) trend = rr, D2 the (n-2) x n second-difference matrix; D2'D2 is banded (diagonals
    1 5 6 ... 6 5 1, -2 -4 ... -4 -2 and 1 ... 1), so the whole ride is one banded solve. `bad` beats count
    as their neighbours' median for the trend, so one long gap doesn't bend it."""
    from scipy.linalg import solveh_banded
    n = rr.size
    if n < 3:
        return rr
    ones = np.ones(n - 2)
    ab = np.zeros((3, n))                                   # upper bands, as solveh_banded takes them
    ab[0, 2:] = lam ** 2 * ones
    ab[1, 1:] = lam ** 2 * np.convolve(ones, [-2, -2])
    ab[2] = 1 + lam ** 2 * np.convolve(ones, [1, 4, 1])
    x = rr if bad is None or not bad.any() else np.where(bad, local_median(np.where(bad, np.nan, rr)), rr)
    return rr - solveh_banded(ab, x) + x.mean()


def corrected(rr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(intervals, artifact flags): runs of odd beats replaced by the whole number (up to RESOLVES) of equal beats
    they span (flag 1), odd runs that don't resolve kept as they are (flag 2); 0 = a normal beat."""
    import pandas as pd
    rr = np.asarray(rr, float)
    if not rr.size:
        return rr, np.zeros(0, int)
    med = local_median(rr)
    dev = pd.Series(rr - med).rolling(SPREAD_BEATS, center=True, min_periods=11)
    qd = (dev.quantile(0.75) - dev.quantile(0.25)).to_numpy() / 2
    off = np.abs(rr - med) > np.maximum(SPREAD_K * np.nan_to_num(qd), OFF_MIN * med)
    run = np.cumsum(np.r_[True, ~(off[1:] & off[:-1])]) - 1          # odd neighbours share a run
    total = np.bincount(run, rr)
    ref = np.bincount(run, med) / np.bincount(run)
    m = np.maximum(np.round(total / ref), 1)
    fits = (np.abs(total / (m * ref) - 1) <= SPANS) & (m <= RESOLVES)
    fix = fits & (np.bincount(run, off) > 0)                           # runs of odd beats that resolve
    g = np.where(fix[run], run, -1 - np.arange(rr.size))                # the others: each beat by itself
    _, first, gid = np.unique(g, return_index=True, return_inverse=True)
    order = np.argsort(first)                                           # groups in beat order
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    gid = rank[gid]
    n = np.where(fix[run[first[order]]], m[run[first[order]]], 1).astype(int)
    out = np.repeat(np.bincount(gid, rr) / n, n)
    odd = np.bincount(gid, off) > 0
    flag = np.repeat(np.where(odd, np.where(fix[run[first[order]]], 1, 2), 0), n)
    return out, flag


def box_residuals(rr: np.ndarray, n: int) -> np.ndarray:
    """for every start i: the sum of squared residuals of a straight line fitted to the profile over beats
    i .. i+n-1 (len(rr) - n + 1 values). Inside a box the profile is the running sum of the intervals after its
    first beat; offsets and linear terms don't change a line fit's residuals, so each box uses only its own beats."""
    w = np.lib.stride_tricks.sliding_window_view(rr, n)
    z = np.cumsum(w - w.mean(axis=1, keepdims=True), axis=1)    # the box's profile (up to a line)
    x = np.arange(n) - (n - 1) / 2
    zc = z - z.mean(axis=1, keepdims=True)
    return (zc * zc).sum(axis=1) - (zc @ x) ** 2 / (x @ x)


def alpha1(rr: np.ndarray, starts: np.ndarray, counts: np.ndarray, bad: np.ndarray | None = None,
           scales: np.ndarray = SCALES) -> np.ndarray:
    """DFA alpha1 of the windows rr[starts[k] : starts[k] + counts[k]]: boxes laid end to end from each window's
    first beat, those holding a `bad` beat left out. NaN for a window with fewer than MIN_BEATS good beats."""
    rr = np.asarray(rr, float)
    starts, counts = np.asarray(starts, int), np.asarray(counts, int)
    nb = np.r_[0, np.cumsum(np.zeros(rr.size, bool) if bad is None else bad)]
    logF = np.full((starts.size, scales.size), np.nan)
    for j, n in enumerate(scales):               # 13 box sizes; each vectorised over every window
        if rr.size < n:
            continue
        ss = box_residuals(rr, n)
        clean = nb[n:] == nb[:-n]                # boxes without a bad beat
        k = np.arange(max(int((counts // n).max(initial=0)), 1))
        pos = np.minimum(starts[:, None] + n * k[None, :], ss.size - 1)
        use = (k[None, :] < (counts // n)[:, None]) & clean[pos]
        total = np.where(use, ss[pos], 0.0).sum(axis=1)
        with np.errstate(all="ignore"):
            logF[:, j] = 0.5 * np.log(total / (use.sum(axis=1) * n))
    x = np.log(scales) - np.log(scales).mean()
    a = (logF - logF.mean(axis=1, keepdims=True)) @ x / (x @ x)
    good = counts - (nb[np.minimum(starts + counts, rr.size)] - nb[np.minimum(starts, rr.size)])
    return np.where(good >= MIN_BEATS, a, np.nan)


def series(rr: np.ndarray, t0: float, t: np.ndarray, max_artifacts: float = MAX_ARTIFACTS
           ) -> tuple[np.ndarray, np.ndarray]:
    """(alpha1, artifact share) at times `t` from intervals `rr` whose first beat ends at t0 + rr[0]: corrected,
    detrended, windowed (WINDOW_S); NaN alpha1 where a window has too few beats or a larger artifact share than
    `max_artifacts`."""
    rr, flag = corrected(rr)
    beat_t = t0 + np.cumsum(rr)                             # the beats' times from the intervals as they were
    a, k = windows(beat_t, t)
    c = np.r_[0, np.cumsum(flag > 0)]
    with np.errstate(all="ignore"):
        share = (c[a + k] - c[a]) / k
    bad = flag == 2
    al = alpha1(detrended(rr, bad), a, k, bad)
    return np.where(share <= max_artifacts, al, np.nan), share


def windows(beat_t: np.ndarray, t: np.ndarray, length: float = WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
    """(first beat, number of beats) of the window (t - length, t] for each time in `t`; beat_t sorted."""
    a = np.searchsorted(beat_t, np.asarray(t, float) - length, side="right")
    b = np.searchsorted(beat_t, np.asarray(t, float), side="right")
    return a, b - a
