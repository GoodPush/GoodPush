"""Intervals in a ride, from its 1 Hz power: the steps you rode, the efforts they make and the sets.

1. Pieces: the power cut where its average shifts (binary segmentation: each split is the one that
   explains the most variance, kept while that beats a penalty from the power's noise and a floor of
   a 5 % CP shift held about a minute). The least-squares cut lands mid-way up a trainer's ramp, so a
   rep's few seconds of lag don't move its edges. Bursty power (outdoors: roughness > BURSTY that the
   steps don't explain, < STEPS_EXPLAIN of it) is cut on its 30 s average so terrain surges aren't
   steps; micro-intervals (30/15s) are rough too, but their steps explain most of it.
2. Tidying: a short piece between two similar ones is a dip (one piece); between different ones a
   ramp (one cut, at its best place); neighbours within CLOSE of CP are one piece.
3. Zones: the plan's (textbook %FTP edges, ZONE_FRAC) in your watts on the ride's day, through the
   same map as the plan's sessions (planning.intensity: 0.75 -> LT1, 1.0 -> CP, 1.2 -> MAP, ...), so
   threshold sits at your CP, VO2max between CP and MAP, anaerobic above MAP; recovery below ~73 % of
   LT1, endurance split into lower (to ~87 % of LT1) and upper (to LT1). Without a physio fit:
   textbook %FTP.
4. Efforts: runs of pieces from tempo up, each step kept; the unders of over-unders stay in (a piece
   below tempo but not below upper endurance, short, near the work beside it); with W' known, how much
   of W' they used (RideStamina's W' balance).
5. Sets: efforts in a row of about the same length (+-20 %), close together, rests about as long as
   the set's earlier ones. Intervals: the efforts long enough for their zone (MIN_S), or the whole set
   when at least half of it is.

Each split is computed for the whole ride at once (cumulative sums); the loops run over splits and
pieces (tens), not samples. A 4-h ride takes ~10 ms.
"""
from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from .stamina import RideStamina

# zone floors, textbook %FTP; endurance split at 0.65 (lower / upper): easy rides belong in the lower
# half, where being a little off still keeps you below LT1
ZONE_FRAC = np.array([0.0, 0.55, 0.65, 0.76, 0.88, 0.95, 1.06, 1.20])
ZONE_NAMES = ("recovery", "endurance_low", "endurance_high", "tempo", "sweetspot", "threshold", "vo2max", "anaerobic")
MIN_S = np.array([0, 0, 0, 300, 120, 60, 30, 10])    # shortest interval that counts, per zone (s)
WORK_ZONE = 3                                        # tempo and up
OVER_UNDER_ZONE = 5                                  # over CP twice or more, unders between, averaging sweet
                                                     # spot (OVER_UNDER_ZONE - 1): threshold (not bursty power)
BURSTY = 0.2                                         # sd of power around its 30 s average / mean power
STEPS_EXPLAIN = 0.5                                  # share of that the raw steps explain: structured, not bursty
MIN_PIECE_S = 10, 30                                 # shortest piece: raw power (15-s rests), 30 s average
DIP_SHARE, DIP_MAX_S = 0.15, 30                      # a dip shorter than both (x longer neighbour, s): short
CLOSE = 0.04                                         # neighbours closer than this (of CP): one piece
UNDER_ZONE = 2                                       # an under: upper endurance or more,
UNDER_MAX_S = 240                                    # at most this long,
UNDER_SHARE = 0.75                                   # a dip at this share of its longer neighbour: a sag
UNDER_NEAR = 0.65                                    # an under: at least this share of the work beside it
                                                     # (lactate-clearance unders ~86 % FTP after ~122 % overs)
NOISE_PENALTY = 8.0                                  # x noise variance x log(samples)
SHIFT_FLOOR = 0.05                                   # of CP, held SHIFT_FLOOR_S: the smallest step
SHIFT_FLOOR_S = 60
SET_LENGTH = (0.8, 1.25)                             # a rep's length vs the set's first
SET_GAP_REPS, SET_GAP_S = 4, 600                     # reps further apart than both: a new set
SET_REST = 2.5, 60                                   # or a rest over both x / + the set's median rest so far
SET_REPS = 3                                         # a set this long counts (steady power), its reps at
SET_REP_SHARE = 1 / 6                                # least this share of MIN_S (tempo 50 s, VO2max 5 s)


class RideIntervals:
    def __init__(self, power, knots, hr=None, w_prime: float | None = None, bursty: bool | None = None):
        """knots: the ride day's watts at planning.intensity.FRAC (IntensityMap.knots_of / ftp_knots);
        w_prime (J): that day's W', for the W' balance (None: not shown); bursty: cut the 30 s power
        (True), the raw power (False, e.g. to find 30-s reps outdoors), or by the ride's roughness and how
        much of it the raw steps explain (None)."""
        from ..planning.intensity import FRAC              # here: planning imports this module
        self.power = np.nan_to_num(np.asarray(power, float))
        self.knots = np.asarray(knots, float)
        self.cp = float(self.knots[2])
        self.edges = np.interp(ZONE_FRAC, FRAC, self.knots)     # zone floors in watts
        self.hr = None if hr is None else np.asarray(hr, float)
        self.w_prime = float(w_prime) if w_prime else None
        p = self.power
        self.sums = np.r_[0.0, np.cumsum(p)], np.r_[0.0, np.cumsum(p * p)]
        self.roughness = float(np.std(p - self.rolling_mean(p, 30)) / max(p.mean(), 1.0)) if p.size else 0.0
        if not p.size:
            self.bursty, self.cuts = bool(bursty), np.zeros(1, int)
            return
        raw = self.pieces(smooth=False) if bursty is not True else None
        self.bursty = self.unexplained(raw) if bursty is None else bool(bursty)
        self.cuts = self.pieces(smooth=True) if self.bursty else raw

    # --- pieces ----------------------------------------------------------------------------
    @staticmethod
    def rolling_mean(p, w: int) -> np.ndarray:
        """centred w-sample average (shorter at the ends)."""
        c = np.r_[0.0, np.cumsum(p)]
        i = np.arange(p.size)
        a, b = np.clip(i - w // 2, 0, p.size), np.clip(i + w - w // 2, 0, p.size)
        return (c[b] - c[a]) / (b - a)

    @staticmethod
    def cost(sums, i, j):
        """squared error of samples i..j-1 around their mean (i, j: arrays or ints)."""
        s1, s2 = sums
        return (s2[j] - s2[i]) - (s1[j] - s1[i]) ** 2 / np.maximum(j - i, 1)

    @classmethod
    def best_split(cls, sums, a: int, b: int, m: int) -> tuple[float, int]:
        """(gain, cut) of the best single cut in a..b-1, each side at least m samples; cut -1 if none."""
        t = np.arange(a + m, b - m + 1)
        if not t.size:
            return -1.0, -1
        g = cls.cost(sums, a, b) - cls.cost(sums, a, t) - cls.cost(sums, t, b)
        k = int(np.argmax(g))
        return float(g[k]), int(t[k])

    @classmethod
    def split(cls, x, penalty: float, m: int) -> np.ndarray:
        """binary segmentation of x: piece edges (0 .. len) while a split gains more than `penalty`."""
        sums = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
        cuts, todo = [0, x.size], []
        g, t = cls.best_split(sums, 0, x.size, m)
        heapq.heappush(todo, (-g, 0, x.size, t))
        while todo:                                       # one pass per piece: tens, not samples
            g, a, b, t = heapq.heappop(todo)
            if -g < penalty or t < 0:
                continue
            cuts.append(t)
            for lo, hi in ((a, t), (t, b)):
                g2, t2 = cls.best_split(sums, lo, hi, m)
                if t2 >= 0:
                    heapq.heappush(todo, (-g2, lo, hi, t2))
        return np.array(sorted(cuts))

    def pieces(self, smooth: bool) -> np.ndarray:
        """piece edges in samples, tidied (see the module doc); smooth: cut the 30 s average."""
        p = self.power
        x = self.rolling_mean(p, 30) if smooth else p
        noise = np.median(np.abs(np.diff(x))) / (0.6745 * np.sqrt(2)) if x.size > 1 else 0.0
        penalty = max(NOISE_PENALTY * noise ** 2 * np.log(max(x.size, 2)),
                      (SHIFT_FLOOR * self.cp) ** 2 * SHIFT_FLOOR_S / 2)
        min_len = MIN_PIECE_S[int(smooth)]
        return self.tidy(self.split(x, penalty, min_len // 2 + 3), min_len)

    def unexplained(self, cuts) -> bool:
        """bursty: rough (BURSTY) and the steps at `cuts` explain less than STEPS_EXPLAIN of it (terrain,
        not a structured session)."""
        if self.roughness <= BURSTY:
            return False
        left = np.std(self.power - np.repeat(self.means(cuts), np.diff(cuts))) / max(self.power.mean(), 1.0)
        return bool(1 - (left / self.roughness) ** 2 < STEPS_EXPLAIN)

    def means(self, cuts) -> np.ndarray:
        s1 = self.sums[0]
        return (s1[cuts[1:]] - s1[cuts[:-1]]) / np.diff(cuts)

    def tidy(self, cuts, min_len: int) -> np.ndarray:
        """short pieces: a dip joins its neighbours, a ramp becomes one cut; then similar neighbours join.
        Short: under min_len, or a dip below both neighbours, under DIP_MAX_S and DIP_SHARE of the shorter
        one (a dropout inside a 4-min rep, not the 15-s rest between 30-s reps) - of the longer one when
        it stays at UNDER_SHARE of that (the sag after a rep's hard start)."""
        while cuts.size > 2:                              # each pass removes a cut: tens of passes
            mu, L = self.means(cuts), np.diff(cuts)
            longer = np.where(L[:-2] > L[2:], 0, 2) + np.arange(L.size - 2)
            room = np.where(mu[1:-1] >= UNDER_SHARE * mu[longer], L[longer], np.minimum(L[:-2], L[2:]))
            dip = (mu[1:-1] < mu[:-2]) & (mu[1:-1] < mu[2:]) & (L[1:-1] < np.minimum(DIP_SHARE * room, DIP_MAX_S))
            inner = np.flatnonzero((L[1:-1] < min_len) | dip) + 1
            if inner.size:
                k = inner[np.argmin(L[inner])]
                if abs(mu[k - 1] - mu[k + 1]) < CLOSE * self.cp:
                    cuts = np.delete(cuts, [k, k + 1])
                else:
                    a, b, t = cuts[k - 1], cuts[k + 2], np.arange(cuts[k], cuts[k + 1] + 1)
                    best = t[np.argmin(self.cost(self.sums, a, t) + self.cost(self.sums, t, b))]
                    cuts = np.r_[cuts[:k], best, cuts[k + 2:]]
                continue
            if L[0] < min_len or L[-1] < min_len:
                cuts = np.delete(cuts, 1 if L[0] < min_len else cuts.size - 2)
                continue
            d = np.abs(np.diff(mu))
            if d.min() >= CLOSE * self.cp:
                break
            cuts = np.delete(cuts, int(np.argmin(d)) + 1)
        return cuts

    def zone(self, watts) -> np.ndarray:
        return np.maximum(np.searchsorted(self.edges, np.asarray(watts), side="right") - 1, 0)

    def w_balance(self) -> np.ndarray:
        """W' left each second (J): Skiba's W' balance as the ride went (RideStamina.w_balance)."""
        return RideStamina(self.power, self.cp, self.w_prime).w_balance()

    # --- intervals and sets ----------------------------------------------------------------
    def working(self, cuts, mu) -> np.ndarray:
        """which pieces make efforts: tempo and up, and the unders of over-unders - a piece below tempo
        but from upper endurance (UNDER_ZONE), at most UNDER_MAX_S, at least UNDER_NEAR of the work
        beside it, between two work pieces or at the end of a block with unders inside (its valleys, at
        tempo or not) when as long as they are (SET_LENGTH); a block with unders must still average tempo.
        Bursty power: no unders (terrain's ups and downs aren't over-unders)."""
        L = np.diff(cuts)
        work = mu >= self.edges[WORK_ZONE]
        if self.bursty:
            return work
        prev, nxt = np.r_[np.inf, mu[:-1]], np.r_[mu[1:], np.inf]
        near = (mu >= UNDER_NEAR * np.maximum(prev, nxt)) & (L <= UNDER_MAX_S)     # max: inf at the ends fails
        under = ~work & (mu >= self.edges[UNDER_ZONE]) & (L <= UNDER_MAX_S)
        inner = under & np.r_[False, work[:-1]] & np.r_[work[1:], False] & near
        out = work | inner
        valley = out & np.r_[False, out[:-1]] & np.r_[out[1:], False] & (mu < prev) & (mu < nxt) & near
        runs = np.flatnonzero(np.diff(np.r_[0, out.astype(int), 0])).reshape(-1, 2)
        for a, b in runs:                                  # efforts: tens
            u = L[a:b][valley[a:b]]
            if not u.size:
                continue
            for i, w in ((a - 1, a), (b, b - 1)):
                if 0 <= i < mu.size and under[i] and mu[i] >= UNDER_NEAR * mu[w] \
                        and SET_LENGTH[0] <= L[i] / np.median(u) <= SET_LENGTH[1]:
                    out[i] = True
        runs = np.flatnonzero(np.diff(np.r_[0, out.astype(int), 0])).reshape(-1, 2)
        s, e = cuts[runs[:, 0]], cuts[runs[:, 1]]
        for a, b in runs[(self.sums[0][e] - self.sums[0][s]) / (e - s) < self.edges[WORK_ZONE]]:   # a handful
            out[a:b] = work[a:b]
        return out

    def efforts(self) -> list[dict]:
        """every run of pieces from tempo up: start / end (s), watts, % CP, zone (of its average; over-unders -
        over CP at least twice, averaging sweet spot, steady power - threshold), steps [(seconds, watts)],
        heart rate (average, max, over the second half once it has settled, and its drift: last third -
        middle third), steadiness (cv: sd / mean power), the share of W' used (its lowest balance in it;
        None without W') and whether it is long enough for its zone on its own (MIN_S)."""
        cuts = self.cuts
        mu = self.means(cuts)
        work = self.working(cuts, mu).astype(int)
        runs = np.flatnonzero(np.diff(np.r_[0, work, 0])).reshape(-1, 2)
        s, e = cuts[runs[:, 0]], cuts[runs[:, 1]]
        watts = (self.sums[0][e] - self.sums[0][s]) / np.maximum(e - s, 1)
        z = self.zone(watts)
        rises = np.r_[0, np.cumsum(np.diff(np.r_[0, (mu >= self.cp).astype(int)]) == 1)]   # runs over CP so far
        overs = rises[runs[:, 1]] - rises[runs[:, 0]]
        if not self.bursty:                                               # over-unders: the average hides the overs
            z = np.where((overs >= 2) & (z == OVER_UNDER_ZONE - 1), OVER_UNDER_ZONE, z)
        bal = self.w_balance() if self.w_prime else None
        out = []
        for (a, b), s0, e0, w, zi in zip(runs, s, e, watts, z):          # tens
            hr = self.hr[s0:e0] if self.hr is not None else np.zeros(0)
            h = hr[np.isfinite(hr)]
            thirds = [x[np.isfinite(x)] for x in np.array_split(hr, 3)] if hr.size >= 3 else []
            late = hr[hr.size // 2:]
            late = late[np.isfinite(late)]
            out.append({"start": int(s0), "end": int(e0), "seconds": int(e0 - s0), "watts": round(float(w)),
                        "pct_cp": round(float(w / self.cp * 100)), "zone": ZONE_NAMES[zi],
                        "w_prime_used": round(float(1 - bal[s0:e0].min() / self.w_prime) * 100) if bal is not None else None,
                        "steps": [[int(cuts[i + 1] - cuts[i]), round(float(mu[i]))] for i in range(a, b)] if b - a > 1 else [],
                        "avg_hr": round(float(h.mean())) if h.size else None,
                        "max_hr": round(float(h.max())) if h.size else None,
                        "hr_late": round(float(late.mean()), 1) if late.size else None,
                        "hr_drift": round(float(thirds[2].mean() - thirds[1].mean()), 1)
                        if len(thirds) == 3 and thirds[1].size and thirds[2].size else None,
                        "cv": round(float(np.std(self.power[s0:e0]) / max(w, 1.0)), 3),
                        "long_enough": bool(e0 - s0 >= MIN_S[zi])})
        return out

    @staticmethod
    def groups(iv: list[dict]) -> list[tuple[int, int]]:
        """(first, end) of each run of efforts of about the same length (SET_LENGTH), close together, each
        rest no longer than SET_REST allows from the set's rests so far (3 x 12 30/15s: three sets)."""
        if not iv:
            return []
        L = np.array([x["seconds"] for x in iv])
        start, end = np.array([x["start"] for x in iv]), np.array([x["end"] for x in iv])
        gap = np.r_[0, start[1:] - end[:-1]]
        out, a = [], 0
        for i in range(1, len(iv) + 1):                    # efforts: tens
            usual = np.median(gap[a + 1:i]) if i > a + 1 else np.inf
            if i < len(iv) and SET_LENGTH[0] <= L[i] / L[a] <= SET_LENGTH[1] \
                    and gap[i] <= max(SET_GAP_REPS * L[a], SET_GAP_S) \
                    and gap[i] <= max(SET_REST[0] * usual, usual + SET_REST[1]):
                continue
            out.append((a, i))
            a = i
        return out

    def intervals(self) -> list[dict]:
        """the efforts that count: long enough for their zone, or a rep of a set where at least half are
        (a 3-min rep at tempo inside a 12 x 3-min ladder stays; a short surge on its own doesn't), or of a
        set of SET_REPS or more on steady power, each rep SET_REP_SHARE of MIN_S (3 x 1 min at tempo spread
        over an endurance hour; not 20-s blips at tempo)."""
        iv = self.efforts()
        out = []
        for a, b in self.groups(iv):                       # sets: a handful
            ok = [x["long_enough"] for x in iv[a:b]]
            rep = [x["seconds"] >= SET_REP_SHARE * MIN_S[ZONE_NAMES.index(x["zone"])] for x in iv[a:b]]
            whole = 2 * sum(ok) >= len(ok) or (not self.bursty and len(ok) >= SET_REPS and all(rep))
            out += [x for x, k in zip(iv[a:b], ok) if k or whole]
        return out

    def sets(self, intervals: list[dict] | None = None) -> list[dict]:
        """the intervals in sets: {first, count, seconds (median), watts_lo / watts_hi, zone (of their
        average), rest (median seconds between)}."""
        iv = self.intervals() if intervals is None else intervals
        L = np.array([x["seconds"] for x in iv])
        start, end = np.array([x["start"] for x in iv], int), np.array([x["end"] for x in iv], int)
        w = np.array([x["watts"] for x in iv])
        zi = np.array([ZONE_NAMES.index(x["zone"]) for x in iv], int)
        s1 = self.sums[0]
        out = []
        for a, i in self.groups(iv):                       # sets: a handful
            rest = start[a + 1:i] - end[a:i - 1]
            rest_w = (s1[start[a + 1:i]] - s1[end[a:i - 1]]) / np.maximum(rest, 1)
            out.append({"first": a, "count": i - a, "seconds": int(np.median(L[a:i])),
                        "watts_lo": int(w[a:i].min()), "watts_hi": int(w[a:i].max()),
                        "zone": ZONE_NAMES[max(int(self.zone(w[a:i].mean())), int(zi[a:i].min()))],
                        "rest": int(np.median(rest)) if rest.size else None,
                        "rest_watts": round(float(rest_w.mean())) if rest.size else None,
                        "rest_zone": ZONE_NAMES[int(self.zone(rest_w.mean()))] if rest.size else None})
        return out

    def zone_seconds(self) -> np.ndarray:
        """seconds in each zone (ZONE_NAMES), second by second."""
        return np.bincount(self.zone(self.power), minlength=len(ZONE_NAMES)) if self.power.size else np.zeros(len(ZONE_NAMES), int)

    def summary(self) -> dict:
        """for the ride page: intervals, sets, the zones (names, floors in watts) and time in each."""
        iv = self.intervals()
        return {"intervals": iv, "sets": self.sets(iv), "bursty": self.bursty, "zones": list(ZONE_NAMES),
                "zone_floors": np.round(self.edges).astype(int).tolist(), "zone_seconds": self.zone_seconds().tolist()}


class IntervalLog:
    """Every ride's intervals in one table (processed/intervals.parquet), one row per interval with its
    set (set, rep, reps, rest_s). A ride is looked at again when the method (VERSION) changes or its day's
    thresholds (LT1 / CP / MAP) move more than THRESHOLD_STEP W from those its intervals were found with (its
    key, in THRESHOLD_STEP W): a refit's small drift across a step leaves it be, so the physio model's older
    readings - which come from these intervals, found with that model's thresholds - don't change and call
    for another full refit. A ride without intervals keeps one row with start -1, so it isn't looked at again."""
    NAME = "intervals"
    VERSION = 3
    THRESHOLD_STEP = 5.0
    DROP = ("end", "steps", "long_enough")

    @classmethod
    def keys(cls, knots: np.ndarray) -> np.ndarray:
        """per ride (rows of knots): the method version and its LT1 / CP / MAP in THRESHOLD_STEP W."""
        k = np.round(knots[:, 1:4] / cls.THRESHOLD_STEP).astype(int).astype(str)
        return np.char.add(f"v{cls.VERSION}-", np.char.add(np.char.add(np.char.add(k[:, 0], "-"), np.char.add(k[:, 1], "-")), k[:, 2]))

    @classmethod
    def stale(cls, ids: np.ndarray, knots: np.ndarray, old: pd.DataFrame | None) -> np.ndarray:
        """which rides (positions in `ids`) to look at (again): new, an older method, or thresholds (rows of
        `knots`) more than THRESHOLD_STEP W from those in the key their intervals were found with."""
        if old is None or not len(old):
            return np.arange(len(ids))
        used = old.drop_duplicates("id").set_index("id")["key"].reindex(ids)
        parts = used.str.extract(r"^v(\d+)-(-?\d+)-(-?\d+)-(-?\d+)$").astype(float).to_numpy()
        moved = np.abs(knots[:, 1:4] - parts[:, 1:] * cls.THRESHOLD_STEP).max(axis=1) > cls.THRESHOLD_STEP
        return np.flatnonzero((parts[:, 0] != cls.VERSION) | moved)

    @classmethod
    def rows(cls, ri: RideIntervals) -> list[dict]:
        """the ride's intervals with their set, ready for the table."""
        iv = ri.intervals()
        for n, st in enumerate(ri.sets(iv)):                       # sets: a handful
            for r in range(st["count"]):
                iv[st["first"] + r].update(set=n, rep=r, reps=st["count"], rest_s=st["rest"])
        return [{**{k: v for k, v in x.items() if k not in cls.DROP}, "steps": len(x["steps"]), "bursty": ri.bursty}
                for x in iv]

    @classmethod
    def update(cls, processed, acts: pd.DataFrame, progress=print) -> pd.DataFrame:
        """the table brought up to date with `acts` (rides with power); returns it (all rows)."""
        from ..io import store, streamstore
        from ..planning.intensity import IntensityMap
        rides = acts[acts["sport"].eq("ride") & acts["has_power"].astype(bool)]
        ids = rides["id"].astype(str).str.replace(r"\.0$", "", regex=True).to_numpy()
        dates = pd.to_datetime(rides["date"]).dt.normalize()
        hist = IntensityMap.history(processed, 0.0)               # your thresholds on each day, if fitted
        if hist is not None:
            m, w_prime = hist
            i = np.clip(m.days.searchsorted(dates), 0, len(m.days) - 1)
            knots, wp = m.knots[i], w_prime[i]
        else:
            knots = IntensityMap.ftp_knots(1.0)[None, :] * rides["ftp"].fillna(0).to_numpy(float)[:, None]
            wp = np.full(len(rides), np.nan)
        keys = cls.keys(knots)
        old = store.load(processed, cls.NAME)
        todo = cls.stale(ids, knots, old)
        if not todo.size and old is not None:
            return old[old["id"].isin(ids)]
        streams = streamstore.read_many(processed, list(ids[todo]), "ride", ["power", "heart_rate"])
        rows = []
        for j in todo:                                             # rides: each one's work is vectorised
            z = streams.get(ids[j])
            found = cls.rows(RideIntervals(z["power"], knots[j], z["heart_rate"], wp[j] if wp[j] == wp[j] else None)) \
                if z is not None and knots[j][2] > 0 else []
            rows += [{"id": ids[j], "date": dates.iloc[j], "key": keys[j], **x} for x in found or [{"start": -1}]]
        new = pd.DataFrame(rows)
        keep = old[old["id"].isin(ids) & ~old["id"].isin(ids[todo])] if old is not None and len(old) else None
        out = pd.concat([keep, new], ignore_index=True) if keep is not None and len(keep) else new
        store.save(out.reset_index(drop=True), processed, cls.NAME)
        progress(f"intervals: {todo.size} rides looked at, {int((out['start'] >= 0).sum())} intervals in all")
        return out

    @classmethod
    def load(cls, processed) -> pd.DataFrame | None:
        """the intervals (rides without any left out), or None before the first update."""
        from ..io import store
        t = store.load(processed, cls.NAME)
        return None if t is None else t[t["start"] >= 0].reset_index(drop=True)
