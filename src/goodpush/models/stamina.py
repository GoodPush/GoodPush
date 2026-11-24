"""Stamina through a ride from its 1 Hz power: what you have left now (stamina) and the most you'd get back
by easing off (potential). An open take on the two lines of Garmin's Real-Time Stamina, from published
models and the day's fitted CP, W' and LT1.

    potential = (CP_t + W'_t / REF_S) / (CP + W' / REF_S)     your 5-min power now, as a share of it fresh
    stamina   = potential x W'bal_t / W'_t                    0 when W' is spent: nothing left above CP

W' balance, bi-exponential (Chorley et al. 2021, Eur J Appl Physiol, doi 10.1007/s00421-021-04874-3;
trained cyclists, recovery at 50 W): of what you spend above CP, FAST comes back with tau TAU_FAST and
the rest with TAU_SLOW. The slow phase slows with repeated depletions (Chorley et al. 2019, Int J Sports
Physiol Perform 14:149; 2021: 377 -> 549 s from the first recovery to the second, each after a W' to
exhaustion): TAU_SLOW grows by SLOW_GROWTH per W' spent after the first, up to the second W' (the range
measured; growing on would stop long rides with many surges from ever recovering). Recovering closer to
CP is slower: both taus scale with Skiba's tau(D_CP) = 546 exp(-0.01 D_CP) + 316 s (Skiba et al. 2012, Med Sci Sports Exerc 44:1526) against its value at
REF_GAP, the gap below CP of Chorley's recoveries. No refill above CP. Each second uses only the ride up
to it, as you'd see it live.

Durability: CP and W' fall with time above LT1 (the heavy domain and up). Clark et al. 2019 (J Appl
Physiol 127:726; Am J Physiol Regul Integr Comp Physiol 317:R59): after 2 h of heavy cycling CP was
~9-11 % lower (unchanged at 40 and 80 min) and W' ~20 % lower (already at 80 min). Codebase constants
from that: CP -CP_DROP per hour above LT1 past CP_ONSET_S, W' -W_DROP per hour from the start. Beyond
~2 h above LT1 this extrapolates; carbohydrate during the ride prevented the CP drop there.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class RideStamina:
    REF_S = 300.0               # s: potential is the power for this long (5 min, MAP)
    FAST = 0.477                # Chorley 2021 (individual fits): share of the W' deficit in the fast phase
    TAU_FAST = 22.0             # s, fast phase (unchanged with repeated bouts)
    TAU_SLOW = 377.0            # s, slow phase on the first recovery
    SLOW_GROWTH = 0.456         # slow tau x (1 + this) once a second W' is spent (549 / 377 s); no further
    REF_GAP = 250.0             # W below CP of Chorley's recoveries (50 W, CP ~300 W)
    TAU = (546.0, 0.01, 316.0)  # Skiba 2012: tau = a exp(-b D_CP) + c, for the recovery-power scaling
    CP_ONSET_S = 3600.0         # Clark 2019: CP held for 80 min, -9 % at 2 h above LT1
    CP_DROP = 0.10              # share of CP lost per hour above LT1 past the onset
    W_DROP = 0.10               # share of W' lost per hour above LT1 (-18 % at 80 min, -23 % at 2 h)
    BLOCK = 3600                # s per pass of a recursion: keeps exp(-cumsum log a) < e^190 (tau >= 19 s)

    def __init__(self, power, cp: float, w_prime: float, lt1: float | None = None):
        """power (W, 1 Hz, NaN = 0); the day's CP (W), W' (J) and LT1 (W; None: no durability decline)."""
        self.power = np.nan_to_num(np.asarray(power, float))
        self.cp, self.w_prime = float(cp), float(w_prime)
        self.lt1 = None if lt1 is None else float(lt1)

    @classmethod
    def for_day(cls, processed: Path, date, power) -> RideStamina | None:
        """with the fitted CP, W' and LT1 of the ride's day (processed/physio_states.parquet, its nearest
        day); None without a fit."""
        from .physio import PhysioRun
        try:
            st = pd.read_parquet(Path(processed) / PhysioRun.STATES, columns=["cp", "w_prime", "lt1"])
        except (OSError, ValueError, KeyError):
            return None
        if not len(st):
            return None
        i = int(np.clip(st.index.searchsorted(pd.Timestamp(date).normalize()), 0, len(st) - 1))
        cp, wp, lt1 = st.iloc[i].to_numpy(float)
        ok = np.isfinite([cp, wp, lt1]).all() and min(cp, wp, lt1) > 0
        return cls(power, cp, wp, lt1) if ok else None

    def heavy_hours(self) -> np.ndarray:
        """hours above LT1 so far, each second."""
        if self.lt1 is None:
            return np.zeros_like(self.power)
        return np.cumsum(self.power > self.lt1) / 3600.0

    def cp_now(self) -> np.ndarray:
        """CP each second (W), lowered by time above LT1 past CP_ONSET_S."""
        h = np.maximum(self.heavy_hours() - self.CP_ONSET_S / 3600.0, 0.0)
        return self.cp * np.maximum(1.0 - self.CP_DROP * h, 0.0)

    def w_prime_now(self) -> np.ndarray:
        """W' each second (J), lowered by time above LT1."""
        return self.w_prime * np.maximum(1.0 - self.W_DROP * self.heavy_hours(), 0.0)

    def skiba_tau(self, gap) -> np.ndarray:
        """Skiba 2012's recovery time constant (s) at `gap` W below CP."""
        a, b, c = self.TAU
        return a * np.exp(-b * np.asarray(gap, float)) + c

    def decaying_sum(self, b: np.ndarray, tau: np.ndarray, recovering: np.ndarray) -> np.ndarray:
        """D_t = a_t D_(t-1) + b_t with a_t = exp(-1/tau_t) where recovering, else 1: D_t = e^(L_t) cumsum(b e^(-L)),
        L = cumsum(log a), per BLOCK (a handful of passes, not a loop over seconds)."""
        log_a = np.where(recovering, -1.0 / tau, 0.0)
        out = np.empty_like(b)
        carry = 0.0
        for s in range(0, b.size, self.BLOCK):
            L = np.cumsum(log_a[s:s + self.BLOCK])
            d = np.exp(L) * (carry + np.cumsum(b[s:s + self.BLOCK] * np.exp(-L)))
            out[s:s + self.BLOCK] = d
            carry = d[-1]
        return out

    def w_balance(self) -> np.ndarray:
        """W' left each second (J): W'_t minus the fast and slow deficits (bi-exponential recovery)."""
        p, cp = self.power, self.cp_now()
        below = p < cp
        spent = np.maximum(p - cp, 0.0)
        k = self.skiba_tau(np.where(below, cp - p, self.REF_GAP)) / self.skiba_tau(self.REF_GAP)
        repeats = np.clip(np.cumsum(spent) / self.w_prime - 1.0, 0.0, 1.0)       # W's after the first, to 1
        fast = self.decaying_sum(self.FAST * spent, self.TAU_FAST * k, below)
        slow = self.decaying_sum((1 - self.FAST) * spent, self.TAU_SLOW * (1 + self.SLOW_GROWTH * repeats) * k, below)
        return self.w_prime_now() - fast - slow

    def fresh(self) -> float:
        """the REF_S power fresh (W): CP + W' / REF_S."""
        return self.cp + self.w_prime / self.REF_S

    def potential(self) -> np.ndarray:
        """share of your fresh REF_S power you'd have with W' full again (1 at the start)."""
        return (self.cp_now() + self.w_prime_now() / self.REF_S) / self.fresh()

    def current(self) -> np.ndarray:
        """stamina each second (0..1): potential x the share of W' left (0 once W' is spent)."""
        wp = np.maximum(self.w_prime_now(), 1e-9)
        return self.potential() * np.clip(self.w_balance() / wp, 0.0, 1.0)
