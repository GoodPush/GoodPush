"""Power-duration curve from the physiological state."""
from __future__ import annotations

import jax.numpy as jnp


class PowerCurve:
    """Omni-domain power-duration model (Puchowicz, Baker & Clarke 2020):

        P(t) = W'/t * (1 - exp(-t (Pmax - CP) / W')) + CP              t <= 30 min
               ... - fade * CP * ln(t / 30 min)                         t >  30 min

    Short end -> Pmax, middle -> CP + W'/t, long end falls by `fade` x CP per e-fold of duration
    past 30 min (durability: a durable rider has a small fade). All arguments broadcast.
    """
    T_CP = 1800.0

    NEWTON = 8          # steps of w_prime_for: from its start it converges in ~4 (float64)

    @classmethod
    def power(cls, t, cp, w_prime, pmax, fade):
        t = jnp.asarray(t, float)
        span = jnp.maximum(pmax - cp, 1.0)
        short = w_prime / t * (1.0 - jnp.exp(-t * span / w_prime)) + cp
        return short - jnp.where(t > cls.T_CP, fade * cp * jnp.log(jnp.maximum(t, cls.T_CP) / cls.T_CP), 0.0)

    @classmethod
    def w_prime_for(cls, t, p, cp, pmax):
        """W' that puts P(t) at p (t <= 30 min): W'/t (1 - exp(-t (Pmax - CP)/W')) = p - CP. With y = t (Pmax - CP)/W'
        that is (1 - e^-y)/y = rho := (p - CP)/(Pmax - CP), solved for y by Newton steps in log y from
        y = 1/rho (exact as rho -> 0); rho is kept in (0.001, 0.95) (p between CP and Pmax)."""
        span = jnp.maximum(pmax - cp, 1.0)
        log_rho = jnp.log(jnp.clip((p - cp) / span, 1e-3, 0.95))
        u = -log_rho
        for _ in range(cls.NEWTON):                 # unrolled: a fixed handful of steps
            y = jnp.exp(u)
            g = jnp.log(-jnp.expm1(-y)) - u - log_rho
            u = u - g / (y / jnp.expm1(y) - 1.0)
        return t * span / jnp.exp(u)
