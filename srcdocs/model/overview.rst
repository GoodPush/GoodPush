Overview
========

The model follows one athlete day by day. Each morning the athlete has a **state**: what they could do if fresh
(capacity) and how tired they are (fatigue). Each day's riding moves the state; rides and threshold readings are
noisy, partial observations of it. Everything unknown (starting state, how training acts, how fast it fades, how
fatigue costs, how noisy each kind of observation is, and what the model can't explain week to week) is one
parameter vector with a prior, so fitting is a Bayesian inversion: the posterior over all of it given the data.
The structure is an impulse-response model of training :cite:p:`banister1975,calvert1976,busso2003` written as a
state-space model, as in fitness-fatigue models with Kalman-filter feedback :cite:p:`kolossa2017`; fusing several
kinds of observation, each with its own bias, into one latent state is GoodPush's own combination of these parts.

.. code-block:: text

   training (daily power histograms)
        |  doses per intensity domain (domain edges from today's thresholds)
        v
   adaptation pool --(rate x exp(-gamma F))--> capacity states --(OmPD)--> power-duration curve
        ^                                         |  detraining to a floor      |  x form exp(-s_F F)
        |                                         |                             v
   fatigue F  <-- daily load (P / CP)^2 hours     |                  ride best powers (censored)
                                                  +--> LT1, CP, MAP, 1-h power  <-- threshold readings

States
------

Five capacity states, on unconstrained scales so any real value is a valid state, plus fatigue:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - State
     - Scale
     - Meaning
   * - :math:`x_1`
     - :math:`\log \mathrm{MAP}`
     - the aerobic ceiling (W): power for 5 min all-out, about the top of a ramp test
   * - :math:`x_2`
     - :math:`\operatorname{logit} f_E`, :math:`f_E = \mathrm{LT1}/P(1\ \mathrm{h})`
     - first lactate / ventilatory threshold as a share of 1-hour power (below)
   * - :math:`x_3`
     - :math:`\operatorname{logit} f_T`, :math:`f_T = \mathrm{CP}/\mathrm{MAP}`
     - critical power :cite:p:`monod1965,jones2010` (the boundary between heavy and severe exercise; close to, but not the same as, LT2 / "FTP") as a share of MAP
   * - :math:`x_4`
     - :math:`\log P_{max}`
     - peak (sprint) power (W)
   * - :math:`x_5`
     - :math:`\log \delta`
     - durability :math:`\delta`: how slowly power falls past 30 min
   * - :math:`F`
     - hours at CP-equivalent
     - fatigue: lowers what you can do on the day, and slows adaptation

:math:`x_t = (x_1, \dots, x_5)` is the state vector on the morning of day :math:`t`; :math:`\mathrm{CP}_t`,
:math:`\mathrm{LT1}_t` etc. are the capacities it implies on that day, and :math:`F_t` is the fatigue.

The thresholds are a **chain of ceilings**: MAP is the ceiling, CP a share of it, LT1 a share of what you could
hold all-out for an hour, :math:`P(1\ \mathrm{h})` (the power-duration curve below):

.. math::

   \mathrm{MAP} = e^{x_1}, \qquad \mathrm{CP} = \sigma(x_3)\,\mathrm{MAP}, \qquad \mathrm{LT1} = \sigma(x_2)\,P(1\ \mathrm{h})

with :math:`\sigma(v) = 1/(1 + e^{-v})` the logistic function, so each share lies between 0 and 1.
LT1 < 1-hour power < CP < MAP always holds, and training can raise the ceiling (MAP, carrying CP and LT1 with it)
separately from filling the gaps below it (raising a share toward 1, which gets harder as it nears 1). LT1 is a
share of 1-hour power rather than of CP because 1-hour power can sit well below CP for a rider who fades over
long efforts, and LT1 can't be above it. See :doc:`dynamics` for which riding does what.

The power-duration curve
------------------------

The whole curve, from a few seconds to several hours, follows from the states through the omni-domain
power-duration model (OmPD, :cite:t:`puchowicz2020`):

.. math::

   P(\Delta) = \frac{W'}{\Delta}\left(1 - e^{-\Delta\,(P_{max} - \mathrm{CP})/W'}\right) + \mathrm{CP}
          \;-\; \mathbb{1}[\Delta > T]\;\frac{\mathrm{CP}}{\delta}\,\ln\frac{\Delta}{T}

where :math:`P(\Delta)` is the power (W) you could hold all-out for a duration :math:`\Delta` (s) when fresh;
:math:`\mathrm{CP}` (W), :math:`P_{max}` (W) and the durability :math:`\delta` come from the states above, and
:math:`W'` (J, the work capacity above CP) is whatever puts the curve's 5-min power at MAP
(:meth:`goodpush.physio.PowerCurve.w_prime_for`: one equation in one unknown, solved by a few Newton steps);
:math:`T` = 30 min; and :math:`\mathbb{1}[\cdot]` is 1 when the condition holds, else 0. The short end tends to
:math:`P_{max}`, the middle to :math:`\mathrm{CP} + W'/\Delta`, and past 30 min power falls by
:math:`\mathrm{fade} \times \mathrm{CP}` per e-fold of duration, where :math:`\mathrm{fade} = 1/\delta`. Also derived:

* **1-h power** :math:`P(60\ \mathrm{min})`: what an all-out hour would give (an "FTP" in the original sense, :cite:t:`allen2019`).

What you could do all-out on day :math:`t` is the capacity curve times a **form** factor :math:`e^{-s_F F_t}`,
where :math:`s_F` (fitted) is what a unit of fatigue costs, as a fraction of power: fatigue lowers every
duration by the same factor.

Where it is in the code
-----------------------

* :mod:`goodpush.physio` - the model itself (pure numpy / JAX / NumPyro; plain arrays in, nothing else of goodpush);
* :mod:`goodpush.models.physio` - builds the model's inputs from goodpush's tables, keeps the fit current;
* :mod:`goodpush.planning.physio_load` and :mod:`goodpush.planning.intensity` - the plan's use of it.

Run it with ``goodpush physio`` (posterior mode), ``goodpush physio --nuts --samples 2500 --ess 400
--max-minutes 540`` (with posterior draws, once), ``goodpush physio --refresh`` (today's ranges from those draws),
or let the app keep it current (``[plan] load_model = "physio"``, ``[physio] ranges_every_days``).
