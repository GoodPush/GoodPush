Planning with the model
=======================

With ``[plan] load_model = "physio"`` the plan's weekly load comes from the model; the daily schedule (which
session on which day) is still the plan's optimiser.

Weekly load: model-predictive control
-------------------------------------

The plan's template fixes the structure: phases, blocks and recovery weeks, key days, tests, races, which days
are easy. For each week :math:`w` the model chooses two numbers (:class:`goodpush.physio.plan.DosePlan`):

* :math:`H_w` - easy hours (spread over the week's easy days, the long day weighted double);
* :math:`\beta_w \in [0, 1]` - how much of the week's key sessions to keep (1 = all; less = an unloading week).

Those choices give each future day's power histogram; the model forecasts the states and the fatigue
:math:`F_t` for every day :math:`t` from today on (no process noise), and the choices minimise

.. math::

   -\sum_i \omega_i \,\overline{\log P_i}
   \;+\; c_T\,H_{tot}
   \;+\; \lambda_F \sum_t \max(0, F_t - F_{cap})^2
   \;+\; \lambda_K \sum_w (1 - \beta_w)^2
   \;+\; \lambda_R\, B(H, \beta)
   \;+\; \lambda_S \sum_w \max(0, S_w - H_w)^2

with:

* **Targets** :math:`i`, each with a weight :math:`\omega_i`; :math:`\overline{\log P_i}` is the mean log power over
  the target's durations on its day. A races (weight 1) and B races (0.3) use all-out power with fatigue on the
  race morning, over durations that suit the race type (e.g. a road race 5, 20, 60 min; a time trial 20, 60 min);
  fitness at the end of the plan uses capacity at 5, 20, 60 min (weight 0.3, or 1 without races).
* **Time cost**: :math:`H_{tot}` is the hours ridden over the whole plan, :math:`c_T` = 0.0002 (0.02 % of target
  power per hour): an hour has to buy something.
* **Fatigue backstop** :math:`F_{cap}`, weight :math:`\lambda_F` = 1: 10 % above the highest fatigue in your history, so each block may go a
  little beyond anything you have done and the ceiling rises with you. Only with 120 days of rides behind it;
  with less history there is none. Otherwise the cost of fatigue is what the model learned (slower adaptation, less
  on the day), not a set limit.
* **Keep**, weight :math:`\lambda_K` = 0.002: keep the template's key sessions unless dropping them helps the
  forecast.
* **Block shape** :math:`B(H, \beta)`, weight :math:`\lambda_R` = 0.05: the sum of squared hours by which the weeks
  miss the block shape (loading weeks, then a recovery week, as coaches periodise :cite:p:`friel2018`) - within a
  block each loading week at least 8 % more hours than the week before, the
  recovery week at most 65 % of the week before it.
* **Steady hours** :math:`S_w`, weight :math:`\lambda_S` = 0.001 per hour² short: time on the bike grows steadily.
  For a loading week :math:`S_w` is your average week of the last 4 weeks, 10 % more after each loading week (the
  volume ramp, one step below its cap), within the week's hours; recovery weeks, recovery phases and tapers have
  none. A week far from the target days barely moves the forecast - without this its hours would be wherever the
  optimiser happens to stop; with it a flat week follows the ramp, and a week where the forecast clearly gains
  still goes its own way.

Smooth in :math:`(H, \beta)`, so L-BFGS-B :cite:p:`byrd1995` with box bounds and JAX gradients; a few dozen variables, under a second.
The weekly hours become TSS targets for the plan's daily optimiser
(:class:`goodpush.planning.physio_load.PhysioLoad`).

Targets in your watts
---------------------

Workouts are written in textbook %FTP. :class:`goodpush.planning.intensity.IntensityMap` turns them into watts
through your thresholds **on the session's day** (the model's forecast for that day), by straight lines between:

.. list-table::
   :header-rows: 1

   * - %FTP
     - 0
     - 0.75
     - 1.00
     - 1.20
     - 1.50
     - 2.50
   * - watts
     - 0
     - LT1
     - CP
     - MAP
     - 0.8 × all-out 1 min
     - 0.9 × Pmax

So endurance stays below your LT1 wherever it is, threshold sits at your CP, and VO2max work between CP and your
MAP. Tests keep their own protocol. Forecasts change as the model takes in new rides.
