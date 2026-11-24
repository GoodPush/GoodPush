Design choices and limits
=========================

What the model assumes, where those assumptions are weak, and what could replace them.

What is linear, and what isn't
------------------------------

* **Linear**: the dose → adaptation → state chain. Given the day's doses, each state moves by a fixed multiple of
  them (the gain matrix), and decays toward its floor at a fixed rate. In the long run a steady daily dose
  :math:`d` moves the states by :math:`E\,d` (:math:`E` the training effects, :doc:`dynamics`): twice the dose,
  twice the effect. This is a first-order linear system per state (a Banister-type impulse response
  :cite:p:`banister1975,calvert1976` with a delay through the pool, as in PerPot :cite:p:`perl2001`).
* **Nonlinear**: almost everything around it. The states are logs and logits, so effects are *relative* and
  the shares LT1 / 1-h power and CP / MAP saturate near 1. The domain edges move with the state, so the same ride is a different dose as you get
  fitter. Adaptation slows with fatigue (:math:`e^{-\gamma F}`). The power curve is a nonlinear function of the
  states, and fatigue load goes with the square of intensity.

The chain of ceilings (CP a share of MAP, LT1 a share of CP) is a structural choice, not a linear dependency: it
guarantees LT1 < CP < MAP and lets training raise the ceiling separately from filling the gaps below it.
Independent states would need constraints to keep them ordered, and the data (few LT1 readings, few maximal
5-min efforts) couldn't separate them.

Where linear is a real weakness
-------------------------------

* **No diminishing returns per dose.** Within one state there is no ceiling except the floor / detraining
  equilibrium. In reality the tenth hour of threshold work in a week buys less than the first, and an adaptation
  that is near its limit responds less. The shares (LT1 / 1-h power, CP / MAP) do saturate through their logits, so
  the value of threshold work falls as CP nears MAP; the ceiling (log MAP) does not. Options: a dose-dependent response :cite:p:`busso2003`, or a saturating dose
  response (a Hill function, :math:`E\,d/(d + d_{50})`, where :math:`d_{50}` is the dose that gives half the
  maximum effect).
* **Fixed gain pattern.** Which domain may train which state is fixed (e.g. VO2max intervals cannot change
  LT1's share directly). The sizes are learned; the pattern isn't.
* **W' is not trained on its own.** It follows from MAP, CP and Pmax, so 1-3 min power can't improve separately
  from them.

The chain of ceilings replaced an earlier structure (states CP, LT1 / CP, W', Pmax, durability; MAP followed from
the curve), which couldn't express "raise the ceiling" versus "fill the gap". Fitted from scratch at four cut-off
dates and scored on the next 12 weeks of best powers and readings, the chain predicted better at three of four,
with one fewer training effect (October 2026, on one athlete's data).

Identifying the ceiling's response needs maximal 3-8 min efforts (ramp tests) often enough to see MAP move. Work
above MAP (short, very hard reps) that a rider seldom does leaves its effect at the prior.

Do we model the entire power curve?
-----------------------------------

Yes, through one parametric family (OmPD) with four parameters (CP, W', Pmax, fade; W' set by MAP), observed at
12 durations from 5 s to 2 h each riding day. Limits:

* the curve's **shape** is fixed by OmPD: if a rider's 1-3 min power sits above or below what CP, W' and Pmax
  imply, the model can't show it;
* **fatigue** scales every duration by the same factor; in reality fatigue costs repeated high-intensity power and
  long-duration power differently :cite:p:`skiba2012,clark2019a,maunder2021`;
* the sprint end (Pmax) and long end (durability) are seldom ridden all-out, so they are mostly lower bounds: they
  only **drift** (process noise) rather than respond to training. With training effects on them, the sampler's
  chains didn't mix.

The likelihood
--------------

See :doc:`likelihood`. Its main weaknesses:

* **Repeated evidence.** One hard ride gives up to 12 best powers (one effort seen at several durations). They are
  now down-weighted - a day counts as at most three observations (:doc:`likelihood`) - but not solved: the windows
  still overlap, and three is a round number, not something the data chose. A likelihood on the whole daily curve
  with correlated errors would be the real answer.
* **The all-out / not all-out split** is a mixture with a probability per ride kind, not per ride: a ride that was
  a genuine all-out 5 min inside an "easy" day is only recognised by fitting the curve.
* **Training is taken as exact.** Power meter calibration drift and missing rides go into the process noise.
* **The shortfall's shape over durations is too stiff.** How far an ordinary ride's best power sits below capacity
  is one Gamma distribution whose scale changes log-linearly with duration, times one factor per ride kind. Real
  riding isn't that regular: intervals put 3-5 min power close to capacity while the same ride's 20-60 min power
  is far below it, and an easy ride may hold one sprint. An offset per duration on the shortfall's scale was tried
  and taken out: it let the shortfall's shape sharpen and the ceiling rise well above tested power. The stiff line
  is what holds the ceiling down to what tests and maxed efforts show.
* **Races are hard rides, not tests.** A race's best powers come inside repeated efforts (surges, sprints, time in
  the wheels), not from one fresh steady effort: with races counted as tests their 20 min and longer sat far below
  the curve and pulled it. A race's maxed efforts (heart rate at its maximum) still count as all-out.

Are the ranges honest?
----------------------

Checked by rolling-origin :cite:p:`tashman2000` coverage (``goodpush physio --coverage``): at each of 24 monthly cut-offs the
posterior is sampled from the data before it, and every threshold reading and best power of the next six weeks is
placed in its predictive distribution (later weeks' noise drawn from the prior, training as ridden). On one
athlete's data (November 2026):

* a 90 % range held about 90 % of the held-out threshold readings and of the best powers, and a 50 % range 54-58 %;
  the same 2, 4 and 6 weeks ahead, so the size of the weekly noise is about right for forecasts;
* threshold readings came in slightly above the forecast (inside the range): a small level error, not a width
  error;
* by duration the best powers are **not** evenly calibrated: 5-s and 3-5 min powers of ordinary rides beat the
  95th percentile two to five times as often as they should, 20-60 min powers too seldom. Tests and
  heart-rate-confirmed maximal efforts at 3-20 min were not under-predicted, so this is the shortfall model above,
  not the ceiling (since: races as hard rides; not yet re-checked);
* the sprint end is not calibrated at all: treat Pmax and its range as an assumption.

So the ranges of LT1, CP and MAP can be read as stated; the predictive range of a single ride's best power at one
duration cannot.

Recovery markers
----------------

Sleep, HRV, resting heart rate and bedtime track recovery :cite:p:`plews2013,fullagar2015`; in the model they were
tried two ways and taken out:

* as **causes** of recovery (better sleep → faster adaptation) they fitted backwards: you sleep more *because* you
  are tired, so more sleep looked like slower adaptation;
* as **signals** of fatigue (a measurement model, day by day) thousands of strongly autocorrelated days swamped
  everything and turned fatigue into a slow drift (time constant 115-340 days).

They act through the plan's readiness instead (two short nights or late bedtimes → an easier day). A measurement
model on weekly averages is the next thing to try.

Identifiability
---------------

With one athlete's data, a parameter is only learned where the training varied: a domain that was always ridden
about the same amount (e.g. little work above MAP for years) leaves its effect close to the prior (impulse-response models are known to be poorly identified from one
athlete's data, :cite:t:`hellard2006`). LT1 is only
observed through threshold readings (alpha1, ramp tests); in years without them the LT1 line is the model's
extrapolation from its fitted effects, not a measurement. Compare the posterior with the prior before reading
anything into a parameter, and treat results as hypotheses about one person.
