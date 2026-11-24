Dynamics
========

One day, from the morning of day :math:`t` to the next (:meth:`goodpush.physio.PhysioModel.step`). The day starts
with the state vector :math:`x_t` (:doc:`overview`), the adaptation pool :math:`\pi_t` (one entry per state:
adaptation earned but not yet realised) and the fatigue :math:`F_t`.

Doses: the day's training by intensity domain
---------------------------------------------

The day's riding is a power histogram: :math:`h_{t,b}` hours in power bin :math:`b`, whose centre is
:math:`P_b` watts (10 W bins). It is split into intensity domains at four edges :math:`\ell_1 < \dots < \ell_4`
(log watts) set by the **current** state, so "threshold work" moves up as you get fitter:

.. math::

   \ell = \log\left(\mathrm{LT1}_t,\ \mathrm{CP}_t,\ \mathrm{MAP}_t,\ \mathrm{MAP}_t + \tfrac12(P_{max,t} - \mathrm{MAP}_t)\right),
   \qquad a_k(b) = \sigma\!\left(\frac{\log P_b - \ell_k}{0.05}\right)

Here :math:`\sigma(v) = 1/(1 + e^{-v})` is the logistic function, so :math:`a_k(b)` is the share of bin
:math:`b` above edge :math:`k`: the edges are soft, 5 % wide in log power, which keeps the model differentiable.
The fourth edge is the **sprint line**, halfway from MAP to :math:`P_{max}`. A bin's share in each domain is
:math:`1 - a_1,\ a_1 - a_2,\ a_2 - a_3,\ a_3 - a_4,\ a_4` (below LT1 ... above the sprint line), and a domain's
dose :math:`d_{t,j}` is its hours (the bins' hours times their shares in it, summed) divided by the domain's unit
:math:`u_j`:

.. list-table::
   :header-rows: 1
   :widths: 12 38 50

   * - Dose
     - Domain
     - Unit (a dose of 1)
   * - z1
     - below LT1
     - 1 h
   * - z2
     - LT1 - CP (tempo, sweet spot, threshold)
     - 1 h
   * - z3
     - CP - MAP (severe: VO2max intervals, about 105-120 % FTP); the band LT1 - MAP split by power and heart rate (below)
     - 10 min
   * - z4
     - MAP - sprint line (above 5-min power: 30/30s, 1-2 min reps, hard surges)
     - 5 min
   * - sprint
     - above the sprint line
     - 1 min
   * - long
     - hours past 2 h in a ride
     - 1 h

The units put a typical session's dose near 1, so the priors on training effects are comparable across domains.

**Heavy or severe: power and heart rate together.** The band LT1 - MAP is split into heavy (z2) and severe (z3) by a
severe share per second,

.. math::

   s = \sigma\!\left(\frac{\log P - \log \mathrm{CP}_t}{0.05} + e\right), \qquad
   z_2 \mathrel{+}= (a_1 - a_3)(1 - s), \quad z_3 \mathrel{+}= (a_1 - a_3)\, s,

the soft CP edge plus heart-rate evidence :math:`e`, added as log-odds (z1, z4, sprint and long as above). Either alone
fails. Power within a few % of CP can't tell steady riding from severe - a VO2max rep with a hard start ridden right at
CP is booked half heavy, a steady stretch a little above it as severe. Heart rate alone counts long hot rides and long
climbs (heart rate high for reasons other than the effort) and books history by one rule and planned sessions, which
have no heart rate, by another. Together, heart rate moves seconds near CP to the side it shows, and power keeps far
from CP what it says.

The evidence comes from the data only (``metrics.severe_hr``): heart rate corrected for its lag
(:math:`H + \tau\,\dot H`, :math:`\tau` the rider's own first-order time constant, the one at which lag-filtered power
lines up best with heart rate) and for each ride's drift (the upward trend of heart rate less what power explains,
fitted on its seconds below CP), against threshold heart rate then (an input, not fitted):
:math:`e = (H_{corr} - \mathrm{LTHR}_t) / w`, :math:`w` half the scatter of corrected heart rate around a straight line
(heart rate on log power) in steady riding between LT1 and CP. It enters the model binned (12 bins, clipped to
:math:`\pm 6`): hours per day, power bin and evidence bin. Seconds without heart rate - no strap, a dropout, every
planned or simulated session - have :math:`e = 0`, the plain power edge, so history and plan share one formula. With
:math:`e = 0` the split differs from cutting the band at the CP edge (:math:`a_1 - a_2`, :math:`a_2 - a_3`) only where
two soft edges overlap.
:math:`d_t = (d_{t,1}, \dots, d_{t,6})` is the day's dose vector, in the order of the table.

**Compared with the usual zone models.** The domains are the classic exercise-intensity domains :cite:p:`poole2012,jones2010` (moderate below
LT1, heavy LT1 - CP, severe CP - MAP), with the extreme domain split at the sprint line. Standard models fix their
zones as % FTP (or lactate / heart rate); these edges are the athlete's own thresholds, so in % FTP they move with
the athlete. The % FTP columns use typical ratios: LT1 about 75-85 % FTP, CP 100-105 %, MAP (5-min power)
115-130 %, the sprint line 250-350 %. The three-zone model is Seiler's :cite:p:`seiler2006`, the five-zone one
Olympiatoppen's :cite:p:`seiler2010`, the seven-zone one Coggan's :cite:p:`allen2019`.

.. list-table::
   :header-rows: 1
   :widths: 9 14 21 26 30

   * - Dose
     - Typical % FTP
     - 3 zones (Seiler: LT1 / LT2)
     - 5 zones (Olympiatoppen I1-I5, lactate)
     - 7 zones (Coggan, % FTP)
   * - z1
     - < 75-85
     - Z1 (below LT1)
     - I1 (< ~2 mmol/l)
     - 1 recovery, 2 endurance, lower 3 tempo
   * - z2
     - 75-85 to 100-105
     - Z2 (LT1 - LT2), up to about CP
     - I2, I3 (2 - 4 mmol/l), up to about CP
     - upper 3 tempo, sweet spot, 4 threshold
   * - z3
     - 100-105 to 115-130
     - Z3 (above LT2)
     - I4, I5 (VO2 rising to max)
     - 5 VO2max, lower 6
   * - z4
     - 115-130 to 250-350
     - Z3
     - above I5 (anaerobic)
     - 6 anaerobic, lower 7
   * - sprint
     - > 250-350
     - Z3
     - above I5
     - upper 7 neuromuscular

So a Coggan "VO2max" interval (106-120 % FTP) is z3, which may raise both MAP and CP's share of MAP (the fit
says how much of each); reps above 5-min power (Coggan 6) are z4, which may raise MAP only. "long" has no counterpart: it counts duration, not intensity.

**CP is not LT2.** The 3- and 5-zone models put their upper edge at a lactate or ventilatory threshold (LT2,
~4 mmol/l, the maximal lactate steady state, the respiratory compensation point; :cite:t:`jones2019`); the model's z2 / z3 edge is CP, the
power-duration asymptote. They are measured differently (blood or breath vs the best-power curve) and usually lie
within a few percent of each other, CP more often a little above MLSS, but the gap depends on the athlete's
glycolytic profile (VLamax, :cite:t:`mader1986`): a high glycolytic rate pulls lactate thresholds below CP. So "Seiler Z2 = z2" holds
only roughly near the top; goodpush's alpha1 LT2 (:mod:`goodpush.alpha`) is a separate estimate, not an input here.

Adaptation: pool, rate, detraining
----------------------------------

.. math::

   r_t &= 1 - \exp\!\left(-\rho \odot e^{-\gamma F_t}\right) \\
   x_{t+1} &= x_t + r_t\,\pi_t - m \odot \frac{x_t - \phi}{\tau} + \varepsilon_t \\
   \pi_{t+1} &= (1 - r_t)\,\pi_t + G\,d_t

Symbols, from the top:

* :math:`r_t \in (0, 1)` - the share of the pool that turns into capacity today;
* :math:`\rho` (per day, fitted, one per trained state) - that rate when fresh: how fast the reservoir of
  training turns into capacity. Physiology sets its prior: the aerobic ceiling (MAP) in about 1-2 weeks, CP / MAP
  in 2-3, LT1 / 1-h power (mitochondria, fat use) in 4-6. One rate shared by all states fitted to about a day,
  following the old heart-rate readings' day-to-day swings; :math:`\gamma` (fitted) - how much fatigue slows it;
* :math:`\odot` - element-wise product; :math:`m` - 1 for the states that detrain, 0 for the others;
* :math:`\phi` - each state's detrained floor; :math:`\tau` (days) - each state's detraining time constant (both
  fitted; the division is element-wise);
* :math:`\varepsilon_t` - process noise (below);
* :math:`G` - the **gain matrix** (states × doses), non-zero only where training is allowed to act:

  .. list-table::
     :header-rows: 1
     :widths: 20 10 10 10 10 10 10

     * - State
       - z1
       - z2
       - z3
       - z4
       - sprint
       - long
     * - MAP
       - x
       - x
       - x
       - x
       -
       -
     * - LT1 / 1-h power
       - x
       - x
       -
       -
       -
       -
     * - CP / MAP
       -
       - x
       - x
       -
       -
       -
     * - Pmax
       -
       -
       -
       -
       -
       -
     * - durability
       -
       -
       -
       -
       -
       -

  Any aerobic work may raise the ceiling (the data decide how much each does); threshold-type work (LT1 - CP and
  CP - MAP) fills CP toward MAP; endurance hours (easy and tempo, any ride length) fill LT1 toward your 1-hour power.
  The sprint dose trains nothing: W' follows from MAP, CP and Pmax, and Pmax only drifts. On LT1 / 1-h power, z1 and z2
  share **one** training effect (an hour below LT1 counts as much as an hour LT1 - CP): they come together on nearly
  every easy ride, so the data fix only their sum - as two effects they slid along a ridge, and one forecasts as well.
  Hours past 2 h no longer train LT1 on their own: they are endurance hours too, and with both terms the posterior
  split between "endurance hours" and "long rides" (either alone fit as well).
* :math:`\pi_t` - the **adaptation pool**: training adds to it at once, it turns into capacity at rate
  :math:`r_t`: a delay between load and performance like PerPot's :cite:p:`perl2001` and the delayed fitness
  term of :cite:t:`calvert1976`. The rate falls with fatigue (:math:`e^{-\gamma F}`): you adapt while recovering, slower while
  tired. How much fatigue costs (:math:`\gamma`) is learned, not a set limit.
* :math:`\phi, \tau` - each trained state decays toward a **detrained floor** with its own time constant
  (:math:`m` = 1 for MAP, LT1 / 1-h power and CP/MAP; 0 for Pmax and durability, which only drift - see :doc:`limits`).
* The fitted quantity is the **training effect** :math:`E` (:math:`E_{ij} = G_{ij}\,\tau_i` for state :math:`i`
  and dose :math:`j`): how far a steady daily dose of 1 moves a state in the long run. With the same dose vector
  :math:`d` every day the states settle at :math:`x - \phi = E\,d`. The data pin down this product, not gain and
  time constant separately, so :math:`E` and :math:`\tau` are the parameters.

Because the states are logs (and a logit), the dynamics are linear in *log* capacity: a training effect is a
*relative* change (0.01 ≈ 1 %), and the shares LT1 / 1-h power and CP / MAP saturate as they approach 1.

Fatigue
-------

.. math::

   F_{t+1} = F_t\, e^{-1/\tau_F} + \sum_b h_{t,b} \left(\frac{P_b}{\mathrm{CP}_t}\right)^2

The day's load (the sum) is hours weighted by the square of intensity relative to CP: an hour at CP adds 1, an
hour at LT1 about 0.6 - the weighting of TSS, intensity factor squared :cite:p:`allen2019`, with CP for FTP. As
the fatigue term of an impulse-response model :cite:p:`banister1975`, fatigue clears exponentially with the time
constant :math:`\tau_F` (days, fitted). It does not lower
capacity; it lowers what you can do on the day by the form factor :math:`e^{-s_F F_t}` (:doc:`overview`) and
slows adaptation through :math:`r_t`.

Process noise
-------------

.. math::

   \varepsilon_t = \tfrac{1}{7}\,k_{w(t)}\,\nu \odot \eta_{w(t)}, \qquad \eta_w \sim \mathcal N(0, I)

where :math:`w(t)` is the week of day :math:`t` (a week's unexplained change comes in equal parts on its seven
days; as one step each Monday the fitted paths were staircases), :math:`\eta_w` one vector per week (fitted, with a standard
normal prior; :math:`I` is the identity matrix), :math:`\nu` the noise size per state and :math:`k_w` 1, or 5 in a
week that overlaps a severe illness (from onset to a week after the comeback; illnesses are found from the rider's
notes and resting heart rate). An illness takes fitness the training doesn't explain; with the usual size the fit read
the drop as fast detraining (on one athlete's data a lung infection moved the MAP time constant from about 230 to
300 days). One number a week shared
by all states was tried: near-max power forecast as well, but LT1 and CP / MAP could then only move in step with
MAP, and the fit made up for it with implausible dynamics (detraining in days) and a large reading bias. The noise absorbs what
the model doesn't explain: illness, heat, a new bike position, missing rides. Its size is **fixed**, not fitted,
at :math:`\nu = (0.005, 0.015, 0.015, 0.01, 0.015)` per week for (log MAP, logit LT1 / 1-h power, logit CP/MAP,
log Pmax, log durability). With a fitted size the posterior mode runs off to large noise that matches every
reading exactly. The size was chosen by backtest: at twice this the noise took over changes that training explains
(forecasts from planned training too flat, held-out LT1 readings predicted worse); at a fifth of it or none the
model could no longer follow real changes, and LT1's level went into its readings' bias.

Implementation
--------------

The whole history is one ``jax.lax.scan`` over days (:meth:`goodpush.physio.PhysioModel.run`). Arrays are padded to
buckets of 28 days and 64 observations, so the compiled code is reused as days are added; JAX keeps compiled code
on disk between runs.
