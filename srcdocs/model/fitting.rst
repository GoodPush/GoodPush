Fitting
=======

All of it works on **standardised parameters** :math:`w = (z - \mu_0)/s_0` (element-wise), where :math:`z` is
the flat vector of every parameter on its unconstrained scale and :math:`\mu_0, s_0` are the prior means and
standard deviations (:doc:`priors`). Every coordinate then starts on the same scale
(:class:`goodpush.physio.PhysioFit`).

Posterior mode
--------------

L-BFGS-B :cite:p:`byrd1995` on the negative log posterior with JAX gradients :cite:p:`bradbury2018`, from the last fit (warm start, new weeks at their
prior mean) or from the priors (``--fresh``). After a bad line search L-BFGS's memory can mislead it, so it
restarts (up to 5 times) while that still improves the posterior by more than 0.1. This is ``goodpush physio``,
and what the plan uses.

Uncertainty
-----------

* **NUTS** (``goodpush physio --samples N --nuts``): NumPyro's :cite:p:`phan2019` No-U-Turn sampler :cite:p:`hoffman2014`, 4 chains in parallel (one CPU
  device each). Two changes of coordinates come first, each leaving the posterior the same (Jacobian-corrected). The
  training effects of a state are sampled as their log total at the rider's average dose mix and the shares of its
  domains: the data fix what the usual mix does, not each domain alone. For MAP and CP / MAP that total is taken
  after :math:`H` = 84 days of the usual dose instead of in the long run,
  :math:`r = \log(\text{total}) + \log(1 - e^{-H/\tau})`: for a short :math:`\tau` the long-run effect, for a long one
  the first gain :math:`\times H`, the quantity the data pin down either way (with the long-run total the chains froze
  for long stretches at long time constants). And the detrained floor is sampled as the level there, floor +
  :math:`e^r`: years of rides fix where the state sits under the usual training, far better than how it splits into
  floor and training lift, and with the lift on a log scale that ridge is curved (divergent draws gathered where the
  lift was large and the floor low). Both are shears, so their Jacobian is 1; the preconditioner's Jacobian of the
  map, :math:`J^\top H J`, includes the log :math:`\tau` columns they depend on. NUTS then samples whitened coordinates :math:`v = v_\text{mode} + L u` with an identity mass,
  :math:`L = L_1 L_2 \cdots`, each stage :math:`L_i = \operatorname{diag}(s_i)(I + V_i \operatorname{diag}(c_i) V_i^\top)`
  in low rank (a step costs :math:`D \times` rank, not :math:`D^2`). The first stage is the inverse Hessian's
  symmetric square root at the mode, in the directions with curvature above 1.1. The Hessian is local: a long or
  curved ridge is wider than the curvature at the mode says, and NUTS then needs up to 1023 steps a draw. So each
  warm-up block (50, 100, 150 steps) adds a stage learned from where the chains went. For a Gaussian with covariance
  :math:`S` in the current coordinates, every draw and its gradient satisfy :math:`x = S g`, so
  :math:`x - g = (I - S^{-1}) x` is zero wherever the whitening is right; the singular vectors of the draws'
  :math:`x - g` give the wrong directions (wide or narrow), and in that subspace the variance is the geometric mean
  of the draws' covariance and the inverse of the gradients' (exact for a Gaussian; a principal-component estimate
  from the draws alone is off by about :math:`D / n` with :math:`n` draws in :math:`D` dimensions). Directions with
  a variance more than 2x off are corrected, then each coordinate's scale. A last block (100 steps) tunes only the
  step size; the draws use the chains' median for every chain (a chain's own tuned step only says where it was
  while tuning). Every block reports steps a draw, the share at the tree limit and the step size per chain, with a
  warning for a chain that sees another shape. Split R-hat :cite:p:`gelman1992` and the smallest effective sample size are reported; the R-hat target of 1.01 is
  the one :cite:t:`vehtari2021` recommend. ``N``
  draws per chain, after ``--warmup`` steps (default 400).
* **Speed** (:mod:`goodpush.physio.numba_posterior`): the potential and its gradient are computed in numba :cite:p:`lam2015`, with a
  hand-written backward (adjoint) pass over the days (W' differentiated implicitly at its Newton solution). They
  equal the JAX model's to ~1e-10 (tested); one evaluation takes ~7 ms instead of ~37 ms on one core for years of
  data, and the preconditioner's Hessian (central differences of the exact gradient) ~35 s instead of ~4 min.
  The run is **checkpointed** (``data/processed/physio_nuts.ckpt``) after every warm-up block and every 200
  draws; running the same command again goes on from there, with the data as it was when the run started (newer
  days follow in the daily updates). ``--fresh`` starts over.
  The finished run's checkpoint is kept, so ``--ess T`` can go on drawing (blocks of 200) until the smallest
  effective sample size (all but Pmax's own parameters, reported apart) reaches ``T`` and R-hat <= 1.01 (or ``--max-minutes``, or 4x ``--samples`` draws). Without a
  checkpoint it goes on from the saved draws: they are mapped into the sampler's coordinates, the chains start at
  their last draws, and the warm-up runs again from there. The ESS is what sets the
  precision of the ranges (Monte Carlo error = posterior sd / sqrt(ESS)): beyond ~400 more draws change little.
* **The stored whitening.** A full NUTS run starts from the Hessian stage alone, takes the full warm-up, and at the
  end of its learning blocks writes the stages it learned (everything after the Hessian stage) with the parameter
  names, the model's structure and the sampler's name to ``data/processed/physio_whitening.pkl`` (kept on this
  machine), replacing what was there; it never reads that file. A refresh (below) reads it: the Hessian stage fresh
  (the mode moved), the stored stages mapped onto the new parameter vector by name (parameters the stored run
  didn't have, such as new weeks, get no correction), and a 200-step warm-up instead of 400: one 100-step learning
  block, which corrects what moved, for that run only, and the 100-step step-size block. A refresh never writes
  the file, so the stages don't pile up day after day; the next full run replaces them. A run going on from its own
  checkpoint keeps the checkpoint's whitening. Any linear map leaves the posterior the same, so a stale whitening
  costs speed, never correctness.
* There is no quick approximation: a Gaussian at the mode (Laplace) was tried and taken out - checked by
  importance sampling on real data, its draws were worth about one in 200. ``--samples N`` without ``--nuts`` fits
  the mode only and says that ranges need ``--nuts``.

The daily states with posterior draws get 5 / 95 % bands (``*_p05``, ``*_p95`` in ``physio_states.parquet``).

Staying current: incremental updates
------------------------------------

A full NUTS run over years of data takes about an hour, so new days are taken in cheaply
(:class:`goodpush.models.physio.PhysioRun`):

* **The mode**: a full posterior-mode fit over everything, warm-started from the stored fit by parameter name (new
  weeks' noise at its prior mean). From a nearby optimum L-BFGS needs few iterations: about a second for a week of
  new days, and it ends at the same optimum as any full fit. When data older than 28 days changed (a late sync, a
  re-import) the same fit runs; only the draws' warm-up differs (below).
* **The ranges** (``PhysioRun.refresh``, ``goodpush physio --refresh``): a short NUTS run on today's data, its
  chains started from the last draw of each chain of the stored run (aligned by name; new weeks' noise at the new
  mode). Those start points belong to another day's posterior, so they are not kept as draws. With the stored
  whitening the warm-up is 200 steps (one learning block, one step-size block); when older data changed, the full
  warm-up. Then 800 draws a chain, more until the smallest ESS reaches 200 and the worst R-hat is <= 1.02 (a full
  run asks 1.01), for at most 12 minutes of drawing. A refresh that ends short of those targets saves nothing: the
  stored ranges keep their date, and the same command goes on from its checkpoint. Its
  checkpoint is its own: a refresh never goes on from a full run's, nor the other way round.
* **Between refreshes** the stored draws are kept as they are, dated by the data day of the run that made them
  (``draws_day``); the app gives that date. A full run
  (``goodpush physio --nuts --samples 2500 --ess 400 --max-minutes 540``) is only needed when there are no draws
  or the model changed.
* **In the app** (``[physio] ranges_every_days = 7``, 0 = never) the refresh runs on the home computer, in its own
  process at below-normal priority, once the ranges are that many data days older than the best fit, or sooner
  when today's best-fit CP, LT1 or MAP lies outside its stored 5-95 % range (a test or a maxed effort moved the
  fit, and the old bands would sit beside the number). The ranges move slowly; the best fit follows every ride. At
  most one start per data day, and never while a job, a ride or another NUTS run (a file lock) is going on. Nothing the app serves waits for it: pages use the mode and the
  stored draws, and read the new ranges once they are saved (files are replaced whole).

Importance re-weighting of the stored draws was used until 2026-11-10 and taken out: as the data move away from
the run that made the draws, the weights degenerate.

What the fit saw is stored with it (``physio_fit.npz``: a key over days, efforts and readings, and one over the
history before the last 28 days), so nothing is recomputed while the data are the same.
