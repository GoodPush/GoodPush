# Methods in short

How GoodPush gets its numbers, outside the physiological model (which has [its own pages](../model/overview.rst)).
The papers and books behind each method are listed in [References](../references.rst) and in the app:
**Fitness > Sources**.

All of GoodPush's own findings about you (what drove your gains, illness links, how fast you can ramp) come from
your data alone: n = 1 and observational, so read them as hypotheses to test, not proof.

* **Ride streams**: resampled to one sample a second. Gaps of 10 s or less are filled forward, gaps of 10-300 s are
  stops (0 W), longer gaps are cut.
* **Per ride**: normalized power, IF, TSS, best powers from 1 s to 2 h, time in power and heart-rate zones,
  efficiency factor (NP / HR) and aerobic decoupling {cite:p}`allen2019,friel2018`. Rides without power use
  heart-rate TSS.
* **FTP** (automatic): fitted to Garmin's own FTP estimates and your entered tests, as a weighted mix of your best
  3-60 min powers (older efforts fading) and your power at threshold heart rate on steady rides. With too few of
  those to fit, it uses critical power {cite:p}`monod1965,jones2010` from the last 90 days instead. `[ftp] mode = "manual"` uses your tests.
* **Workout types**: each ride is classed as recovery, endurance, long, tempo, sweet spot, threshold, VO2max,
  anaerobic, race or test, from its sustained efforts relative to FTP, checked against heart rate and the workout's
  name.
* **Load**: CTL (42-day), ATL (7-day), TSB (form) {cite:p}`banister1975,allen2019`, the acute:chronic ratio
  {cite:p}`hulin2016,gabbett2016`, monotony and strain {cite:p}`foster1998`, ramp rate {cite:p}`friel2018`.
* **Performance marker**: the best 20-minute-equivalent power on hard days in the previous 6 weeks. Weeks without
  hard riding are "untested", not "detrained". Aerobic efficiency on steady rides is a second marker that needs no
  hard efforts.
* **What drove your progress**: correlations between 8-week training features (volume, intensity, consistency,
  rest, sleep...) and the change in the marker, with a cross-validated model as a check on how much they predict.
* **Overreaching** {cite:p}`meeusen2013,aubry2014`: weeks with load warning signs (a sharp load jump, deep fatigue, high monotony), each judged by
  what happened in the next 6 weeks (performance, a forced break, resting HR, HRV, sleep, fatigue notes).
* **Illness**: sickness words in ride names and descriptions (flu, covid, cough, sinus...) mark comeback rides; each
  illness is rebuilt backwards from the note (the training gap plus a resting-HR rise), then checked for warning
  signs in the weeks before. Add or correct illnesses as `[[events]] type = "illness"`.
* **Fitness-fatigue model**: your personal fitness and fatigue time constants (Banister), fitted to your tested
  performances {cite:p}`banister1975,calvert1976`; from one athlete's data they are poorly identified
  {cite:p}`hellard2006`.

## DFA alpha1

If you record the **alphaHRV** Connect IQ data field, or ride with GoodPush's trainer bar and a chest strap, GoodPush
reads alpha1 (how organised your beat-to-beat rhythm is: about 1.0 easy, falling through 0.75 near LT1 and 0.5 near
LT2; {cite:alp}`peng1995,gronwald2020,rogers2021a,rogers2021b`).

* **Warm-up check**: each ride's alpha1 and heart rate in the warm-up (minutes 5-20) and as the session gets going
  (20-40) against what your last 6 weeks say they should be *at the same power*, from steady riding with clean data
  only. Clearly lower or higher than normal is flagged; a warm-up well off normal makes the next day amber.
* **Thresholds**: LT1 (alpha1 0.75) and LT2 (0.5) with your heart rate at each. In ordinary rides alpha1 also falls
  with time into a ride, so these read low; a **ramp test** (power climbing steadily for 10+ minutes to above FTP,
  then stopping) gives better ones, and GoodPush spots ramp tests in your rides by itself.
* VLamax {cite:p}`mader1986` can't be estimated from power alone (it needs blood lactate); a **glycolytic profile** is shown instead:
  your best 15 / 30 / 60 s power relative to CP and W'.
