// Sources: where the methods come from (SourcesPage).
import {Page} from "./page.js";
import {esc} from "../util.js";

// where goodpush's methods come from: [topic, what goodpush uses it for, references]
const SOURCES = [
  ["Training load: TSS, CTL, ATL, TSB", "Load per ride, long-term load (CTL), fatigue and form; FTP from a 20-minute test (95 %); the power zones",
    ["Allen H, Coggan AR, McGregor S (2019). Training and Racing with a Power Meter, 3rd ed. VeloPress.",
     "Foster C (1998). Monitoring training in athletes with reference to overtraining syndrome. Med Sci Sports Exerc 30:1164-1168.",
     "Hulin BT, Gabbett TJ, Lawson DW, Caputi P, Sampson JA (2016). The acute:chronic workload ratio predicts injury: high chronic workload may decrease injury risk in elite rugby league players. Br J Sports Med 50:231-236."]],
  ["Fitness-fatigue (impulse-response) model", "Your personal fitness and fatigue time constants (Banister model) and the physiological model's structure",
    ["Banister EW, Calvert TW, Savage MV, Bach T (1975). A systems model of training for athletic performance. Aust J Sports Med 7:57-61.",
     "Calvert TW, Banister EW, Savage MV, Bach T (1976). A systems model of the effects of training on physical performance. IEEE Trans Syst Man Cybern SMC-6:94-102.",
     "Busso T (2003). Variable dose-response relationship between exercise training and performance. Med Sci Sports Exerc 35:1188-1195.",
     "Hellard P, Avalos M, Lacoste L, Barale F, Chatard JC, Millet GP (2006). Assessing the limitations of the Banister model in monitoring training. J Sports Sci 24:509-520.",
     "Perl J (2001). PerPot: a metamodel for simulation of load performance interaction. Eur J Sport Sci 1(2):1-13.",
     "Kolossa D, Bin Azhar MA, Rasche C, et al. (2017). Performance estimation using the fitness-fatigue model with Kalman filter feedback. Int J Comput Sci Sport 16:117-129."]],
  ["Critical power and W'", "CP / W' from your best efforts; the power-duration curve; the intensity domains; why CP and lactate thresholds differ",
    ["Monod H, Scherrer J (1965). The work capacity of a synergic muscular group. Ergonomics 8:329-338.",
     "Puchowicz MJ, Baker J, Clarke DC (2020). Development and field validation of an omni-domain power-duration model. J Sports Sci 38:801-813.",
     "Jones AM, Vanhatalo A, Burnley M, Morton RH, Poole DC (2010). Critical power: implications for determination of VO2max and exercise tolerance. Med Sci Sports Exerc 42:1876-1890.",
     "Poole DC, Jones AM (2012). Oxygen uptake kinetics. Compr Physiol 2:933-996.",
     "Jones AM, Burnley M, Black MI, Poole DC, Vanhatalo A (2019). The maximal metabolic steady state: redefining the 'gold standard'. Physiol Rep 7:e14098.",
     "Mader A, Heck H (1986). A theory of the metabolic origin of \"anaerobic threshold\". Int J Sports Med 7 Suppl 1:45-65."]],
  ["DFA alpha1", "How organised your beat-to-beat rhythm is: about 1.0 easy, falling through 0.75 near your first threshold (LT1) and 0.5 near the second (LT2); warm-up checks, LT1 / LT2 from steady riding and from ramp tests. Indoor rides use goodpush's own alpha1 from your chest strap's beat-to-beat intervals; outdoor rides your Garmin's alphaHRV field (a closed tool: goodpush reads what it recorded, it doesn't copy how it works). goodpush's own: DFA over boxes of 4-16 beats, the last 2 minutes every 5 s. With these box sizes DFA reads uncorrelated beats as about 0.58, not 0.5 (a known property of the method). Slow heart-rate drift (a ramp, a climb) is taken out first (smoothness priors, lambda 500), or it lifts alpha1. Odd beats: further from their neighbours than the local spread allows (adapts to the strap and the intensity, so breathing swings at rest aren't odd); a missed or extra beat is repaired when it spans a whole number of beats (up to 3), anything longer is never filled with invented beats - only the boxes holding it are left out. A window with more than 5 % repaired or bad beats has no value. On a Polar H10 ramp goodpush's alpha1 agreed with alphaHRV at r = 0.97, about 0.06 higher; LT1 / LT2 heart rates within 2-3 bpm. A ramp test reads LT1 high (alpha1 lags the rising power) and steady riding reads it low (alpha1 also falls with time into a ride): the physio model weighs both with each kind's own bias.",
    ["Peng CK, Havlin S, Stanley HE, Goldberger AL (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. Chaos 5:82-87.",
     "Kantelhardt JW, Koscielny-Bunde E, Rego HHA, Havlin S, Bunde A (2001). Detecting long-range correlations with detrended fluctuation analysis. Physica A 295:441-454.",
     "Gronwald T, Rogers B, Hoos O (2020). Fractal correlation properties of heart rate variability: a new biomarker for intensity distribution in endurance exercise and training prescription? Front Physiol 11:550572.",
     "Rogers B, Giles D, Draper N, Hoos O, Gronwald T (2021). A new detection method defining the aerobic threshold for endurance exercise and training prescription based on fractal correlation properties of heart rate variability. Front Physiol 11:596567.",
     "Rogers B, Giles D, Draper N, Mourot L, Gronwald T (2021). Detection of the anaerobic threshold in endurance sports: validation of a new method using correlation properties of heart rate variability. J Funct Morphol Kinesiol 6:38.",
     "Rogers B, Giles D, Draper N, Mourot L, Gronwald T (2021). Influence of artefact correction and recording device type on the practical application of a non-linear heart rate variability biomarker for aerobic threshold determination. Sensors 21:821.",
     "Lipponen JA, Tarvainen MP (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat classification. J Med Eng Technol 43:173-181.",
     "Tarvainen MP, Ranta-aho PO, Karjalainen PA (2002). An advanced detrending method with application to HRV analysis. IEEE Trans Biomed Eng 49:172-175.",
     "Ma QDY, Bartsch RP, Bernaola-Galván P, Yoneyama M, Ivanov PCh (2010). Effect of extreme data loss on long-range correlated and anticorrelated signals quantified by detrended fluctuation analysis. Phys Rev E 81:031101."]],
  ["Ramp tests", "The plan's ramp to failure: 2-minute steps of 6 % of FTP from 50 %. Top of the ramp: the last step held to the end (10 s short still counts) plus the share of the next one you rode. Heart rate at each step is its last 30 s (heart rate lags the power), giving your heart rate at your thresholds' watts. Heart-rate recovery: the drop 1 and 2 minutes after you stop. Without alpha1, LT1 / LT2 are read as the power where the ramp reached your LT1 / LT2 heart rate (from alpha1 on earlier rides) - goodpush's own rule, which reads high; the physio model learns by how much. A slower ramp tops out lower than a 1-minute one, so compare a test with your other ramp tests.",
    ["Kuipers H, Verstappen FTJ, Keizer HA, Geurten P, van Kranenburg G (1985). Variability of aerobic performance in the laboratory and its physiologic correlates. Int J Sports Med 6:197-201.",
     "Cole CR, Blackstone EH, Pashkow FJ, Snader CE, Lauer MS (1999). Heart-rate recovery immediately after exercise as a predictor of mortality. N Engl J Med 341:1351-1357.",
     "Daanen HAM, Lamberts RP, Kallen VL, Jin A, Van Meeteren NLU (2012). A systematic review on heart-rate recovery to monitor changes in training status in athletes. Int J Sports Physiol Perform 7:251-260."]],
  ["LSCT", "The submaximal test in each recovery week: 6 min at 60 %, 6 min at 80 % and 3 min at 90 % of max heart rate, then 90 s sitting still. Ridden in resistance mode: you hold each heart rate, as in the published test. Per stage goodpush takes the mean power and heart rate over its second half; heart-rate recovery is the heart rate over the last 5 s of the last stage less the heart rate one minute into sitting still.",
    ["Lamberts RP, Swart J, Capostagno B, Noakes TD, Lambert MI (2010). Heart rate recovery as a guide to monitor fatigue and predict changes in performance parameters. Scand J Med Sci Sports 20:449-457.",
     "Lamberts RP, Swart J, Noakes TD, Lambert MI (2011). A novel submaximal cycle test to monitor fatigue and predict cycling performance. Br J Sports Med 45:797-804."]],
  ["Beat-to-beat recording", "How goodpush gets your beats: an ANT+ chest strap sends the time of its last beat (1/1024 s) and a beat count four times a second, page 4 also the beat before. goodpush rebuilds every interval; messages lost on the radio become one interval over the beats it spans, never a guessed beat. A Bluetooth strap sends the intervals themselves (1/1024 s). Saved in the ride's FIT file as standard hrv messages. Chest straps time each beat from the heart's electrical signal; optical sensors (wrist, arm) don't time beats well enough for alpha1.",
    ["ANT Wireless (Garmin Canada). ANT+ Device Profile: Heart Rate Monitor. thisisant.com.",
     "Garmin. FIT SDK: Flexible and Interoperable Data Transfer protocol (hrv message). developer.garmin.com/fit."]],
  ["Intensity distribution", "Polarized and pyramidal plans; the three-zone model next to the five- and seven-zone ones",
    ["Seiler KS, Kjerland GØ (2006). Quantifying training intensity distribution in elite endurance athletes: is there evidence for an \"optimal\" distribution? Scand J Med Sci Sports 16:49-56.",
     "Stöggl T, Sperlich B (2014). Polarized training has greater impact on key endurance variables than threshold, high intensity, or high volume training. Front Physiol 5:33.",
     "Seiler S (2010). What is best practice for training intensity and duration distribution in endurance athletes? Int J Sports Physiol Perform 5:276-291.",
     "Seiler S, Jøranson K, Olesen BV, Hetlelid KJ (2013). Adaptations to aerobic interval training: interactive effects of exercise intensity and total work duration. Scand J Med Sci Sports 23:74-83.",
     "Rønnestad BR, Hansen J, Ellefsen S (2014). Block periodization of high-intensity aerobic intervals provides superior training effects in trained cyclists. Scand J Med Sci Sports 24:34-42."]],
  ["Periodization, ramp rate and volume", "Base / build / specialty phases, recovery weeks, how fast load builds (the usual 1.5-5 CTL a week; ~10 % more hours a week is a rule of thumb, not a finding)",
    ["Friel J (2018). The Cyclist's Training Bible, 5th ed. VeloPress.",
     "Gabbett TJ (2016). The training-injury prevention paradox: should athletes be training smarter and harder? Br J Sports Med 50:273-280."]],
  ["Tapering", "Load before A and B races",
    ["Mujika I, Padilla S (2003). Scientific bases for precompetition tapering strategies. Med Sci Sports Exerc 35:1182-1187.",
     "Aubry A, Hausswirth C, Louis J, Coutts AJ, Le Meur Y (2014). Functional overreaching: the key to peak performance during the taper? Med Sci Sports Exerc 46:1769-1777."]],
  ["Overreaching", "Weeks with load warning signs, judged by what followed",
    ["Meeusen R, Duclos M, Foster C, et al. (2013). Prevention, diagnosis, and treatment of the overtraining syndrome: joint consensus statement of the European College of Sport Science and the American College of Sports Medicine. Med Sci Sports Exerc 45:186-205.",
     "Le Meur Y, Pichon A, Schaal K, et al. (2013). Evidence of parasympathetic hyperactivity in functionally overreached athletes. Med Sci Sports Exerc 45:2061-2071."]],
  ["Detraining", "How fast fitness fades without training: the physiological model's detraining priors",
    ["Coyle EF, Martin WH, Sinacore DR, Joyner MJ, Hagberg JM, Holloszy JO (1984). Time course of loss of adaptations after stopping prolonged intense endurance training. J Appl Physiol 57:1857-1864.",
     "Coyle EF, Martin WH, Bloomfield SA, Lowry OH, Holloszy JO (1985). Effects of detraining on responses to submaximal exercise. J Appl Physiol 59:853-859.",
     "Mujika I, Padilla S (2000). Detraining: loss of training-induced physiological and performance adaptations. Part I and II. Sports Med 30:79-87, 145-154."]],
  ["Durability", "The durability finish on long rides",
    ["Maunder E, Seiler S, Mildenhall MJ, Kilding AE, Plews DJ (2021). The importance of 'durability' in the physiological profiling of endurance athletes. Sports Med 51:1619-1628."]],
  ["Stamina through a ride",
    "The ride page's stamina lines, an open take on the idea behind Garmin's Real-Time Stamina (a closed tool: "
    + "goodpush doesn't copy how it works), from your CP, W′ and LT1 that day. Potential: the power you could "
    + "hold for 5 minutes now (CP + W′ / 300 s) as a share of it fresh. Stamina: potential × the share of W′ you "
    + "have left, so 0 % once W′ is spent. W′ is spent above CP and refills below it in two phases (Chorley et "
    + "al.): about half within a minute or so (22 s time constant), the rest over several minutes (377 s, slowing "
    + "to 549 s once you've spent a second W′, as they measured on repeated efforts to exhaustion; not further). "
    + "Recovering closer to CP is slower, by Skiba's dependence on how far below CP you ride. Long hard riding "
    + "lowers CP and W′ themselves: from Clark et al., CP 10 % an hour above LT1 after the first hour, W′ 10 % an "
    + "hour from the start (their 2 h of heavy riding: CP -9-11 %, W′ about -20 %; carbohydrate during the ride "
    + "kept CP up). Riding below LT1 costs nothing here, and past about 2 h above LT1 it's an extrapolation. "
    + "goodpush's own combination of these: an estimate to test, not a validated model.",
    ["Chorley A, et al. (2021). Bi-exponential modelling of W′ reconstitution kinetics in "
     + "trained cyclists. Eur J Appl Physiol, doi 10.1007/s00421-021-04874-3.",
     "Chorley A, Bott RP, Marwood S, Lamb KL (2019). Slowing the reconstitution of W′ in recovery with repeated "
     + "bouts of maximal exercise. Int J Sports Physiol Perform 14:149-155.",
     "Skiba PF, Chidnok W, Vanhatalo A, Jones AM (2012). Modeling the expenditure and reconstitution of work "
     + "capacity above critical power. Med Sci Sports Exerc 44:1526-1532.",
     "Clark IE, Vanhatalo A, Thompson C, et al. (2019). Dynamics of the power-duration relationship during "
     + "prolonged endurance exercise and influence of carbohydrate ingestion. J Appl Physiol 127:726-736.",
     "Clark IE, Vanhatalo A, et al. (2019). Changes in the power-duration relationship following prolonged "
     + "exercise: estimation using conventional and all-out protocols and relationship to muscle glycogen. "
     + "Am J Physiol Regul Integr Comp Physiol 317:R59-R67."]],
  ["Readiness: HRV and sleep", "Morning readiness from HRV, resting HR and sleep",
    ["Plews DJ, Laursen PB, Stanley J, Kilding AE, Buchheit M (2013). Training adaptation and heart rate variability in elite endurance athletes: opening the door to effective monitoring. Sports Med 43:773-781.",
     "Fullagar HHK, Skorski S, Duffield R, Hammes D, Coutts AJ, Meyer T (2015). Sleep and athletic performance: the effects of sleep loss on exercise performance, and physiological and cognitive responses to exercise. Sports Med 45:161-186."]],
  ["Model fitting", "The physiological model's best fit and uncertainty (NUTS sampling), its checks, and the plan's optimisers",
    ["Hoffman MD, Gelman A (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. J Mach Learn Res 15:1593-1623.",
     "Phan D, Pradhan N, Jankowiak M (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv:1912.11554.",
     "Byrd RH, Lu P, Nocedal J, Zhu C (1995). A limited memory algorithm for bound constrained optimization. SIAM J Sci Comput 16:1190-1208.",
     "Bradbury J, Frostig R, Hawkins P, et al. (2018). JAX: composable transformations of Python+NumPy programs. github.com/jax-ml/jax.",
     "Lam SK, Pitrou A, Seibert S (2015). Numba: a LLVM-based Python JIT compiler. Proc. 2nd Workshop on the LLVM Compiler Infrastructure in HPC, 1-6.",
     "Gelman A, Rubin DB (1992). Inference from iterative simulation using multiple sequences. Stat Sci 7:457-472.",
     "Vehtari A, Gelman A, Simpson D, Carpenter B, Bürkner PC (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. Bayesian Anal 16:667-718.",
     "Tashman LJ (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. Int J Forecast 16:437-450.",
     "Huangfu Q, Hall JAJ (2018). Parallelizing the dual revised simplex method. Math Prog Comp 10:119-142."]],
];

class SourcesPage extends Page {
  html() {
    return `<h1>Sources</h1>
  <p class="sub">Where goodpush's methods come from. Your own results (ramp ranges, illness links, what drove your gains) are from your data alone: n = 1 and observational, so read them as hints to test, not findings.</p>
  ${SOURCES.map(([topic, use, refs]) => `<div class="card sources"><h3>${esc(topic)}</h3><p class="small muted">${esc(use)}</p>
    <ul>${refs.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`).join("")}`;
  }
}
export const sourcesPage = new SourcesPage();
