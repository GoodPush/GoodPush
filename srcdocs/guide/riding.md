# Riding the plan

## Through intervals.icu (watch, bike computer, Zwift, MyWhoosh)

GoodPush sends your planned workouts to [intervals.icu](https://intervals.icu), and intervals.icu passes them on to
the devices and apps you connect there. GoodPush doesn't talk to Garmin, Wahoo or Zwift itself for this.

1. Make a free intervals.icu account.
2. In intervals.icu, **Settings > Connections**: connect your Garmin Connect, Wahoo, Zwift or MyWhoosh account and
   switch on sending planned workouts to it. Garmin then puts each workout in its calendar and on your watch or
   Edge after the next sync.
3. In intervals.icu, **Settings > Developer Settings**: generate an API key and paste it into GoodPush under
   **Settings > API keys**.
4. In GoodPush, **Settings > intervals.icu > Send plan now**. It sends today and the next 13 days. Send again after
   the plan changes: GoodPush's workouts there are updated, dropped ones removed, your own left alone.

* Targets are % of FTP: keep your FTP in intervals.icu the same as in GoodPush.
* MyWhoosh takes the next 7 days; ramps ride as a steady average, without cadence targets.
* intervals.icu is free and paid for by the riders who support it. If it gets your plan to your watch, please
  consider supporting it.

## On a Garmin watch or bike computer, without intervals.icu

Each session is written as a structured workout (`reports/plan/workouts/gp_*.fit`). Plug the watch in by USB and
copy the files into `GARMIN/NewFiles`; after unplugging they appear under *Training > Workouts* (cycling). Pair a
power source first (power pedals outdoors; a smart trainer as a power sensor or ANT+ FE-C trainer). Or import them in
Garmin Connect (web: Training & Planning > Workouts > Import) and send them to the watch from there.

## In Incyclist

GoodPush can write your sessions straight into Incyclist's own workout calendar on this computer (no account,
nothing online): **Settings > Incyclist**, tick *Write my plan into Incyclist* and Save (or `goodpush incyclist on`).
After every plan update the next two weeks go into Incyclist's `workouts/calendar.json`; Incyclist shows them under
*Workouts > Scheduled Workouts*.

* Incyclist reads that file when it starts: restart it after a plan change.
* Targets are % of FTP: set your FTP in Incyclist too.
* Only GoodPush's entries are replaced. If you connect an online calendar inside Incyclist, its sync replaces the list.
* Incyclist installed somewhere else? Set its data folder in Settings > Incyclist (`goodpush incyclist --folder PATH`).

Every session is also a Zwift workout (`workouts/gp_*.zwo`) to import by hand into Zwift or Incyclist.

## GoodPush's trainer bar

**▶ Ride it now** on Today (or `goodpush ride`): a small see-through bar that stays on top of every window, so you can
watch something while you ride. It shows power (3 s) against the target, heart rate, cadence, the step with its
countdown, the next step and the session's shape, and steers the trainer in ERG over ANT+ FE-C. All-out steps
(sprints, the 20-min test) and free rides use resistance mode instead. The workout clock only runs while you pedal.

* **You need** an ANT+ USB stick ([its driver on Windows](../start/install.md#ant-usb-stick-for-riding-on-a-trainer)).
  From source: the `[ride]` extra.
* **The first time**, a window lists the trainers, power meters, cadence sensors and heart-rate straps it hears; pick
  yours (remembered per computer; `goodpush ride --pick` to change them, `goodpush ride scan` to list what's around).
* **While riding**: drag the bar where you like. Right-click it (or use the tray icon) to pause, skip a step, go harder
  or easier, turn ERG off, change its look, switch devices, or finish. Mouse wheel = 1 % harder / easier,
  ↑ / ↓ = 5 %, space = pause, → = skip the step.
* **Finishing** saves a FIT file (a lap per step) in `data/raw/goodpush/`. It's taken in like any other ride, linked
  to that day's session, and the plan is rebuilt. If your watch recorded the same ride, the watch's copy is dropped.
  A crash loses at most a minute: the next ride saves what was left.
* `goodpush ride --free` = no workout, `--session 2026-11-03` = another day's session, `--sim` = a pretend trainer to
  try it out (saved to a temporary folder only).
* Keep the stick close to the trainer (a USB extension cable): with a weak signal readings drop out and ERG changes
  arrive late.
* macOS: the bar shows on every desktop and over full-screen apps; while you ride, GoodPush has no Dock icon (use the
  menu-bar icon).

## Tests

* **FTP test** {cite:p}`allen2019`: a 20-minute test every `test_every_weeks` (default 6). Record one from elsewhere on Plan > Record an
  FTP test.
* **Ramp test** with alphaHRV, every `ramp_every_weeks` (default 6): 2-minute steps of about 6 % of FTP {cite:p}`kuipers1985`, for LT1 / LT2 {cite:p}`rogers2021a,rogers2021b`.
  GoodPush recognises a ramp test in your rides by itself.
