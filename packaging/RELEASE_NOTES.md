## goodpush 0.1.0: the first release

Analyse your cycling training and plan the next weeks, on your own computer, from your Strava / Garmin exports
and the Garmin Personal API.

- **Fitness:** training load, FTP, thresholds (LT1 / LT2 from alpha1), power curve, a physiological model of how
  your fitness changes with training.
- **Plan:** an adaptive plan built around your races, the hours you have and how you're recovering; interval
  progression; sessions in your own watts; Incyclist and .zwo / .fit workouts.
- **Calendar:** drag sessions to any day, copy / paste them, drop a ride on a session to link it (or unlink it).
- **Rides:** power, heart rate and alpha1 in riding time (stops left out), intervals in zone colours, the planned
  session's targets lined up with what you rode.
- **Recovery:** sleep, HRV and resting heart rate; readiness that flags short nights and late bedtimes.
- **Ride on the trainer** (ANT+): ERG workouts with a small on-top overlay, recorded as .fit files.

## Download

| Your computer | File |
|---|---|
| Mac with Apple Silicon (M1 or later) | `goodpush-mac-apple-silicon.dmg` |
| Mac with an Intel processor | `goodpush-mac-intel.dmg` |
| Windows 10 / 11 | `goodpush-windows-setup.exe` |
| Linux (64-bit) | `goodpush-linux.tar.gz` |

Not sure which Mac you have? Apple menu > About This Mac: "Chip: Apple M…" is Apple Silicon, "Processor: Intel" is Intel.

## Why your computer warns you, and how to open it anyway

goodpush is free and isn't **code-signed**. Signing means paying Apple and Microsoft every year to vouch for the
developer; without it, your computer can't check who made the app, so it warns you the first time. The warning is
about the missing signature, not about anything goodpush does. It runs only on your computer and your data never
leaves it.

- **Mac:** open the .dmg and drag goodpush to Applications. Open it: macOS says it can't be verified, click **Done**.
  Then go to **System Settings > Privacy & Security**, scroll down to "goodpush was blocked" and click **Open Anyway**.
  Only the first time.
- **Windows:** run the setup. If "Windows protected your PC" appears, click **More info**, then **Run anyway**.
- **Linux:** unpack it (`tar -xzf goodpush-linux.tar.gz`), then `sh goodpush/install.sh`.

goodpush opens in your web browser, with an icon in the menu bar (Mac) or system tray (Windows) to open it again or quit.

## Licences

goodpush is free software under the GNU AGPL v3.0 or later, with the attribution terms in `NOTICE`. Each download includes `THIRD_PARTY_NOTICES.txt` for the libraries it bundles. The Windows
installer includes Zadig (GPLv3, unmodified) for the ANT+ USB driver; its source for this exact version is the
`libwdi-…-source.tar.gz` file below.
