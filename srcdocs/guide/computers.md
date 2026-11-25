# Several athletes, several computers

## Several athletes on one computer

Each athlete has their own GoodPush folder: settings, data, plan, model fits and reports. In the app, click the
GoodPush logo (top left) to switch. From the command line, register them once and pick one with `--athlete` (or
`GOODPUSH_ATHLETE`):

```
goodpush athlete add sam ~/goodpush/sam                    # a new folder (gets its own goodpush.toml)
goodpush athlete add alex ~/goodpush/alex --credentials    # an existing folder; keeps this computer's sign-ins for it
goodpush athlete                                           # list them (* = the default)
goodpush --athlete sam app --port 8766                     # one app per athlete, each on its own port
```

Garmin / Strava / Google sign-ins are kept per athlete on each computer (`athletes/<name>/` in GoodPush's own folder, see
[where things are kept](privacy.md)), never in the
folder, so a folder can be copied to another computer as it is.

## One athlete on several computers

Pair the two apps once: on the computer that's on most, **Settings > Other computers > Show a code**; on the other,
**Join** with the address and code it shows (same network, or both on Tailscale), then **Allow** it on the first one
when it asks. Either computer can then sync with the other: **Send to** (rides and data from here to there),
**Get from** (the other way) or **Both ways**; your plan, settings and other choices always merge both ways.

* The computer that showed the code is **home**: it downloads from Garmin, takes in what arrives from the others
  (new rides, your changes) and passes it on. The others also sync when you press **Sync** (top right), after each
  ride, and every 2 hours while both apps are open (Settings, per computer; 0 = only when you press Sync).
* **Your decisions** (settings, the plan's sessions, moves, days away, your workouts, deleted rides) are merged entry
  by entry, so changes made on both survive; where both changed the same entry, home wins and the sync says so.
  Settings that belong to one computer (its folders, Incyclist, where the ride bar sits) stay as they are on each.
* **Data** (downloads, ride files) is copied; changed on both, the newer wins and the other is kept in
  `.goodpush-sync/old/`. **Analysis results and model fits** are copied too, so the other computer is current
  without re-running.
* **Never copied**: sign-ins and keys, and the caches each computer builds for itself.

**Security.** The code is never sent: both sides prove they know it, tied to the home computer's certificate, which
the other then pins (and the other way round); the connection is encrypted. Each computer only answers your home
network and Tailscale, never the open internet, and only listens while a code is shown or computers are paired with
it (switch it off per computer in the same card).

From a terminal: `goodpush peer join ADDRESS CODE`, then `goodpush peer sync NAME [--send | --get] [--dry-run]`.

### Over ssh instead

Started from your main computer; copy the folder over once. The other computer needs an ssh server (on a Mac: System
Settings > General > Sharing > Remote Login), and you connect to it once with `ssh` so it is trusted:

```
goodpush peer add laptop laptop.local ~/goodpush/athletes/NAME   # once (USER@HOST if your user differs); this computer becomes home
goodpush peer sync laptop                                        # shows what would move, then asks
goodpush peer sync laptop --dry-run                              # only show
```

Close the app on the other computer while syncing over ssh.
