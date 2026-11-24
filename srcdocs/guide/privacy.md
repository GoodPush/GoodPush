# Privacy

GoodPush is local and per person: it runs on your computer, keeps your data in your GoodPush folder, and has no
server, account or tracking of its own. Nothing is sent anywhere unless you set it up.

**When it goes online**, and to whom:

* **Your own data downloads**, straight from Garmin, Strava or Google Health to your computer, with the sign-ins you
  gave it.
* **Map tiles**: when you open a ride, its map comes from OpenStreetMap / CyclOSM (with Leaflet), so those servers
  see which area you're looking at. Nothing is stored; offline the route shows without a map.
* **Your other computers**, if you pair them: only on your home network or Tailscale, encrypted
  ([details](computers.md#one-athlete-on-several-computers)).
* **Your phone**, if you start the app with `--lan`: only on your Wi-Fi, with a private key.
* **`reports/dashboard.html`** loads its charting library from the internet when you open it.
* **GitHub**, only when you click a link to it (Report a problem, the guide). The app fills in its version and your
  type of computer in the form; nothing is posted until you press Submit on GitHub, and you see everything first.

**Where things are kept**: your data, settings and reports in your GoodPush folder; keys and sign-ins in
`~/.goodpush/` in your user folder (never in the GoodPush folder, so the folder can be copied or backed up as it is).

If you put your GoodPush folder under version control or in a shared folder, keep `data/` and `reports/` private:
they hold your health data.
