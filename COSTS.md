# What GoodPush costs

GoodPush is free and will stay free: no ads, no tracking, no accounts, no paid tier. It runs on your computer with
your own free sign-ins to Garmin, Strava or Google, so there are no servers to pay for.

A few things that would make GoodPush easier and safer to install *do* cost money. This page lists each one, what
it costs and what it would change for you. None of them are in place yet.

## What it costs today: nothing

| What | Cost | Why it's free |
|---|---|---|
| Code, issues, releases | 0 | GitHub, free for open-source projects |
| Building the installers (Mac, Windows, Linux) | 0 | GitHub Actions, free for public repositories |
| This guide | 0 | GitHub Pages |
| Your data | 0 | stays on your computer; Garmin's Personal API, Strava's export and Google Health are free to you |
| The maintainer's time | 0 | unpaid |

## What would cost money

Prices as published in November 2026, in US dollars; they change, and this table is updated when they do.

| What | What it changes for you | Cost |
|---|---|---|
| **Mac: Apple Developer Program** | The Mac app is *notarised*: no "can't be verified" warning, no trip to Privacy & Security. Also needed for any iPhone app. | $99 a year |
| **Windows: code signing** | The installer carries a verified publisher instead of "Unknown publisher", and the "Windows protected your PC" warning fades as the signature builds a reputation. | free to $300 a year (below) |
| **Android: Google Play account** | Needed to publish an Android app on Google Play. | $25 once |
| **A web address** (e.g. goodpush.app) | A short, permanent address for the guide and downloads. Nice to have. | about $15 a year |

**Windows signing, cheapest first.** [SignPath Foundation](https://signpath.org) signs open-source projects for
free. Otherwise: Microsoft's
[Artifact Signing](https://learn.microsoft.com/windows/apps/package-and-deploy/code-signing-options) (about $10 a
month, only for individuals in the USA and Canada) or a certificate from a certificate authority ($150-300 a year).

**Phone apps don't exist yet.** GoodPush already works in your phone's browser (`goodpush app --lan`). A real
Android or iPhone app is a lot of work beyond the store fees, so those costs only matter once someone builds one.
They're listed so that you can see the whole picture.

**First year, everything:** $139 with free Windows signing, up to $439 with a paid certificate. **After that:**
$114-414 a year.

Questions about any of this: [open an issue](https://github.com/GoodPush/GoodPush/issues/new?template=idea.yml).
