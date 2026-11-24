// goodpush: download your Garmin data from a signed-in Garmin Connect tab.
// 1. Open https://connect.garmin.com in your browser and sign in.
// 2. Press F12, open the Console tab, paste this whole script and press Enter.
//    (Chrome may ask you to type "allow pasting" first.)
// 3. A file goodpush_garmin_<start>_<end>.json is downloaded. Then run:
//      goodpush garmin-import
// It only reads data (sleep, resting HR, HRV, readiness, weight and your activities with
// their original .fit files); nothing in your Garmin account is changed.
(async () => {
  const START = "__START__";             // first day of sleep/HRV data to fetch (YYYY-MM-DD)
  const ACT_START = "__ACT_START__";     // first day of activities to fetch
  const END = new Date().toISOString().slice(0, 10);
  const PAUSE_MS = 400;
  const tokEl = document.querySelector('meta[name="csrf-token"]');
  if (!location.hostname.endsWith("connect.garmin.com") || !tokEl) {
    console.error("goodpush: run this on a signed-in https://connect.garmin.com page.");
    return;
  }
  const tok = tokEl.content;
  const get = async (path) => {
    const r = await fetch("/gc-api/" + path, {credentials: "include",
      headers: {"connect-csrf-token": tok, "Accept": "application/json"}});
    if (r.status === 429) throw new Error("rate-limited (429) - try again later");
    if (!r.ok) return null;
    const t = await r.text();
    try { return t ? JSON.parse(t) : null; } catch (e) { return null; }
  };
  const getFile = async (path) => {
    const r = await fetch("/gc-api/" + path, {credentials: "include",
      headers: {"connect-csrf-token": tok}});
    if (!r.ok) return null;
    const blob = await r.blob();
    return await new Promise((res) => {
      const fr = new FileReader();
      fr.onload = () => res(String(fr.result).split(",")[1] || null);
      fr.onerror = () => res(null);
      fr.readAsDataURL(blob);
    });
  };
  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

  // 1. sleep / resting HR / HRV / readiness, one day at a time
  const prof = await get("userprofile-service/socialProfile");
  const dn = prof && prof.displayName;
  const days = {};
  const d0 = new Date(START + "T12:00:00Z"), d1 = new Date(END + "T12:00:00Z");
  for (let d = d0; d <= d1; d = new Date(d.getTime() + 864e5)) {
    const ds = d.toISOString().slice(0, 10);
    const s = await get(`sleep-service/sleep/dailySleepData?date=${ds}&nonSleepBufferMinutes=60`);
    const sleepTrim = s ? {dailySleepDTO: s.dailySleepDTO, restingHeartRate: s.restingHeartRate,
      avgOvernightHrv: s.avgOvernightHrv, hrvStatus: s.hrvStatus,
      bodyBatteryChange: s.bodyBatteryChange} : null;
    const summary = dn ? await get(`usersummary-service/usersummary/daily/${dn}?calendarDate=${ds}`) : null;
    const hrv = await get(`hrv-service/hrv/${ds}`);
    const ready = await get(`metrics-service/metrics/trainingreadiness/${ds}`);
    days[ds] = {sleep: sleepTrim, summary, hrv: hrv ? {hrvSummary: hrv.hrvSummary} : null,
                readiness: ready};
    console.log("goodpush: wellness", ds);
    await sleep(PAUSE_MS);
  }
  const weight = await get(`weight-service/weight/range/${START}/${END}?includeAll=true`);

  // 2. activities: the list, then each one's original file (.fit inside a zip)
  const list = [];
  for (let start = 0; ; start += 100) {
    const page = await get(`activitylist-service/activities/search/activities?startDate=${ACT_START}&endDate=${END}&start=${start}&limit=100`);
    if (!Array.isArray(page) || !page.length) break;
    list.push(...page);
    if (page.length < 100) break;
  }
  const activities = [];
  for (const a of list) {
    const file = await getFile(`download-service/files/activity/${a.activityId}`);
    activities.push({summary: a, file_b64: file});
    console.log("goodpush: activity", a.startTimeLocal, a.activityName, file ? "" : "(no file)");
    await sleep(PAUSE_MS);
  }

  const out = {source: "goodpush-garmin-browser", version: 2, start: START, end: END,
               act_start: ACT_START, days, weight, activities};
  const blob = new Blob([JSON.stringify(out)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `goodpush_garmin_${START}_${END}.json`;
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`goodpush: done - ${Object.keys(days).length} days and ${activities.length} activities saved to ${a.download}`);
})();
