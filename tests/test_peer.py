"""goodpush peer: syncing an athlete folder with another machine (both sides local folders here; ssh is the
same calls over SFTP)."""
import json
import os
import socket
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from goodpush.sync.peer import STATE, Devices, FolderSide, HttpSide, Pairing, PeerCert, Peers, PeerServer, PeerSync

TOML = """[athlete]
name = "A"   # who

[paths]
downloads = "{dl}"   # this machine's

[plan]
hours_per_week = 8.0

[ftp.history]
"2026-10-22" = 250
"""


def put(root, rel, data, mtime):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    os.utime(p, (mtime, mtime))


def sync(here, there, home_here=True):
    s = PeerSync(FolderSide(here), FolderSide(there), "mac", home_here)
    out = s.run(s.plan(), progress=False)
    return s, out


def test_first_sync_copies_both_ways_and_never_caches_or_credentials(tmp_path):
    here, there = tmp_path / "pc", tmp_path / "mac"
    put(here, "data/raw/garmin/a.fit", "A", 1_000)
    put(here, "data/raw/garmin/same.fit", "S", 1_000)
    put(there, "data/raw/garmin/same.fit", "S", 1_000)
    put(there, "data/raw/strava/b.fit", "B", 1_000)
    for rel in ("key.txt", "data/processed/jax_cache/x.bin", "data/processed/physio_nuts.ckpt", "data/processed/r.pkl"):
        put(here, rel, "private", 1_000)
    s = PeerSync(FolderSide(here), FolderSide(there), "mac")
    steps = s.plan()
    assert sorted((x.action, x.rel) for x in steps) == [("get", "data/raw/strava/b.fit"), ("put", "data/raw/garmin/a.fit")]
    s.run(steps, progress=False)
    assert (there / "data/raw/garmin/a.fit").read_text() == "A" and (here / "data/raw/strava/b.fit").read_text() == "B"
    assert os.stat(there / "data/raw/garmin/a.fit").st_mtime == 1_000                  # the time travels with it
    assert not (there / "key.txt").exists() and not (there / "data/processed/jax_cache").exists()
    assert not list(there.rglob("*.pkl")) and not list(there.rglob("physio_nuts*"))
    assert PeerSync(FolderSide(here), FolderSide(there), "mac").plan() == []          # in step now
    assert (here / STATE / "mac.json").exists() and not (there / STATE).exists()


def test_both_machines_changes_survive_and_nothing_is_lost(tmp_path):
    here, there = tmp_path / "pc", tmp_path / "mac"
    state = {"2026-11-04": {"type": "easy"}, "2026-11-05": {"type": "vo2max"}}
    for root, dl in ((here, "dl-pc"), (there, "dl-mac")):
        put(root, "goodpush.toml", TOML.format(dl=dl), 1_000)
        put(root, "data/processed/plan_state.json", json.dumps(state), 1_000)
        put(root, "data/deleted_activities.json", json.dumps(["1"]), 1_000)
        put(root, "data/raw/goodpush/activities_goodpush.csv", "id,name\n1,Ride 1\n", 1_000)
        put(root, "data/raw/garmin/day.json", "old", 1_000)
        put(root, "data/raw/garmin/gone.fit", "x", 1_000)
        put(root, "data/processed/daily.parquet", "old", 1_000)
    sync(here, there)
    # this machine (home): a session moved, an hour more, a ride recorded; the other: a day away, an FTP entry,
    # a ride recorded, a file deleted
    put(here, "data/processed/plan_state.json", json.dumps({**state, "2026-11-04": {"type": "rest"},
                                                            "2026-11-05": {"type": "threshold"}}), 2_000)
    put(there, "data/processed/plan_state.json", json.dumps({**state, "away": {"2026-11-10": "trip"},
                                                             "2026-11-05": {"type": "tempo"}}), 2_000)
    put(here, "goodpush.toml", TOML.format(dl="dl-pc").replace("8.0", "10.0"), 2_000)
    put(there, "goodpush.toml", TOML.format(dl="dl-mac") + '"2026-11-03" = 240\n', 2_000)
    put(here, "data/deleted_activities.json", json.dumps(["1", "7"]), 2_000)
    put(there, "data/deleted_activities.json", json.dumps(["1", "9"]), 2_000)
    put(here, "data/raw/goodpush/activities_goodpush.csv", "id,name\n1,Ride 1\n2,Ride 2\n", 2_000)
    put(there, "data/raw/goodpush/activities_goodpush.csv", "id,name\n1,Ride 1\n3,Ride 3\n", 2_000)
    put(here, "data/raw/garmin/day.json", "pc", 2_000)
    put(there, "data/raw/garmin/day.json", "mac, later", 3_000)
    put(here, "data/processed/daily.parquet", "pc", 2_000)
    put(there, "data/processed/daily.parquet", "mac, later", 3_000)
    (there / "data/raw/garmin/gone.fit").unlink()
    s, out = sync(here, there)
    for root in (here, there):
        st = json.loads((root / "data/processed/plan_state.json").read_text())
        assert st["2026-11-04"] == {"type": "rest"} and st["away"] == {"2026-11-10": "trip"}
        assert st["2026-11-05"] == {"type": "threshold"}                       # both changed it: home's
        cfg = tomllib.loads((root / "goodpush.toml").read_text())
        assert cfg["plan"]["hours_per_week"] == 10.0 and cfg["ftp"]["history"] == {"2026-10-22": 250, "2026-11-03": 240}
        assert json.loads((root / "data/deleted_activities.json").read_text()) == ["1", "7", "9"]
        assert (root / "data/raw/goodpush/activities_goodpush.csv").read_text().split() == ["id,name", "1,Ride", "1",
                                                                                          "2,Ride", "2", "3,Ride", "3"]
        assert (root / "data/raw/garmin/day.json").read_text() == "mac, later"       # data: the newer
        assert (root / "data/processed/daily.parquet").read_text() == "pc"           # derived: home's
        assert not (root / "data/raw/garmin/gone.fit").exists()
    assert "# who" in (here / "goodpush.toml").read_text()                            # comments kept
    assert tomllib.loads((here / "goodpush.toml").read_text())["paths"]["downloads"] == "dl-pc"
    assert tomllib.loads((there / "goodpush.toml").read_text())["paths"]["downloads"] == "dl-mac"
    old = here / STATE / "old"                                                        # the losing data and the
    assert [p.read_text() for p in old.rglob("day.json")] == ["pc"]                   # deletion kept, not lost
    assert [p.read_text() for p in old.rglob("gone.fit")] == ["x"]
    assert any("2026-11-05" in c for c in out["clashes"])
    assert PeerSync(FolderSide(here), FolderSide(there), "mac").plan() == []


def test_first_sync_takes_the_home_machines_plan_whole(tmp_path):
    here, there = tmp_path / "pc", tmp_path / "mac"
    put(here, "data/processed/plan_state.json", json.dumps({"planned": {"2099-11-04": {"type": "rest"}}}), 2_000)
    put(there, "data/processed/plan_state.json", json.dumps({"planned": {"2099-11-04": {"type": "easy"},
                                                                         "2099-11-05": {"type": "stale"},
                                                                         "2020-10-01": {"type": "ridden"}}}), 1_000)
    sync(here, there)
    for root in (here, there):     # an old copy's plans aren't revived (no last sync to tell new from deleted),
        st = json.loads((root / "data/processed/plan_state.json").read_text())        # but past days never go
        assert st == {"planned": {"2099-11-04": {"type": "rest"}, "2020-10-01": {"type": "ridden"}}}


def test_a_copy_started_afresh_never_wipes_your_history(tmp_path):
    """2026-11-03: the laptop's plan state was a new one; the merge took everything it lacked as deleted and wiped
    the history, days away and levels on both computers."""
    here, there = tmp_path / "mac", tmp_path / "pc"
    full = {"start": "2020-10-19", "away": {"2099-11-12": "work trip"}, "levels": {"threshold": 2},
            "planned": {"2020-10-27": {"type": "endurance", "status": "done", "ride": "g1"},
                        "2099-11-05": {"type": "ramp"}}}
    for root in (here, there):
        put(root, "data/processed/plan_state.json", json.dumps(full), 1_000)
    sync(here, there, home_here=False)                                          # in step: that's the base
    fresh = {"levels": {"threshold": 0}, "planned": {"2099-11-05": {"type": "vo2max"}}}
    put(here, "data/processed/plan_state.json", json.dumps(fresh), 2_000)       # the laptop starts afresh
    s, out = sync(here, there, home_here=False)
    for root in (here, there):
        st = json.loads((root / "data/processed/plan_state.json").read_text())
        assert st["away"] == full["away"] and st["start"] == full["start"]                # sections stay
        assert st["planned"]["2020-10-27"] == full["planned"]["2020-10-27"]               # history stays
    assert any("lacked" in c for c in out["clashes"])
    kept = list((there / STATE).rglob("plan_state.json")) + list((here / STATE / "old").rglob("plan_state.json"))
    assert any(json.loads(p.read_text()) == full for p in kept)                 # the copies it replaced are kept


def test_only_the_home_machine_syncs_garmin_on_its_own(tmp_path):
    from goodpush.app.server import _auto_hours
    me = socket.gethostname()
    assert PeerSync.home_here_for({}) and PeerSync.home_here_for({"sync": {"home": me}})
    assert not PeerSync.home_here_for({}, joined=True)          # a new machine never overwrites the one it joined
    assert not PeerSync.home_here_for({"sync": {"home": "another-machine"}})
    cfg = {"garmin": {"auto_sync_hours": 8}}
    assert _auto_hours(SimpleNamespace(cfg=cfg)) == 8
    assert _auto_hours(SimpleNamespace(cfg={**cfg, "sync": {"home": "another-machine"}})) == 0
    p = Peers(tmp_path)
    p.add("mac", host="mac-host", folder="~/goodpush/athletes/rider")
    p.add("dl-pc.local", kind="app", address="10.0.0.2:8865", fingerprint="ab", token="t")    # a dotted name
    assert Peers(tmp_path).get("mac") == {"host": "mac-host", "folder": "~/goodpush/athletes/rider"}
    assert Peers(tmp_path).get("dl-pc.local")["address"] == "10.0.0.2:8865"


def serve(root, secrets, ask=False, **kw):
    return PeerServer(root, 0, secrets, host="127.0.0.1", ask=ask, **kw).start()


def answer_when_asked(pairing, allow):
    """you, on the home computer: allow / deny the machine once it asks (the name it had, in the future)"""
    def answer():
        for _ in range(500):
            if pairing.waiting():
                return pairing.answer(allow)
            time.sleep(0.01)
    return ThreadPoolExecutor(1).submit(answer)


def test_a_computer_that_knows_the_code_still_needs_your_ok(tmp_path, monkeypatch):
    srv = serve(tmp_path / "pc", tmp_path / "s_pc", ask=True)
    addr = f"127.0.0.1:{srv.port}"
    try:
        for allow in (False, True):
            code = srv.pairing.new()
            answer = answer_when_asked(srv.pairing, allow)
            if not allow:
                with pytest.raises(ValueError, match="Not allowed"):
                    HttpSide.pair(addr, code, "stranger")
            else:
                assert HttpSide.pair(addr, code, "mac")["kind"] == "app"
            assert answer.result() == ("mac" if allow else "stranger")
        assert list(Devices(tmp_path / "s_pc").peers) == ["mac"]       # the denied one isn't paired
        monkeypatch.setattr(Pairing, "WAIT", 0.2)
        code = srv.pairing.new()
        with pytest.raises(ValueError, match="Not allowed"):            # nobody answered
            HttpSide.pair(addr, code, "late")
        assert srv.pairing.waiting() is None and list(Devices(tmp_path / "s_pc").peers) == ["mac"]
    finally:
        srv.stop()


def test_pairing_needs_the_code_and_pins_the_certificate(tmp_path):
    home = tmp_path / "pc"
    put(home, "data/raw/garmin/a.fit", "A", 1_000)
    put(home, "key.txt", "private", 1_000)
    put(home, "data/processed/jax_cache/x.bin", "cache", 1_000)
    srv = serve(home, tmp_path / "s_pc")
    addr = f"127.0.0.1:{srv.port}"
    try:
        with pytest.raises(ValueError):                                  # no code shown
            HttpSide.pair(addr, "ABCDE-FGHJK", "mac")
        code = srv.pairing.new()
        with pytest.raises(ValueError):
            HttpSide.pair(addr, "ABCDE-FGHJK", "mac")
        entry = HttpSide.pair(addr, code.lower().replace("-", " "), "mac")      # typed loosely
        assert entry["kind"] == "app" and entry["fingerprint"] == PeerCert(tmp_path / "s_pc").fingerprint()
        assert Devices(tmp_path / "s_pc").check(entry["token"]) == "mac"
        with pytest.raises(ValueError):                                  # used up
            HttpSide.pair(addr, code, "mac")
        srv.pairing.new()
        for _ in range(Pairing.TRIES):                                   # guessing voids it
            with pytest.raises(ValueError):
                HttpSide.pair(addr, "22222-22222", "mac")
        assert srv.pairing.active() is None
        side = HttpSide(addr, entry["fingerprint"], entry["token"], "pc")
        assert side.read("data/raw/garmin/a.fit") == b"A" and "key.txt" not in side.files()
        for rel in ("key.txt", "KEY.TXT", "../outside.txt", "data/processed/jax_cache/x.bin", ".goodpush-sync/mac.json"):
            with pytest.raises(PermissionError):
                side.read(rel)
        side.close()
        with pytest.raises(PermissionError):                             # not paired
            HttpSide(addr, entry["fingerprint"], "made-up", "pc").files()
        with pytest.raises(ConnectionError, match="paired with"):        # another machine on that address
            HttpSide(addr, "00" * 32, entry["token"], "pc").files()
    finally:
        srv.stop()


def test_paired_apps_sync_and_the_home_takes_in_new_rides(tmp_path):
    home, mac = tmp_path / "pc", tmp_path / "mac"
    big = os.urandom(3_000_000)
    put(home, "data/raw/garmin/day.json", "sleep", 1_000)
    put(home, "data/processed/plan_state.json", json.dumps({"2026-11-04": {"type": "easy"}}), 1_000)
    put(mac, "data/raw/goodpush/ride.fit", big, 2_000)
    put(mac, "data/raw/goodpush/activities_goodpush.csv", "id,name\n5,Trainer\n", 2_000)
    put(home, "goodpush.toml", TOML.format(dl="dl-pc").replace("8.0", "12.0"), 1_000)       # the rider's settings
    put(mac, "goodpush.toml", TOML.format(dl="dl-mac"), 2_000)                                # a new machine's defaults
    took, busy = [], iter([True, False])

    def taken_in(name, kinds, wait):  # the home machine analyses what it got: here, a new plan
        took.append((name, sorted(kinds)))
        put(home, "reports/plan/plan.json", json.dumps({"rides": 1}), 3_000)
        return True
    srv = serve(home, tmp_path / "s_pc", on_done=taken_in, busy=lambda: next(busy, False))
    try:
        srv.pairing.new()
        entry = HttpSide.pair(f"127.0.0.1:{srv.port}", srv.pairing.active(), "mac")
        there = HttpSide(entry["address"], entry["fingerprint"], entry["token"], entry["name"])
        s = PeerSync(FolderSide(mac), there, "pc", home_here=PeerSync.home_here_for({}, joined=True))
        s.run(s.plan(), progress=False)
        for root in (home, mac):                     # no home set: the machine it joined is home, its settings win
            assert tomllib.loads((root / "goodpush.toml").read_text())["plan"]["hours_per_week"] == 12.0
        assert (home / "data/raw/goodpush/ride.fit").read_bytes() == big                # streamed up
        assert (mac / "data/raw/garmin/day.json").read_text() == "sleep"
        assert os.stat(home / "data/raw/goodpush/ride.fit").st_mtime == 2_000
        again = s.hand_over(log=lambda *a: None, poll=0)
        assert took == [("mac", ["data", "rows"])] and again is not None              # its settings didn't change
        assert json.loads((mac / "reports/plan/plan.json").read_text()) == {"rides": 1}   # its result came back
        there.close()
        assert PeerSync(FolderSide(mac), HttpSide(entry["address"], entry["fingerprint"], entry["token"]),
                        "pc", home_here=False).plan() == []
    finally:
        srv.stop()


class QuietJobs:
    """Jobs that only note what would start"""

    def __init__(self):
        self.started = []

    def start(self, name, fn, queue=False):
        self.started.append(name)
        return True

    def status(self):
        return {"running": False, "queued": None}


def test_the_app_pairs_joins_and_sets_the_home_machine(tmp_path):
    from goodpush.app.server import Data, PeerHub
    from goodpush.config import init_project, load_project
    pc, mac = init_project(tmp_path / "pc"), init_project(tmp_path / "mac")
    hub_pc = PeerHub(Data(pc), QuietJobs(), 0, tmp_path / "s_pc")
    hub_pc.server.host, hub_pc.server.port = "127.0.0.1", 0
    try:
        st = hub_pc.pair()
        assert st["code"] and st["home"] == socket.gethostname() and st["home_here"]
        port = hub_pc.server.port
        hub_mac = PeerHub(Data(mac), QuietJobs(), 0, tmp_path / "s_mac")
        with pytest.raises(ValueError):
            hub_mac.join(f"127.0.0.1:{port}", "")
        with pytest.raises(ValueError, match="Both computers are called"):    # both would take themselves for home
            hub_mac.join(f"127.0.0.1:{port}", st["code"])
        hub_mac.me = "dl-mac"                     # (both run on this test machine)
        answer = answer_when_asked(hub_pc.server.pairing, True)
        name = hub_mac.join(f"127.0.0.1:{port}", st["code"])
        assert answer.result() == "dl-mac"                               # you allowed it on the home computer
        assert name == socket.gethostname() and hub_mac.jobs.started == [f"Syncing with {name}"]
        assert load_project(mac.root).cfg["sync"]["home"] == name          # the machine it joined is home
        assert hub_mac.status()["joined"][0]["kind"] == "app"
        assert [d["name"] for d in hub_pc.status()["devices"]] == ["dl-mac"]
        hub_mac.forget(name)
        assert hub_mac.status()["joined"] == []
    finally:
        hub_pc.server.stop()


def test_a_paired_computer_never_sets_where_this_one_reads_or_writes(tmp_path):
    home = tmp_path / "pc"
    mine = TOML.format(dl="dl-pc") + '\n[incyclist]\nsync = true\nfolder = "inc-pc"\n\n[sync]\nhome = "pc"\nserve = true\n'
    put(home, "goodpush.toml", mine, 1_000)
    srv = serve(home, tmp_path / "s_pc")
    try:
        srv.pairing.new()
        e = HttpSide.pair(f"127.0.0.1:{srv.port}", srv.pairing.active(), "mac")
        side = HttpSide(e["address"], e["fingerprint"], e["token"], e["name"])
        theirs = (TOML.format(dl="elsewhere").replace("8.0", "10.0")
                  + '\n[incyclist]\nsync = true\nfolder = "elsewhere"\n\n[sync]\nhome = "pc"\nserve = false\n')
        side.write("goodpush.toml", theirs.encode(), 2_000)        # sent whole, not merged
        side.close()
    finally:
        srv.stop()
    cfg = tomllib.loads((home / "goodpush.toml").read_text())
    assert cfg["plan"]["hours_per_week"] == 10.0                  # the shared settings arrive
    assert cfg["paths"]["downloads"] == "dl-pc" and cfg["incyclist"]["folder"] == "inc-pc" and cfg["sync"]["serve"]
    assert "# who" in (home / "goodpush.toml").read_text()


def test_only_from_your_own_network():
    for a in ("192.168.1.5", "10.0.0.2", "172.16.4.1", "100.101.2.3", "127.0.0.1", "::1", "fe80::1%12"):
        assert PeerServer.nearby(a), a                            # home networks, Tailscale, this computer
    for a in ("8.8.8.8", "203.0.114.7", "2606:4700::1111", "nonsense"):
        assert not PeerServer.nearby(a), a


def test_the_app_listens_only_while_needed(tmp_path):
    from goodpush.app.server import Data, PeerHub
    from goodpush.config import init_project
    hub = PeerHub(Data(init_project(tmp_path / "pc")), QuietJobs(), 0, tmp_path / "s_pc")
    hub.server.host, hub.server.port = "127.0.0.1", 0
    assert not hub.server.httpd                                   # nothing paired: not listening
    try:
        hub.pair()
        assert hub.server.httpd and hub.status()["serve"]
        hub.server.pairing.code = None                            # the code ran out, nobody paired
        hub.listen_if_needed()
        assert not hub.server.httpd
        hub.server.devices_now().pair("mac")
        hub.listen_if_needed()
        assert hub.server.httpd
        hub.set_serve(False)                                      # switched off on this computer
        assert not hub.server.httpd and not hub.status()["serve"]
        hub.set_serve(True)
        hub.forget("mac")                                         # the last one removed
        assert not hub.server.httpd
    finally:
        hub.server.stop()


def test_syncs_when_you_say_after_a_ride_and_every_few_hours(tmp_path, monkeypatch):
    from goodpush.app.server import Data, PeerHub
    from goodpush.config import init_project
    hub = PeerHub(Data(init_project(tmp_path / "mac")), QuietJobs(), 0, tmp_path / "s_mac")
    with pytest.raises(ValueError):                                  # joined nothing yet
        hub.sync_all_job()
    hub.peers().add("pc", kind="app", address="10.0.0.2:8865", fingerprint="ab", token="t")
    monkeypatch.setattr(PeerHub, "reachable", staticmethod(lambda address: True))
    started = hub.jobs.started
    assert hub.every_hours() == 2.0
    hub.auto()
    assert started == []                                             # not due yet
    hub.after_ride()
    hub.auto()
    assert started == ["Syncing with pc"]
    hub.last_auto -= 2 * 3600
    hub.auto()
    assert len(started) == 2
    hub.set_every(0)                                                 # only when you press Sync
    hub.last_auto -= 100 * 3600
    hub.auto()
    assert len(started) == 2 and hub.status()["every_hours"] == 0
    hub.after_ride()                                                 # ... and after a ride
    hub.auto()
    assert len(started) == 3
    hub.sync_all_job()
    assert started[-1] == "Syncing"
    with pytest.raises(ValueError):
        hub.set_every(-1)


def test_renaming_a_computer_only_changes_what_you_see(tmp_path, monkeypatch):
    from goodpush.app.server import Data, PeerHub
    from goodpush.config import init_project
    hub = PeerHub(Data(init_project(tmp_path / "mac")), QuietJobs(), 0, tmp_path / "s_mac")
    hub.peers().add("DESKTOP-X1", kind="app", address="10.0.0.2:8865", fingerprint="ab", token="t")
    hub.server.devices_now().pair("Rider-MBP")
    hub.set_home("DESKTOP-X1")
    with pytest.raises(ValueError):
        hub.rename("nobody", "x")
    hub.rename("DESKTOP-X1", "  Desktop   upstairs ")
    hub.rename("Rider-MBP", "Laptop")
    st = hub.status()
    assert [(p["name"], p["label"]) for p in st["joined"]] == [("DESKTOP-X1", "Desktop upstairs")]
    assert [(d["name"], d["label"]) for d in st["devices"]] == [("Rider-MBP", "Laptop")]
    assert st["home"] == "DESKTOP-X1" and st["home_label"] == "Desktop upstairs"    # home still by its own name
    hub.sync_job("DESKTOP-X1")
    assert hub.jobs.started == ["Syncing with Desktop upstairs"]
    assert hub.peers().side("DESKTOP-X1").label == "Desktop upstairs"
    assert hub.peers().get("DESKTOP-X1")["token"] == "t"                          # still the same pairing
    hub.rename("Rider-MBP", "")                                                   # its own name again
    assert hub.status()["devices"][0]["label"] == "Rider-MBP"


def test_send_and_get_move_files_one_way_and_keep_the_rest_for_later(tmp_path):
    here, there = tmp_path / "mac", tmp_path / "pc"
    put(here, "data/raw/goodpush/ride.fit", "ride", 1_000)                  # trained here
    put(there, "data/raw/garmin/day.json", "sleep", 1_000)                 # downloaded there
    put(here, "data/plans.json", json.dumps({"a": 1}), 1_000)
    put(there, "data/plans.json", json.dumps({"b": 2}), 1_000)
    s = PeerSync(FolderSide(here), FolderSide(there), "pc", home_here=False)
    s.run(s.plan("send"), progress=False)
    assert (there / "data/raw/goodpush/ride.fit").exists() and not (here / "data/raw/garmin/day.json").exists()
    for root in (here, there):                                             # your decisions merge both ways
        assert json.loads((root / "data/plans.json").read_text()) == {"b": 2, "a": 1}   # (a section never goes)
    s = PeerSync(FolderSide(here), FolderSide(there), "pc", home_here=False)
    assert [(x.action, x.rel) for x in s.plan("send")] == []
    assert [(x.action, x.rel) for x in s.plan("get")] == [("get", "data/raw/garmin/day.json")]  # still to do
    s.run(s.plan("get"), progress=False)
    assert (here / "data/raw/garmin/day.json").read_text() == "sleep"
    assert PeerSync(FolderSide(here), FolderSide(there), "pc", home_here=False).plan() == []


def test_paired_both_ways_and_home_passes_on_what_it_gets(tmp_path):
    from goodpush.app.server import Data, PeerHub
    from goodpush.config import init_project
    pc, mac = init_project(tmp_path / "pc"), init_project(tmp_path / "mac")
    hub_pc = PeerHub(Data(pc), QuietJobs(), 0, tmp_path / "s_pc")
    hub_mac = PeerHub(Data(mac), QuietJobs(), 0, tmp_path / "s_mac")
    hub_mac.me = "dl-mac"                                # (both run on this test machine)
    for h in (hub_pc, hub_mac):
        h.server.host, h.server.port = "127.0.0.1", 0
    hub_mac.server.start()
    try:
        st = hub_pc.pair()
        answer_when_asked(hub_pc.server.pairing, True)
        name = hub_mac.join(f"127.0.0.1:{hub_pc.server.port}", st["code"])
        assert hub_mac.server.httpd                                          # it listens for the home one now
        back = hub_pc.peers().get("dl-mac")
        assert back["address"] == f"127.0.0.1:{hub_mac.server.port}" and back["kind"] == "app"
        assert [(c["name"], c["can_start"], c["lets_in"]) for c in hub_pc.status()["computers"]] == [("dl-mac", True, True)]
        put(pc.root, "data/raw/garmin/day.json", "sleep", 1_000)            # Garmin, downloaded at home
        hub_pc.updated(True, lambda *a: None)                                # ... passed on
        assert (mac.root / "data/raw/garmin/day.json").read_text() == "sleep"
        put(mac.root, "data/raw/goodpush/ride.fit", "ride", 2_000)           # a ride on the laptop, sent home
        hub_mac.sync(name, lambda *a: None, "send")
        assert (pc.root / "data/raw/goodpush/ride.fit").read_text() == "ride"
        assert hub_pc.jobs.started == ["Adding what came from dl-mac"]       # home takes it in, then passes it on
        hub_pc.jobs.started.clear()
        hub_mac.jobs.started.clear()                                         # (the first sync joining queued)
        assert not hub_mac.taken_in(name, {"data"}, False) and hub_mac.jobs.started == []   # not home: nothing to do
    finally:
        hub_pc.server.stop()
        hub_mac.server.stop()


def test_a_computer_that_moved_is_found_again(tmp_path):
    srv = PeerServer(tmp_path / "pc", 0, tmp_path / "s_pc")
    peers = Peers(tmp_path / "s_pc")
    peers.add("mac", kind="app", address="192.168.1.20:8865", fingerprint="ab", token="t")
    peers.add("named", kind="app", address="laptop.local:8865", fingerprint="ab", token="t")
    srv.seen("mac", "192.168.1.31")
    srv.seen("named", "192.168.1.31")                     # an address you typed as a name stays
    assert Peers(tmp_path / "s_pc").get("mac")["address"] == "192.168.1.31:8865"
    assert Peers(tmp_path / "s_pc").get("named")["address"] == "laptop.local:8865"


def test_the_page_shows_who_waits_and_knows_when_the_app_restarted(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from goodpush.app.server import Data, PeerHub, make_handler
    from goodpush.config import init_project
    proj = init_project(tmp_path / "pc")
    hub = PeerHub(Data(proj), QuietJobs(), 0, tmp_path / "s_pc")
    assert hub.status()["waiting"] is None
    hub.server.pairing.pending = {"name": "mac.local", "event": threading.Event(), "allow": False}
    assert hub.status()["waiting"] == "mac.local"                  # Settings shows it next to the list
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proj, None, False, warm=False, port=0))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/api/job"
        runs = [json.loads(urllib.request.urlopen(url).read())["started"] for _ in range(2)]
        assert runs[0] and runs[0] == runs[1]                       # the same run: the page stays
    finally:
        srv.shutdown()
        srv.server_close()


def test_merged_files_read_the_same_in_any_encoding(tmp_path):
    """2026-11-03: the merge wrote "×" as raw UTF-8; the app (Windows, cp1252 by default) read it as "Ã—" and the
    next syncs garbled it further. Merged files are escaped like goodpush writes them; the plan state reads UTF-8."""
    from goodpush.planning import PlanState
    here, there = tmp_path / "mac", tmp_path / "pc"
    title = {"planned": {"2099-10-24": {"title": "Over-unders 2x6 × 1 min"}}}
    for root, extra in ((here, {"2099-10-25": {"title": "Rest"}}), (there, {"2099-10-26": {"title": "Easy"}})):
        put(root, "data/processed/plan_state.json", json.dumps({"planned": {**title["planned"], **extra}}), 1_000)
    sync(here, there)
    for root in (here, there):
        raw = (root / "data/processed/plan_state.json").read_bytes()
        assert raw.isascii() and json.loads(raw.decode("cp1252"))["planned"]["2099-10-24"]["title"].endswith("× 1 min")
    p = tmp_path / "plan_state.json"
    p.write_bytes(json.dumps(title, ensure_ascii=False).encode("utf-8"))      # raw UTF-8 from an older sync
    assert PlanState.load(p)["planned"]["2099-10-24"]["title"] == "Over-unders 2x6 × 1 min"


def test_garmin_downloaded_on_both_computers_merges(tmp_path):
    """2026-11-04: Update now pressed on a computer that isn't home too - every Garmin activity is kept, the same file
    fetched on both is no clash, the combined files are home's (it rebuilds them from every day's download)."""
    here, there = tmp_path / "pc", tmp_path / "mac"
    act = "data/raw/garmin/activities/activities_garmin.csv"
    for root in (here, there):
        put(root, act, "id,name\ngarmin_1,Ride 1\n", 1_000)
        put(root, "data/raw/garmin/api/api_sleepData.json", "[1]", 1_000)
        put(root, "data/raw/garmin/activities/export_activity_files.json", json.dumps({"a.zip": {"files": []}}), 1_000)
    sync(here, there)
    put(here, act, "id,name\ngarmin_1,Ride 1\ngarmin_2,Ride 2\n", 2_000)               # each downloaded a new one
    put(there, act, "id,name\ngarmin_1,Ride 1\ngarmin_3,Ride 3\n", 2_100)
    put(here, "data/raw/garmin/activities/garmin_2.fit", b"FIT2", 2_000)              # the same file on both,
    put(there, "data/raw/garmin/activities/garmin_2.fit", b"FIT2", 2_200)             # fetched at different times
    put(here, "data/raw/garmin/api/raw/2026-11-04.json", '{"sleep": 1}', 2_000)
    put(there, "data/raw/garmin/api/raw/2026-11-04.json", '{"sleep": 1, "hrv": 2}', 2_300)   # later: more of the day
    put(here, "data/raw/garmin/api/api_sleepData.json", "[1, 2]", 2_000)
    put(there, "data/raw/garmin/api/api_sleepData.json", "[1, 3]", 2_300)
    put(here, "data/raw/garmin/activities/export_activity_files.json", json.dumps({"a.zip": {"files": []}, "b.zip": {}}), 2_000)
    put(there, "data/raw/garmin/activities/export_activity_files.json", json.dumps({"a.zip": {"files": []}, "c.zip": {}}), 2_000)
    s, out = sync(here, there)
    for root in (here, there):
        ids = [r.split(",")[0] for r in (root / act).read_text().splitlines()[1:]]
        assert sorted(ids) == ["garmin_1", "garmin_2", "garmin_3"]                    # every activity kept
        assert json.loads((root / "data/raw/garmin/api/raw/2026-11-04.json").read_text()) == {"sleep": 1, "hrv": 2}
        assert (root / "data/raw/garmin/api/api_sleepData.json").read_text() == "[1, 2]"     # home's (rebuilt)
        assert set(json.loads((root / "data/raw/garmin/activities/export_activity_files.json").read_text())) == \
            {"a.zip", "b.zip", "c.zip"}
    assert not list((here / STATE / "old").rglob("garmin_2.fit"))                      # identical: no clash
    from goodpush.sync.peer import PeerRules
    assert PeerRules.kind("data/raw/garmin/api/api_sleepData.json") == "derived"
    assert PeerRules.kind("data/raw/garmin/api/raw/2026-11-04.json") == "data"


def test_update_now_on_a_computer_that_isnt_home_asks_home(tmp_path, monkeypatch):
    from goodpush.app.server import Data, PeerHub
    from goodpush.config import init_project
    from goodpush.sync import garmin_api
    pc, mac = init_project(tmp_path / "pc"), init_project(tmp_path / "mac")
    hub_pc = PeerHub(Data(pc), QuietJobs(), 0, tmp_path / "s_pc")
    hub_mac = PeerHub(Data(mac), QuietJobs(), 0, tmp_path / "s_mac")
    hub_mac.me = "dl-mac"
    for h in (hub_pc, hub_mac):
        h.server.host, h.server.port = "127.0.0.1", 0
    hub_mac.server.start()
    monkeypatch.setattr(garmin_api, "has_key", lambda: True)
    try:
        st = hub_pc.pair()
        answer_when_asked(hub_pc.server.pairing, True)
        hub_mac.join(f"127.0.0.1:{hub_pc.server.port}", st["code"])
        local, log = [], []
        hub_mac.garmin_via_home(log.append, lambda lg: local.append(1))
        assert hub_pc.jobs.started[-1].startswith("Updating from Garmin (asked by") and not local   # home downloads
        hub_pc.server.stop()
        hub_mac.garmin_via_home(log.append, lambda lg: local.append(1))
        assert local == [1] and "isn't reachable" in log[-1]                           # home away: here, merged later
    finally:
        hub_pc.server.stop()
        hub_mac.server.stop()


def test_a_one_way_sync_never_sends_a_rides_row_without_its_file(tmp_path):
    """2026-11-04: Get from home on the laptop with a new ride on it - the trainer-rides list merged both ways, so home
    had the ride's row but not its file. Lists of rides go the sync's way only; the next both-ways sync merges them."""
    home, mac = tmp_path / "pc", tmp_path / "mac"
    rows = "data/raw/goodpush/activities_goodpush.csv"
    for root in (home, mac):
        put(root, rows, "id,name\n1,Ride 1\n", 1_000)
    sync(mac, home, home_here=False)
    put(mac, rows, "id,name\n1,Ride 1\n2,Ride 2\n", 2_000)                     # ridden on the laptop
    put(mac, "data/raw/goodpush/2.fit", b"FIT", 2_000)
    put(home, rows, "id,name\n1,Ride 1\n3,Ride 3\n", 2_000)                    # one recorded at home
    s = PeerSync(FolderSide(mac), FolderSide(home), "pc", home_here=False)
    s.run(s.plan("get"), progress=False)
    assert "2,Ride 2" not in (home / rows).read_text() and not (home / "data/raw/goodpush/2.fit").exists()
    assert "3,Ride 3" in (mac / rows).read_text() and "2,Ride 2" in (mac / rows).read_text()   # received, kept its own
    s = PeerSync(FolderSide(mac), FolderSide(home), "pc", home_here=False)
    s.run(s.plan("both"), progress=False)                                      # then both ways: everything, everywhere
    for root in (home, mac):
        assert sorted((root / rows).read_text().splitlines()[1:]) == ["1,Ride 1", "2,Ride 2", "3,Ride 3"]
    assert (home / "data/raw/goodpush/2.fit").read_bytes() == b"FIT"


def test_computers_sync_only_when_they_write_the_same_files(tmp_path, monkeypatch):
    """2026-11-04: a laptop on older goodpush merged with the desktop and wrote files the desktop read otherwise. Each
    computer says which goodpush it runs; another sync format is refused (told to update), other code only noted."""
    from goodpush.sync import peer as P
    home = tmp_path / "pc"
    srv = serve(home, tmp_path / "s_pc")
    try:
        srv.pairing.new()
        e = HttpSide.pair(f"127.0.0.1:{srv.port}", srv.pairing.active(), "mac")
        there = HttpSide(e["address"], e["fingerprint"], e["token"], "pc")
        s = PeerSync(FolderSide(tmp_path / "mac"), there, "pc", home_here=False)
        assert s.plan() == [] and s.note == ""                               # the same goodpush: fine, silent
        real = there.hello()                     # both computers are this process: the other one answers otherwise
        monkeypatch.setattr(there, "hello", lambda: {**real, "code": "other"})
        s.plan()
        assert "other goodpush code" in s.note and "other goodpush code" in s.describe([])   # synced, but said
        monkeypatch.setattr(there, "hello", lambda: {**real, "format": P.SYNC_FORMAT + 1, "version": "9.9"})
        with pytest.raises(ConnectionError, match="Update goodpush on both"):
            PeerSync(FolderSide(tmp_path / "mac"), there, "pc", home_here=False).plan()   # refused before any file
        there.close()
    finally:
        srv.stop()


def test_merged_files_are_written_exactly_as_goodpush_writes_them(tmp_path):
    """The bytes of the synced files are part of the sync format: a merge writes a file exactly as the app does (the
    2026-11-03 garbled title: the merge wrote raw UTF-8, the app escaped text). Changing these bytes means bumping
    SYNC_FORMAT, so an older computer is told to update instead of being merged with."""
    from goodpush.planning import PlanState
    from goodpush.sync.peer import SYNC_FORMAT, Merge
    assert SYNC_FORMAT == 4                      # bump together with the expected bytes below
    state = {"planned": {"2026-11-05": {"title": "Over-unders 2x6 × 1 min", "tss": 80}}, "away": {}}
    p = tmp_path / "plan_state.json"
    PlanState(state, p).save()
    app = p.read_bytes()
    assert app == (b'{\n "levels": {\n  "threshold": 0,\n  "vo2max": 0,\n  "sweetspot": 0,\n  "tempo": 0,\n  "anaerobic": 0'
                   b',\n  "durability": 0\n },\n "ladders": 2,\n "planned": {\n  "2026-11-05": {\n   '
                   b'"title": "Over-unders 2x6 \\u00d7 1 min'
                   b'",\n   "tss": 80\n  }\n },\n "away": {}\n}')
    merged = Merge(True).json(None, app, app, "data/processed/plan_state.json")
    assert merged == app and merged.isascii()       # what the merge writes is what the app writes


def test_a_sync_copies_in_parallel_and_asks_nothing_twice(tmp_path, monkeypatch):
    """2026-11-04 (syncs were slow): copies go over several connections at once, and the sync record comes from what
    the sync already knows - no size / time asked again per file."""
    home, mac = tmp_path / "pc", tmp_path / "mac"
    for i in range(40):
        put(home, f"reports/plan/workouts/gp_{i:02d}.zwo", f"w{i}" * 50, 1_000 + i)
    put(mac, "data/raw/goodpush/ride.fit", b"FIT" * 1000, 2_000)
    srv = serve(home, tmp_path / "s_pc")
    calls = []
    real = HttpSide.call
    monkeypatch.setattr(HttpSide, "call", lambda self, method, path, *a, **k: (calls.append(path), real(self, method, path, *a, **k))[1])
    try:
        srv.pairing.new()
        e = HttpSide.pair(f"127.0.0.1:{srv.port}", srv.pairing.active(), "mac")
        calls.clear()
        there = HttpSide(e["address"], e["fingerprint"], e["token"], "pc")
        s = PeerSync(FolderSide(mac), there, "pc", home_here=False)
        s.run(s.plan(), progress=False)
        assert all((mac / f"reports/plan/workouts/gp_{i:02d}.zwo").read_text() == f"w{i}" * 50 for i in range(40))
        assert (home / "data/raw/goodpush/ride.fit").read_bytes() == b"FIT" * 1000
        assert "/peer/stat" not in calls                                  # nothing asked again
        assert calls.count("/peer/file") == 41                            # one request per file
        there.close()
        again = PeerSync(FolderSide(mac), HttpSide(e["address"], e["fingerprint"], e["token"], "pc"), "pc", home_here=False)
        assert again.plan() == []                                         # the record matches what's there
    finally:
        srv.stop()


def test_the_app_lets_a_page_that_went_away_go_quietly(tmp_path):
    """2026-11-04: a reload while the app answered printed ConnectionAbortedError tracebacks (sending the error
    about the closed connection over that same connection). A page that went away isn't an error."""
    from goodpush.app.server import make_handler
    from goodpush.config import init_project

    class Gone:
        def write(self, b):
            raise ConnectionAbortedError(10053, "aborted")

        def flush(self):
            pass

    H = make_handler(init_project(tmp_path / "pc"), None, False, warm=False, port=0)
    h = H.__new__(H)
    h.wfile, h.request_version, h.requestline, h.command = Gone(), "HTTP/1.1", "GET /api/calendar HTTP/1.1", "GET"
    h.client_address, h.close_connection = ("127.0.0.1", 1), False
    h._send(200, b"{}")                                   # no exception
    assert h.close_connection
