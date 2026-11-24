"""Keep an athlete folder in step with its copy on another machine: between two goodpush apps paired with a code
(Settings > Other devices; `goodpush peer join`), or over SSH (`goodpush peer add / sync`).

The folder holds three kinds of files, each synced its own way:

* your decisions - goodpush.toml, plans.json, plan_state.json, my_workouts.json, deleted_activities.json,
  dip_labels.json: merged entry by entry against the copy from the last sync, so changes made on both machines
  survive; where both changed the same entry, the home machine's wins (and it is reported).
  activities_goodpush.csv (rides recorded on the trainer) is merged by ride.
* data - raw downloads and ride files: new or changed files are copied; changed on both sides, the newer wins and
  the other is kept in .goodpush-sync/old/ on its machine.
* derived - data/processed and reports (tables, streams.h5, model fits): copied like data, but changed on both
  sides the home machine's wins (they can be rebuilt).

Never synced: machine caches (jax_cache, the NUTS checkpoint, *.pkl), credentials (key.txt, tokens: those stay in
~/.goodpush on each machine) and settings that belong to one machine ([paths] downloads, [incyclist], where the ride
overlay sits on screen).

The home machine ([sync] home in goodpush.toml; `goodpush peer add` makes it the machine you run it on) downloads from
Garmin and Strava; the other machines' apps don't sync on their own. This machine always starts the sync. What the
last sync saw is kept in .goodpush-sync/ (on this machine only).

Paired apps: the machine that shows the code serves its folder to the machines that joined it (`PeerServer`: HTTPS
on the app's port + 100, its own self-signed certificate). Joining proves both sides know the code without sending
it, bound to that certificate (scrypt + HMAC, so a machine in the middle can't relay or guess it in time); the
joiner then pins the certificate and gets a token of its own. The server answers only addresses that aren't on the
open internet (home networks, Tailscale), listens only while a code is shown or paired machines may sync ([sync]
serve, per machine), and a goodpush.toml sent to it keeps this machine's own settings (`Merge.machine_kept`). Afterwards the home machine takes in what it got (new
rides, your changes) and the joiner fetches the result. Over SSH: the other needs an SSH server (macOS: Remote
Login) and must be in your known_hosts (connect once with ssh).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fnmatch
import functools
import hashlib
import hmac
import http.client
import io
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import ssl
import stat
import tempfile
import threading
import time
import tomllib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd

from . import secrets_dir
from ..app import tomledit

STATE = ".goodpush-sync"
CONFIG = "goodpush.toml"
# How the synced files are written (plan state and your other decisions, settings, the fit files, the ride and Garmin
# lists): two computers sync only when they write them the same way. Bump it with any change to those files' format
# (tests/test_peer.py pins the bytes) - an older computer is then told to update instead of being merged with.
SYNC_FORMAT = 4     # 4 goodpush (goodpush.toml, .goodpush-sync, data/raw/goodpush)


class PeerRules:
    """What kind of file a path in the athlete folder is (relative, with /)."""
    SKIP_PATHS = (STATE, "data/processed/jax_cache", "data/processed/physio_nuts.*",
                  "data/processed/physio_draws.npz",         # posterior draws: hundreds of MB, this machine's own
                  "data/processed/ladder/*",                 # the physio checks' results (goodpush.checks;
                                                             # rebuildable)
                  "logs/*")                                  # this computer's run logs (Project.logs)
    SKIP_NAMES = ("key.txt", "app_key", "*token*", "*secret*", "*credential*", "*.pkl", "*.peer-tmp", "*.lock", "*.log",
                  ".DS_Store", "Thumbs.db", "__pycache__", "*.pyc")
    DECISIONS = (CONFIG, "data/plans.json", "data/my_workouts.json", "data/deleted_activities.json",
                 "data/processed/plan_state.json", "data/processed/dip_labels.json")
    HISTORY = ("data/processed/plan_state.json",)      # days before today are history: a merge never drops one
    ROWS = {"activities_goodpush.csv": "id",            # file name -> the column that identifies a row
            "activities_garmin.csv": "id"}             # (Garmin downloaded on both computers: every activity kept)
    INDEX = ("data/raw/garmin/activities/export_activity_files.json",)   # JSON indexes: both sides' entries kept
    DERIVED = ("data/processed/", "reports/")
    REBUILT = "data/raw/garmin/api/"                   # Garmin's combined files: rebuilt from the per-day ones (raw/)
    MACHINE = {"paths": ("downloads",), "incyclist": None, "ride": ("x", "y", "scale", "opacity", "click_through"),
               "sync": ("serve", "every_hours")}

    @classmethod
    def kind(cls, rel: str) -> str:
        """skip | decision | rows | index | derived | data"""
        name = PurePosixPath(rel).name
        if any(fnmatch.fnmatch(rel, p) for p in cls.SKIP_PATHS) or any(fnmatch.fnmatch(name, p) for p in cls.SKIP_NAMES):
            return "skip"
        if rel in cls.DECISIONS:
            return "decision"
        if name in cls.ROWS:
            return "rows"
        if rel in cls.INDEX:
            return "index"
        if rel.startswith(cls.DERIVED) or (rel.startswith(cls.REBUILT) and "/" not in rel[len(cls.REBUILT):]):
            return "derived"
        return "data"

    @classmethod
    def machine_key(cls, section: str, key: str) -> bool:
        keys = cls.MACHINE.get(section, ())
        return section in cls.MACHINE and (keys is None or key in keys)


class Peers:
    """Machines in a toml file with the credentials, name -> plain values. peers.toml = the ones this machine syncs
    with: an ssh host and its folder there, or a paired app (address, certificate fingerprint, token)."""
    FILE = "peers.toml"

    def __init__(self, folder: Path | None = None):
        self.file = Path(folder or secrets_dir()) / self.FILE
        self.peers = tomllib.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}

    def add(self, name: str, **entry) -> dict:
        self.peers[name] = {k: str(v) for k, v in entry.items()}
        return self.save()[name]

    def update(self, name: str, **entry) -> None:
        if name in self.peers:
            self.peers[name].update({k: str(v) for k, v in entry.items()})
            self.save()

    def remove(self, name: str) -> None:
        if self.peers.pop(name, None) is not None:
            self.save()

    def save(self) -> dict:
        lines = [f"[{json.dumps(k)}]\n" + "".join(f"{x} = {json.dumps(y)}\n" for x, y in v.items())
                 for k, v in sorted(self.peers.items())]
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text("\n".join(lines), encoding="utf-8")
        return self.peers

    def get(self, name: str) -> dict:
        if name not in self.peers:
            known = ", ".join(sorted(self.peers)) or "none yet"
            raise KeyError(f"No machine '{name}' ({known}): pair it in Settings > Other devices, "
                           f"or `goodpush peer add {name} USER@HOST FOLDER`")
        return self.peers[name]

    def joined(self) -> bool:
        """this machine joined a paired app (PeerSync.home_here_for)"""
        return any(p.get("kind") == "app" for p in self.peers.values())

    def side(self, name: str) -> "FolderSide":
        p = self.get(name)
        if p.get("kind") == "app":
            return HttpSide(p["address"], p["fingerprint"], p["token"], p.get("label") or name)
        return SshSide(p["host"], p["folder"])


class Devices(Peers):
    """The machines allowed to sync with this one (paired.toml): name -> sha256 of the token it was given."""
    FILE = "paired.toml"

    def pair(self, name: str) -> str:
        token = secrets.token_urlsafe(32)
        self.add(name, token=hashlib.sha256(token.encode()).hexdigest(), paired=dt.datetime.now().isoformat("T", "seconds"))
        return token

    def check(self, token: str) -> str | None:
        """the paired machine's name for this token, or None"""
        h = hashlib.sha256(token.encode()).hexdigest()
        return next((k for k, v in self.peers.items() if token and hmac.compare_digest(v.get("token", ""), h)), None)


class FolderSide:
    """An athlete folder on this machine."""
    FORKABLE = True                            # PeerSync.transfer may use it from several threads

    def fork(self) -> "FolderSide":
        return self

    def __init__(self, root: Path):
        self.root = Path(root)
        self.label = "this machine"

    def files(self) -> dict[str, tuple[int, int]]:
        """rel -> (size, mtime in s) of everything that syncs."""
        out = {}
        for top, dirs, names in os.walk(self.root):
            base = Path(top).relative_to(self.root).as_posix()
            pre = "" if base == "." else base + "/"
            dirs[:] = [d for d in dirs if PeerRules.kind(pre + d) != "skip"]
            for n in names:
                if PeerRules.kind(pre + n) != "skip":
                    st = os.stat(os.path.join(top, n))
                    out[pre + n] = (st.st_size, int(st.st_mtime))
        return out

    def stat(self, rel: str) -> tuple[int, int]:
        st = (self.root / rel).stat()
        return st.st_size, int(st.st_mtime)

    def open(self, rel: str):
        return open(self.root / rel, "rb")

    def read(self, rel: str) -> bytes:
        return (self.root / rel).read_bytes()

    def write(self, rel: str, src, mtime: int) -> None:
        """src: a binary file or bytes; written next to it first, then put in place."""
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".peer-tmp")
        with open(tmp, "wb") as f:
            f.write(src) if isinstance(src, bytes) else shutil.copyfileobj(src, f, 1 << 20)
        os.utime(tmp, (mtime, mtime))
        os.replace(tmp, p)

    def remove(self, rel: str) -> None:
        (self.root / rel).unlink(missing_ok=True)

    def retire(self, rel: str, stamp: str) -> None:
        """keep this machine's version in .goodpush-sync/old/<stamp>/ instead of losing it"""
        dest = self.root / STATE / "old" / stamp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root / rel, dest)

    def close(self) -> None:
        pass


class SshSide(FolderSide):
    """The athlete folder on the other machine, over SFTP (paramiko; your ssh keys / agent and ~/.ssh/config)."""
    FORKABLE = False                           # one SFTP channel: the copies take turns

    def __init__(self, host: str, folder: str):
        import paramiko
        cfg = paramiko.SSHConfig()
        conf = Path.home() / ".ssh" / "config"
        if conf.exists():
            cfg = paramiko.SSHConfig.from_path(str(conf))
        user, _, name = host.rpartition("@")
        o = cfg.lookup(name)
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())   # only machines you've trusted with ssh
        try:
            self.client.connect(o.get("hostname", name), port=int(o.get("port", 22)), username=user or o.get("user"),
                                key_filename=o.get("identityfile"), timeout=15)
        except (paramiko.SSHException, OSError) as e:
            raise ConnectionError(f"{host}: {e}. Connect once with `ssh {host}` so it's in your known_hosts, "
                                  "and check the other machine accepts ssh (macOS: Remote Login).") from e
        self.sftp = self.client.open_sftp()
        home = self.sftp.normalize(".")
        self.root = folder if folder.startswith("/") else f"{home}/{folder.removeprefix('~/')}"
        self.label = host
        self.made: set[str] = set()

    def path(self, rel: str) -> str:
        return f"{self.root}/{rel}"

    def files(self) -> dict[str, tuple[int, int]]:
        out, todo = {}, [""]
        try:
            self.sftp.stat(self.root)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{self.label}: no folder {self.root}") from e
        while todo:
            pre = todo.pop()
            for a in self.sftp.listdir_attr(self.path(pre) if pre else self.root):
                rel = pre + a.filename
                if PeerRules.kind(rel) == "skip":
                    continue
                if stat.S_ISDIR(a.st_mode):
                    todo.append(rel + "/")
                else:
                    out[rel] = (a.st_size, int(a.st_mtime))
        return out

    def stat(self, rel: str) -> tuple[int, int]:
        a = self.sftp.stat(self.path(rel))
        return a.st_size, int(a.st_mtime)

    def open(self, rel: str):
        f = self.sftp.open(self.path(rel), "rb")
        f.prefetch()
        return f

    def read(self, rel: str) -> bytes:
        with self.open(rel) as f:
            return f.read()

    def mkdirs(self, rel_dir: str) -> None:
        parts = [p for p in rel_dir.split("/") if p]
        for i in range(1, len(parts) + 1):
            d = "/".join(parts[:i])
            if d in self.made:
                continue
            try:
                self.sftp.stat(self.path(d))
            except FileNotFoundError:
                self.sftp.mkdir(self.path(d))
            self.made.add(d)

    def write(self, rel: str, src, mtime: int) -> None:
        if "/" in rel:
            self.mkdirs(str(PurePosixPath(rel).parent))
        tmp = self.path(rel) + ".peer-tmp"
        with self.sftp.open(tmp, "wb") as f:
            f.set_pipelined(True)
            f.write(src) if isinstance(src, bytes) else shutil.copyfileobj(src, f, 1 << 20)
        self.sftp.utime(tmp, (mtime, mtime))
        self.sftp.posix_rename(tmp, self.path(rel))

    def remove(self, rel: str) -> None:
        try:
            self.sftp.remove(self.path(rel))
        except FileNotFoundError:
            pass

    def retire(self, rel: str, stamp: str) -> None:
        dest = f"{STATE}/old/{stamp}/{rel}"
        self.mkdirs(str(PurePosixPath(dest).parent))
        self.sftp.posix_rename(self.path(rel), self.path(dest))

    def close(self) -> None:
        self.client.close()


class PeerCert:
    """This machine's certificate for syncing: self-signed, made once (secrets folder); a machine that pairs with
    this one pins its fingerprint."""
    CERT, KEY = "peer_cert.pem", "peer_key.pem"

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder or secrets_dir())
        self.cert, self.key = self.folder / self.CERT, self.folder / self.KEY

    def ensure(self) -> "PeerCert":
        if self.cert.exists() and self.key.exists():
            return self
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "goodpush peer")])
        now = dt.datetime.now(dt.timezone.utc)
        cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
                .not_valid_after(now + dt.timedelta(days=3650)).sign(key, hashes.SHA256()))
        self.folder.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.key, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
        self.cert.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        return self

    def fingerprint(self) -> str:
        return self.fingerprint_of(ssl.PEM_cert_to_DER_cert(self.ensure().cert.read_text()))

    @staticmethod
    def fingerprint_of(der: bytes) -> str:
        return hashlib.sha256(der).hexdigest()

    def context(self) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(self.ensure().cert, self.key)
        return ctx


class Pairing:
    """The one-off code a machine shows to pair (valid MINUTES, TRIES wrong guesses). It never travels: each side
    proves it knows it with a key derived from the code and the server's certificate (scrypt: slow to guess).
    A machine that knew it still waits (`ask`, up to WAIT s) until you allow it on this one (`answer`)."""
    ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"       # no 0/O, 1/I/L  # pragma: allowlist secret
    LENGTH, MINUTES, TRIES, WAIT = 10, 10, 5, 120

    def __init__(self):
        self.lock = threading.Lock()
        self.code, self.until, self.tries = None, 0.0, 0
        self.pending: dict | None = None

    def ask(self, name: str) -> bool:
        """wait until you allow or deny `name` here (denied after WAIT s)"""
        p = {"name": name, "event": threading.Event(), "allow": False}
        with self.lock:
            if self.pending:                   # one at a time: a second knew the code too - don't guess which
                self.pending["event"].set()
                self.pending = None
                return False
            self.pending = p
        p["event"].wait(self.WAIT)
        with self.lock:
            if self.pending is p:
                self.pending = None
            return p["allow"]

    def answer(self, allow: bool) -> str | None:
        with self.lock:
            p, self.pending = self.pending, None
        if p:
            p["allow"] = bool(allow)
            p["event"].set()
        return p and p["name"]

    def waiting(self) -> str | None:
        """the machine waiting for you to allow it"""
        with self.lock:
            return self.pending and self.pending["name"]

    def new(self) -> str:
        with self.lock:
            self.code = "".join(secrets.choice(self.ALPHABET) for _ in range(self.LENGTH))
            self.until, self.tries = time.time() + self.MINUTES * 60, 0
            return self.shown(self.code)

    def active(self) -> str | None:
        with self.lock:
            return self.shown(self.code) if self.code and time.time() < self.until else None

    def shown(self, code: str) -> str:
        return f"{code[:5]}-{code[5:]}"

    @staticmethod
    def normal(code: str) -> str:
        return re.sub(r"[^0-9A-Z]", "", code.upper())

    @staticmethod
    def key(code: str, fingerprint: str) -> bytes:
        return hashlib.scrypt(Pairing.normal(code).encode(), salt=b"goodpush-pair" + bytes.fromhex(fingerprint),
                              n=2 ** 14, r=8, p=1, dklen=32)

    @staticmethod
    def proof(key: bytes, *parts: str) -> str:
        return hmac.new(key, "|".join(parts).encode(), hashlib.sha256).hexdigest()

    def accept(self, fingerprint: str, name: str, proof: str) -> bytes | None:
        """the key, when `proof` shows the joining machine knows the code (the code is used up then)"""
        with self.lock:
            if not self.code or time.time() >= self.until:
                return None
            key = self.key(self.code, fingerprint)
            if hmac.compare_digest(self.proof(key, "join", name), proof):
                self.code = None
                return key
            self.tries += 1
            if self.tries >= self.TRIES:
                self.code = None
            return None


class RequestBody:
    """The first `n` bytes of a request, read like a file."""

    def __init__(self, f, n: int):
        self.f, self.left = f, n

    def read(self, size: int = -1) -> bytes:
        if self.left <= 0:
            return b""
        b = self.f.read(self.left if size < 0 else min(size, self.left))
        if not b:
            raise ConnectionError("the upload was cut short")
        self.left -= len(b)
        return b


class PeerHandler(BaseHTTPRequestHandler):
    """/peer/* for paired machines: a token on every request but the pairing one; paths checked by PeerServer."""
    protocol_version = "HTTP/1.1"
    server_version = "goodpush-peer"
    timeout = 120

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.route("GET")

    def do_PUT(self):
        self.route("PUT")

    def do_POST(self):
        self.route("POST")

    def reply(self, code: int, obj=None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        if code >= 400:                       # an upload refused mid-way leaves its bytes unread
            self.close_connection = True
            self.send_header("Connection", "close")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(min(n, 100_000)) or b"{}")

    def route(self, method: str) -> None:
        peer: PeerServer = self.server.peer
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if not PeerServer.nearby(self.client_address[0]):
            return self.reply(403, {"error": "only from your own network"})
        try:
            if (method, u.path) == ("POST", "/peer/pair"):
                return self.pair(peer)
            who = peer.devices_now().check(self.headers.get("Authorization", "").removeprefix("Bearer "))
            if not who:
                return self.reply(403, {"error": "not paired with this machine"})
            if u.path == "/peer/hello":
                return self.reply(200, {"name": socket.gethostname(), "busy": bool(peer.busy()), **PeerSync.identity()})
            if u.path == "/peer/files":                # a sync starts: where that machine is now
                peer.seen(who, self.client_address[0])
                return self.reply(200, peer.side.files())
            if (method, u.path) == ("POST", "/peer/done"):
                b = self.body()
                return self.reply(200, {"job": bool(peer.on_done(who, b.get("kinds") or [], b.get("wait", True)))})
            if (method, u.path) == ("POST", "/peer/garmin"):     # a computer that isn't home: download from Garmin
                return self.reply(200, {"job": bool(peer.on_garmin(who))})
            if (method, u.path) == ("POST", "/peer/link"):
                peer.link(who, self.client_address[0], self.body())
                return self.reply(200, {"ok": True})
            rel = peer.checked(q.get("rel", ""))
            if (method, u.path) == ("GET", "/peer/stat"):
                return self.reply(200, peer.side.stat(rel))
            if (method, u.path) == ("GET", "/peer/file"):
                return self.send_file(peer.side.root / rel)
            # the app may be saving your decisions meanwhile (plan state): only they wait for it
            with peer.lock if PeerRules.kind(rel) == "decision" else contextlib.nullcontext():
                if (method, u.path) == ("PUT", "/peer/file"):
                    body = RequestBody(self.rfile, int(self.headers.get("Content-Length") or 0))
                    if rel == CONFIG:                  # this computer's own settings stay as they are
                        if body.left > 1 << 20:
                            raise ValueError("that's not a settings file")
                        old = (peer.side.root / CONFIG).read_bytes() if (peer.side.root / CONFIG).exists() else None
                        body = Merge.machine_kept(old, body.read())
                    peer.side.write(rel, body, int(q.get("mtime") or time.time()))
                elif (method, u.path) == ("POST", "/peer/remove"):
                    peer.side.remove(rel)
                elif (method, u.path) == ("POST", "/peer/retire"):
                    peer.side.retire(rel, re.sub(r"[^0-9_-]", "", q.get("stamp", "")) or "old")
                else:
                    return self.reply(404, {"error": "not found"})
            return self.reply(200, {"ok": True})
        except PermissionError as e:
            return self.reply(403, {"error": f"not shared: {e}"})
        except FileNotFoundError as e:
            return self.reply(404, {"error": f"no such file: {e}"})
        except (ConnectionError, OSError, ValueError) as e:
            return self.reply(400, {"error": str(e)})

    def pair(self, peer: "PeerServer") -> None:
        b = self.body()
        name = re.sub(r"[^\w.-]", "", str(b.get("name") or ""))[:64] or "device"
        if name == socket.gethostname():         # both would take themselves for home
            return self.reply(403, {"error": f"both computers are called {name}: rename one (its computer name in "
                                             "the system settings), then pair again"})
        key = peer.pairing.accept(peer.cert.fingerprint(), name, str(b.get("proof") or ""))
        if key is None:
            return self.reply(403, {"error": "wrong or expired code: show a new one on the other computer"})
        if peer.ask and not peer.pairing.ask(name):
            return self.reply(403, {"error": f"not allowed on {socket.gethostname()}: show a new code there and "
                                             "allow this computer when it asks"})
        token = peer.devices_now().pair(name)
        me = socket.gethostname()
        self.reply(200, {"name": me, "token": token, "proof": Pairing.proof(key, "home", me, token), **PeerSync.identity()})

    def send_file(self, p: Path) -> None:
        with open(p, "rb") as f:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(os.fstat(f.fileno()).st_size))
            self.end_headers()
            shutil.copyfileobj(f, self.wfile, 1 << 20)


class PeerServer:
    """This athlete folder for the machines that paired with this one: HTTPS on the app's port + PORT_OFFSET.
    on_done(name, kinds, wait) = a machine finished a sync that changed these kinds of files here, and waits for the
    result or not (start taking them in; true if a job started); busy() = this machine is working on something (the
    other waits for it); ask = a machine that knew the code waits until you allow it here; on_garmin(name) = a machine
    asks this one (home) to download from Garmin (true if it started)."""
    PORT_OFFSET = 100

    def __init__(self, root: Path, port: int, folder: Path | None = None, on_done=None, busy=None, lock=None,
                 host: str = "0.0.0.0", ask: bool = True, on_garmin=None):
        self.side, self.port, self.host, self.folder, self.ask = FolderSide(root), port, host, folder, ask
        self.cert, self.pairing = PeerCert(folder), Pairing()
        self.on_done = on_done or (lambda name, kinds, wait: False)
        self.on_garmin = on_garmin or (lambda name: False)
        self.busy = busy or (lambda: False)
        self.lock = lock or threading.RLock()
        self.httpd: ThreadingHTTPServer | None = None

    def devices_now(self) -> Devices:
        """the paired machines (read again only when paired.toml changed: every request asks)"""
        f = Path(self.folder or secrets_dir()) / Devices.FILE
        stamp = f.stat().st_mtime_ns if f.exists() else 0
        if getattr(self, "_devices", (None,))[0] != stamp:
            self._devices = (stamp, Devices(self.folder))
        return self._devices[1]

    def link(self, name: str, ip: str, b: dict) -> None:
        """a machine that paired with this one lets it start syncs too: kept in peers.toml (its certificate pinned)"""
        fp, port, token = str(b.get("fingerprint") or ""), int(b.get("port") or 0), str(b.get("token") or "")
        if not (re.fullmatch(r"[0-9a-f]{64}", fp) and 0 < port < 65536 and len(token) >= 32):
            raise ValueError("not a link")
        peers = Peers(self.folder)
        label = (peers.peers.get(name) or {}).get("label") or (self.devices_now().peers.get(name) or {}).get("label")
        peers.add(name, kind="app", address=HttpSide.join_address(ip, port), fingerprint=fp, token=token,
                  **({"label": label} if label else {}))

    def seen(self, name: str, ip: str) -> None:
        """the machine `name` is at `ip` now (an address that changed since it linked: DHCP)"""
        peers = Peers(self.folder)
        p = peers.peers.get(name)
        if not p or p.get("kind") != "app":
            return
        host, port = HttpSide.split(p["address"])
        try:
            ipaddress.ip_address(host)           # only an address it was given as an IP, never a name you typed
        except ValueError:
            return
        if host != ip:
            peers.update(name, address=HttpSide.join_address(ip, port))

    def start(self) -> "PeerServer":
        if self.httpd:
            return self
        httpd = ThreadingHTTPServer((self.host, self.port), PeerHandler)
        httpd.daemon_threads = True
        httpd.socket = self.cert.context().wrap_socket(httpd.socket, server_side=True, do_handshake_on_connect=False)
        httpd.peer = self
        httpd.handle_error = lambda request, address: None      # a machine dropping off mid-way isn't news
        threading.Thread(target=httpd.serve_forever, daemon=True, name="goodpush-peer").start()
        self.httpd = httpd
        self.port = httpd.server_address[1]
        return self

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    def addresses(self) -> list[str]:
        """how another machine can reach this one: its address on the network, its name"""
        from ..app.server import _lan_ip
        return list(dict.fromkeys([f"{_lan_ip()}:{self.port}", f"{socket.gethostname()}:{self.port}"]))

    @staticmethod
    def nearby(address: str) -> bool:
        """a home network, Tailscale (100.64/10) or this computer - never an address on the open internet"""
        try:
            return not ipaddress.ip_address(address.split("%")[0]).is_global
        except ValueError:
            return False

    def checked(self, rel: str) -> str:
        """`rel` if it's a file this folder shares (inside it, not a credential, cache or sync record)"""
        p = PurePosixPath(rel)
        parts = ["/".join(p.parts[:i + 1]) for i in range(len(p.parts))]
        if (not rel or "\\" in rel or ":" in rel or p.is_absolute() or ".." in p.parts or rel.startswith(STATE)
                or any(PeerRules.kind(x) == "skip" or PeerRules.kind(x.lower()) == "skip" for x in parts)):
            raise PermissionError(rel)
        root = self.side.root.resolve()
        if not (root / rel).resolve().is_relative_to(root):
            raise PermissionError(rel)
        return rel


class HttpSide(FolderSide):
    """The athlete folder of a paired machine's app, over HTTPS: its certificate must be the one seen at pairing."""
    TIMEOUT = 30

    def __init__(self, address: str, fingerprint: str, token: str = "", label: str | None = None):
        self.address = address
        self.host, self.port = self.split(address)
        self.fingerprint, self.token = fingerprint, token
        self.label, self.root = label or address, address
        self.conn: http.client.HTTPSConnection | None = None

    @staticmethod
    def split(address: str) -> tuple[str, int]:
        from ..launcher import PORT
        host, _, port = address.strip().removeprefix("https://").rstrip("/").rpartition(":")
        if not host or not port.isdigit():                 # no port given
            return address.strip().strip("[]"), PORT + PeerServer.PORT_OFFSET
        return host.strip("[]"), int(port)

    @staticmethod
    def join_address(host: str, port: int) -> str:
        return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"

    def connect(self) -> str:
        """connect; the fingerprint seen (checked against the pinned one)"""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE      # pinned below instead
        conn = http.client.HTTPSConnection(self.host, self.port, context=ctx, timeout=self.TIMEOUT)
        try:
            conn.connect()
        except OSError as e:
            raise ConnectionError(f"{self.label} ({self.host}:{self.port}): {e}. Is goodpush open there, and on "
                                  "the same network?") from e
        seen = PeerCert.fingerprint_of(conn.sock.getpeercert(binary_form=True))
        if self.fingerprint and not hmac.compare_digest(seen, self.fingerprint):
            conn.close()
            raise ConnectionError(f"{self.host}:{self.port} isn't the {self.label} you paired with (its certificate "
                                  "differs). If you reinstalled goodpush there, pair again.")
        self.conn = conn
        return seen

    def call(self, method: str, path: str, body=None, headers=None, **query) -> http.client.HTTPResponse:
        url = path + ("?" + urlencode(query) if query else "")
        h = {"Authorization": f"Bearer {self.token}", **(headers or {})}
        for attempt in (0, 1):
            if self.conn is None:
                self.connect()
            try:
                self.conn.request(method, url, body=body, headers=h)
                r = self.conn.getresponse()
                break
            except (http.client.HTTPException, ConnectionError) as e:
                self.conn.close()
                self.conn = None
                if attempt or not (body is None or isinstance(body, bytes)):   # a stream can't be sent twice
                    raise ConnectionError(f"{self.label}: {e}") from e
        if r.status >= 400:
            msg = (json.loads(r.read() or b"{}") or {}).get("error") or f"HTTP {r.status}"
            raise {403: PermissionError, 404: FileNotFoundError}.get(r.status, ConnectionError)(f"{self.label}: {msg}")
        return r

    def json(self, method: str, path: str, obj=None, **query):
        body = None if obj is None else json.dumps(obj).encode()
        return json.loads(self.call(method, path, body, {"Content-Type": "application/json"} if body else None,
                                    **query).read())

    def files(self) -> dict[str, tuple[int, int]]:
        return {k: tuple(v) for k, v in self.json("GET", "/peer/files").items()}

    def stat(self, rel: str) -> tuple[int, int]:
        return tuple(self.json("GET", "/peer/stat", rel=rel))

    def open(self, rel: str):
        return self.call("GET", "/peer/file", rel=rel)

    def read(self, rel: str) -> bytes:
        with self.open(rel) as r:
            return r.read()

    def write(self, rel: str, src, mtime: int) -> None:
        size = len(src) if isinstance(src, bytes) else os.fstat(src.fileno()).st_size
        self.call("PUT", "/peer/file", src, {"Content-Length": str(size)}, rel=rel, mtime=mtime).read()

    def remove(self, rel: str) -> None:
        self.json("POST", "/peer/remove", rel=rel)

    def retire(self, rel: str, stamp: str) -> None:
        self.json("POST", "/peer/retire", rel=rel, stamp=stamp)

    def fork(self) -> "HttpSide":
        """another connection to the same machine (PeerSync.transfer: copies in parallel)"""
        return HttpSide(self.address, self.fingerprint, self.token, self.label)

    def hello(self) -> dict:
        """its name, whether it's busy, and which goodpush it runs (PeerSync.identity)"""
        return self.json("GET", "/peer/hello")

    def busy(self) -> bool:
        return bool(self.hello().get("busy"))

    def done(self, kinds, wait: bool = True) -> bool:
        """tell it the sync changed these kinds of files there (and whether this machine waits for the result);
        true if it started taking them in"""
        return bool(self.json("POST", "/peer/done", {"kinds": sorted(kinds), "wait": bool(wait)}).get("job"))

    def garmin(self) -> bool:
        """ask it (home) to download from Garmin; it sends the result to its computers. True if it started."""
        return bool(self.json("POST", "/peer/garmin").get("job"))

    def link(self, fingerprint: str, port: int, token: str) -> None:
        """after pairing: let it start syncs too - this machine's certificate, port and a token of its own"""
        self.json("POST", "/peer/link", {"fingerprint": fingerprint, "port": int(port), "token": token})

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    @classmethod
    def pair(cls, address: str, code: str, name: str) -> dict:
        """join the machine at `address` showing `code`: its peers.toml entry (with its name as "name")"""
        side = cls(address, "")
        side.TIMEOUT = Pairing.WAIT + 15            # the other computer asks you first
        try:
            seen = side.connect()
            key = Pairing.key(code, seen)
            try:
                r = side.json("POST", "/peer/pair", {"name": name, "proof": Pairing.proof(key, "join", name)})
            except PermissionError as e:
                why = str(e).removeprefix(f"{address}: ")
                raise ValueError(why[:1].upper() + why[1:] + ".") from e
        finally:
            side.close()
        if not hmac.compare_digest(str(r.get("proof") or ""), Pairing.proof(key, "home", str(r.get("name")), str(r.get("token")))):
            raise ConnectionError(f"{address} answered without knowing the code: not paired.")
        PeerSync.compatible(r, address)
        if r["name"] == name:              # both would take themselves for home ([sync] home is a computer name)
            raise ValueError(f"Both computers are called {name}: rename one (its computer name in the system "
                             "settings), then pair again.")
        return {"name": r["name"], "kind": "app", "address": address, "fingerprint": seen, "token": r["token"]}


@dataclass
class Step:
    """get: theirs -> here, put: here -> theirs, merge: both, drop_here / drop_there: deleted on the other side;
    retire: keep the version it replaces (in .goodpush-sync/old)."""
    action: str
    rel: str
    kind: str
    size: int = 0
    retire: bool = False


class Merge:
    """Three-way merges of your decisions: base = what both had at the last sync."""
    MISSING = object()

    def __init__(self, home_here: bool):
        self.home_here = home_here
        self.clashes: list[str] = []

    def value(self, b, l, r, where: str = ""):
        """JSON-like values: dicts entry by entry, lists of plain values as sets, anything else whole."""
        M = self.MISSING
        if l == r:
            return l
        if l == b:
            return r
        if r == b:
            return l
        if isinstance(l, dict) and isinstance(r, dict):
            b = b if isinstance(b, dict) else {}
            out = {}
            for k in list(l) + [k for k in r if k not in l]:
                v = self.value(b.get(k, M), l.get(k, M), r.get(k, M), f"{where}/{k}")
                if v is not M:
                    out[k] = v
            return out
        if isinstance(l, list) and isinstance(r, list) and self.plain(l) and self.plain(r):
            b = b if isinstance(b, list) and self.plain(b) else []
            gone = (set(b) - set(l)) | (set(b) - set(r))
            return [x for x in dict.fromkeys(l + r) if x not in gone]
        self.clashes.append(where or "/")
        return l if self.home_here else r

    @staticmethod
    def plain(xs: list) -> bool:
        return all(isinstance(x, (str, int, float, bool)) for x in xs)

    def json(self, b: bytes | None, l: bytes, r: bytes, rel: str) -> bytes:
        base = json.loads(b) if b else self.MISSING
        L, R = json.loads(l), json.loads(r)
        out = self.value(base, L, R, rel)
        self.keep(out, L, R, rel)
        return json.dumps(out, indent=1, default=str).encode()   # escaped like goodpush writes it: any reader gets it right

    def keep(self, out, L, R, rel: str) -> None:
        """What a merge never takes away, even when one side lacks it (a copy started afresh looks like it deleted
        everything): a whole section, and in PeerRules.HISTORY files the days before today. Noted in `clashes`."""
        if not (isinstance(out, dict) and isinstance(L, dict) and isinstance(R, dict)):
            return
        first, second = (L, R) if self.home_here else (R, L)
        today = dt.date.today().isoformat()
        sections, days = [], 0
        for k in list(first) + [k for k in second if k not in first]:
            if k not in out:
                out[k] = first[k] if k in first else second[k]
                sections.append(k)
            elif rel in PeerRules.HISTORY and isinstance(out[k], dict):
                for side in (first, second):
                    for d, v in (side.get(k) if isinstance(side.get(k), dict) else {}).items():
                        if re.match(r"\d{4}-\d{2}-\d{2}", d) and d[:10] < today and d not in out[k]:
                            out[k][d] = v
                            days += 1
        if sections or days:
            self.clashes.append(f"{rel}: one computer's copy lacked " + " and ".join(
                ([f"[{', '.join(sections)}]"] if sections else []) + ([f"{days} past day(s)"] if days else []))
                + " - kept (a merge never drops those)")

    def rows(self, b: bytes | None, l: bytes, r: bytes, key: str) -> bytes:
        """a table by row: rows added on either side kept, rows deleted on either side dropped"""
        L, R = (pd.read_csv(io.BytesIO(x), dtype=str, keep_default_na=False) for x in (l, r))
        B = pd.read_csv(io.BytesIO(b), dtype=str, keep_default_na=False) if b else L.iloc[0:0]
        gone = set(B[key]) - (set(L[key]) & set(R[key]))
        first, second = (L, R) if self.home_here else (R, L)
        out = pd.concat([first, second], ignore_index=True).drop_duplicates(key, keep="first")
        return out[~out[key].isin(gone)].to_csv(index=False).encode()

    def settings(self, b: bytes | None, l: bytes, r: bytes) -> tuple[bytes, bytes]:
        """goodpush.toml key by key, each machine's own keys left alone; each side keeps its comments"""
        tb, tl, tr = (x.decode("utf-8") for x in (b or b"", l, r))
        fb, fl, fr = self.flat(tb), self.flat(tl), self.flat(tr)
        M, merged = self.MISSING, {}
        for k in list(fl) + [k for k in fr if k not in fl]:
            if PeerRules.machine_key(*k):
                continue
            v = self.value(fb.get(k, M), fl.get(k, M), fr.get(k, M), f"{CONFIG} [{k[0]}] {k[1]}")
            if v is not M:
                merged[k] = v
        return self.apply(tl, fl, merged), self.apply(tr, fr, merged)

    @staticmethod
    def flat(text: str) -> dict:
        """(section, key) -> value; arrays of tables ([[events]]) as ("[[", name) -> list"""
        if not text.strip():
            return {}
        data, out = tomllib.loads(text), {}
        sections = set(re.findall(r"(?m)^\s*\[([^\[\]]+)\]\s*$", text))
        arrays = set(re.findall(r"(?m)^\s*\[\[([^\[\]]+)\]\]\s*$", text))
        todo = [("", data)]
        while todo:
            path, obj = todo.pop()
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                if isinstance(v, dict) and any(s == p or s.startswith(p + ".") for s in sections):
                    todo.append((p, v))
                elif isinstance(v, list) and p in arrays:
                    out[("[[", p)] = v
                elif path:
                    out[(path, k)] = v
        return out

    @classmethod
    def machine_kept(cls, old: bytes | None, new: bytes) -> bytes:
        """`new` (a goodpush.toml another computer sent) with this computer's own settings put back as `old` has
        them: what arrives never says where this computer reads or writes (Incyclist's folder, downloads, ...)"""
        tn, to = new.decode("utf-8"), (old or b"").decode("utf-8")
        fn, fo = cls.flat(tn), cls.flat(to)
        theirs = {k: v for k, v in fn.items() if PeerRules.machine_key(*k)}
        mine = {k: v for k, v in fo.items() if PeerRules.machine_key(*k)}
        if theirs == mine:
            return new
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / CONFIG
            p.write_text(tn, encoding="utf-8")
            for sec in dict.fromkeys(k[0] for k in list(theirs) + list(mine)):
                shared = {k[1]: v for k, v in fn.items() if k[0] == sec and not PeerRules.machine_key(*k)}
                tomledit.set_table(p, sec, {**shared, **{k[1]: v for k, v in mine.items() if k[0] == sec}})
            return p.read_bytes()

    def apply(self, text: str, flat: dict, merged: dict) -> bytes:
        """`text` with the merged values (tomledit: comments and the rest untouched)"""
        mine = {k: v for k, v in flat.items() if not PeerRules.machine_key(*k)}
        if mine == merged:
            return text.encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / CONFIG
            p.write_text(text, encoding="utf-8")
            for sec in dict.fromkeys(k[0] for k in list(mine) + list(merged) if k[0] != "[["):
                new = {k[1]: v for k, v in merged.items() if k[0] == sec}
                if any(k[0] == sec and k not in merged for k in mine):          # a key taken out: the whole section
                    keep = {k[1]: v for k, v in flat.items() if k[0] == sec and PeerRules.machine_key(*k)}
                    tomledit.set_table(p, sec, {**new, **keep})
                elif changed := {k: v for k, v in new.items() if mine.get((sec, k), self.MISSING) != v}:
                    tomledit.set_values(p, sec, changed)
            for name in dict.fromkeys(k[1] for k in list(mine) + list(merged) if k[0] == "[["):
                if mine.get(("[[", name)) != merged.get(("[[", name)):
                    tomledit.set_tables(p, name, merged.get(("[[", name)) or [])
            return p.read_bytes()


class PeerSync:
    """One sync between this folder and the other machine's: plan() what to do, run() it."""

    def __init__(self, here: FolderSide, there: FolderSide, peer: str, home_here: bool = True):
        self.here, self.there, self.peer, self.home_here = here, there, peer, home_here
        self.state = Path(here.root) / STATE
        self.record_file = self.state / f"{peer}.json"
        self.record = json.loads(self.record_file.read_text()) if self.record_file.exists() else {}
        self.seen: dict = {}
        self.clashes: list[str] = []
        self.changed: dict[str, set] = {"here": set(), "there": set()}
        self.held: set[str] = set()
        self.one_way: set[str] = set()
        self.direction = "both"

    @staticmethod
    def home_here_for(cfg: dict, joined: bool = False, me: str | None = None) -> bool:
        """this machine (`me`: its name) is the athlete's home: [sync] home = its name; none set, unless it `joined`
        the other one (a paired app: that one serves, so it's home - a new machine never overwrites the folder it
        joins)"""
        home = str((cfg.get("sync") or {}).get("home") or "")
        return home == (me or socket.gethostname()) if home else not joined

    @staticmethod
    def same(a, b) -> bool:
        return bool(a and b) and a[0] == b[0] and abs(a[1] - b[1]) <= 1

    DIRECTIONS = {"both": None, "send": ("put", "drop_there", "merge"), "get": ("get", "drop_here", "merge")}

    @staticmethod
    def identity() -> dict:
        """which goodpush this is: the sync format (must match), its version and a fingerprint of its code"""
        from .. import __version__
        return {"format": SYNC_FORMAT, "version": __version__, "code": PeerSync.code_fingerprint()}

    @staticmethod
    @functools.cache
    def code_fingerprint() -> str:
        """goodpush's own code (every .py file of the package): two computers on other code still sync if they write
        the same format, but the sync says so"""
        root = Path(__file__).resolve().parents[1]
        h = hashlib.sha256()
        for f in sorted(root.rglob("*.py")):
            h.update(f.relative_to(root).as_posix().encode() + f.read_bytes())
        return h.hexdigest()[:12]

    @staticmethod
    def compatible(other: dict, label: str) -> str:
        """refuse a computer that writes the synced files another way (raises ConnectionError, saying what to do);
        "" if it runs this very code, else a note that it runs other code"""
        mine = PeerSync.identity()
        if other.get("format") != mine["format"]:
            raise ConnectionError(
                f"{label} runs goodpush {other.get('version', 'older')} (sync format {other.get('format', 'none')}), this "
                f"computer {mine['version']} (format {mine['format']}): they write your files differently. Update goodpush "
                "on both (git pull / the installer), restart their apps, then sync.")
        if other.get("code") == mine["code"]:
            return ""
        return f"{label} runs other goodpush code (same file format, so syncing is safe); update both when you can."

    def plan(self, direction: str = "both") -> list[Step]:
        """what to do; `direction` send / get = only files to the other machine / from it (your decisions still
        merge both ways: small, and a merge loses nothing). Files held back keep their record, so they're still to do
        next time. Lists of rides (trainer rides, Garmin's activities) go the sync's way only - a ride's row never
        arrives without its file - and keep their record too, so the next both-ways sync merges them in full."""
        self.note = self.compatible(self.there.hello(), self.there.label) if isinstance(self.there, HttpSide) else ""
        steps = self.compare()
        keep = self.DIRECTIONS[direction]
        self.direction = direction
        self.held = {s.rel for s in steps if keep and s.action not in keep}
        self.one_way = {s.rel for s in steps if keep and s.action == "merge" and s.kind in ("rows", "index")}
        return [s for s in steps if s.rel not in self.held]

    def compare(self) -> list[Step]:
        L, R = self.here.files(), self.there.files()
        self.seen = {"here": L, "there": R}
        steps = []
        for rel in sorted(set(L) | set(R) | set(self.record)):
            kind = PeerRules.kind(rel)
            if kind == "skip":
                continue
            l, r, rec = L.get(rel), R.get(rel), self.record.get(rel)
            bl, br = (tuple(rec["l"]) if rec and rec.get("l") else None), (tuple(rec["r"]) if rec and rec.get("r") else None)
            lch, rch = rec is None or l != bl, rec is None or r != br
            if l and r:
                if not (lch or rch):
                    continue
                if kind in ("decision", "rows", "index"):  # small files: their contents decide, not size and time
                    if self.here.read(rel) != self.there.read(rel):
                        steps.append(Step("merge", rel, kind, l[0] + r[0]))
                    continue
                if self.same(l, r):
                    continue
                if lch and rch and kind == "data" and l[0] == r[0] and self.here.read(rel) == self.there.read(rel):
                    continue                               # the same file fetched on both (a Garmin download)
                if not rch:
                    steps.append(Step("put", rel, kind, l[0]))
                elif not lch:
                    steps.append(Step("get", rel, kind, r[0]))
                elif kind == "derived":
                    steps.append(Step("put", rel, kind, l[0]) if self.home_here else Step("get", rel, kind, r[0]))
                else:
                    steps.append(Step("put", rel, kind, l[0], True) if l[1] >= r[1] else Step("get", rel, kind, r[0], True))
            elif l:
                steps.append(Step("put", rel, kind, l[0]) if rec is None or lch else Step("drop_here", rel, kind))
            elif r:
                steps.append(Step("get", rel, kind, r[0]) if rec is None or rch else Step("drop_there", rel, kind))
        return steps

    def describe(self, steps: list[Step]) -> str:
        note = f"\n  {self.note}" if getattr(self, "note", "") else ""
        if not steps:
            return f"Already in step with {self.there.label}." + note
        mb = lambda n: f"{n / 1e6:,.1f} MB"   # noqa: E731
        lines = []
        for action, what in (("get", f"from {self.there.label}"), ("put", f"to {self.there.label}"),
                             ("merge", "merged on both"), ("drop_here", "deleted there, delete here"),
                             ("drop_there", "deleted here, delete there")):
            s = [x for x in steps if x.action == action]
            if s:
                kinds = ", ".join(f"{sum(x.kind == k for x in s)} {k}" for k in ("decision", "rows", "data", "derived")
                                  if any(x.kind == k for x in s))
                lines.append(f"  {len(s):5d} {what} ({kinds}; {mb(sum(x.size for x in s))})")
        kept = [x.rel for x in steps if x.retire]
        if kept:
            lines.append(f"  changed on both machines, the newer wins, the other kept in {STATE}/old: " +
                         ", ".join(kept[:5]) + (" ..." if len(kept) > 5 else ""))
        return "\n".join(lines) + note

    def run(self, steps: list[Step], progress: bool = True) -> dict:
        from tqdm import tqdm
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        merge = Merge(self.home_here)
        L, R = self.seen.get("here") or self.here.files(), self.seen.get("there") or self.there.files()
        rec = {rel: {"l": L[rel], "r": R[rel]} for rel in set(L) & set(R) if PeerRules.kind(rel) != "skip"}
        for rel in self.held | self.one_way:       # not this time (or only one side): as the last sync left them
            rec.pop(rel, None)
            if rel in self.record:
                rec[rel] = self.record[rel]
        moved = {"get": [], "put": []}
        self.changed = {"here": set(), "there": set()}
        copies = [s for s in steps if s.action in ("get", "put")]
        with tqdm(total=sum(s.size for s in copies), unit="B", unit_scale=True,
                  disable=not progress, desc=f"sync {self.peer}") as bar:
            for s in steps:                        # merges and deletions first (few, small), then the copies
                if s.action == "merge":
                    stats = self.merge(s, merge, stamp, L, R)
                    if stats:
                        rec[s.rel] = stats
                elif s.action in ("drop_here", "drop_there"):
                    side = self.here if s.action == "drop_here" else self.there
                    self.changed[s.action.removeprefix("drop_")].add(s.kind)
                    side.retire(s.rel, stamp) if s.kind in ("data", "decision", "rows", "index") else side.remove(s.rel)
                    rec.pop(s.rel, None)
            self.transfer(copies, L, R, stamp, bar)
        for s in copies:                           # a copy has its source's size and time (no asking again)
            st = (R if s.action == "get" else L)[s.rel]
            rec[s.rel] = {"l": st, "r": st}
            moved[s.action].append(s)
            self.changed["here" if s.action == "get" else "there"].add(s.kind)
        self.clashes = merge.clashes
        for rel in rec:                            # in step now: what both have is the next merge's base
            if PeerRules.kind(rel) == "decision" and not self.base(rel).exists():
                self.base(rel).parent.mkdir(parents=True, exist_ok=True)
                self.base(rel).write_bytes(self.here.read(rel))
        self.state.mkdir(parents=True, exist_ok=True)
        self.record_file.write_text(json.dumps(rec))
        return {"from": moved["get"], "to": moved["put"], "clashes": merge.clashes}

    WORKERS = 4                                # connections at once for the copies (a paired app; SSH: one)

    def transfer(self, steps: list[Step], L: dict, R: dict, stamp: str, bar) -> None:
        """the copies - over WORKERS connections at once when both sides can open more (latency, not bandwidth,
        is what a few hundred small files over Wi-Fi cost)"""
        from concurrent.futures import ThreadPoolExecutor
        if len(steps) < 2 or not (self.here.FORKABLE and self.there.FORKABLE):
            for s in steps:
                self.copy(s, self.here, self.there, L, R, stamp, bar)
            return
        n = min(self.WORKERS, len(steps))
        sides = [(self.here.fork(), self.there.fork()) for _ in range(n)]
        try:
            with ThreadPoolExecutor(n) as pool:
                list(pool.map(lambda k: [self.copy(s, *sides[k], L, R, stamp, bar) for s in steps[k::n]], range(n)))
        finally:
            for side in {id(x): x for pair in sides for x in pair}.values():
                if side is not self.here and side is not self.there:
                    side.close()

    @staticmethod
    def copy(s: Step, here, there, L: dict, R: dict, stamp: str, bar) -> None:
        from tqdm.utils import CallbackIOWrapper
        src, dst = (there, here) if s.action == "get" else (here, there)
        if s.retire:
            dst.retire(s.rel, stamp)
        with src.open(s.rel) as f:
            dst.write(s.rel, CallbackIOWrapper(bar.update, f, "read"), (R if s.action == "get" else L)[s.rel][1])

    def hand_over(self, log=print, wait: bool = True, wait_minutes: float = 15, poll: float = 5) -> dict | None:
        """After a run with a paired app: tell it what changed there. If it's home it takes that in (new rides,
        your changes) and passes it on to its other computers; with `wait`, this machine then fetches what came of
        it (analysis, plan) - the second run's result. Without, home sends it here too when it's done."""
        changed = self.changed["there"] - {"derived"}
        if not isinstance(self.there, HttpSide) or self.home_here or not changed:
            return None
        if not self.there.done(changed, wait) or not wait:
            return None
        log(f"{self.there.label} is taking in what it got ...")
        t0 = time.time()
        while self.there.busy():
            if time.time() - t0 > wait_minutes * 60:
                log(f"{self.there.label} is still busy: its results come with the next sync.")
                return None
            time.sleep(poll)
        return self.run(self.plan(), progress=False)

    def base(self, rel: str) -> Path:
        return self.state / "base" / self.peer / rel

    def merge(self, s: Step, merge: Merge, stamp: str = "", L: dict | None = None, R: dict | None = None) -> dict | None:
        """both sides' changes in one; a side's copy that changes is first kept in .goodpush-sync/old/<stamp>. The
        file's record afterwards ({"l", "r": (size, mtime)}; None: keep the last one - a one-way sync)."""
        b = self.base(s.rel).read_bytes() if self.base(s.rel).exists() else None
        l, r = self.here.read(s.rel), self.there.read(s.rel)
        if b is None and s.kind == "decision":     # first sync: can't tell added from deleted - the home's whole
            b = r if self.home_here else l
        if s.rel == CONFIG:
            nl, nr = merge.settings(b, l, r)
        elif s.kind == "rows":
            nl = nr = merge.rows(b, l, r, PeerRules.ROWS[PurePosixPath(s.rel).name])
        elif s.kind == "index":                    # both sides' entries (an index only grows)
            nl = nr = merge.json(None, l, r, s.rel)
        else:
            nl = nr = merge.json(b, l, r, s.rel)
        now = int(time.time())
        receive = {"get": "here", "send": "there"}.get(self.direction) if s.rel in self.one_way else None
        stats = {"l": (L or {}).get(s.rel) or (len(l), now), "r": (R or {}).get(s.rel) or (len(r), now)}
        for where, side, old, new in (("here", self.here, l, nl), ("there", self.there, r, nr)):
            if receive and where != receive:       # a one-way sync: the sending side's list stays as it was
                continue
            if new != old:
                side.retire(s.rel, stamp or time.strftime("%Y-%m-%d_%H%M%S"))
                side.write(s.rel, new, now)
                self.changed[where].add(s.kind)
                stats["l" if where == "here" else "r"] = (len(new), now)
        if receive:                                # the base stays the last both-ways one
            return None
        self.base(s.rel).parent.mkdir(parents=True, exist_ok=True)
        self.base(s.rel).write_bytes(nl)
        return stats
