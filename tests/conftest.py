import os
import tempfile
from pathlib import Path

# compiled JAX code kept between test runs (each test's own processed/jax_cache was compiled anew and thrown away:
# ~1300 compiles in the slowest test). Set before jax is imported; PhysioRun.keep_compiled leaves it alone.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(tempfile.gettempdir()) / "goodpush-tests-jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")

import pytest  # noqa: E402

from goodpush.athletes import Athletes  # noqa: E402
from goodpush.folders import Folders  # noqa: E402

# for the whole session, not per test: a thread that outlives its test (an app server's warm-up or sync) must not
# reach this computer's own folders either, once a test's monkeypatch is undone
Folders.migrated, Folders.LEGACY = True, None
os.environ[Folders.CONFIG_ENV] = tempfile.mkdtemp(prefix="goodpush-tests-config-")


@pytest.fixture(autouse=True)
def no_real_athletes(tmp_path_factory, monkeypatch):
    """the tests never see this machine's athletes (athletes.toml in its machine folder) or its credentials / paired
    machines, and never move its folders (Folders.migrate)."""
    monkeypatch.setenv(Folders.CONFIG_ENV, str(tmp_path_factory.mktemp("config")))
    monkeypatch.setattr(Athletes, "HOME", tmp_path_factory.mktemp("machine"))
    monkeypatch.setenv("GOODPUSH_SECRETS", str(tmp_path_factory.mktemp("secrets")))
