"""goodpush - personal cycling training analysis."""
import hashlib
from pathlib import Path

try:
    from ._version import __version__        # hatch-vcs writes it from the git tag on install / build
except ImportError:                          # a source tree never installed
    __version__ = "0.0.0"


class CodeStamp:
    """A folder's code by its content (line endings aside), not its file times: touching a file, a checkout or a
    hook rewriting line endings is no new code (2026-11-18: each such touch re-made the plan at app start). The
    hash is worked out again only when a file's time or size changes (the app asks on every job poll)."""
    seen: dict[Path, tuple] = {}             # folder -> (its files' times and sizes, their hash)

    @classmethod
    def of(cls, folder: Path) -> str:
        files = sorted(folder.rglob("*.py"))
        stats = tuple((f, s.st_mtime_ns, s.st_size) for f in files for s in [f.stat()])
        if cls.seen.get(folder, (None,))[0] != stats:
            h = hashlib.sha1()
            for f in files:
                h.update(f.relative_to(folder).as_posix().encode() + b"\0")
                h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
            cls.seen[folder] = (stats, h.hexdigest()[:16])
        return cls.seen[folder][1]


def code_on_disk(folder: Path | None = None) -> str:
    """goodpush's code (or one of its folders') as it is on disk now (`CodeStamp`). A process that loaded other
    code than this needs a restart to run it."""
    return CodeStamp.of(Path(folder or Path(__file__).parent))
