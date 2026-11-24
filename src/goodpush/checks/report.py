"""Where the checks keep their results: processed/ladder/<name>.json (the folder keeps the model ladder's name,
so the results already there stay valid), one section per check, replaced on its own."""
from __future__ import annotations

import json
from pathlib import Path


class CheckReport:
    """processed/ladder/<name>.json (e.g. physio_calibrated: its backtest; physio_coverage.json is the coverage
    check's own file)."""
    FOLDER = "ladder"

    def __init__(self, processed: Path):
        self.folder = Path(processed) / self.FOLDER

    def read(self, name: str) -> dict:
        try:
            return json.loads((self.folder / f"{name}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save(self, name: str, section: str, value: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        out = self.read(name) | {section: value}
        (self.folder / f"{name}.json").write_text(json.dumps(out, ensure_ascii=True, default=float), encoding="utf-8")
