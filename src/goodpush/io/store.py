"""Table persistence: parquet when pyarrow is installed, else gzipped CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # pragma: no cover - depends on environment
    import pyarrow  # noqa: F401
    HAVE_PARQUET = True
except Exception:  # pragma: no cover
    HAVE_PARQUET = False


def _path(base: Path, name: str) -> Path:
    return base / (f"{name}.parquet" if HAVE_PARQUET else f"{name}.csv.gz")


def save(df: pd.DataFrame, base: Path, name: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    p = _path(base, name)
    if HAVE_PARQUET:
        df.to_parquet(p)
    else:
        df.to_csv(p, index=True)
    return p


def load(base: Path, name: str) -> pd.DataFrame | None:
    pq, csv = base / f"{name}.parquet", base / f"{name}.csv.gz"
    if HAVE_PARQUET and pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        df = pd.read_csv(csv, index_col=0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def load_tables(processed: Path) -> dict:
    """The stored analysis tables (daily, activities): enough to rebuild the plan or learn markers
    without a full analysis."""
    daily, acts = load(processed, "daily"), load(processed, "activities")
    if daily is None or acts is None:
        raise FileNotFoundError("no analysis yet - run `goodpush run` first")
    daily.index = pd.to_datetime(daily.index)
    acts["date"] = pd.to_datetime(acts["date"])
    for c in ("name", "description", "private_note", "workout", "sport"):
        if c in acts:
            acts[c] = acts[c].where(acts[c].notna(), "").astype(str).replace("nan", "")
    return {"daily": daily, "acts": acts}
