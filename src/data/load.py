"""Load registry tables from data/interim/ (written by scripts/convert_raw.py).

Harness-internal: the agent never calls this. Agents reach temporal tables only
through src/data/accessor.py.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import INTERIM_DIR
from src.data.schema import TABLES


@lru_cache(maxsize=None)
def _load(name: str, columns: tuple[str, ...] | None) -> pd.DataFrame:
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; known: {sorted(TABLES)}")
    path = INTERIM_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `uv run python scripts/convert_raw.py`")
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def load_table(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a copy so callers cannot mutate the cache."""
    return _load(name, tuple(columns) if columns else None).copy()
