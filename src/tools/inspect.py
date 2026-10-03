"""inspect(table, op, column=None, as_of_day=None, n=5) — quick looks at data through the accessor.

Runs in the harness (no code execution). Temporal tables are viewed as of `as_of_day`
(default and max: the research horizon). Output is plain text, truncated.
"""

from __future__ import annotations

import pandas as pd

from src.data.accessor import AGENT_NAMES, AsOf
from src.data.splits import research_visible_day

OPS = ("columns", "shape", "head", "describe", "nunique", "value_counts")
MAX_CHARS = 4000


def inspect(table: str, op: str, column: str | None = None, as_of_day: int | None = None, n: int = 5) -> str:
    if table not in AGENT_NAMES:
        raise ValueError(f"unknown table {table!r}; tables: {', '.join(AGENT_NAMES)}")
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; ops: {', '.join(OPS)}")
    horizon = research_visible_day()
    day = horizon if as_of_day is None else int(as_of_day)
    if day > horizon:
        raise ValueError(f"as_of_day must be <= {horizon} (research horizon)")
    df: pd.DataFrame = getattr(AsOf(day), table)
    if column is not None and column not in df.columns:
        raise ValueError(f"{table} has no column {column!r}; columns: {', '.join(df.columns)}")
    n = max(1, min(int(n), 50))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        if op == "columns":
            out = "\n".join(f"{c}: {t}" for c, t in df.dtypes.items())
        elif op == "shape":
            out = f"{df.shape[0]:,} rows x {df.shape[1]} columns (as of day {day})"
        elif op == "head":
            out = df.head(n).to_string()
        elif op == "describe":
            out = (df[[column]] if column else df).describe(include="all").T.to_string()
        elif op == "nunique":
            out = (df[[column]] if column else df).nunique().to_string()
        else:
            if column is None:
                raise ValueError("value_counts needs a column")
            out = df[column].value_counts(dropna=False).head(n).to_string()
    return out[:MAX_CHARS] + ("\n...[truncated]" if len(out) > MAX_CHARS else "")
