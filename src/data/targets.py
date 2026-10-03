"""Canonical target table: household_key × snapshot_day → future_spend_4w.

    future_spend_4w = sum(sales_value) over days in (snapshot_day, snapshot_day + horizon]

Eligibility is a property of the row, identical in every split:
  * the future window is complete: snapshot_day + horizon <= last observed day
  * the household's first observed transaction is <= snapshot_day - min_history_days
Eligible rows with no purchases in the window are kept with target 0.
"""

from __future__ import annotations

import pandas as pd

from src.config import load_config
from src.data.load import load_table
from src.data.schema import DAY_COL, HOUSEHOLD_KEY, SPEND_COL

TARGET = "future_spend_4w"
KEYS = [HOUSEHOLD_KEY, "snapshot_day"]


def snapshot_days(last_day: int, horizon: int, cadence: int, earliest: int) -> list[int]:
    """Grid anchored at the last day with a complete horizon, stepping back by cadence."""
    days = list(range(last_day - horizon, earliest - 1, -cadence))
    return sorted(days)


def build_targets() -> pd.DataFrame:
    cfg = load_config()["task"]
    horizon, cadence, min_hist = cfg["horizon_days"], cfg["cadence_days"], cfg["min_history_days"]
    tx = load_table("transaction_data", [HOUSEHOLD_KEY, DAY_COL, SPEND_COL])
    first_day = tx.groupby(HOUSEHOLD_KEY)[DAY_COL].min()
    last_day = int(tx[DAY_COL].max())
    grid = snapshot_days(last_day, horizon, cadence, earliest=int(first_day.min()) + min_hist)

    daily = tx.groupby([HOUSEHOLD_KEY, DAY_COL])[SPEND_COL].sum()
    rows = []
    for s in grid:
        eligible = first_day.index[first_day.values <= s - min_hist]
        window = daily[(daily.index.get_level_values(DAY_COL) > s) & (daily.index.get_level_values(DAY_COL) <= s + horizon)]
        spend = window.groupby(level=HOUSEHOLD_KEY).sum().reindex(eligible, fill_value=0.0)
        rows.append(pd.DataFrame({HOUSEHOLD_KEY: eligible, "snapshot_day": s, TARGET: spend.values}))
    out = pd.concat(rows, ignore_index=True)
    out[HOUSEHOLD_KEY] = out[HOUSEHOLD_KEY].astype("int32")
    out["snapshot_day"] = out["snapshot_day"].astype("int16")
    out[TARGET] = out[TARGET].round(2)  # cents; removes float-summation-order noise
    return out.sort_values(KEYS, ignore_index=True)


def load_targets() -> pd.DataFrame:
    """Harness-only: the persisted table with labels for every split (incl. test)."""
    from src.config import PROCESSED_DIR

    path = PROCESSED_DIR / "targets.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `uv run python scripts/build_targets.py`")
    return pd.read_parquet(path)
