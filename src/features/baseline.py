"""E000 — weak root baseline: household demographics + snapshot calendar info only.

Deliberately excludes recent spend, recency, frequency, trends, category, promotion,
campaign and rolling-window features: those are the research search space.
"""

from __future__ import annotations

import pandas as pd

from src.data.accessor import AsOf, build_features
from src.data.schema import HOUSEHOLD_KEY, week_of_day

DEMOGRAPHIC_COLUMNS = [
    "classification_1", "classification_2", "classification_3", "classification_4",
    "classification_5", "homeowner_desc", "kid_category_desc",
]


def baseline_fn(view: AsOf, snapshot_day: int) -> pd.DataFrame:
    demo = view.demographics.set_index(HOUSEHOLD_KEY)[DEMOGRAPHIC_COLUMNS]
    out = demo.reindex(view.households)
    out["has_demographics"] = out.index.isin(demo.index)
    out["snapshot_day_index"] = snapshot_day
    out["week_of_year"] = (week_of_day(snapshot_day) - 1) % 52 + 1  # no calendar dates; 52-week cycle
    return out


def build_baseline_features(splits: tuple[str, ...] = ("train", "validation")) -> pd.DataFrame:
    return build_features(baseline_fn, splits=splits)
