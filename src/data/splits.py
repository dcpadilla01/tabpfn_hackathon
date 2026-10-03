"""Temporal split by snapshot day — same boundaries for every household. Never random."""

from __future__ import annotations

import pandas as pd

from src.config import load_config

SPLITS = ("train", "validation", "test")


def split_boundaries(snapshot_days: list[int], fractions: list[float] | None = None) -> dict[str, list[int]]:
    fr = fractions or load_config()["task"]["split_fractions"]
    days = sorted(snapshot_days)
    n = len(days)
    n_train = round(fr[0] * n)
    n_val = round(fr[1] * n)
    return {
        "train": days[:n_train],
        "validation": days[n_train:n_train + n_val],
        "test": days[n_train + n_val:],
    }


def assign_split(targets: pd.DataFrame) -> pd.DataFrame:
    bounds = split_boundaries(targets["snapshot_day"].unique().tolist())
    lookup = {d: name for name, days in bounds.items() for d in days}
    out = targets.copy()
    out["split"] = pd.Categorical(out["snapshot_day"].map(lookup), categories=list(SPLITS))
    return out
