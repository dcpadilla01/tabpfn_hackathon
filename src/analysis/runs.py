"""Load MVP run logs (Env-1: arms b, a_prime; seeds 0-2) into tidy tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR

ARMS = ("b", "a_prime")
SEEDS = (0, 1, 2)
ARM_LABEL = {"b": "B (TabPFN)", "a_prime": "A′ (XGBoost)"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def experiments() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            for r in _jsonl(RESULTS_DIR / arm / str(seed) / "experiments.jsonl"):
                rows.append({**r, "arm": arm, "seed": seed})
    df = pd.DataFrame(rows)
    df["idx"] = df["experiment_id"].str[1:].astype(int)
    return df.sort_values(["arm", "seed", "idx"]).reset_index(drop=True)


def calls() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            for c in _jsonl(RESULTS_DIR / arm / str(seed) / "calls.jsonl"):
                rows.append({**c, "arm": arm, "seed": seed})
    return pd.DataFrame(rows)


def transcript(arm: str, seed: int) -> list[dict]:
    return _jsonl(RESULTS_DIR / arm / str(seed) / "transcript.jsonl")


def best_experiments(exp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Final candidate per run, chosen on validation MAE only (E000 excluded)."""
    exp = experiments() if exp is None else exp
    ok = exp[(exp["status"] == "ok") & (exp["experiment_id"] != "E000")]
    return ok.loc[ok.groupby(["arm", "seed"])["mae"].idxmin()].reset_index(drop=True)


def predictions(arm: str, seed: int, experiment_id: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / arm / str(seed) / experiment_id / "predictions.parquet")
