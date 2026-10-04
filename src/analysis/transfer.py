"""Phase 11 representation transfer: each run's best feature table on both backends.

Separates "found better features" from "fits the same features better".
    uv run python -m src.analysis.transfer
"""

from __future__ import annotations

import json

import pandas as pd

from src.analysis.runs import best_experiments
from src.config import RESULTS_DIR, ROOT
from src.evaluation.evaluator import evaluate

OUT = ROOT / "experiments" / "analysis" / "transfer.csv"
BACKEND = {"b": "tabpfn", "a_prime": "xgb"}


def main() -> None:
    rows = []
    for _, r in best_experiments().iterrows():
        ft = pd.read_parquet(RESULTS_DIR / r["arm"] / str(r["seed"]) / r["experiment_id"] / "features.parquet")
        own = BACKEND[r["arm"]]
        other = "xgb" if own == "tabpfn" else "tabpfn"
        res = evaluate(ft, other)
        assert res["status"] == "ok", res
        row = {"arm": r["arm"], "seed": r["seed"], "experiment_id": r["experiment_id"], "n_features": r["n_features"],
               f"mae_{own}": r["mae"], f"mae_{other}": res["mae"]}
        rows.append(row)
        print(json.dumps(row))
    df = pd.DataFrame(rows)[["arm", "seed", "experiment_id", "n_features", "mae_tabpfn", "mae_xgb"]]
    df["tabpfn_minus_xgb"] = (df["mae_tabpfn"] - df["mae_xgb"]).round(3)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print("\nmean by arm of origin:\n", df.groupby("arm")[["mae_tabpfn", "mae_xgb", "tabpfn_minus_xgb"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
