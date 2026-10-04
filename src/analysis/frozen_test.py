"""Phase 12 summary: frozen test per run + paired household-clustered bootstrap on test.

    uv run python -m src.analysis.frozen_test
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.analysis.compare_runs import cluster_bootstrap
from src.analysis.runs import ARMS, SEEDS
from src.config import ROOT
from src.data.targets import KEYS, TARGET, load_targets

DIR = ROOT / "experiments" / "analysis" / "frozen_test"


def main() -> None:
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            r = json.loads((DIR / f"{arm}_{seed}.json").read_text())
            rows.append({k: r.get(k) for k in ("arm", "seed", "experiment_id", "val_mae", "test_mae_fit_train_val",
                                               "test_r2_fit_train_val", "test_mae_fit_train", "val_to_test_degradation_same_fit",
                                               "reproduction_kind")})
    df = pd.DataFrame(rows)
    df.to_csv(DIR.parent / "frozen_test.csv", index=False)
    print(df.to_string(index=False))
    agg = df.groupby("arm")[["val_mae", "test_mae_fit_train_val", "test_mae_fit_train", "val_to_test_degradation_same_fit"]]
    print("\nmean (sd), all six runs:\n", agg.agg(["mean", "std"]).round(3).T.to_string())
    exact = df[df["reproduction_kind"] == "exact"]
    print("\nmean, exact reproductions only:\n", exact.groupby("arm")[["val_mae", "test_mae_fit_train_val"]].mean().round(3).to_string())

    t = load_targets()
    test = t[t["split"] == "test"][KEYS + [TARGET]].reset_index(drop=True)
    errs = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = pd.read_parquet(DIR / f"{arm}_{seed}_test_predictions.parquet")
            m = test.merge(p, on=KEYS, how="left")
            assert m["prediction"].notna().all()
            errs[(arm, seed)] = (m["prediction"] - m[TARGET]).abs().to_numpy()
    d = pd.Series(np.mean([errs[("b", s)] for s in SEEDS], 0) - np.mean([errs[("a_prime", s)] for s in SEEDS], 0))
    boot = {"arm_mean_abs_error_test": cluster_bootstrap(d, test["household_key"]),
            "pairs": {f"b{sb}-a'{sa}": cluster_bootstrap(pd.Series(errs[("b", sb)] - errs[("a_prime", sa)]), test["household_key"])
                      for sb in SEEDS for sa in SEEDS}}
    (DIR.parent / "bootstrap_test.json").write_text(json.dumps(boot, indent=2))
    print("\ntest paired bootstrap (B − A′):", boot["arm_mean_abs_error_test"])
    print("pairs CI upper bounds:", {k: round(v["ci95"][1], 3) for k, v in boot["pairs"].items()})


if __name__ == "__main__":
    main()
