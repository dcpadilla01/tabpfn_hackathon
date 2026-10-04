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
    e000 = {r["backend"]: r for r in json.loads((DIR / "e000.json").read_text())}
    backend = {"b": "tabpfn", "a_prime": "xgb"}
    regimes = {"train+validation (primary)": ("test_mae_fit_train_val", "_test_predictions.parquet"),
               "train-only": ("test_mae_fit_train", "_test_predictions_fit_train.parquet")}
    table, boots = [], {}
    for regime, (col, suffix) in regimes.items():
        errs = {}
        for arm in ARMS:
            for seed in SEEDS:
                p = pd.read_parquet(DIR / f"{arm}_{seed}{suffix}")
                m = test.merge(p, on=KEYS, how="left")
                assert m["prediction"].notna().all()
                errs[(arm, seed)] = (m["prediction"] - m[TARGET]).abs().to_numpy()
        for arm in ARMS:
            g = df[df["arm"] == arm]
            e = e000[backend[arm]]
            e_test = e[col]
            table.append({"regime": regime, "arm": arm, "val_mae": round(g["val_mae"].mean(), 2),
                          "test_mae": round(g[col].mean(), 2), "delta": round(g[col].mean() - g["val_mae"].mean(), 2),
                          "E000_val": e["val_mae"], "E000_test": e_test, "E000_delta": round(e_test - e["val_mae"], 2)})
        d = pd.Series(np.mean([errs[("b", s)] for s in SEEDS], 0) - np.mean([errs[("a_prime", s)] for s in SEEDS], 0))
        boots[regime] = {"arm_mean_abs_error_test": cluster_bootstrap(d, test["household_key"]),
                         "pairs": {f"b{sb}-a'{sa}": cluster_bootstrap(pd.Series(errs[("b", sb)] - errs[("a_prime", sa)]),
                                                                      test["household_key"]) for sb in SEEDS for sa in SEEDS}}
    tbl = pd.DataFrame(table)
    tbl.to_csv(DIR.parent / "frozen_test_2x2.csv", index=False)
    (DIR.parent / "bootstrap_test.json").write_text(json.dumps(boots, indent=2))
    print("\n2x2 (means over 3 runs; delta = test − validation):\n", tbl.to_string(index=False))
    for regime, bt in boots.items():
        a = bt["arm_mean_abs_error_test"]
        worst = max(v["ci95"][1] for v in bt["pairs"].values())
        print(f"\n{regime}: B − A′ = {a['mean']:+.3f}  CI95 [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]; "
              f"pairs with CI below 0: {sum(v['ci95'][1] < 0 for v in bt['pairs'].values())}/9 (max upper {worst:+.3f})")


if __name__ == "__main__":
    main()
