"""Env-2 Phase 12 summary over the runs that produced a test score (n varies by arm; stated).

Cross-arm primary: train-only (pre-declared 2026-10-05 05:51). Replication: train+validation for A′ and B.
Arm-level paired, household-clustered bootstrap over the available runs of each arm.

    uv run python -m src.analysis.frozen_test_env2
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from src.analysis.compare_runs import cluster_bootstrap
from src.config import ANALYSIS_DIR
from src.data.targets import KEYS, TARGET, load_targets

DIR = ANALYSIS_DIR / "frozen_test"
REGIMES = {"train-only (primary)": ("test_mae_fit_train", "_test_predictions_fit_train.parquet", ("b", "a_prime", "a")),
           "train+validation (replication)": ("test_mae_fit_train_val", "_test_predictions.parquet", ("b", "a_prime"))}


def main() -> None:
    runs = {}
    for f in sorted(DIR.glob("*_[012].json")):
        r = json.loads(f.read_text())
        runs[(r["arm"], r["seed"])] = r
    t = load_targets(); test = t[t["split"] == "test"][KEYS + [TARGET]].reset_index(drop=True)
    table, boots = [], {}
    for regime, (col, suffix, arms) in REGIMES.items():
        errs = {}
        for arm in arms:
            ok = [s for s in (0, 1, 2) if runs.get((arm, s), {}).get(col) is not None]
            for s in ok:
                p = pd.read_parquet(DIR / f"{arm}_{s}{suffix}")
                m = test.merge(p, on=KEYS, how="left"); assert m["prediction"].notna().all()
                errs[(arm, s)] = (m["prediction"] - m[TARGET]).abs().to_numpy()
            vals = [runs[(arm, s)][col] for s in ok]
            table.append({"regime": regime, "arm": arm, "n_runs": len(ok), "seeds": ok,
                          "val_mae": round(np.mean([runs[(arm, s)]["val_mae"] for s in ok]), 2),
                          "test_mae": round(np.mean(vals), 2), "delta": round(np.mean(vals) - np.mean([runs[(arm, s)]["val_mae"] for s in ok]), 2),
                          "kinds": [runs[(arm, s)].get("reproduction_kind", "exact") for s in ok]})
        mean_err = {a: np.mean([e for (aa, _), e in errs.items() if aa == a], axis=0) for a in arms}
        boots[regime] = {f"{x} − {y}": cluster_bootstrap(pd.Series(mean_err[x] - mean_err[y]), test["household_key"])
                         for x, y in combinations(arms, 2)}
    tbl = pd.DataFrame(table)
    tbl.to_csv(ANALYSIS_DIR / "frozen_test_env2.csv", index=False)
    (ANALYSIS_DIR / "bootstrap_test_env2.json").write_text(json.dumps(boots, indent=2))
    print(tbl.to_string(index=False))
    for regime, b in boots.items():
        print(f"\n{regime}:")
        for k, v in b.items():
            print(f"  {k}: {v['mean']:+.3f}  CI95 [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
