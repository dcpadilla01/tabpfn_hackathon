"""Env-2 Phase 12 for Arm A (pre-declared in docs/decisions.md, 2026-10-05 05:51).

Each A run's final candidate (chosen on validation only) is re-run from its own code with test rows switched on
(same replay as scripts/evaluate_test.py). Rules, applied literally:
  1. the replayed validation predictions must reproduce the logged ones (rtol = atol = 1e-6), else
     "not reproduced" (difference reported, no test score);
  2. the replayed predictions must cover every test row with a finite value, else "not replayable";
  3. otherwise score test once. Regime: train-only by construction (the agent's own fit on train labels).
No workarounds, no hand edits.

    uv run python scripts/evaluate_test_arm_a.py [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_test import OUT, replay  # noqa: E402
from src.analysis.runs import best_experiments  # noqa: E402
from src.config import RESULTS_DIR  # noqa: E402
from src.data.targets import KEYS, TARGET, load_targets  # noqa: E402
from src.evaluation.metrics import mae, r2  # noqa: E402


def evaluate_a(seed: int, experiment_id: str, val_mae: float) -> dict:
    print(f"== a/{seed} {experiment_id} (val MAE {val_mae})", flush=True)
    path, meta = replay("a", seed, experiment_id)
    res = {"arm": "a", "seed": seed, "experiment_id": experiment_id, "val_mae": val_mae, "replay": meta,
           "regime": "train-only (agent's own fit)"}
    if not path.exists():
        res["status"] = "not replayable: final predictions file not produced"
        return res
    p = pd.read_parquet(path)
    if not set(KEYS + ["prediction"]) <= set(p.columns):
        res["status"] = f"not replayable: predictions file lacks {sorted(set(KEYS + ['prediction']) - set(p.columns))}"
        return res
    t = load_targets()
    logged = pd.read_parquet(RESULTS_DIR / "a" / str(seed) / experiment_id / "predictions.parquet")
    m = logged.merge(p[KEYS + ["prediction"]], on=KEYS, how="left", suffixes=("_logged", "_replay"))
    a, b = m["prediction_logged"].to_numpy(float), m["prediction_replay"].to_numpy(float)
    found = int(np.isfinite(b).sum())
    max_diff = float(np.nanmax(np.abs(a - b))) if found else None
    res["reproduction"] = {"validation_rows": len(m), "rows_found": found, "max_abs_diff": max_diff,
                           "reproduced": bool(found == len(m) and np.allclose(a, b, rtol=1e-6, atol=1e-6))}
    if not res["reproduction"]["reproduced"]:
        res["status"] = "not reproduced"
        return res
    test = t[t["split"] == "test"][KEYS + [TARGET]]
    mt = test.merge(p[KEYS + ["prediction"]], on=KEYS, how="left")
    n_ok = int(np.isfinite(mt["prediction"].to_numpy(float)).sum())
    res["test_rows_predicted"] = n_ok
    if n_ok != len(test):
        res["status"] = f"not replayable: {n_ok}/{len(test)} test rows predicted"
        return res
    res.update({"status": "ok", "test_mae_fit_train": round(mae(mt[TARGET], mt["prediction"]), 4),
                "test_r2_fit_train": round(r2(mt[TARGET], mt["prediction"]), 4)})
    res["val_to_test_degradation_same_fit"] = round(res["test_mae_fit_train"] - val_mae, 4)
    mt[KEYS + ["prediction"]].to_parquet(OUT / f"a_{seed}_test_predictions_fit_train.parquet", index=False)
    print(f"   test MAE (agent's own train-only fit) {res['test_mae_fit_train']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    best = best_experiments()
    best = best[best["arm"] == "a"]
    if args.seed is not None:
        best = best[best["seed"] == args.seed]
    for _, r in best.iterrows():
        res = evaluate_a(int(r["seed"]), r["experiment_id"], float(r["mae"]))
        (OUT / f"a_{int(r['seed'])}.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != "replay"}, default=str), flush=True)


if __name__ == "__main__":
    main()
