"""Phase 0 smoke test.

    uv run python scripts/smoke_test_tabpfn.py --backend local   # few hundred rows, CPU/MPS
    uv run python scripts/smoke_test_tabpfn.py --backend api     # + row-budget timing at planned size

Prints one regression prediction per backend. With --backend api it also
times a fit+predict at the planned train size and records credits used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR, load_config, package_versions  # noqa: E402
from src.evaluation.backends import fit_predict_tabpfn  # noqa: E402


def synthetic(n_rows: int, n_features: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"x{i}" for i in range(n_features)])
    # skewed, zero-inflated target, roughly like 4-week spend
    y = np.exp(1 + 0.5 * X["x0"] - 0.3 * X["x1"] + 0.2 * rng.normal(size=n_rows)) * 50
    y[rng.random(n_rows) < 0.1] = 0.0
    return X, y


def run(backend: str, n_train: int, n_eval: int, n_features: int, seed: int) -> dict:
    X, y = synthetic(n_train + n_eval, n_features, seed)
    Xtr, ytr, Xev, yev = X.iloc[:n_train], y[:n_train], X.iloc[n_train:], y[n_train:]
    usage_before = _usage() if backend == "api" else None
    t0 = time.perf_counter()
    preds, meta = fit_predict_tabpfn(Xtr, ytr, Xev, backend=backend)
    wall = time.perf_counter() - t0
    usage_after = _usage() if backend == "api" else None
    return {
        "backend": backend,
        "n_train": n_train,
        "n_eval": n_eval,
        "n_features": n_features,
        "seed": seed,
        "wall_clock_seconds": round(wall, 2),
        "mae": float(np.mean(np.abs(preds - yev))),
        "first_prediction": float(preds[0]),
        "usage_before": usage_before,
        "usage_after": usage_after,
        "versions": package_versions(),
        "meta": meta,
    }


def _usage() -> str:
    import tabpfn_client

    return tabpfn_client.get_api_usage()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["api", "local"], required=True)
    ap.add_argument("--budget-rows", type=int, default=35_000, help="planned train size for the API row-budget check")
    ap.add_argument("--budget-eval-rows", type=int, default=9_000)
    ap.add_argument("--features", type=int, default=30)
    ap.add_argument("--skip-budget", action="store_true")
    args = ap.parse_args()
    seed = load_config()["seed"]

    out = {"smoke": run(args.backend, 300, 50, 10, seed)}
    print(json.dumps(out["smoke"], indent=2, default=str))

    if args.backend == "api" and not args.skip_budget:
        out["row_budget"] = run("api", args.budget_rows, args.budget_eval_rows, args.features, seed)
        print(json.dumps(out["row_budget"], indent=2, default=str))
        print(f"\nROW-BUDGET: {args.budget_rows} train rows x {args.features} features "
              f"-> {out['row_budget']['wall_clock_seconds']} s per fit+predict")

    path = RESULTS_DIR / "phase0" / f"smoke_{args.backend}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"saved {path.relative_to(RESULTS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
