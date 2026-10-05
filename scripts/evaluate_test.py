"""Phase 12: frozen test evaluation of each run's final candidate (chosen on validation only).

For each run: replay, in order, every executed cell that called save_table up to the
final experiment (only saved tables persist between cells), with build_features in
harness-only frozen-test mode so tables also carry test-snapshot rows. Verify the
replayed train+validation rows reproduce the logged features.parquet, then fit once:
  primary:  fit train+validation → score test
  also:     fit train only       → score test   (clean validation→test degradation)

    uv run python scripts/evaluate_test.py                 # all Env-1 runs
    uv run python scripts/evaluate_test.py --arm b --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.runs import best_experiments, transcript  # noqa: E402
from src.config import ANALYSIS_DIR, RESULTS_DIR, ROOT  # noqa: E402
from src.data.targets import KEYS  # noqa: E402
from src.evaluation.evaluator import evaluate_frozen_test  # noqa: E402
from src.researcher.arms import arm_config  # noqa: E402
from src.tools.run_python import run_python  # noqa: E402

OUT = ANALYSIS_DIR / "frozen_test"
BACKEND = {"b": "tabpfn", "a_prime": "xgb"}


def strip_asserts(code: str) -> str:
    """Remove `assert` statements: agents assert research-time row counts (e.g. == 36426), which
    test rows break. Asserts compute nothing, so feature values are unchanged; the reproduction
    check against the logged table guards this."""
    import ast

    class _Strip(ast.NodeTransformer):
        def visit_Assert(self, node):
            return ast.Pass()

    try:
        return ast.unparse(_Strip().visit(ast.parse(code)))
    except SyntaxError:
        return code


def replay(arm: str, seed: int, experiment_id: str) -> tuple[Path, dict]:
    T = transcript(arm, seed)
    exp_call = [t for t in T if t["event"] == "tool_call" and t["tool"] in ("experiment", "score")
                and t["experiment_id"] == experiment_id][-1]
    cells = [t for t in T if t["event"] == "tool_call" and t["tool"] == "run_python" and t["status"] != "rejected"
             and t["step"] < exp_call["step"] and "save_table" in t["args"].get("code", "")]
    work = OUT / f"{arm}_{seed}"
    shutil.rmtree(work, ignore_errors=True)
    ws, cells_dir = work / "workspace", work / "cells"
    ws.mkdir(parents=True)
    log = []
    t0 = time.perf_counter()
    for t in cells:
        r = run_python(strip_asserts(t["args"]["code"]), ws, cells_dir, arm_config(arm)["allow_modeling_imports"],
                       timeout=1800, extra_env={"AGENT_API_MODE": "frozen_test"})
        log.append({"step": t["step"], "orig_status": t["status"], "replay_status": r.status,
                    "error": None if r.status == "ok" else r.output[-400:]})
        print(f"  {arm}/{seed} step {t['step']:4} orig={t['status']:5} replay={r.status}", flush=True)
    meta = {"cells_replayed": len(cells), "replay_seconds": round(time.perf_counter() - t0, 1), "cells": log,
            "table_path": exp_call["args"].get("table_path") or exp_call["args"].get("predictions_path")}
    return ws / meta["table_path"], meta


def compare_to_logged(replayed: pd.DataFrame, logged: pd.DataFrame) -> dict:
    """Replayed train+validation rows vs the logged feature table (column-wise)."""
    r = logged[KEYS].merge(replayed, on=KEYS, how="left")
    cols = [c for c in logged.columns if c not in KEYS]
    missing_cols = [c for c in cols if c not in r.columns]
    worst = 0.0
    mismatched = []
    for c in cols:
        if c in missing_cols:
            continue
        a, b = logged[c], r[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.to_numpy(float), b.to_numpy(float)
            same_nan = np.isnan(a) == np.isnan(b)
            diff = np.nanmax(np.abs(a - b) / (1 + np.abs(a))) if (~np.isnan(a)).any() else 0.0
            if not same_nan.all() or diff > 1e-6:
                mismatched.append(c)
            worst = max(worst, float(diff if np.isfinite(diff) else np.inf))
        elif not a.astype("string").fillna("<NA>").equals(b.astype("string").fillna("<NA>")):
            mismatched.append(c)
    return {"n_logged_rows": len(logged), "rows_found": int(r[cols[0]].notna().sum() if cols else 0),
            "missing_columns": missing_cols, "mismatched_columns": mismatched, "max_rel_diff": worst,
            "reproduced": not missing_cols and not mismatched}


def evaluate_run(arm: str, seed: int, experiment_id: str, val_mae: float, reuse: bool = False,
                 accept_approx: bool = False) -> dict:
    print(f"== {arm}/{seed} {experiment_id} (val MAE {val_mae})", flush=True)
    prev = OUT / f"{arm}_{seed}.json"
    if reuse and prev.exists():
        import json as _json
        meta = _json.loads(prev.read_text())["replay"]
        path = OUT / f"{arm}_{seed}" / "workspace" / meta["table_path"]
    else:
        path, meta = replay(arm, seed, experiment_id)
    res = {"arm": arm, "seed": seed, "experiment_id": experiment_id, "val_mae": val_mae, "replay": meta}
    if not path.exists():
        res["status"] = "replay_failed: final table not produced"
        return res
    table = pd.read_parquet(path)
    logged_path = RESULTS_DIR / arm / str(seed) / experiment_id / "features.parquet"
    if not logged_path.exists():
        # Fresh clone without `make artifacts`: the logged feature table is not in git. Score the replayed
        # table on the columns the experiment logged, and mark the reproduction check as unchecked.
        rec = next(json.loads(l) for l in (RESULTS_DIR / arm / str(seed) / "experiments.jsonl").read_text().splitlines()
                   if json.loads(l)["experiment_id"] == experiment_id)
        logged = table[KEYS + rec["feature_columns"]].iloc[:0]
        res["reproduction"] = {"reproduced": None, "note": "unchecked: logged features.parquet not present (run `make artifacts`)"}
        print("   reproduction check: UNCHECKED (logged feature table not present; run `make artifacts`)", flush=True)
    else:
        logged = pd.read_parquet(logged_path)
        res["reproduction"] = compare_to_logged(table, logged)
    n_test = int((table["snapshot_day"] >= 571).sum())
    res["test_rows_in_table"] = n_test
    rep = res["reproduction"]
    if rep["reproduced"] is None:
        res["reproduction_kind"] = "unchecked"
        rep = {"reproduced": True, "missing_columns": [], "rows_found": 0, "n_logged_rows": 0}
    # Approximate: same rows and columns, values differ only through statistics the agent computed over
    # the whole table (now including test-period FEATURE values; never labels). Opt-in, flagged.
    approx_ok = accept_approx and not rep["missing_columns"] and rep["rows_found"] == rep["n_logged_rows"]
    if n_test == 0 or (not rep["reproduced"] and not approx_ok):
        res["status"] = "not_reproduced" if not rep["reproduced"] else "no_test_rows"
        return res
    res.setdefault("reproduction_kind", "exact" if rep["reproduced"] else "approximate")
    table = table[[c for c in logged.columns]]  # exactly the evaluated columns
    backend = BACKEND[arm]
    tv = evaluate_frozen_test(table, backend, ("train", "validation"))
    tr = evaluate_frozen_test(table, backend, ("train",))
    tv["predictions"].to_parquet(OUT / f"{arm}_{seed}_test_predictions.parquet", index=False)
    tr["predictions"].to_parquet(OUT / f"{arm}_{seed}_test_predictions_fit_train.parquet", index=False)
    res.update({
        "status": "ok", "backend": backend,
        "test_mae_fit_train_val": tv["mae"], "test_r2_fit_train_val": tv["r2"],
        "test_mae_fit_train": tr["mae"], "test_r2_fit_train": tr["r2"],
        "val_to_test_degradation_same_fit": round(tr["mae"] - val_mae, 4),
        "model_meta": {"train_val": tv["meta"].get("api_meta", {}).get("tabpfn_config", {}).get("model_path"),
                       "n_train_rows": tv["n_train"], "n_test_rows": tv["n_eval"]},
    })
    print(f"   test MAE (fit train+val) {tv['mae']}, (fit train) {tr['mae']}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--reuse-replay", action="store_true", help="score the existing replayed table")
    ap.add_argument("--accept-approx", action="store_true",
                    help="score tables whose train+val values differ only via table-wide statistics (flagged)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    best = best_experiments()
    if args.arm:
        best = best[best["arm"] == args.arm]
    if args.seed is not None:
        best = best[best["seed"] == args.seed]
    for _, b in best.iterrows():
        res = evaluate_run(b["arm"], int(b["seed"]), b["experiment_id"], float(b["mae"]),
                           reuse=args.reuse_replay, accept_approx=args.accept_approx)
        (OUT / f"{b['arm']}_{int(b['seed'])}.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k not in ("replay",)}, default=str), flush=True)


if __name__ == "__main__":
    main()
