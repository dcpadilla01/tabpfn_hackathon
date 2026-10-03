"""Phase 5: validate the research loop by hand — accessor → feature table → evaluate().

Each hypothesis extends its parent's feature function. Nothing here touches the
evaluator, split or target; this is exactly the path the agent will use.

    uv run python scripts/run_manual.py              # E001-E003, both backends
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.accessor import AsOf, build_features  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.evaluation.logger import ExperimentLog  # noqa: E402
from src.features.baseline import baseline_fn  # noqa: E402


def spend_in_window(view: AsOf, days: int) -> pd.Series:
    tx = view.transactions
    recent = tx[tx["day"] > view.day - days]
    return recent.groupby("household_key")["sales_value"].sum()


def e001(view: AsOf, day: int) -> pd.DataFrame:
    out = baseline_fn(view, day)
    out["spend_last_4w"] = spend_in_window(view, 28).reindex(out.index, fill_value=0.0)
    return out


def e002(view: AsOf, day: int) -> pd.DataFrame:
    out = e001(view, day)
    out["spend_last_26w"] = spend_in_window(view, 182).reindex(out.index, fill_value=0.0)
    return out


def e003(view: AsOf, day: int) -> pd.DataFrame:
    out = e002(view, day)
    tx = view.transactions
    tx = tx[tx["day"] > day - 84]
    promoted = (tx["retail_disc"] < 0) | (tx["coupon_disc"] < 0) | (tx["coupon_match_disc"] < 0)
    by_hh = tx.assign(promo_spend=tx["sales_value"].where(promoted, 0.0)).groupby("household_key")[["promo_spend", "sales_value"]].sum()
    out["promo_share_12w"] = (by_hh["promo_spend"] / by_hh["sales_value"].where(by_hh["sales_value"] > 0)).reindex(out.index)
    return out


EXPERIMENTS = [
    ("E001", "E000", e001, "Recent purchasing behavior predicts future spend.",
     "+ spend_last_4w: sum of sales_value over (snapshot_day-28, snapshot_day]."),
    ("E002", "E001", e002, "Long-term value adds signal beyond recent behavior.",
     "+ spend_last_26w: sum of sales_value over (snapshot_day-182, snapshot_day]."),
    ("E003", "E002", e003, "Promotion sensitivity.",
     "+ promo_share_12w: share of last-12-week spend on line items with any discount (retail, coupon or coupon match); NaN if no spend."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tabpfn", "xgb", "both"], default="both")
    args = ap.parse_args()
    backends = ["tabpfn", "xgb"] if args.backend == "both" else [args.backend]
    seed = load_config()["seed"]
    for exp_id, parent, fn, hypothesis, transformation in EXPERIMENTS:
        ft = build_features(fn)
        for backend in backends:
            log = ExperimentLog(f"manual_{backend}", seed)
            res = evaluate(ft, backend, log=log, experiment_id=exp_id, parent_id=parent,
                           hypothesis=hypothesis, transformation_description=transformation)
            print(json.dumps(res))


if __name__ == "__main__":
    main()
