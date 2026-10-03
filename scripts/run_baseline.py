"""Phase 3: E000 weak root baseline, both backends, logged under experiments/results/manual/<seed>/.

    uv run python scripts/run_baseline.py                  # both backends
    uv run python scripts/run_baseline.py --backend xgb
    uv run python scripts/run_baseline.py --reproduce      # re-evaluate, assert equal to logged E000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.evaluation.evaluator import evaluate  # noqa: E402
from src.evaluation.logger import ExperimentLog  # noqa: E402
from src.features.baseline import build_baseline_features  # noqa: E402

HYPOTHESIS = "Who the household is (demographics) and when the snapshot is (calendar) predict 4-week spend."
TRANSFORMATION = "Demographic codes (NaN + has_demographics flag when absent), snapshot day index, week-of-year."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tabpfn", "xgb", "both"], default="both")
    ap.add_argument("--reproduce", action="store_true")
    args = ap.parse_args()
    backends = ["tabpfn", "xgb"] if args.backend == "both" else [args.backend]
    ft = build_baseline_features()
    for backend in backends:
        log = ExperimentLog(f"manual_{backend}", load_config()["seed"])
        if args.reproduce:
            logged = next(r for r in log.records() if r["experiment_id"] == "E000")
            res = evaluate(ft, backend)
            same = (res["mae"], res["r2"]) == (logged["mae"], logged["r2"])
            print(f"{backend}: logged mae={logged['mae']} r2={logged['r2']} | now mae={res['mae']} r2={res['r2']} "
                  f"-> {'REPRODUCED' if same else 'DIFFERS'}")
            continue
        res = evaluate(ft, backend, log=log, experiment_id="E000", hypothesis=HYPOTHESIS,
                       transformation_description=TRANSFORMATION)
        print(json.dumps(res))


if __name__ == "__main__":
    main()
