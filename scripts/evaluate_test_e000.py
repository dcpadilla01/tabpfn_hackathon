"""Phase 12 reference: E000 (demographics + calendar) on test, both backends, both fitting regimes.

    uv run python scripts/evaluate_test_e000.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT  # noqa: E402
from src.evaluation.evaluator import evaluate, evaluate_frozen_test  # noqa: E402
from src.features.baseline import build_baseline_features  # noqa: E402

OUT = ROOT / "experiments" / "analysis" / "frozen_test"


def main() -> None:
    ft_all = build_baseline_features(splits=("train", "validation", "test"))
    ft_research = build_baseline_features()
    rows = []
    for backend in ("tabpfn", "xgb"):
        val = evaluate(ft_research, backend)["mae"]
        tr = evaluate_frozen_test(ft_all, backend, ("train",))
        tv = evaluate_frozen_test(ft_all, backend, ("train", "validation"))
        rows.append({"backend": backend, "val_mae": val, "test_mae_fit_train": tr["mae"], "test_mae_fit_train_val": tv["mae"],
                     "test_r2_fit_train": tr["r2"], "test_r2_fit_train_val": tv["r2"]})
        print(rows[-1])
    (OUT / "e000.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
