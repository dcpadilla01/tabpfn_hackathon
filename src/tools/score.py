"""score() — Arm A's research primitive.

    score(predictions_path, hypothesis, parent, mutation, reasoning_summary) -> metrics

Reads the agent's validation predictions (household_key, snapshot_day, prediction) from its workspace and
hands them to the evaluator, which checks the key set equals the validation split, computes MAE/R², logs,
and never reveals labels. One call = one experiment (same accounting as experiment()).
"""

from __future__ import annotations

import shutil

import pandas as pd

from src.evaluation.evaluator import score
from src.tools.experiment import ExperimentTool


class ScoreTool(ExperimentTool):
    """Same workspace confinement, metadata checks, code persistence and rollup as experiment()."""

    def __call__(self, predictions_path: str, hypothesis: str, parent: str, mutation: str,
                 reasoning_summary: str = "") -> dict:
        experiment_id = self.log.next_id()
        known = {r["experiment_id"] for r in self.log.records()} | {"E000"}
        problems = []
        if not hypothesis.strip():
            problems.append("hypothesis is required")
        if not mutation.strip():
            problems.append("mutation (the transformation you applied) is required")
        if parent not in known:
            problems.append(f"parent {parent!r} is not a known experiment; known: {sorted(known)}")
        try:
            path = self._resolve(predictions_path)
            preds = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            problems.append(str(e))
            path, preds = None, None
        exp_dir = self.log.experiment_dir(experiment_id)
        extra = self._rollup_extra(experiment_id) | self._persist_code(exp_dir) | {"reasoning_summary": reasoning_summary}
        if problems:
            return score(pd.DataFrame(), log=self.log, experiment_id=experiment_id, parent_id=parent,
                         hypothesis=hypothesis, transformation_description=mutation, extra=extra,
                         invalid_reason="; ".join(problems))
        shutil.copy(path, exp_dir / "submitted_predictions.parquet")
        return score(preds, log=self.log, experiment_id=experiment_id, parent_id=parent, hypothesis=hypothesis,
                     transformation_description=mutation, extra=extra, save_dir=exp_dir)

    def log_failed(self, reason: str, hypothesis: str = "", parent: str | None = None) -> dict:
        experiment_id = self.log.next_id()
        exp_dir = self.log.experiment_dir(experiment_id)
        extra = self._rollup_extra(experiment_id) | self._persist_code(exp_dir)
        extra["tool_calls"] -= 1
        return score(pd.DataFrame(), log=self.log, experiment_id=experiment_id, parent_id=parent,
                     hypothesis=hypothesis, transformation_description="", extra=extra, invalid_reason=reason)
