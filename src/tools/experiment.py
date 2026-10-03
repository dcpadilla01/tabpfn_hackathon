"""experiment() — the research primitive for Arms A′ and B.

    experiment(table_path, hypothesis, parent, mutation, reasoning_summary) -> metrics

Reads the agent's feature table from its workspace, hands it to the fixed evaluator
with the arm's backend, logs, and returns metrics. It does NOT invent, select, join,
window or tune features: whatever the table holds is what gets evaluated.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.evaluation.evaluator import evaluate
from src.evaluation.logger import ExperimentLog
from src.researcher.trace import RunTrace

ROOT_PARENT = "E000"


class ExperimentTool:
    def __init__(self, backend: str, log: ExperimentLog, workspace: Path, trace: RunTrace | None = None):
        self.backend, self.log, self.trace = backend, log, trace
        self.workspace = Path(workspace).resolve()

    def _resolve(self, table_path: str) -> Path:
        p = (self.workspace / table_path).resolve()
        if not p.is_relative_to(self.workspace):
            raise ValueError("table_path must be inside your workspace")
        if p.suffix != ".parquet":
            raise ValueError("table_path must be a .parquet file written with DataFrame.to_parquet")
        if not p.exists():
            raise ValueError(f"no file at {table_path!r} in your workspace")
        return p

    def __call__(self, table_path: str, hypothesis: str, parent: str, mutation: str,
                 reasoning_summary: str = "") -> dict:
        experiment_id = self.log.next_id()
        known = {r["experiment_id"] for r in self.log.records()} | {ROOT_PARENT}
        problems = []
        if not hypothesis.strip():
            problems.append("hypothesis is required")
        if not mutation.strip():
            problems.append("mutation (the transformation you applied) is required")
        if parent not in known:
            problems.append(f"parent {parent!r} is not a known experiment; known: {sorted(known)}")
        try:
            path = self._resolve(table_path)
            table = pd.read_parquet(path)
        except Exception as e:
            problems.append(str(e))
            path, table = None, None

        exp_dir = self.log.experiment_dir(experiment_id)
        extra = self._rollup_extra(experiment_id)
        if problems:  # still consumes an experiment: logged as invalid, not evaluated
            return evaluate(pd.DataFrame(), self.backend, log=self.log, experiment_id=experiment_id,
                            parent_id=parent, hypothesis=hypothesis, transformation_description=mutation,
                            extra=extra | {"reasoning_summary": reasoning_summary},
                            invalid_reason="; ".join(problems))

        shutil.copy(path, exp_dir / "features.parquet")
        return evaluate(table, self.backend, log=self.log, experiment_id=experiment_id, parent_id=parent,
                        hypothesis=hypothesis, transformation_description=mutation,
                        extra=extra | {"reasoning_summary": reasoning_summary}, save_dir=exp_dir)

    def _rollup_extra(self, experiment_id: str) -> dict:
        if self.trace is None:
            return {}
        r = self.trace.rollup(experiment_id)
        return {
            "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
            "cached_tokens": r["cached_tokens"], "reasoning_tokens": r["reasoning_tokens"],
            "uncached_equivalent_tokens": r["uncached_equivalent_tokens"],
            "llm_calls": r["llm_calls"], "llm_cost_usd": r["llm_cost_usd"],
            "tool_calls": r["tool_calls"] + 1,  # + this experiment() call, logged by the loop after it returns
            "tool_calls_rejected": r["tool_calls_rejected"],
            "wall_clock_seconds": round(self.trace.seconds_since_experiment_began(), 2),
        }
