"""Experiment log: one JSON line per experiment in <results>/<arm>/<seed>/experiments.jsonl.

Every record carries the Phase 4c fields; fields that do not apply (e.g. token counts
for a manual run) are present and null, so every log has the same shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import RESULTS_DIR


@dataclass
class ExperimentRecord:
    experiment_id: str
    parent_id: str | None
    arm: str
    backend: str
    seed: int
    hypothesis: str
    transformation_description: str
    feature_columns: list[str]
    mae: float | None = None
    r2: float | None = None
    n_features: int | None = None
    runtime_seconds: float | None = None
    status: str = "ok"                     # ok | invalid | error
    error: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_calls: int | None = None
    generated_code_size: int | None = None
    wall_clock_seconds: float | None = None
    # rollup snapshot from calls.jsonl at experiment time (Phase 11 recomputes from calls.jsonl)
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    uncached_equivalent_tokens: int | None = None
    llm_calls: int | None = None
    llm_cost_usd: float | None = None
    tool_calls_rejected: int | None = None
    tabpfn_estimated_credits: int | None = None
    reasoning_summary: str | None = None
    feature_table_hash: str | None = None
    n_train: int | None = None
    n_eval: int | None = None
    eval_split: str = "validation"
    versions: dict[str, str] = field(default_factory=dict)
    backend_meta: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


class ExperimentLog:
    def __init__(self, arm: str, seed: int, root: Path = RESULTS_DIR):
        self.arm, self.seed = arm, seed
        self.dir = root / arm / str(seed)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "experiments.jsonl"

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def next_id(self) -> str:
        return f"E{len(self.records()):03d}"

    def experiment_dir(self, experiment_id: str) -> Path:
        d = self.dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def append(self, record: ExperimentRecord) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")
