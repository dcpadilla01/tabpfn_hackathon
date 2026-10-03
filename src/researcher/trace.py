"""Per-run call trace: every LLM call and tool call, for cost accounting and audit.

Files in <results>/<arm>/<seed>/:
    calls.jsonl       one line per event (llm_call | tool_call) — single source of truth for cost
    transcript.jsonl  full request/response and tool I/O, reasoning text stripped
    reasoning.jsonl   raw reasoning text, keyed by step + generation_id; never read by the agent

Experiment-level token and tool-call counts are rollups over calls.jsonl (`rollup()`),
never counted separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import RESULTS_DIR

REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _get(d: Any, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or d.get(k) is None:
            return default
        d = d[k]
    return d


def parse_usage(response: dict) -> dict:
    """Normalise an OpenAI-compatible (OpenRouter) usage block. Missing fields → None/0.
    Convention (OpenAI/OpenRouter): prompt_tokens INCLUDES cached tokens, so
    uncached-equivalent = prompt_tokens + completion_tokens. Re-verify on a live
    response in Phase 7; the raw usage block is logged alongside."""
    u = response.get("usage") or {}
    prompt = u.get("prompt_tokens") or 0
    completion = u.get("completion_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": _get(u, "prompt_tokens_details", "cached_tokens", default=0),
        "cache_write_tokens": _get(u, "prompt_tokens_details", "cache_write_tokens"),
        "reasoning_tokens": _get(u, "completion_tokens_details", "reasoning_tokens", default=0),
        "uncached_equivalent_tokens": prompt + completion,
        "cost_usd": u.get("cost"),
        "usage_raw": u,
    }


def split_reasoning(response: dict) -> tuple[dict, list]:
    """Return (response without reasoning text, [reasoning payloads per choice])."""
    clean = copy.deepcopy(response)
    extracted = []
    for choice in clean.get("choices") or []:
        msg = choice.get("message") or {}
        found = {k: msg.pop(k) for k in REASONING_KEYS if msg.get(k) is not None}
        extracted.append(found)
    return clean, extracted


class RunTrace:
    def __init__(self, arm: str, seed: int, root: Path = RESULTS_DIR, run_id: str | None = None):
        self.arm, self.seed = arm, seed
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.dir = root / arm / str(seed)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.dir / "calls.jsonl"
        self.transcript_path = self.dir / "transcript.jsonl"
        self.reasoning_path = self.dir / "reasoning.jsonl"
        self.step = 0
        self.experiment_id: str | None = None   # experiment currently being worked towards
        self.tool_calls_since_experiment = 0
        self.experiment_started = time.perf_counter()

    # ----------------------------------------------------------------- writers
    def _write(self, path: Path, rec: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def _base(self, event: str) -> dict:
        return {"run_id": self.run_id, "arm": self.arm, "seed": self.seed, "step": self.step,
                "event": event, "experiment_id": self.experiment_id, "timestamp": _now()}

    def log_llm_call(self, *, request: dict, response: dict, wall_clock_seconds: float,
                     retries: int = 0, provider: str | None = None) -> dict:
        """`request`: the kwargs sent (model, temperature, messages delta, tools...).
        `response`: the raw response as a dict (e.g. `resp.model_dump()`)."""
        self.step += 1
        clean, reasoning = split_reasoning(response)
        choice0 = (response.get("choices") or [{}])[0]
        rec = self._base("llm_call") | {
            "model_requested": request.get("model"),
            "model_returned": response.get("model"),
            "provider": provider or response.get("provider"),
            "generation_id": response.get("id"),
            "finish_reason": choice0.get("finish_reason"),
            "temperature": request.get("temperature"),
            "retries": retries,
            "wall_clock_seconds": round(wall_clock_seconds, 3),
            **{k: v for k, v in parse_usage(response).items() if k != "usage_raw"},
        }
        self._write(self.calls_path, rec | {"usage_raw": parse_usage(response)["usage_raw"]})
        self._write(self.transcript_path, self._base("llm_call") | {"request": request, "response": clean})
        if any(reasoning):
            self._write(self.reasoning_path, self._base("llm_call") | {"generation_id": response.get("id"), "reasoning": reasoning})
        return rec

    def log_tool_call(self, *, tool: str, args: dict, output: Any, status: str,
                      wall_clock_seconds: float, error: str | None = None) -> dict:
        """status: ok | error | rejected. Every call (incl. rejected) counts toward the cap."""
        self.step += 1
        self.tool_calls_since_experiment += 1
        args_json = json.dumps(args, sort_keys=True, default=str)
        out_json = output if isinstance(output, str) else json.dumps(output, default=str)
        rec = self._base("tool_call") | {
            "tool": tool,
            "status": status,
            "error": error,
            "args_hash": hashlib.md5(args_json.encode()).hexdigest(),
            "args_size": len(args_json),
            "output_size": len(out_json),
            "tool_calls_since_last_experiment": self.tool_calls_since_experiment,
            "wall_clock_seconds": round(wall_clock_seconds, 3),
        }
        self._write(self.calls_path, rec)
        self._write(self.transcript_path, self._base("tool_call") | {"tool": tool, "args": args, "output": output, "status": status, "error": error})
        return rec

    def begin_experiment(self, experiment_id: str) -> None:
        """Mark the start of work towards `experiment_id` (after the previous one closed)."""
        self.experiment_id = experiment_id
        self.tool_calls_since_experiment = 0
        self.experiment_started = time.perf_counter()

    def seconds_since_experiment_began(self) -> float:
        return time.perf_counter() - self.experiment_started

    # ----------------------------------------------------------------- readers
    def calls(self) -> list[dict]:
        if not self.calls_path.exists():
            return []
        return [json.loads(l) for l in self.calls_path.read_text().splitlines() if l.strip()]

    def rollup(self, experiment_id: str) -> dict:
        """Totals for every event attributed to `experiment_id` in this run."""
        ev = [c for c in self.calls() if c["experiment_id"] == experiment_id and c["run_id"] == self.run_id]
        llm = [c for c in ev if c["event"] == "llm_call"]
        tools = [c for c in ev if c["event"] == "tool_call"]
        costs = [c["cost_usd"] for c in llm if c.get("cost_usd") is not None]
        return {
            "llm_calls": len(llm),
            "tokens_in": sum(c["prompt_tokens"] for c in llm),
            "tokens_out": sum(c["completion_tokens"] for c in llm),
            "cached_tokens": sum(c["cached_tokens"] or 0 for c in llm),
            "reasoning_tokens": sum(c["reasoning_tokens"] or 0 for c in llm),
            "uncached_equivalent_tokens": sum(c["uncached_equivalent_tokens"] for c in llm),
            "llm_cost_usd": round(sum(costs), 6) if costs else None,
            "tool_calls": len(tools),
            "tool_calls_rejected": sum(c["status"] == "rejected" for c in tools),
            "tool_calls_error": sum(c["status"] == "error" for c in tools),
        }
