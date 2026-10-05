"""V0 autonomous researcher: a hand-written sequential loop. No framework.

Per experiment:
    context = system prompt (frozen) + one user turn (compact history table + budget)
    loop: LLM call → tool calls → tool results … until the evaluation tool is called
          (or the 15-tool-call cap is hit → failed experiment)
Context resets after every experiment, so history is the compact table, never code.
Assistant messages are fed back without reasoning text (it goes to reasoning.jsonl only).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from openai import OpenAI

from src.config import load_config
from src.evaluation.evaluator import evaluate
from src.evaluation.logger import ExperimentLog
from src.features.baseline import build_baseline_features
from src.data.splits import research_visible_day
from src.researcher import prompts
from src.researcher.agent_api import snapshot_days
from src.researcher.arms import arm_config
from src.researcher.history import render_history
from src.researcher.trace import RunTrace
from src.tools.experiment import ExperimentTool
from src.tools.inspect import inspect as inspect_tool
from src.tools.run_python import run_python
from src.tools.score import ScoreTool

MAX_NUDGES = 3
LLM_RETRIES = 3


class Researcher:
    def __init__(self, arm: str, seed: int, budget: int, overwrite: bool = False):
        self.arm, self.seed, self.budget = arm, seed, budget
        self.cfg = arm_config(arm)
        llm = load_config()["llm"]
        self.model, self.temperature = llm["model"], llm["temperature"]
        self.max_tokens, self.cap = llm["max_tokens"], llm["max_tool_calls_between_experiments"]
        self.client = OpenAI(base_url=llm["base_url"], api_key=os.environ[llm["api_key_env"]])

        self.log = ExperimentLog(arm, seed)
        if self.log.records() or any(self.log.dir.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.log.dir} already has a run; pass overwrite=True to replace it")
            shutil.rmtree(self.log.dir)
            self.log = ExperimentLog(arm, seed)
        self.trace = RunTrace(arm, seed)
        self.workspace = self.log.dir / "workspace"
        self.harness_dir = self.log.dir / "cells"
        self.workspace.mkdir(parents=True, exist_ok=True)
        tool_cls = ScoreTool if "score" in self.cfg["tools"] else ExperimentTool
        self.tool = tool_cls(self.cfg["backend"], self.log, self.workspace, self.trace)

        self.system = prompts.system_prompt(self.cfg, budget, snapshot_days(), research_visible_day())
        self.tools = prompts.tool_schemas(self.cfg["tools"])
        (self.log.dir / "system_prompt.txt").write_text(self.system)
        (self.log.dir / "tools.json").write_text(json.dumps(self.tools, indent=2))

    # ------------------------------------------------------------------ run
    def run(self) -> list[dict]:
        self._root()
        while self._used() < self.budget:
            self._one_experiment()
        return self.log.records()

    def _used(self) -> int:
        return sum(r["experiment_id"] != "E000" for r in self.log.records())

    def _root(self) -> None:
        """E000 with this arm's backend: the root every run starts from (not budgeted)."""
        self.trace.begin_experiment("E000")
        # Arm A has no harness model: its root is the baseline table scored by the fixed XGBoost evaluator
        # (identical to A′'s E000), recorded as a harness reference.
        evaluate(build_baseline_features(), self.cfg["backend"] or "xgb", log=self.log, experiment_id="E000",
                 hypothesis="Who the household is (demographics) and when the snapshot is (calendar) predict 4-week spend.",
                 transformation_description="Baseline: demographic codes, has_demographics, snapshot day index, week-of-year.")

    def _one_experiment(self) -> None:
        exp_id = self.log.next_id()
        self.trace.begin_experiment(exp_id)
        code_blocks: list[str] = []
        saved = sorted(p.name for p in self.workspace.glob("*.parquet"))
        user = prompts.turn_message(render_history(self.log.records(), self.budget, self._used()), saved)
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": user}]
        nudges = 0
        while True:
            msg = self._llm(messages)
            tool_calls = msg.get("tool_calls") or []
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls") and v is not None})
            if not tool_calls:
                nudges += 1
                if nudges > MAX_NUDGES:
                    self.tool.pending_code = "\n\n# ---- cell ----\n".join(code_blocks)
                    self.tool.log_failed("agent stopped calling tools")
                    return
                messages.append({"role": "user", "content": "Continue by calling a tool."})
                continue
            for i, tc in enumerate(tool_calls):
                name = tc["function"]["name"]
                if name not in self.cfg["tools"]:
                    out, status = f"unknown tool {name!r}", "error"
                    self._log_tool(name, {}, out, status, 0.0, out)
                elif name in ("experiment", "score"):
                    self.tool.pending_code = "\n\n# ---- cell ----\n".join(code_blocks)
                    args, err = self._args(tc)
                    t0 = time.perf_counter()
                    first = "predictions_path" if name == "score" else "table_path"
                    fields = (first, "hypothesis", "parent", "mutation", "reasoning_summary")
                    res = self.tool(**{f: str(args.get(f) or "") for f in fields})
                    self._log_tool(name, args, res, "ok" if res["status"] == "ok" else "error",
                                   time.perf_counter() - t0, res.get("error"))
                    return  # experiment closed; remaining tool calls in this message are dropped
                else:
                    if self.trace.tool_calls_since_experiment >= self.cap:
                        self.tool.pending_code = "\n\n# ---- cell ----\n".join(code_blocks)
                        self.tool.log_failed(f"exceeded {self.cap} tool calls without calling the evaluation tool")
                        return
                    out = self._run_tool(name, tc, code_blocks)
                    out += self._budget_note()
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})

    # ---------------------------------------------------------------- tools
    def _args(self, tc) -> tuple[dict, str | None]:
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
            return (args, None) if isinstance(args, dict) else ({}, "arguments must be a JSON object")
        except json.JSONDecodeError as e:
            return {}, f"invalid JSON arguments: {e}"

    def _run_tool(self, name: str, tc: dict, code_blocks: list[str]) -> str:
        args, err = self._args(tc)
        t0 = time.perf_counter()
        if err:
            out, status, error = err, "error", err
        elif name == "run_python":
            code = args.get("code", "")
            r = run_python(code, self.workspace, self.harness_dir, self.cfg["allow_modeling_imports"])
            if r.status != "rejected":
                code_blocks.append(code)
            out, status, error = r.output, r.status, r.error
        elif name == "inspect":
            try:
                out, status, error = inspect_tool(**args), "ok", None
            except Exception as e:  # noqa: BLE001
                out, status, error = f"{type(e).__name__}: {e}", "error", str(e)
        else:
            out, status, error = f"unknown tool {name!r}", "error", "unknown tool"
        self._log_tool(name, args, out, status, time.perf_counter() - t0, error)
        return out

    def _budget_note(self) -> str:
        """Tell the agent how many tool calls remain before the evaluation tool must be called."""
        used = self.trace.tool_calls_since_experiment
        left = self.cap - used
        note = f"\n\n[tool calls since last experiment: {used}/{self.cap}"
        if left <= 3:
            note += f" — {left} left; call the evaluation tool before the limit or this experiment fails"
        return note + "]"

    def _log_tool(self, name, args, output, status, seconds, error):
        self.trace.log_tool_call(tool=name, args=args, output=output, status=status,
                                 wall_clock_seconds=seconds, error=error)

    # ------------------------------------------------------------------ llm
    def _llm(self, messages: list[dict]) -> dict:
        request = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "seed": self.seed, "tools": self.tools, "messages": messages}
        last_err = None
        for attempt in range(LLM_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**request, extra_body={"usage": {"include": True}})
                d = resp.model_dump()
                if not d.get("choices"):
                    raise RuntimeError(f"empty response: {str(d)[:300]}")
                logged = {k: v for k, v in request.items() if k not in ("messages", "tools")}
                logged["messages"] = messages[1:]  # system prompt and tools are stored once per run
                self.trace.log_llm_call(request=logged, response=d, wall_clock_seconds=time.perf_counter() - t0,
                                        retries=attempt)
                return d["choices"][0]["message"]
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {LLM_RETRIES + 1} attempts: {last_err}")


def run(arm: str, seed: int, budget: int, overwrite: bool = False) -> list[dict]:
    return Researcher(arm, seed, budget, overwrite).run()
