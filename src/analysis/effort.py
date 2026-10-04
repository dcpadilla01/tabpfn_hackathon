"""Phase 11 effort classification from ACTIONS only (tool calls and the code they ran),
never from reasoning text. Rules, first match wins:

  evaluation          experiment() / score() call
  debugging           run_python right after a failed/rejected tool call in the same experiment
  model engineering   code fits the agent's own model or computes a local MAE on labels
  feature construction  code calls build_features or save_table
  preprocessing       code encodes/scales/imputes (dummies, category casts, fillna, standardise, clip, log)
  data exploration    everything else (inspect calls, describe, value_counts, correlations, prints)

Effort share = share of tool calls, and share of uncached-equivalent tokens of the LLM call
that produced each tool call (split evenly when one LLM call emits several tool calls).

    uv run python -m src.analysis.effort
"""

from __future__ import annotations

import re

import pandas as pd

from src.analysis.runs import ARMS, SEEDS, transcript
from src.config import ROOT

OUT = ROOT / "experiments" / "analysis"
LABEL = re.compile(r"train_targets|future_spend_4w|\bTARGET\b")
FIT = re.compile(r"linalg\.(solve|lstsq|pinv)|lstsq|def \w*(fit|gbm|boost|tree|ridge)\w*\(|\bridge\b|boost|build_tree")
LOCAL_SCORE = re.compile(r"np\.abs\([^)]*-[^)]*\)\.mean\(\)|\bmae\b", re.I)
FEATURE = re.compile(r"build_features\(|save_table\(")
PREPROC = re.compile(r"get_dummies|astype\(['\"]category|fillna\(|\.clip\(|np\.log1p|\.std\(\)|/\s*sd\b|standardi|rank\(pct")
CATEGORIES = ["data exploration", "feature construction", "preprocessing", "model engineering", "debugging", "evaluation"]


def classify(tool: str, code: str, prev_failed: bool) -> str:
    if tool in ("experiment", "score"):
        return "evaluation"
    if tool == "inspect":
        return "data exploration"
    if prev_failed:
        return "debugging"
    if FIT.search(code) or (LABEL.search(code) and LOCAL_SCORE.search(code)):
        return "model engineering"
    if FEATURE.search(code):
        return "feature construction"
    if PREPROC.search(code):
        return "preprocessing"
    return "data exploration"


def classify_run(arm: str, seed: int) -> pd.DataFrame:
    T = transcript(arm, seed)
    rows, pending_tokens, n_tools_in_turn = [], 0, 0
    prev_failed_by_exp: dict[str, bool] = {}
    # tokens of each llm_call are attributed to the tool calls that follow it (until the next llm_call)
    turn_tools: list[int] = []
    for t in T:
        if t["event"] == "llm_call":
            u = (t.get("response") or {}).get("usage") or {}
            pending_tokens = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
            turn_tools = []
            rows.append(None)  # marker
            continue
        exp = t["experiment_id"]
        code = (t.get("args") or {}).get("code", "") if t["tool"] == "run_python" else ""
        cat = classify(t["tool"], code, prev_failed_by_exp.get(exp, False))
        prev_failed_by_exp[exp] = t["status"] in ("error", "rejected")
        rows.append({"arm": arm, "seed": seed, "experiment_id": exp, "step": t["step"], "tool": t["tool"],
                     "status": t["status"], "category": cat, "turn_tokens": pending_tokens})
    # split each LLM turn's tokens over its tool calls
    out, group = [], []
    for r in rows + [None]:
        if r is None:
            for g in group:
                g["tokens"] = g["turn_tokens"] / len(group)
                out.append(g)
            group = []
        else:
            group.append(r)
    return pd.DataFrame(out)


def main() -> None:
    df = pd.concat([classify_run(a, s) for a in ARMS for s in SEEDS], ignore_index=True)
    df.to_csv(OUT / "effort_actions.csv", index=False)
    calls = (df.groupby(["arm", "category"]).size() / df.groupby("arm").size()).unstack().reindex(columns=CATEGORIES)
    toks = (df.groupby(["arm", "category"])["tokens"].sum() / df.groupby("arm")["tokens"].sum()).unstack().reindex(columns=CATEGORIES)
    per_run = (df.groupby(["arm", "seed", "category"]).size() / df.groupby(["arm", "seed"]).size()).unstack().reindex(columns=CATEGORIES)
    share = pd.concat({"share of tool calls": calls, "share of tokens": toks})
    share.round(3).to_csv(OUT / "effort_share.csv")
    per_run.round(3).to_csv(OUT / "effort_share_per_run.csv")
    with pd.option_context("display.width", 200):
        print((share * 100).round(1).to_string())
        print("\nper run, share of tool calls (%):")
        print((per_run * 100).round(1).to_string())


if __name__ == "__main__":
    main()
