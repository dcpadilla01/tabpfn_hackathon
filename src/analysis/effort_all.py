"""One effort table over both environments (15 runs), identical classifier and definitions.

Measures per environment × arm:
  - primary category share of tool calls (one category per call; rules in src/analysis/effort.py)
  - share of executed run_python cells that contain any model fit (FIT anywhere in the cell)   ← robust measure
  - share of executed cells that fit a model AND use labels (Env-1's "41%" definition)
  - rejected cells and cell timeouts per run

    uv run python -m src.analysis.effort_all
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analysis.effort import CATEGORIES, FIT, LABEL, classify
from src.config import ROOT

ENVS = {"Env-1": ("results", ("b", "a_prime")), "Env-2": ("results_env2", ("b", "a_prime", "a"))}
LABELS = {"b": "B (TabPFN)", "a_prime": "A′ (XGBoost)", "a": "A (free-form)"}


def run_actions(results: str, arm: str, seed: int) -> pd.DataFrame:
    rows, prev_failed = [], {}
    for t in map(json.loads, open(ROOT / "experiments" / results / arm / str(seed) / "transcript.jsonl")):
        if t["event"] != "tool_call":
            continue
        code = (t.get("args") or {}).get("code", "") if t["tool"] == "run_python" else ""
        cat = classify(t["tool"], code, prev_failed.get(t["experiment_id"], False))
        prev_failed[t["experiment_id"]] = t["status"] in ("error", "rejected")
        rows.append({"tool": t["tool"], "status": t["status"], "category": cat, "code": code,
                     "timeout": (t.get("error") or "") == "timeout"})
    return pd.DataFrame(rows)


def main() -> None:
    out = []
    for env, (results, arms) in ENVS.items():
        for arm in arms:
            df = pd.concat([run_actions(results, arm, s) for s in (0, 1, 2)], ignore_index=True)
            share = df["category"].value_counts(normalize=True).reindex(CATEGORIES).fillna(0)
            cells = df[(df["tool"] == "run_python") & (df["status"] != "rejected")]
            any_fit = cells["code"].apply(lambda c: bool(FIT.search(c))).mean()
            fit_label = cells["code"].apply(lambda c: bool(FIT.search(c)) and bool(LABEL.search(c))).mean()
            out.append({"env": env, "arm": LABELS[arm], **{c: round(100 * share[c], 1) for c in CATEGORIES},
                        "model eng + debugging": round(100 * (share["model engineering"] + share["debugging"]), 1),
                        "cells with any model fit %": round(100 * any_fit, 1),
                        "cells fitting on labels %": round(100 * fit_label, 1),
                        "rejected / run": round((df["status"] == "rejected").sum() / 3, 1),
                        "timeouts / run": round(df["timeout"].sum() / 3, 1)})
    t = pd.DataFrame(out)
    path = ROOT / "experiments" / "analysis_env2" / "effort_both_envs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()
