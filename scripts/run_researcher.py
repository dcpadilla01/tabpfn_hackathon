"""Run one autonomous research session.

    uv run python scripts/run_researcher.py --arm b --seed 0 --budget 20
    make run ARM=b SEED=0 BUDGET=3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.researcher.agent import run  # noqa: E402
from src.researcher.trace import RunTrace  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "a_prime", "b"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    records = run(args.arm, args.seed, args.budget, args.overwrite)
    print(f"\n{'id':5} {'parent':6} {'status':8} {'mae':>9}  hypothesis")
    for r in records:
        mae = f"{r['mae']:.3f}" if r["mae"] is not None else "—"
        print(f"{r['experiment_id']:5} {r['parent_id'] or '—':6} {r['status']:8} {mae:>9}  {(r['hypothesis'] or '')[:70]}")
    calls = RunTrace(args.arm, args.seed).calls()
    llm = [c for c in calls if c["event"] == "llm_call"]
    cost = sum(c.get("cost_usd") or 0 for c in llm)
    print(f"\nLLM calls {len(llm)}, uncached-equivalent tokens {sum(c['uncached_equivalent_tokens'] for c in llm):,}, "
          f"cost ${cost:.4f}, tool calls {sum(c['event'] == 'tool_call' for c in calls)}")


if __name__ == "__main__":
    main()
