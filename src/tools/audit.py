"""Post-run audit: re-check every persisted code cell with the same static rules.

    uv run python -m src.tools.audit [experiments/results/<arm>/<seed> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import RESULTS_DIR
from src.researcher.arms import arm_config
from src.tools.run_python import check_code


def audit_run(run_dir: Path) -> list[str]:
    arm = run_dir.parent.name
    allow = arm_config(arm)["allow_modeling_imports"] if arm in ("a", "a_prime", "b") else False
    findings = []
    transcript = run_dir / "transcript.jsonl"
    if transcript.exists():
        for line in transcript.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "tool_call" and rec.get("tool") == "run_python" and rec.get("status") != "rejected":
                problems = check_code((rec.get("args") or {}).get("code", ""), allow)
                if problems:
                    findings.append(f"{run_dir} step {rec['step']}: EXECUTED code violates rules: {problems}")
    for code in sorted(run_dir.glob("E*/code.py")):
        for cell in code.read_text().split("# ---- cell ----"):
            if cell.strip() and (problems := check_code(cell, allow)):
                findings.append(f"{code}: {problems}")
    return findings


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [d for arm in RESULTS_DIR.iterdir() if arm.name in ("a", "a_prime", "b")
                                              for d in arm.iterdir() if d.is_dir()]
    all_findings = [f for d in dirs for f in audit_run(d)]
    for f in all_findings:
        print(f)
    print(f"audited {len(dirs)} run(s): {'CLEAN' if not all_findings else f'{len(all_findings)} finding(s)'}")
    sys.exit(1 if all_findings else 0)


if __name__ == "__main__":
    main()
