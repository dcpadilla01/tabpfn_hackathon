"""Label-access audit over every executed run_python cell of the MVP runs.

For each cell: does it touch labels (train_targets / TARGET / future_spend_4w)? Which
functions are passed to build_features, and do those bodies print, touch labels, or
reference validation days? Which snapshot days are used to split data for local scoring?
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

VAL_DAYS = {459, 487, 515, 543}
LABEL = re.compile(r"train_targets|future_spend_4w|\bTARGET\b")


def fn_args_to_build_features(tree):
    names, lambdas = set(), []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", None)) == "build_features" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Name):
                names.add(a.id)
            elif isinstance(a, ast.Lambda):
                lambdas.append(a)
    return names, lambdas


def body_flags(node, src):
    seg = ast.get_source_segment(src, node) or ""
    calls = {getattr(c.func, "id", getattr(c.func, "attr", "")) for c in ast.walk(node) if isinstance(c, ast.Call)}
    consts = {c.value for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, int)}
    return {"print": "print" in calls, "labels": bool(LABEL.search(seg)), "val_day_literal": bool(consts & VAL_DAYS)}


def snapshot_split_literals(src):
    """Integer literals compared against something named *snapshot*/sd/day in this cell."""
    out = set()
    for m in re.finditer(r"(snapshot_day|\bsd\b|\bs\b|\bday\b)\s*(<=|<|>=|>|==|!=)\s*(\d{2,3})\b", src):
        out.add(int(m.group(3)))
    for m in re.finditer(r"\b(\d{2,3})\s*(<=|<|>=|>)\s*(snapshot_day|\bsd\b)", src):
        out.add(int(m.group(1)))
    return out


def main(dirs):
    totals = Counter()
    for d in dirs:
        T = [json.loads(l) for l in open(d / "transcript.jsonl")]
        cells = [t for t in T if t["event"] == "tool_call" and t["tool"] == "run_python" and t["status"] != "rejected"]
        c = Counter()
        flagged = []
        split_days = Counter()
        for t in cells:
            src = t["args"].get("code", "")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            uses_labels = bool(LABEL.search(src))
            c["cells"] += 1
            c["cells_with_labels"] += uses_labels
            names, lambdas = fn_args_to_build_features(tree)
            fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef,)) and n.name in names] + lambdas
            for f in fns:
                fl = body_flags(f, src)
                c["fn_bodies"] += 1
                c["fn_print"] += fl["print"]
                c["fn_labels"] += fl["labels"]
                c["fn_val_day_literal"] += fl["val_day_literal"]
                if fl["labels"] or fl["val_day_literal"]:
                    flagged.append((t["step"], t["experiment_id"], getattr(f, "name", "<lambda>"), fl))
            if uses_labels:
                for day in snapshot_split_literals(src):
                    split_days[day] += 1
        totals.update(c)
        print(f"\n## {d.parent.name}/{d.name}: {dict(c)}")
        print("   literals compared to snapshot/day in label-using cells:", dict(sorted(split_days.items())))
        bad = {k: v for k, v in split_days.items() if k in VAL_DAYS}
        print("   ... of which validation days:", bad or "none")
        for f in flagged:
            print("   FLAG fn body:", f)
    print("\nTOTAL", dict(totals))


if __name__ == "__main__":
    root = Path("experiments/results")
    dirs = [Path(a) for a in sys.argv[1:]] or [root / a / str(s) for a in ("b", "a_prime") for s in (0, 1, 2)]
    main(dirs)
