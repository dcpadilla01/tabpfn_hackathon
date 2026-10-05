"""run_python(code) — execute agent code in a subprocess with `agent_api` pre-imported.

Static enforcement plus audit, not a sandbox:
  1. AST import allowlist (primary): only ALLOWED_IMPORTS (+ modelling libs in Arm A).
  2. AST blocklist: file/IO builtins, reflection builtins, dunder attributes, and any
     pandas/numpy read_*/to_*/load/save style attribute (catches aliasing like `x.read_csv`).
  3. String-pattern denylist (secondary tripwire).
  4. Subprocess with a stripped environment (PATH only), cwd = the run's workspace,
     timeout, and harness paths scrubbed from any output.
Rejected code never runs; the rejection is returned to the agent and logged.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import ROOT

ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "collections", "itertools", "functools",
    "operator", "re", "datetime", "typing", "dataclasses", "warnings", "agent_api",
}
MODELLING_IMPORTS = {"sklearn", "xgboost", "scipy"}

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview",
}
BLOCKED_ATTR = re.compile(
    r"^(read_\w+|to_(csv|parquet|pickle|json|feather|hdf|sql|excel|orc|stata|xml|html|latex|clipboard)"
    r"|load|loadtxt|genfromtxt|fromfile|save|savez|savez_compressed|savetxt|memmap|tofile"
    r"|open|system|popen|remove|unlink|listdir|scandir|walk|environ|getenv"
    r"|os|sys|io|subprocess|builtins|pathlib|shutil|socket|importlib|pickle)$"
)
# Names agent code may import from / access on agent_api: its public API only.
from src.researcher.agent_api_public import PUBLIC_API  # noqa: E402
DENY_PATTERNS = [
    r"read_csv", r"read_parquet", r"open\(", r"\bglob\b", r"data/", r"os\.listdir",
    r"pyarrow", r"duckdb", r"polars", r"pathlib", r"subprocess", r"socket",
    r"importlib", r"__builtins__", r"\bos\.", r"\bsys\.", r"shutil", r"pickle",
]

TIMEOUT_SECONDS = 300  # per cell, every arm (unchanged from Env-1)
# Every arm, every environment: OpenMP single-threaded (as in Env-1). Env-2 adds a joblib/loky process cap,
# which matters only where joblib is importable (Arm A's sklearn), so concurrent runs cannot starve each other.
SUBPROCESS_ENV = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "2"}
MAX_OUTPUT_CHARS = 6000


@dataclass
class RunResult:
    status: str          # ok | error | rejected
    output: str          # what the agent sees
    error: str | None = None


def check_code(code: str, allow_modelling: bool = False) -> list[str]:
    """Return a list of violations (empty = allowed)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    allowed = ALLOWED_IMPORTS | (MODELLING_IMPORTS if allow_modelling else set())
    problems = []
    api_aliases = {"agent_api"} | {
        a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.name == "agent_api"
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "agent_api":
            for a in node.names:
                if a.name != "*" and a.name not in PUBLIC_API:
                    problems.append(f"agent_api has no public name {a.name!r}")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in api_aliases \
                and node.attr not in PUBLIC_API:
            problems.append(f"agent_api has no public name {node.attr!r}")
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""] if node.level == 0 else ["<relative import>"]
        else:
            mods = []
        for m in mods:
            if m.split(".")[0] not in allowed:
                problems.append(f"import of {m!r} is not allowed (allowed: {', '.join(sorted(allowed))})")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            problems.append(f"use of {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                problems.append(f"dunder attribute {node.attr!r} is not allowed")
            elif BLOCKED_ATTR.match(node.attr):
                problems.append(f"file I/O via .{node.attr} is not allowed; use agent_api (save_table) instead")
    for pat in DENY_PATTERNS:
        if re.search(pat, code):
            problems.append(f"code matches a blocked pattern ({pat!r})")
    return sorted(set(problems))


RUNNER = """\
import sys
sys.path.insert(0, {root!r})
import src  # noqa: F401  (xgboost before torch; see src/__init__.py)
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from src.researcher import agent_api
sys.modules["agent_api"] = agent_api
_ns = {{"__name__": "__main__", "agent_api": agent_api}}
exec("from agent_api import *", _ns)
_code = open({code_path!r}).read()
exec(compile(_code, "<your code>", "exec"), _ns)
"""


def _scrub(text: str, workspace: Path) -> str:
    # run dir before ROOT: its path names the arm and seed, which the agent must not see
    for p, repl in ((str(workspace), "."), (str(workspace.parent), "<run>"), (str(ROOT), "<harness>"),
                    (sys.prefix, "<python>")):
        text = text.replace(p, repl)
    if len(text) > MAX_OUTPUT_CHARS:
        half = MAX_OUTPUT_CHARS // 2
        text = text[:half] + f"\n... [{len(text) - MAX_OUTPUT_CHARS} chars truncated] ...\n" + text[-half:]
    return text


def run_python(code: str, workspace: Path, harness_dir: Path, allow_modelling: bool = False,
               timeout: int = TIMEOUT_SECONDS, extra_env: dict | None = None) -> RunResult:
    problems = check_code(code, allow_modelling)
    if problems:
        return RunResult("rejected", "REJECTED (code was not run):\n- " + "\n- ".join(problems), "; ".join(problems))
    harness_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S") + f"_{time.perf_counter_ns() % 10**6}"
    code_path = harness_dir / f"cell_{stamp}.py"
    runner_path = harness_dir / f"runner_{stamp}.py"
    code_path.write_text(code)
    runner_path.write_text(RUNNER.format(root=str(ROOT), code_path=str(code_path)))
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(runner_path)],
            cwd=workspace, env=SUBPROCESS_ENV | (extra_env or {}),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return RunResult("error", f"TIMEOUT after {timeout}s", "timeout")
    finally:
        runner_path.unlink(missing_ok=True)
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
    out = _scrub(out.strip() or "(no output — use print() to see results)", workspace)
    if proc.returncode != 0:
        return RunResult("error", out, f"exit code {proc.returncode}")
    return RunResult("ok", out)
