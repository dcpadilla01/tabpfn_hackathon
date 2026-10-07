"""Load config/default.yaml and .env; expose package versions for experiment logs."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"


load_dotenv(ROOT / ".env")
# The local `tabpfn` package reads TABPFN_TOKEN for the weight-download licence check.
if os.environ.get("TABPFN_API_KEY") and not os.environ.get("TABPFN_TOKEN"):
    os.environ["TABPFN_TOKEN"] = os.environ["TABPFN_API_KEY"]

PINNED_PACKAGES = ("tabpfn", "tabpfn-client", "xgboost", "openai", "pandas", "scikit-learn")


@lru_cache
def load_config(path: str | Path = ROOT / "config" / "default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Experiment environment (see docs/build_plan_2026-10.md "Environments and primary results"). Env-1 = the reported runs in
# experiments/results; Env-2 = the post-audit harness, kept in experiments/results_env2. Never mixed.
ENVIRONMENT = os.environ.get("EXPERIMENT_ENV") or load_config().get("environment", "env1")
if ENVIRONMENT not in ("env1", "env2"):
    raise ValueError(f"unknown environment {ENVIRONMENT!r}")
_SUFFIX = "" if ENVIRONMENT == "env1" else "_env2"
RESULTS_DIR = ROOT / "experiments" / f"results{_SUFFIX}"
ANALYSIS_DIR = ROOT / "experiments" / f"analysis{_SUFFIX}"


def package_versions() -> dict[str, str]:
    return {p: version(p) for p in PINNED_PACKAGES}


def tabpfn_token() -> str:
    token = os.environ.get("TABPFN_API_KEY") or os.environ.get("TABPFN_TOKEN")
    if not token:
        raise RuntimeError("Set TABPFN_API_KEY in .env (see .env.example).")
    return token
