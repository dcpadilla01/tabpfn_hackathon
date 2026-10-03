"""Arms differ only in their tool list (and, for experiment(), the fixed backend)."""

from __future__ import annotations

ARMS = {
    "b":       {"tools": ["inspect", "run_python", "experiment"], "backend": "tabpfn", "allow_modeling_imports": False},
    "a_prime": {"tools": ["inspect", "run_python", "experiment"], "backend": "xgb",    "allow_modeling_imports": False},
    "a":       {"tools": ["inspect", "run_python", "score"],      "backend": None,     "allow_modeling_imports": True},
}


def arm_config(arm: str) -> dict:
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return ARMS[arm]
