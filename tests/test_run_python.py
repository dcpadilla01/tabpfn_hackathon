"""run_python static enforcement and the agent_api isolation guarantees."""

import pytest

from src.tools.run_python import check_code, run_python


@pytest.mark.parametrize("code", [
    "import os", "import pyarrow.parquet as pq", "from pathlib import Path", "import sklearn",
    "open('x')", "pd.read_csv('a')", "df.to_csv('a')", "x = getattr(pd, 'read_' + 'csv')",
    "x.__class__", "np.load('a')", "import xgboost", "__import__('os')", "import duckdb",
])
def test_rejected(code):
    assert check_code(code)


@pytest.mark.parametrize("code", [
    "import pandas as pd\nprint(pd.__name__ if False else 1)" .replace("pd.__name__ if False else ", ""),
    "from agent_api import build_features", "global x\nx = 1", "spread_x = 1\nprint(spread_x)",
    "import numpy as np\nimport math\nfrom collections import Counter",
])
def test_allowed(code):
    assert check_code(code) == []


def test_modelling_imports_only_when_allowed():
    assert check_code("import sklearn") and check_code("import sklearn", allow_modelling=True) == []


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "ws").mkdir()
    return tmp_path / "ws", tmp_path / "harness"


def test_runs_and_scrubs_paths(ws):
    w, h = ws
    r = run_python("v = snapshot()\nprint(v.day, len(v.transactions) > 0)\nraise ValueError('boom')", w, h)
    assert r.status == "error" and "boom" in r.output and "459 True" in r.output
    assert "/Users/" not in r.output


def test_direct_access_capped_at_research_horizon(ws):
    w, h = ws
    r = run_python("snapshot(543)", w, h)
    assert r.status == "error" and "research horizon" in r.output


@pytest.mark.parametrize("code", [
    "from agent_api import os as o\nprint(o.environ)",
    "import agent_api as a\nprint(a.os)",
    "import pandas as pd\nx = pd.io.common.os",
    "import agent_api\nagent_api._load_table('transaction_data')",
])
def test_escape_hatches_rejected(code):
    assert check_code(code)


def test_subprocess_env_is_stripped():
    import inspect

    from src.tools import run_python as rp

    assert 'env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"}' in inspect.getsource(rp.run_python)


def test_stash_across_snapshots_does_not_leak(ws):
    """Harvest later-snapshot data into a global on one pass, inject on the next: must fail."""
    w, h = ws
    code = '''
stash = {}
def harvest(view, day):
    stash[day] = view.transactions["day"].max()
    return pd.DataFrame({"x": 0.0}, index=view.households)
build_features(harvest)
print("stash after build:", len(stash))
def inject(view, day):
    return pd.DataFrame({"x": float(len(stash))}, index=view.households)
ft = build_features(inject)
print("injected max:", ft["x"].max())
print("saved:", save_table(ft, "t"))
'''
    r = run_python(code, w, h)
    assert r.status == "ok", r.output
    assert "stash after build: 0" in r.output and "injected max: 0.0" in r.output
    assert (w / "t.parquet").exists()


def test_file_write_inside_fn_is_denied(ws):
    w, h = ws
    code = '''
def fn(view, day):
    save_table(view.transactions.head(), "leak")
    return pd.DataFrame({"x": 0.0}, index=view.households)
build_features(fn)
'''
    r = run_python(code, w, h)
    assert r.status == "error" and "cannot be called inside fn" in r.output


def test_agent_api_name_is_bound(ws):
    w, h = ws
    r = run_python("print(agent_api.RESEARCH_VISIBLE_DAY, RESEARCH_VISIBLE_DAY)", w, h)
    assert r.status == "ok" and "459 459" in r.output


def test_save_table_accepts_parquet_suffix(ws):
    w, h = ws
    r = run_python("print(save_table(pd.DataFrame({'a': [1]}), 'x.y.parquet'))", w, h)
    assert r.status == "ok" and "xy.parquet" in r.output and (w / "xy.parquet").exists()
    r = run_python("print(save_table(pd.DataFrame({'a': [1]}), 'e003_mix.parquet'))", w, h)
    assert "e003_mix.parquet" in r.output and (w / "e003_mix.parquet").exists()


def test_run_dir_not_revealed(tmp_path):
    run_dir = tmp_path / "experiments" / "results" / "b" / "1"
    w = run_dir / "workspace"; w.mkdir(parents=True)
    r = run_python("raise ValueError('x')", w, run_dir / "cells")
    assert "results/b/1" not in r.output and "<run>" in r.output


def test_fn_output_suppressed_at_validation_snapshots(ws):
    w, h = ws
    code = """
def fn(view, day):
    print("SNAP", day, "SECRET", view.transactions["day"].max())
    return pd.DataFrame({"x": 0.0}, index=view.households)
build_features(fn)
"""
    r = run_python(code, w, h)
    assert r.status == "ok" and "SNAP 431" in r.output
    assert "SNAP 459" not in r.output and "SNAP 543" not in r.output


def test_fn_error_message_suppressed_at_validation_snapshots(ws):
    w, h = ws
    code = """
def fn(view, day):
    if day >= 459:
        raise ValueError(f"leaky {view.transactions['sales_value'].sum()}")
    return pd.DataFrame({"x": 0.0}, index=view.households)
build_features(fn)
"""
    r = run_python(code, w, h)
    assert r.status == "error" and "ValueError at line" in r.output and "leaky" not in r.output
