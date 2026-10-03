"""experiment() primitive: contract, logging, rollup. XGBoost backend (local, no API)."""

import json

import pandas as pd
import pytest

from src.data.accessor import build_features
from src.evaluation.logger import ExperimentLog
from src.researcher.trace import RunTrace
from src.tools.experiment import ExperimentTool


@pytest.fixture(scope="module")
def table():
    return build_features(lambda v, d: pd.DataFrame({"n_lines": v.transactions.groupby("household_key").size()}))


@pytest.fixture
def setup(tmp_path, table):
    ws = tmp_path / "workspace"; ws.mkdir()
    table.to_parquet(ws / "f.parquet", index=False)
    log = ExperimentLog("a_prime", 0, root=tmp_path / "results")
    trace = RunTrace("a_prime", 0, root=tmp_path / "results", run_id="r")
    return ExperimentTool("xgb", log, ws, trace), log, trace, ws


def test_valid_experiment_logs_and_persists(setup):
    tool, log, trace, ws = setup
    trace.begin_experiment(log.next_id())
    trace.log_tool_call(tool="run_python", args={"code": "..."}, output="", status="ok", wall_clock_seconds=0.1)
    res = tool("f.parquet", "volume predicts spend", "E000", "+ n_lines", "first try")
    assert res["status"] == "ok" and res["mae"] > 0 and res["experiment_id"] == "E000"
    rec = log.records()[-1]
    assert rec["tool_calls"] == 2 and rec["parent_id"] == "E000" and rec["reasoning_summary"] == "first try"
    assert rec["wall_clock_seconds"] >= rec["runtime_seconds"]
    d = log.dir / "E000"
    preds = pd.read_parquet(d / "predictions.parquet")
    assert "future_spend_4w" not in preds.columns and (d / "features.parquet").exists()


@pytest.mark.parametrize("path, msg", [("../outside.parquet", "inside your workspace"),
                                       ("f.csv", ".parquet"), ("missing.parquet", "no file")])
def test_bad_paths_are_invalid_but_consume_an_experiment(setup, path, msg):
    tool, log, _, _ = setup
    res = tool(path, "h", "E000", "m")
    assert res["status"] == "invalid" and msg in res["error"]
    assert log.records()[-1]["status"] == "invalid" and log.next_id() == "E001"


def test_unknown_parent_and_missing_metadata(setup):
    tool, _, _, _ = setup
    res = tool("f.parquet", "", "E042", "")
    assert res["status"] == "invalid"
    assert "hypothesis is required" in res["error"] and "E042" in res["error"] and "mutation" in res["error"]
