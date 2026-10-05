"""Arm A: score() contract, labels never revealed, accounting identical to experiment()."""

import pandas as pd
import pytest

from src.data.accessor import target_keys
from src.evaluation.logger import ExperimentLog
from src.researcher import prompts
from src.researcher.arms import arm_config
from src.researcher.trace import RunTrace
from src.tools.score import ScoreTool


@pytest.fixture
def setup(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    val = target_keys(("validation",))[["household_key", "snapshot_day"]]
    val.assign(prediction=75.0).to_parquet(ws / "p.parquet", index=False)
    log = ExperimentLog("a", 0, root=tmp_path / "r")
    trace = RunTrace("a", 0, root=tmp_path / "r", run_id="x")
    return ScoreTool(None, log, ws, trace), log, trace, ws, val


def test_valid_score_logs_and_hides_labels(setup):
    tool, log, trace, ws, _ = setup
    trace.begin_experiment(log.next_id())
    res = tool("p.parquet", "constant guess", "E000", "predict 75 for everyone", "baseline check")
    assert res["status"] == "ok" and res["mae"] > 0 and set(res) == {"experiment_id", "status", "error", "mae", "r2", "backend"}
    rec = log.records()[-1]
    assert rec["backend"] == "agent" and rec["tool_calls"] == 1 and rec["n_eval"] == 9989
    saved = pd.read_parquet(log.dir / "E000" / "predictions.parquet")
    assert "future_spend_4w" not in saved.columns


@pytest.mark.parametrize("mutate, msg", [
    (lambda v: v.iloc[1:].assign(prediction=1.0), "missing"),
    (lambda v: v.assign(prediction=float("nan")), "NaN"),
    (lambda v: v.assign(prediction="x"), "numeric"),
    (lambda v: v.assign(pred=1.0), "missing columns"),
    (lambda v: pd.concat([v, target_keys(("train",))[["household_key", "snapshot_day"]].head(3)]).assign(prediction=1.0), "unexpected"),
])
def test_invalid_predictions_consume_an_experiment(setup, mutate, msg):
    tool, log, _, ws, val = setup
    mutate(val).to_parquet(ws / "bad.parquet", index=False)
    res = tool("bad.parquet", "h", "E000", "m")
    assert res["status"] == "invalid" and msg in res["error"] and log.next_id() == "E001"


def test_prompts_identical_except_tool_section():
    days = {"train": [95], "validation": [459]}
    a = prompts.system_prompt(arm_config("a"), 20, days, 459)
    ap = prompts.system_prompt(arm_config("a_prime"), 20, days, 459)
    b = prompts.system_prompt(arm_config("b"), 20, days, 459)
    assert ap == b
    obj = lambda s: s.split("# Tools")[0].replace("`score`", "`EVAL`").replace("`experiment`", "`EVAL`")
    assert obj(a) == obj(ap)                      # objective identical up to the evaluation tool's name
    assert "score(predictions_path" in a and "experiment(table_path" not in a
    assert "sklearn" in a and "sklearn" not in ap  # modelling imports only in Arm A
