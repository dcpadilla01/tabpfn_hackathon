"""Agent loop control flow with a scripted fake LLM (no API calls). Backend: xgb (local)."""

import json
from types import SimpleNamespace

import pytest

import src.researcher.agent as agent_mod
from src.evaluation import logger as logger_mod
from src.researcher import trace as trace_mod

SAVE = "ft = build_features(lambda v, d: pd.DataFrame({'n': v.transactions.groupby('household_key').size()}))\nprint(save_table(ft, 'f'))"


def tc(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls, content=None):
    return {"id": "gen", "model": "fake", "provider": "fake",
            "choices": [{"finish_reason": "tool_calls" if calls else "stop",
                         "message": {"role": "assistant", "content": content, "tool_calls": list(calls) or None,
                                     "reasoning": "thinking..."}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.seen = []

    def create(self, **kw):
        self.seen.append(json.loads(json.dumps(kw)))
        d = self.script.pop(0)
        return SimpleNamespace(model_dump=lambda: d)


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(logger_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(trace_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(agent_mod, "ExperimentLog", lambda arm, seed: logger_mod.ExperimentLog(arm, seed, root=tmp_path))
    monkeypatch.setattr(agent_mod, "RunTrace", lambda arm, seed: trace_mod.RunTrace(arm, seed, root=tmp_path))

    def _make(script, budget):
        r = agent_mod.Researcher("a_prime", 0, budget)
        r.client = FakeClient(script)
        return r
    return _make


def test_happy_path_cap_and_nudges(make):
    exp = {"table_path": "f.parquet", "hypothesis": "volume", "parent": "E000", "mutation": "+n", "reasoning_summary": "r"}
    script = [
        reply(tc(1, "run_python", {"code": SAVE})),
        reply(tc(2, "experiment", exp)),                                   # E001 ok
        *[reply(tc(10 + i, "inspect", {"table": "products", "op": "shape"})) for i in range(16)],  # E002 cap
        *[reply(content="hmm") for _ in range(4)],                          # E003 stops calling tools
    ]
    r = make(script, budget=3)
    recs = r.run()
    assert [x["experiment_id"] for x in recs] == ["E000", "E001", "E002", "E003"]
    assert recs[1]["status"] == "ok" and recs[1]["tool_calls"] == 2 and recs[1]["generated_code_size"] > 0
    assert recs[2]["status"] == "invalid" and "exceeded 15 tool calls" in recs[2]["error"] and recs[2]["tool_calls"] == 15
    assert recs[3]["status"] == "invalid" and "stopped calling tools" in recs[3]["error"]
    assert recs[1]["tokens_in"] == 200 and recs[1]["llm_calls"] == 2
    # reasoning never fed back to the model
    assert all("reasoning" not in m for kw in r.client.seen for m in kw["messages"])
    # context resets per experiment: first E002 call has only system + user
    assert len(r.client.seen[2]["messages"]) == 2
    # tool results carry the call counter; warning near the cap
    e002_last = r.client.seen[2 + 15]["messages"]
    tool_msgs = [m["content"] for m in e002_last if m["role"] == "tool"]
    assert tool_msgs[0].endswith("[tool calls since last experiment: 1/15]")
    assert "1 left" in tool_msgs[13] and "0 left" in tool_msgs[14]


def test_bad_experiment_args_consume_budget(make):
    r = make([reply(tc(1, "experiment", {"table_path": "nope.parquet", "bogus": 1}))], budget=1)
    recs = r.run()
    assert recs[-1]["status"] == "invalid" and "hypothesis is required" in recs[-1]["error"]


def test_arm_a_scores_its_own_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "ExperimentLog", lambda arm, seed: logger_mod.ExperimentLog(arm, seed, root=tmp_path))
    monkeypatch.setattr(agent_mod, "RunTrace", lambda arm, seed: trace_mod.RunTrace(arm, seed, root=tmp_path))
    code = ("from sklearn.linear_model import Ridge\n"
            "ft = build_features(lambda v, d: pd.DataFrame({'n': v.transactions.groupby('household_key').size()}))\n"
            "tt = train_targets(); tr = ft.merge(tt, on=KEYS)\n"
            "m = Ridge().fit(tr[['n']].fillna(0), tr[TARGET])\n"
            "va = ft[ft.snapshot_day >= 459].copy(); va['prediction'] = m.predict(va[['n']].fillna(0))\n"
            "print(save_table(va[KEYS + ['prediction']], 'p'))")
    args = {"predictions_path": "p.parquet", "hypothesis": "volume", "parent": "E000", "mutation": "ridge on n", "reasoning_summary": "r"}
    r = agent_mod.Researcher("a", 0, 1)
    r.client = FakeClient([reply(tc(1, "run_python", {"code": code})), reply(tc(2, "score", args))])
    recs = r.run()
    assert recs[0]["backend"] == "xgb" and recs[0]["experiment_id"] == "E000"   # harness reference root
    assert recs[1]["status"] == "ok" and recs[1]["backend"] == "agent" and recs[1]["mae"] > 0
