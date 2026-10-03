import json

from src.researcher.trace import RunTrace, parse_usage


def fake_response(gen_id, prompt, completion, cached=0, reasoning_text=None, cost=None):
    msg = {"role": "assistant", "content": "ok"}
    if reasoning_text:
        msg["reasoning"] = reasoning_text
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "prompt_tokens_details": {"cached_tokens": cached},
             "completion_tokens_details": {"reasoning_tokens": 7}}
    if cost is not None:
        usage["cost"] = cost
    return {"id": gen_id, "model": "m", "choices": [{"message": msg, "finish_reason": "stop"}], "usage": usage}


def test_uncached_equivalent_counts_cached_prompt_tokens():
    u = parse_usage(fake_response("g", 1000, 50, cached=900))
    assert u["uncached_equivalent_tokens"] == 1050 and u["cached_tokens"] == 900


def test_missing_usage_is_tolerated():
    assert parse_usage({"choices": []})["uncached_equivalent_tokens"] == 0


def test_trace_rollup_and_reasoning_separation(tmp_path):
    tr = RunTrace("b", 0, root=tmp_path, run_id="r1")
    tr.begin_experiment("E001")
    tr.log_llm_call(request={"model": "m", "temperature": 0.7}, response=fake_response("g1", 100, 10, 80, "secret thoughts", 0.001), wall_clock_seconds=1.0)
    tr.log_tool_call(tool="run_python", args={"code": "x"}, output="out", status="rejected", wall_clock_seconds=0.1, error="import not allowed")
    tr.log_tool_call(tool="experiment", args={}, output={"mae": 1}, status="ok", wall_clock_seconds=5.0)
    tr.begin_experiment("E002")
    tr.log_llm_call(request={"model": "m"}, response=fake_response("g2", 200, 20), wall_clock_seconds=1.0)

    r = tr.rollup("E001")
    assert (r["tokens_in"], r["tokens_out"], r["cached_tokens"], r["tool_calls"], r["tool_calls_rejected"]) == (100, 10, 80, 2, 1)
    assert r["uncached_equivalent_tokens"] == 110 and r["llm_cost_usd"] == 0.001
    assert tr.rollup("E002")["tokens_in"] == 200

    transcript = (tmp_path / "b/0/transcript.jsonl").read_text()
    assert "secret thoughts" not in transcript
    reasoning = [json.loads(l) for l in (tmp_path / "b/0/reasoning.jsonl").read_text().splitlines()]
    assert reasoning[0]["generation_id"] == "g1" and reasoning[0]["reasoning"][0]["reasoning"] == "secret thoughts"
    assert len(reasoning) == 1  # no reasoning in g2 → no line
