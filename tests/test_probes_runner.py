import json
import os
import sys

import pytest

from harness_fit import cli, task
from harness_fit.mockgw import MockGateway
from harness_fit.parsers import parse
from harness_fit.probes import Gateway, load_config, probe_one, thinking_verdict
from harness_fit.protocols import StreamResult, apply_event, build_request, redact, sse_events
from harness_fit.report import aggregate, render, score
from harness_fit.runner import build_cmd, expand_env, run_one

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def gw():
    mock = MockGateway(speed_scale=0.0)
    return Gateway("mock", {"base_url": "https://gateway.example.com", "api_key": "k"},
                   transport=mock.transport())


# ---------- protocols ----------

def test_build_request_shapes():
    p, b = build_request("openai", "m", "hi", tools=True, thinking="high")
    assert p == "/v1/chat/completions" and b["reasoning_effort"] == "high" and b["tools"][0]["type"] == "function"
    p, b = build_request("anthropic", "m", "hi", system="S", cache=True, thinking="xhigh")
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert b["thinking"]["budget_tokens"] == 16384 and b["max_tokens"] > 16384
    p, b = build_request("gemini", "gm", "hi", tools=True)
    assert p.startswith("/v1beta/models/gm:streamGenerateContent")
    assert b["tools"][0]["functionDeclarations"][0]["name"] == "lookup_sku"
    with pytest.raises(ValueError):
        build_request("grpc", "m", "hi")


def test_sse_and_openai_tool_call_fragments():
    lines = [": ping", 'data: {"model":"x","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup_sku","arguments":"{\\"sku\\":"}}]}}]}',
             'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"AB-1234\\"}"}}]}}]}',
             'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}',
             "data: [DONE]"]
    r = StreamResult()
    for ev in sse_events(lines):
        apply_event("openai", ev, r)
    assert r.tool_calls == [{"name": "lookup_sku", "args": '{"sku":"AB-1234"}'}]
    assert (r.input_tokens, r.output_tokens, r.cached_tokens) == (10, 5, 4)


# ---------- probes against mock gateway ----------

def test_probe_full_support(gw):
    row = probe_one(gw, "deepseek-v4", "anthropic", thinking_levels=["low", "high"])
    assert row.connect and row.tools and row.cache_hit
    assert row.cached_tokens > 0 and 0 < row.cache_ratio <= 1
    assert row.ttft_ms is not None and row.tok_s and row.tok_s > 0
    assert row.thinking["low"] < row.thinking["high"]


def test_probe_detects_missing_cache_and_tools(gw):
    g = probe_one(gw, "gemini-3-pro", "openai")
    assert g.connect and g.tools and g.cache_hit is False
    a = probe_one(gw, "argon-1", "openai")
    assert a.connect and a.tools is False and a.cache_hit is False


def test_probe_unroutable_protocol(gw):
    row = probe_one(gw, "argon-1", "anthropic")
    assert not row.connect and row.status == 400 and "not routable" in row.error and row.tools is None
    assert row.error.startswith("HTTP 400")


def test_thinking_alias_and_reject(gw):
    row = probe_one(gw, "gemini-3-pro", "gemini", thinking_levels=["low", "high", "xhigh"])
    assert thinking_verdict(row.thinking) == "low=300 / high=1800 / xhigh==high"
    row = probe_one(gw, "deepseek-v4", "openai", thinking_levels=["high", "xhigh"])
    assert row.thinking["xhigh"] == -1 and "xhigh=rejected" in thinking_verdict(row.thinking)
    assert thinking_verdict({"a": 5, "b": 5}) == "no effect"
    assert thinking_verdict({"a": 0, "b": 0}) == "not reported"


def test_load_config_validates(tmp_path):
    fp = tmp_path / "c.yaml"
    fp.write_text("gateways: {g: {base_url: https://gateway.example.com}}\nmodels: [{id: m, gateway: nope}]\n")
    with pytest.raises(ValueError):
        load_config(str(fp))
    fp.write_text("gateways: {g: {base_url: https://gateway.example.com}}\nmodels: [{id: m, gateway: g}]\n")
    cfg = load_config(str(fp))
    assert cfg["models"][0]["protocols"] == ["openai", "anthropic", "gemini"]


# ---------- parsers ----------

def test_parsers_count_tools_and_errors():
    pi = "\n".join(json.dumps(e) for e in [
        {"type": "tool_execution_end", "toolName": "grep", "isError": True},
        {"type": "tool_execution_end", "toolName": "edit", "isError": False},
        {"type": "message_end", "message": {"role": "assistant", "usage": {"input": 100, "output": 10, "cacheRead": 300},
                                            "content": [{"type": "text", "text": "DONE"}]}}])
    s = parse("pi", "noise\n" + pi)
    assert (s.tool_calls, s.tool_errors, s.tool_success, s.final_text) == (2, 1, 0.5, "DONE")
    assert s.cache_ratio == 0.75
    codex = "\n".join(json.dumps(e) for e in [
        {"type": "item.completed", "item": {"type": "command_execution", "exit_code": 2}},
        {"type": "item.completed", "item": {"type": "file_change", "status": "completed"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 50}}])
    c = parse("codex", codex)
    assert (c.tool_calls, c.tool_errors, c.cached_tokens, c.input_tokens) == (2, 1, 600, 400)
    with pytest.raises(ValueError):
        parse("hermes", "")


def test_parsers_surface_harness_errors():
    cl = json.dumps({"type": "result", "is_error": True, "result": "Not logged in", "usage": {}})
    assert parse("claude", cl).error == "Not logged in"
    cx = json.dumps({"type": "turn.failed", "error": {"message": "401 Unauthorized"}})
    assert parse("codex", cx).error == "401 Unauthorized"
    pi = json.dumps({"type": "message_end", "message": {"role": "assistant", "stopReason": "error",
                                                      "errorMessage": "429 quota", "content": []}})
    assert parse("pi", pi).error == "429 quota"


def test_redact_masks_keys():
    fake = "sk-" + "demo" * 3 + "***"
    out = redact(f"401: Incorrect API key provided: {fake}. Bearer abcdefghijkl")
    assert "demo" not in out and "abcdefghijkl" not in out and out.count("[redacted]") == 2


# ---------- task fixture + runner ----------

def test_fixture_starts_broken_and_grades():
    ws = task.make_workspace()
    try:
        g = task.grade(ws)
        assert not g["solved"] and g["tests_untouched"] and g["passed"] < g["total"]
    finally:
        task.cleanup(ws)


def _fake(kind):
    return {"name": kind, "kind": kind, "env": {"PYTHONPATH": ROOT},
            "cmd": [sys.executable, "-m", "harness_fit.fakeharness", "--kind", kind, "--model", "{model}", "{prompt}"]}


@pytest.mark.parametrize("kind", ["pi", "claude", "codex"])
def test_run_one_with_fake_harness(kind):
    good = run_one(_fake(kind), "deepseek-v4", log=lambda *_: None)
    assert good.solved and good.checks_passed == good.checks_total and good.tool_calls == 4
    assert good.tool_success == 1.0 and good.exit_code == 0
    partial = run_one(_fake(kind), "argon-1", log=lambda *_: None)
    assert not partial.solved and 0 < partial.checks_passed < partial.checks_total
    assert partial.tool_errors == 3


def test_run_one_failures():
    bad = run_one(_fake("pi"), "no-such-model", log=lambda *_: None)
    assert bad.exit_code == 1 and "not available" in bad.error and not bad.solved
    missing = run_one({"name": "x", "kind": "pi", "cmd": ["hfb-definitely-missing-bin"]}, "m", log=lambda *_: None)
    assert missing.error.startswith("command not found")


def test_build_cmd_and_env(monkeypatch):
    monkeypatch.setenv("HFB_TEST_KEY", "abc")
    assert expand_env({"T": "${HFB_TEST_KEY}"}) == {"T": "abc"}
    cmd = build_cmd({"kind": "codex"}, "m1", "do it")
    assert cmd[:2] == ["codex", "exec"] and "m1" in cmd and cmd[-1] == "do it"
    assert build_cmd({"kind": "pi", "cmd": "tool --m {model}"}, "z", "p") == ["tool", "--m", "z"]


# ---------- report + cli ----------

def test_score_ranks_solved_first():
    runs = [
        {"harness": "a", "model": "m", "solved": True, "checks_passed": 6, "checks_total": 6, "tool_calls": 4,
         "tool_errors": 0, "wall_s": 10, "input_tokens": 1000, "output_tokens": 100, "cached_tokens": 0,
         "cache_ratio": 0.0, "error": ""},
        {"harness": "b", "model": "m", "solved": False, "checks_passed": 3, "checks_total": 6, "tool_calls": 9,
         "tool_errors": 5, "wall_s": 2, "input_tokens": 500, "output_tokens": 50, "cached_tokens": 0,
         "cache_ratio": 0.0, "error": "timeout"},
    ]
    rows = score(aggregate(runs))
    assert rows[0]["harness"] == "a" and rows[0]["score"] > rows[1]["score"]
    md = render([], runs)
    assert "Harness leaderboard" in md and "Broken: b + m: timeout" in md


def test_cli_init_and_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "-o", "bench.yaml"]) == 0
    cfg = load_config("bench.yaml")
    assert {h["kind"] for h in cfg["harnesses"]} == {"pi", "claude", "codex"}
    assert cli.main(["init", "-o", "bench.yaml"]) == 1
    from harness_fit.cli import demo_config, execute
    cfg = demo_config()
    cfg["models"] = cfg["models"][:2]
    for h in cfg["harnesses"]:
        h["models"] = ["deepseek-v4", "argon-1"]
    md = execute(cfg, str(tmp_path / "out"), do_probe=True, do_run=True,
                 transport=MockGateway(speed_scale=0.0).transport(), log=lambda *_: None)
    assert "Protocol probes" in md and "Best overall" in md
    assert os.path.isfile(tmp_path / "out" / "runs.json")
    assert cli.main(["report", "-o", str(tmp_path / "out")]) == 0
