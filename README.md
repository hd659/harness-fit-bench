# harness-fit-bench

Pick the best harness x model combo from measurements, not from vibes.

You list your gateways and models. The bench then answers, for every model:

- **Protocols**: does it connect through `openai` (`/v1/chat/completions`), `anthropic` (`/v1/messages`) and `gemini` (`:streamGenerateContent`)? This settles questions like "can Claude Code in 3p mode run gemini" or "can deepseek speak the Claude protocol" on your own gateway.
- **Tool calling**: does the model emit a real `lookup_sku` tool call with the right argument?
- **Prompt cache**: send the same ~2.5k token prefix twice. Does the second call report cached tokens, and what share of the input was cached?
- **Speed**: time to first token (TTFT) and output tokens per second on a streamed answer.
- **Thinking levels**: send `low/high/xhigh` and record the reasoning tokens. The report flags levels that are rejected, that behave the same as another level (`xhigh==high`), or that the gateway never reports.

It then runs **pi / claude / codex** headless on the same small coding task. The task is a two-bug `shop/` package, scored by a hidden checker so editing the tests does not help. For each combo it records solve rate, tool calls, tool success rate, wall time, tokens and cache ratio, then ranks everything on one leaderboard.

## Install

```bash
pip install httpx pyyaml pytest   # Python 3.9+
```

## Quick start

```bash
# 1. Offline demo: mock gateway + scripted harnesses, no keys or network needed (~10 s)
python3 -m harness_fit demo -o hfb-demo

# 2. Real bench
python3 -m harness_fit init -o bench.yaml        # writes an example config
export HFB_GATEWAY_KEY=...                       # keys come from env vars, never the YAML
python3 -m harness_fit probe -c bench.yaml       # protocol / tool / cache / speed / thinking
python3 -m harness_fit run   -c bench.yaml --only pi claude-3p --repeats 2
python3 -m harness_fit all   -c bench.yaml       # probe + run + report
python3 -m harness_fit report -o hfb-out         # re-render report.md from saved JSON
```

Output goes to `hfb-out/`: `probes.json`, `runs.json` and `report.md`.

## Config

```yaml
gateways:
  main:
    base_url: https://gateway.example.com
    api_key_env: HFB_GATEWAY_KEY
    paths:                           # optional per-protocol path override
      gemini: /gemini/v1beta/models/{model}:streamGenerateContent?alt=sse
models:
  - {id: gemini-3-pro, gateway: main, protocols: [openai, anthropic, gemini], thinking: [low, high, xhigh]}
  - {id: deepseek-v4, gateway: main, protocols: [openai, anthropic]}
harnesses:
  - {name: pi, kind: pi, models: [gemini-3-pro, deepseek-v4]}
  - name: claude-3p
    kind: claude
    models: [gemini-3-pro]
    env: {ANTHROPIC_BASE_URL: "https://gateway.example.com", ANTHROPIC_AUTH_TOKEN: "${HFB_GATEWAY_KEY}"}
  - {name: codex, kind: codex, models: [deepseek-v4]}
run: {timeout: 600, keep_workspace: false}
```

Default harness commands (override with `cmd:`, where `{model}` and `{prompt}` are substituted):

| kind | command | events parsed |
|---|---|---|
| pi | `pi -p --mode json --no-session --model {model}` | `tool_execution_end.isError`, `message_end.usage` |
| claude | `claude -p --output-format stream-json --verbose --dangerously-skip-permissions` | `tool_use` / `tool_result.is_error`, `result.usage` |
| codex | `codex exec --json --skip-git-repo-check -s workspace-write -m {model}` | `item.completed` (command / file_change), `turn.completed.usage` |

Each run happens in a fresh temporary copy of the fixture, so the harness only touches that copy. Errors the harness reports itself (not logged in, 401, 429) appear in the report. Key-like strings are masked before anything is written to disk.

## Scoring

```
score = 50*solve_rate + 15*tool_success + 15*speed + 10*token_efficiency + 10*cache_ratio
```

`speed` and `token_efficiency` are relative to the best combo that solved the task, so scores are only comparable within one bench run. Combos that did not solve the task get 0 on both.

## Reading the probe table

| column | meaning |
|---|---|
| connect | streaming request returned content through this protocol |
| tools | the model called the probe tool with the right argument |
| cache | second identical-prefix call reported cached tokens, plus the share of input cached |
| note | HTTP error, or `served as X` when the gateway answered with a different model id |
| thinking | reasoning tokens per level: `no effect`, `not reported`, `xhigh==high` (alias) or `rejected` |

A `no` under cache with `yes` under connect means the request went through but nothing was cached. Comparing the same model across protocols shows whether one particular route loses the cache.

## Layout

```
harness_fit/
  protocols.py   request builders + SSE parsers for the 3 protocols
  probes.py      connectivity / tools / cache / speed / thinking probes
  task.py        shared fixture repo + hidden grader
  parsers.py     pi / claude / codex JSONL event parsers
  runner.py      harness x model matrix runner
  report.py      scoring, leaderboard, verdicts
  mockgw.py      httpx.MockTransport gateway for demo/tests
  fakeharness.py scripted CLI that emits real event shapes
  cli.py         argparse entry point
tests/test_probes_runner.py
```

## Tests

```bash
python3 -m pytest -q
```
