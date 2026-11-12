"""Run every configured harness x model on the shared fixture task and collect stats.

Harness entries in the YAML config:

    harnesses:
      - name: claude-3p
        kind: claude                       # pi | claude | codex (selects parser + default cmd)
        models: [gemini-3-pro, deepseek-v4]
        env: {ANTHROPIC_BASE_URL: "https://gw.example.com", ANTHROPIC_AUTH_TOKEN: "${GW_KEY}"}
        cmd: [...]                         # optional override; {model} and {prompt} are substituted
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import task as fixture
from .parsers import parse
from .protocols import redact

DEFAULT_CMDS = {
    "pi": ["pi", "-p", "--mode", "json", "--no-session", "--model", "{model}", "{prompt}"],
    "claude": ["claude", "-p", "--output-format", "stream-json", "--verbose",
               "--dangerously-skip-permissions", "--model", "{model}", "{prompt}"],
    "codex": ["codex", "exec", "--json", "--skip-git-repo-check", "-s", "workspace-write",
              "-m", "{model}", "{prompt}"],
}


@dataclass
class RunResult:
    harness: str
    kind: str
    model: str
    solved: bool = False
    checks_passed: int = 0
    checks_total: int = 0
    tests_untouched: bool = True
    tool_calls: int = 0
    tool_errors: int = 0
    tool_success: float = 0.0
    by_tool: dict = field(default_factory=dict)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_ratio: float = 0.0
    wall_s: float = 0.0
    exit_code: Optional[int] = None
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def expand_env(env: dict) -> dict:
    """Expand ${VAR} references so keys stay in the shell environment, not in the YAML."""
    return {k: os.path.expandvars(str(v)) for k, v in (env or {}).items()}


def build_cmd(h: dict, model: str, prompt: str) -> list:
    tpl = h.get("cmd") or DEFAULT_CMDS[h["kind"]]
    if isinstance(tpl, str):
        tpl = shlex.split(tpl)
    return [part.replace("{model}", model).replace("{prompt}", prompt) for part in tpl]


def run_one(h: dict, model: str, *, timeout: float = 600, keep: bool = False, log=print) -> RunResult:
    res = RunResult(harness=h["name"], kind=h["kind"], model=model)
    ws = fixture.make_workspace()
    cmd = build_cmd(h, model, h.get("prompt", fixture.TASK_PROMPT))
    env = {**os.environ, **expand_env(h.get("env"))}
    t0 = time.perf_counter()
    stdout = ""
    try:
        p = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        stdout, res.exit_code = p.stdout, p.returncode
        if p.returncode != 0:
            res.error = (p.stderr or p.stdout).strip().splitlines()[-1][:200] if (p.stderr or p.stdout) else \
                f"exit {p.returncode}"
    except FileNotFoundError:
        res.error = f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        res.error = f"timeout after {timeout}s"
    res.wall_s = round(time.perf_counter() - t0, 2)
    st = parse(h["kind"], stdout)
    res.tool_calls, res.tool_errors, res.tool_success = st.tool_calls, st.tool_errors, st.tool_success
    res.by_tool = dict(st.by_tool)
    res.turns = st.turns
    res.input_tokens, res.output_tokens, res.cached_tokens = st.input_tokens, st.output_tokens, st.cached_tokens
    res.cache_ratio = st.cache_ratio
    if st.error:  # the harness's own error is more specific than the last stderr line
        res.error = st.error
    res.error = redact(res.error)[:200]
    g = fixture.grade(ws)
    res.solved, res.checks_passed, res.checks_total = g["solved"], g["passed"], g["total"]
    res.tests_untouched = g["tests_untouched"]
    if keep:
        log(f"[run] workspace kept at {ws}")
    else:
        fixture.cleanup(ws)
    log(f"[run] {h['name']:<12} {model:<24} solved={res.solved} checks={res.checks_passed}/"
        f"{res.checks_total} tools={res.tool_calls} ok={res.tool_success:.0%} "
        f"wall={res.wall_s}s {res.error}")
    return res


def run_matrix(cfg: dict, *, only: Optional[list] = None, repeats: int = 1, log=print) -> list:
    rcfg = cfg.get("run", {})
    timeout = rcfg.get("timeout", 600)
    out = []
    for h in cfg.get("harnesses", []):
        if h.get("kind") not in DEFAULT_CMDS:
            raise ValueError(f"harness {h.get('name')}: kind must be one of {sorted(DEFAULT_CMDS)}")
        if only and h["name"] not in only:
            continue
        models = h.get("models") or [m["id"] for m in cfg.get("models", [])]
        for model in models:
            for _ in range(max(1, repeats)):
                out.append(run_one(h, model, timeout=timeout, keep=rcfg.get("keep_workspace", False), log=log))
    return out
