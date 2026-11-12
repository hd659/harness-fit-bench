"""Command line entry point.

    python -m harness_fit init   [-o bench.yaml]          write an example config
    python -m harness_fit probe  -c bench.yaml [-o out]   protocol / tool / cache / speed probes
    python -m harness_fit run    -c bench.yaml [-o out]   run harnesses on the shared task
    python -m harness_fit all    -c bench.yaml [-o out]   probe + run + report
    python -m harness_fit report -o out                   re-render report from saved JSON
    python -m harness_fit demo   [-o out]                 offline end-to-end demo (mock gateway + fake CLIs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__
from .probes import dump_rows, load_config, run_probes
from .report import render
from .runner import run_matrix

EXAMPLE_CONFIG = """\
# harness-fit-bench config. Keys are read from env vars; never paste them here.
gateways:
  main:
    base_url: https://gateway.example.com
    api_key_env: HFB_GATEWAY_KEY
    # paths:                      # override if your gateway mounts protocols elsewhere
    #   gemini: /gemini/v1beta/models/{model}:streamGenerateContent?alt=sse

models:
  - id: gemini-3-pro
    gateway: main
    protocols: [openai, anthropic, gemini]
    thinking: [low, high, xhigh]  # checks whether levels really change reasoning tokens
  - id: deepseek-v4
    gateway: main
    protocols: [openai, anthropic]
    thinking: [low, high]
  - id: argon-1
    gateway: main
    protocols: [openai, gemini]

probe:
  timeout: 60
  speed_tokens: 300
  cache_repeat: 40                # repeats of the filler prefix; ~2.5k tokens

harnesses:
  - name: pi
    kind: pi
    models: [gemini-3-pro, deepseek-v4, argon-1]
  - name: claude-3p
    kind: claude
    models: [gemini-3-pro, deepseek-v4]
    env:
      ANTHROPIC_BASE_URL: https://gateway.example.com
      ANTHROPIC_AUTH_TOKEN: ${HFB_GATEWAY_KEY}
  - name: codex
    kind: codex
    models: [gemini-3-pro, deepseek-v4]

run:
  timeout: 600
  keep_workspace: false
"""


def demo_config(py: str = sys.executable) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = ["gemini-3-pro", "deepseek-v4", "argon-1"]
    harnesses = []
    for kind, name in (("pi", "pi"), ("claude", "claude-3p"), ("codex", "codex")):
        harnesses.append({
            "name": name, "kind": kind, "models": models,
            "cmd": [py, "-m", "harness_fit.fakeharness", "--kind", kind, "--model", "{model}", "{prompt}"],
            "env": {"PYTHONPATH": root},
        })
    return {
        "gateways": {"mock": {"base_url": "https://gateway.example.com", "api_key": "demo-key"}},
        "models": [
            {"id": "gemini-3-pro", "gateway": "mock", "protocols": ["openai", "anthropic", "gemini"],
             "thinking": ["low", "high", "xhigh"]},
            {"id": "deepseek-v4", "gateway": "mock", "protocols": ["openai", "anthropic", "gemini"],
             "thinking": ["low", "high"]},
            {"id": "argon-1", "gateway": "mock", "protocols": ["openai", "anthropic"]},
        ],
        "probe": {"timeout": 10, "speed_tokens": 120, "cache_repeat": 40},
        "harnesses": harnesses,
        "run": {"timeout": 60},
    }


def _save(out: str, name: str, data) -> str:
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, name)
    if isinstance(data, str):
        with open(fp, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return fp


def _load(out: str, name: str) -> list:
    fp = os.path.join(out, name)
    if not os.path.isfile(fp):
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def execute(cfg: dict, out: str, *, do_probe: bool, do_run: bool, transport=None,
            only=None, repeats: int = 1, log=print) -> str:
    probes = _load(out, "probes.json")
    runs = _load(out, "runs.json")
    if do_probe:
        rows = run_probes(cfg, transport=transport, log=log)
        os.makedirs(out, exist_ok=True)
        dump_rows(rows, os.path.join(out, "probes.json"))
        probes = [r.to_dict() for r in rows]
    if do_run:
        runs = [r.to_dict() for r in run_matrix(cfg, only=only, repeats=repeats, log=log)]
        _save(out, "runs.json", runs)
    md = render(probes, runs)
    fp = _save(out, "report.md", md)
    log(f"\nreport written to {fp}\n")
    log(md)
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness_fit", description="Harness x model fit bench")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init", help="write an example config")
    p_init.add_argument("-o", "--output", default="bench.yaml")
    for name in ("probe", "run", "all"):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", required=True)
        p.add_argument("-o", "--out", default="hfb-out")
        if name != "probe":
            p.add_argument("--only", nargs="*", help="harness names to run")
            p.add_argument("--repeats", type=int, default=1)
    p_rep = sub.add_parser("report")
    p_rep.add_argument("-o", "--out", default="hfb-out")
    p_demo = sub.add_parser("demo", help="offline demo with mock gateway and scripted harnesses")
    p_demo.add_argument("-o", "--out", default="hfb-demo")
    a = ap.parse_args(argv)

    if a.cmd == "init":
        if os.path.exists(a.output):
            print(f"{a.output} already exists; not overwriting", file=sys.stderr)
            return 1
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(EXAMPLE_CONFIG)
        print(f"wrote {a.output}; set HFB_GATEWAY_KEY and edit gateways/models, then run `all`")
        return 0
    if a.cmd == "report":
        execute({}, a.out, do_probe=False, do_run=False)
        return 0
    if a.cmd == "demo":
        from .mockgw import MockGateway
        execute(demo_config(), a.out, do_probe=True, do_run=True,
                transport=MockGateway(speed_scale=0.3).transport())
        return 0
    cfg = load_config(a.config)
    execute(cfg, a.out, do_probe=a.cmd in ("probe", "all"), do_run=a.cmd in ("run", "all"),
            only=getattr(a, "only", None), repeats=getattr(a, "repeats", 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
