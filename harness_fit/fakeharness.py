"""Scripted stand-in for pi / claude / codex, used by the offline demo and tests.

    python -m harness_fit.fakeharness --kind pi --model gemini-3-pro "<prompt>"

It really edits the fixture in cwd (so grading is genuine) and prints the same JSONL
event shapes the real CLIs emit. Behaviour is chosen by model name:
    *gemini*   fixes both bugs, but one failing grep first (tool error)
    *deepseek* fixes both bugs cleanly, fewer steps
    *argon*    fixes only the discount bug, several failing tool calls
    anything else: does nothing and exits 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FIX_DISCOUNT = ("return round(price - percent, 2)", "return round(price * (1 - percent / 100), 2)")
FIX_QTY = ("sum(price for _, price, qty in self.items)", "sum(price * qty for _, price, qty in self.items)")


def _patch(rel: str, old: str, new: str) -> bool:
    fp = os.path.join(os.getcwd(), rel)
    try:
        with open(fp, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return False
    if old not in src:
        return False
    with open(fp, "w", encoding="utf-8") as f:
        f.write(src.replace(old, new))
    return True


def plan(model: str) -> list:
    """Return [(tool_name, is_error, action)] where action is None or (rel, old, new)."""
    m = model.lower()
    if "gemini" in m:
        return [("grep", True, None), ("read", False, None), ("read", False, None),
                ("edit", False, ("shop/pricing.py",) + FIX_DISCOUNT),
                ("edit", False, ("shop/cart.py",) + FIX_QTY), ("bash", False, None)]
    if "deepseek" in m:
        return [("bash", False, None), ("edit", False, ("shop/pricing.py",) + FIX_DISCOUNT),
                ("edit", False, ("shop/cart.py",) + FIX_QTY), ("bash", False, None)]
    if "argon" in m:
        return [("grep", True, None), ("grep", True, None), ("read", False, None),
                ("edit", False, ("shop/pricing.py",) + FIX_DISCOUNT), ("edit", True, None),
                ("bash", False, None), ("bash", False, None)]
    return []


def emit_pi(steps, usage):
    out = []
    for i, (name, err, _) in enumerate(steps):
        out.append({"type": "tool_execution_end", "toolCallId": f"t{i}", "toolName": name, "isError": err})
        out.append({"type": "message_end", "message": {"role": "assistant", "content": [], "usage": usage}})
    out.append({"type": "message_end", "message": {"role": "assistant", "usage": usage,
                                                   "content": [{"type": "text", "text": "DONE"}]}})
    return out


def emit_claude(steps, usage):
    out = [{"type": "system", "subtype": "init"}]
    names = {"grep": "Grep", "read": "Read", "edit": "Edit", "bash": "Bash"}
    for i, (name, err, _) in enumerate(steps):
        out.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"t{i}", "name": names.get(name, name), "input": {}}]}})
        out.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "is_error": err, "content": "..."}]}})
    n = len(steps) + 1
    out.append({"type": "result", "subtype": "success", "result": "DONE", "num_turns": n, "usage": {
        "input_tokens": usage["input"] * n, "output_tokens": usage["output"] * n,
        "cache_read_input_tokens": usage["cacheRead"] * n, "cache_creation_input_tokens": 0}})
    return out


def emit_codex(steps, usage):
    out = [{"type": "thread.started"}, {"type": "turn.started"}]
    for i, (name, err, _) in enumerate(steps):
        kind = "file_change" if name == "edit" else "command_execution"
        item = {"id": f"t{i}", "type": kind, "status": "failed" if err else "completed"}
        if kind == "command_execution":
            item["exit_code"] = 1 if err else 0
        out.append({"type": "item.completed", "item": item})
    out.append({"type": "item.completed", "item": {"type": "agent_message", "text": "DONE"}})
    n = len(steps) + 1
    out.append({"type": "turn.completed", "usage": {
        "input_tokens": (usage["input"] + usage["cacheRead"]) * n,
        "cached_input_tokens": usage["cacheRead"] * n, "output_tokens": usage["output"] * n}})
    return out


EMITTERS = {"pi": emit_pi, "claude": emit_claude, "codex": emit_codex}
# per-kind cache behaviour: what fraction of each turn's prompt is read from cache
CACHE_SHARE = {"pi": 0.0, "claude": 0.8, "codex": 0.6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=sorted(EMITTERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("prompt", nargs="?", default="")
    a = ap.parse_args(argv)
    steps = plan(a.model)
    if not steps:
        print(f"error: model {a.model} is not available on this provider", file=sys.stderr)
        return 1
    for _, err, action in steps:
        if action and not err:
            _patch(*action)
    share = CACHE_SHARE[a.kind] if "argon" not in a.model else 0.0
    prompt_tokens = 3000 if a.kind == "pi" else 4000
    usage = {"input": int(prompt_tokens * (1 - share)), "output": 150, "cacheRead": int(prompt_tokens * share)}
    for ev in EMITTERS[a.kind](steps, usage):
        print(json.dumps(ev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
