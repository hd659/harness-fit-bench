"""Parse JSONL event streams emitted by pi / claude / codex into common run stats.

pi     : `pi -p --mode json`           -> tool_execution_end{toolName,isError}, message_end{message.usage}
claude : `claude -p --output-format stream-json --verbose`
                                       -> assistant tool_use blocks, user tool_result{is_error}, result{usage}
codex  : `codex exec --json`           -> item.completed{item.type,status,exit_code}, turn.completed{usage}
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class RunStats:
    tool_calls: int = 0
    tool_errors: int = 0
    by_tool: Counter = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    turns: int = 0
    final_text: str = ""
    error: str = ""  # error reported by the harness itself (auth, API, turn failure)

    @property
    def tool_success(self) -> float:
        if not self.tool_calls:
            return 0.0
        return round(1 - self.tool_errors / self.tool_calls, 3)

    @property
    def cache_ratio(self) -> float:
        denom = self.input_tokens + self.cached_tokens
        return round(self.cached_tokens / denom, 3) if denom else 0.0


def _events(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def parse_pi(text: str) -> RunStats:
    s = RunStats()
    for ev in _events(text):
        t = ev.get("type")
        if t == "message_end" and (ev.get("message") or {}).get("stopReason") == "error":
            s.error = str((ev.get("message") or {}).get("errorMessage") or "model error")[:200]
        if t == "tool_execution_end":
            s.tool_calls += 1
            s.by_tool[ev.get("toolName", "?")] += 1
            if ev.get("isError"):
                s.tool_errors += 1
        elif t == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            s.turns += 1
            u = msg.get("usage") or {}
            s.input_tokens += u.get("input", 0) or 0
            s.output_tokens += u.get("output", 0) or 0
            s.cached_tokens += u.get("cacheRead", 0) or 0
            texts = [c.get("text", "") for c in msg.get("content") or [] if c.get("type") == "text"]
            if any(texts):
                s.final_text = "".join(texts)
    return s


def parse_claude(text: str) -> RunStats:
    s = RunStats()
    result_usage = None
    for ev in _events(text):
        t = ev.get("type")
        content = (ev.get("message") or {}).get("content") or []
        if t == "assistant":
            s.turns += 1
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    s.tool_calls += 1
                    s.by_tool[c.get("name", "?")] += 1
        elif t == "user":
            for c in content if isinstance(content, list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                    s.tool_errors += 1
        elif t == "result":
            result_usage = ev.get("usage") or {}
            s.final_text = ev.get("result", "") or ""
            if ev.get("is_error") or str(ev.get("subtype", "")).startswith("error"):
                s.error = (s.final_text or ev.get("subtype", "error"))[:200]
            s.turns = ev.get("num_turns", s.turns) or s.turns
    if result_usage:
        s.input_tokens = (result_usage.get("input_tokens", 0) or 0) + \
            (result_usage.get("cache_creation_input_tokens", 0) or 0)
        s.output_tokens = result_usage.get("output_tokens", 0) or 0
        s.cached_tokens = result_usage.get("cache_read_input_tokens", 0) or 0
    return s


CODEX_TOOL_ITEMS = {"command_execution", "file_change", "mcp_tool_call", "web_search"}


def parse_codex(text: str) -> RunStats:
    s = RunStats()
    for ev in _events(text):
        t = ev.get("type")
        if t == "item.completed":
            item = ev.get("item") or {}
            it = item.get("type") or item.get("item_type")
            if it in CODEX_TOOL_ITEMS:
                s.tool_calls += 1
                s.by_tool[it] += 1
                failed = item.get("status") == "failed" or (item.get("exit_code") not in (None, 0))
                if failed:
                    s.tool_errors += 1
            elif it in ("agent_message", "assistant_message"):
                s.final_text = item.get("text", "") or s.final_text
        elif t in ("turn.failed", "error"):
            err = ev.get("error") if isinstance(ev.get("error"), dict) else {}
            s.error = str(err.get("message") or ev.get("message") or t)[:200]
        elif t == "turn.completed":
            s.turns += 1
            u = ev.get("usage") or {}
            cached = u.get("cached_input_tokens", 0) or 0
            s.input_tokens += max((u.get("input_tokens", 0) or 0) - cached, 0)
            s.cached_tokens += cached
            s.output_tokens += u.get("output_tokens", 0) or 0
    return s


PARSERS = {"pi": parse_pi, "claude": parse_claude, "codex": parse_codex}


def parse(kind: str, text: str) -> RunStats:
    if kind not in PARSERS:
        raise ValueError(f"unknown harness kind {kind!r}; expected one of {sorted(PARSERS)}")
    return PARSERS[kind](text)
