"""Request builders and streaming parsers for the three wire protocols.

Every protocol is reduced to the same StreamResult so probes can compare
openai-compatible, anthropic-messages and gemini-native endpoints side by side.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

DEFAULT_PATHS = {
    "openai": "/v1/chat/completions",
    "anthropic": "/v1/messages",
    "gemini": "/v1beta/models/{model}:streamGenerateContent?alt=sse",
}

# thinking level -> token budget for protocols that take a budget instead of an effort label
THINKING_BUDGET = {"off": 0, "minimal": 512, "low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384}

PROBE_TOOL = {
    "name": "lookup_sku",
    "description": "Look up inventory for a product SKU.",
    "parameters": {
        "type": "object",
        "properties": {"sku": {"type": "string", "description": "Product SKU"}},
        "required": ["sku"],
    },
}


@dataclass
class StreamResult:
    ok: bool = False
    status: int = 0
    error: str = ""
    model_echo: str = ""
    text: str = ""
    tool_calls: list = field(default_factory=list)  # [{"name":..., "args": "..."}]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    ttft: Optional[float] = None
    total: float = 0.0

    @property
    def tok_per_s(self) -> Optional[float]:
        if self.ttft is None or self.output_tokens <= 0:
            return None
        gen = self.total - self.ttft
        return round(self.output_tokens / gen, 1) if gen > 0 else None


_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9*_\-]{4,}|AIza[0-9A-Za-z_\-]{10,}|(?i:bearer)\s+[A-Za-z0-9._\-]{8,})")


def redact(text: str) -> str:
    """Mask API-key-like substrings so error messages are safe to write into reports."""
    return _SECRET_RE.sub("[redacted]", text or "")


def auth_headers(protocol: str, api_key: str) -> dict:
    if protocol == "anthropic":
        return {"x-api-key": api_key, "anthropic-version": "2023-06-01", "authorization": f"Bearer {api_key}"}
    if protocol == "gemini":
        return {"x-goog-api-key": api_key}
    return {"authorization": f"Bearer {api_key}"}


def build_request(protocol: str, model: str, prompt: str, *, system: str = "", tools: bool = False,
                  cache: bool = False, thinking: Optional[str] = None, max_tokens: int = 256) -> tuple:
    """Return (path_template_filled, json_body)."""
    if protocol not in DEFAULT_PATHS:
        raise ValueError(f"unknown protocol: {protocol}")
    path = DEFAULT_PATHS[protocol].format(model=model)
    if protocol == "openai":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": model, "messages": msgs, "stream": True, "max_tokens": max_tokens,
                "stream_options": {"include_usage": True}}
        if tools:
            body["tools"] = [{"type": "function", "function": PROBE_TOOL}]
            body["tool_choice"] = "auto"
        if thinking:
            body["reasoning_effort"] = thinking
        return path, body
    if protocol == "anthropic":
        body = {"model": model, "max_tokens": max_tokens, "stream": True,
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            block = {"type": "text", "text": system}
            if cache:
                block["cache_control"] = {"type": "ephemeral"}
            body["system"] = [block]
        if tools:
            t = PROBE_TOOL
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}]
        if thinking and THINKING_BUDGET.get(thinking, 0) > 0:
            budget = THINKING_BUDGET[thinking]
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = budget + max_tokens
        return path, body
    if protocol == "gemini":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [PROBE_TOOL]}]
        if thinking:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": THINKING_BUDGET.get(thinking, 1024), "includeThoughts": True}
        return path, body
    raise ValueError(f"unknown protocol: {protocol}")


def sse_events(lines: Iterable[str]):
    """Yield decoded JSON payloads from `data:` lines; skip comments and [DONE]."""
    for line in lines:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def apply_event(protocol: str, ev: dict, r: StreamResult) -> bool:
    """Fold one stream event into r. Returns True if it carried generated content."""
    got = False
    if protocol == "openai":
        r.model_echo = ev.get("model", r.model_echo)
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                r.text += d["content"]; got = True
            if d.get("reasoning_content") or d.get("reasoning"):
                got = True
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") or {}
                idx = tc.get("index", len(r.tool_calls))
                while len(r.tool_calls) <= idx:
                    r.tool_calls.append({"name": "", "args": ""})
                r.tool_calls[idx]["name"] += fn.get("name") or ""
                r.tool_calls[idx]["args"] += fn.get("arguments") or ""
                got = True
        u = ev.get("usage") or {}
        if u:
            r.input_tokens = u.get("prompt_tokens", 0)
            r.output_tokens = u.get("completion_tokens", 0)
            r.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            r.reasoning_tokens = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    elif protocol == "anthropic":
        t = ev.get("type")
        if t == "message_start":
            m = ev.get("message") or {}
            r.model_echo = m.get("model", "")
            u = m.get("usage") or {}
            r.input_tokens = u.get("input_tokens", 0)
            r.cached_tokens = u.get("cache_read_input_tokens", 0) or 0
            r.cache_write_tokens = u.get("cache_creation_input_tokens", 0) or 0
        elif t == "content_block_start":
            cb = ev.get("content_block") or {}
            if cb.get("type") == "tool_use":
                r.tool_calls.append({"name": cb.get("name", ""), "args": ""}); got = True
        elif t == "content_block_delta":
            d = ev.get("delta") or {}
            if d.get("type") == "text_delta":
                r.text += d.get("text", "")
            elif d.get("type") == "input_json_delta" and r.tool_calls:
                r.tool_calls[-1]["args"] += d.get("partial_json", "")
            elif d.get("type") == "thinking_delta":
                r.reasoning_tokens += max(1, len(d.get("thinking", "")) // 4)
            got = True
        elif t == "message_delta":
            r.output_tokens = (ev.get("usage") or {}).get("output_tokens", r.output_tokens)
        elif t == "error":
            r.error = json.dumps(ev.get("error"))[:200]
    elif protocol == "gemini":
        r.model_echo = ev.get("modelVersion", r.model_echo)
        for cand in ev.get("candidates") or []:
            for p in (cand.get("content") or {}).get("parts") or []:
                if "functionCall" in p:
                    fc = p["functionCall"]
                    r.tool_calls.append({"name": fc.get("name", ""), "args": json.dumps(fc.get("args", {}))})
                elif p.get("text") and not p.get("thought"):
                    r.text += p["text"]
                got = True
        u = ev.get("usageMetadata") or {}
        if u:
            r.input_tokens = u.get("promptTokenCount", 0)
            r.output_tokens = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            r.cached_tokens = u.get("cachedContentTokenCount", 0) or 0
            r.reasoning_tokens = u.get("thoughtsTokenCount", 0) or 0
    return got
