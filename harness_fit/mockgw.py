"""In-process mock gateway (httpx.MockTransport) speaking all three streaming protocols.

Used by `harness_fit demo` and the tests. Each model profile describes how the fake
gateway behaves for it, so the probe code paths are exercised end-to-end without network.
"""
from __future__ import annotations

import json
import time

import httpx

PROFILES = {
    # protocols the gateway accepts, tool support, cache support, chars per second, first-token delay
    "gemini-3-pro": {"protocols": {"openai", "gemini", "anthropic"}, "tools": True, "cache": {"gemini"},
                     "tps": 140, "ttft": 0.04, "thinking": {"low": 300, "high": 1800, "xhigh": 1800}},
    "deepseek-v4": {"protocols": {"openai", "anthropic"}, "tools": True, "cache": {"openai", "anthropic"},
                    "tps": 90, "ttft": 0.02, "thinking": {"low": 200, "high": 900}},
    "argon-1": {"protocols": {"openai"}, "tools": False, "cache": set(),
                "tps": 35, "ttft": 0.08, "thinking": {}},
}


def _sse(objs) -> bytes:
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs)


class _SlowStream(httpx.SyncByteStream):
    def __init__(self, first: bytes, rest: list, ttft: float, gap: float):
        self.first, self.rest, self.ttft, self.gap = first, rest, ttft, gap

    def __iter__(self):
        time.sleep(self.ttft)
        yield self.first
        for chunk in self.rest:
            time.sleep(self.gap)
            yield chunk


def _parse(request: httpx.Request):
    body = json.loads(request.content or b"{}")
    path = request.url.path
    if path.endswith("/chat/completions"):
        return "openai", body.get("model", ""), body
    if path.endswith("/messages"):
        return "anthropic", body.get("model", ""), body
    if ":streamGenerateContent" in path:
        return "gemini", path.split("/models/")[-1].split(":")[0], body
    return "", "", body


def _prompt_chars(proto: str, body: dict) -> int:
    if proto == "anthropic":
        sys = "".join(b.get("text", "") for b in body.get("system") or [])
        return len(sys) + len(json.dumps(body.get("messages")))
    if proto == "gemini":
        return len(json.dumps(body.get("contents"))) + len(json.dumps(body.get("systemInstruction") or {}))
    return len(json.dumps(body.get("messages")))


class MockGateway:
    def __init__(self, profiles: dict = None, speed_scale: float = 1.0):
        self.profiles = profiles or PROFILES
        self.scale = speed_scale
        self.seen_prefixes = set()
        self.calls = 0

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        proto, model, body = _parse(request)
        prof = self.profiles.get(model)
        if not prof:
            return httpx.Response(404, json={"error": {"message": f"model {model} not found"}})
        if proto not in prof["protocols"]:
            return httpx.Response(400, json={"error": {"message": f"{model} not routable via {proto}"}})
        has_tools = bool(body.get("tools"))
        thinking = body.get("reasoning_effort") or ("on" if body.get("thinking") else None) \
            or (body.get("generationConfig", {}).get("thinkingConfig") and "on")
        level = body.get("reasoning_effort")
        if not level and body.get("thinking"):
            level = {1024: "low", 4096: "medium", 8192: "high", 16384: "xhigh"}.get(
                body["thinking"].get("budget_tokens"), "high")
        if not level and thinking:
            b = body["generationConfig"]["thinkingConfig"]["thinkingBudget"]
            level = {1024: "low", 4096: "medium", 8192: "high", 16384: "xhigh"}.get(b, "high")
        reasoning = prof["thinking"].get(level, 0) if level else 0
        if level and prof["thinking"] and level not in prof["thinking"]:
            return httpx.Response(400, json={"error": {"message": f"unsupported effort {level}"}})
        n_in = _prompt_chars(proto, body) // 4
        prefix_key = (model, proto, n_in)
        cached = 0
        if proto in prof["cache"] and n_in > 1000:
            if prefix_key in self.seen_prefixes:
                cached = int(n_in * 0.9)
            self.seen_prefixes.add(prefix_key)
        tool_call = has_tools and prof["tools"]
        text = "" if tool_call else ", ".join(str(i) for i in range(1, 61))[: body.get("max_tokens", 256) * 3]
        n_out = max(1, len(text) // 3) if text else 12
        chunks = [text[i:i + 12] for i in range(0, len(text), 12)] or [""]
        events = self._events(proto, model, chunks, tool_call, n_in, n_out, cached, reasoning)
        gap = (12 / prof["tps"]) * self.scale / 3
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=_SlowStream(events[0], events[1:], prof["ttft"] * self.scale, gap))

    @staticmethod
    def _events(proto, model, chunks, tool_call, n_in, n_out, cached, reasoning) -> list:
        args = {"sku": "AB-1234"}
        if proto == "openai":
            ev = [{"model": model, "choices": [{"delta": {"content": c}}]} for c in chunks if c]
            if tool_call:
                ev = [{"model": model, "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
                    "name": "lookup_sku", "arguments": json.dumps(args)}}]}}]}]
            ev.append({"model": model, "choices": [], "usage": {
                "prompt_tokens": n_in, "completion_tokens": n_out + reasoning,
                "prompt_tokens_details": {"cached_tokens": cached},
                "completion_tokens_details": {"reasoning_tokens": reasoning}}})
            return [_sse([e]) for e in ev] + [b"data: [DONE]\n\n"]
        if proto == "anthropic":
            ev = [{"type": "message_start", "message": {"model": model, "usage": {
                "input_tokens": n_in - cached, "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0 if cached else n_in}}}]
            if reasoning:
                ev.append({"type": "content_block_delta", "delta": {"type": "thinking_delta",
                                                                    "thinking": "x" * reasoning * 4}})
            if tool_call:
                ev.append({"type": "content_block_start", "content_block": {"type": "tool_use", "name": "lookup_sku"}})
                ev.append({"type": "content_block_delta", "delta": {"type": "input_json_delta",
                                                                    "partial_json": json.dumps(args)}})
            else:
                ev += [{"type": "content_block_delta", "delta": {"type": "text_delta", "text": c}} for c in chunks]
            ev.append({"type": "message_delta", "usage": {"output_tokens": n_out}})
            ev.append({"type": "message_stop"})
            return [_sse([e]) for e in ev]
        parts = [[{"functionCall": {"name": "lookup_sku", "args": args}}]] if tool_call else \
            [[{"text": c}] for c in chunks]
        ev = [{"modelVersion": model, "candidates": [{"content": {"parts": p}}]} for p in parts]
        ev[-1]["usageMetadata"] = {"promptTokenCount": n_in, "candidatesTokenCount": n_out,
                                   "cachedContentTokenCount": cached, "thoughtsTokenCount": reasoning}
        return [_sse([e]) for e in ev]
