"""Protocol probes: connectivity, tool calling, prompt cache, speed, thinking-level mapping.

One ProbeRow per (model, protocol). Every probe is a real streaming request, so the
numbers reflect what a harness would actually see through the same gateway.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx
import yaml

from .protocols import StreamResult, apply_event, auth_headers, build_request, redact, sse_events

CACHE_FILLER = (
    "You are a meticulous code reviewer. Rules: keep diffs minimal, never rename public APIs, "
    "prefer standard library, write tests for each bug fix, explain root cause in one line. "
)


@dataclass
class ProbeRow:
    model: str
    protocol: str
    connect: bool = False
    status: int = 0
    error: str = ""
    model_echo: str = ""
    tools: Optional[bool] = None
    cache_hit: Optional[bool] = None
    cached_tokens: int = 0
    cache_ratio: float = 0.0
    ttft_ms: Optional[int] = None
    tok_s: Optional[float] = None
    thinking: dict = field(default_factory=dict)  # level -> reasoning tokens (or -1 if rejected)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("gateways", {})
    cfg.setdefault("models", [])
    cfg.setdefault("harnesses", [])
    cfg.setdefault("probe", {})
    for m in cfg["models"]:
        if m.get("gateway") not in cfg["gateways"]:
            raise ValueError(f"model {m.get('id')} references unknown gateway {m.get('gateway')!r}")
        m.setdefault("protocols", ["openai", "anthropic", "gemini"])
        m.setdefault("thinking", [])
    return cfg


class Gateway:
    def __init__(self, name: str, spec: dict, transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 60.0):
        self.name = name
        self.base_url = spec["base_url"].rstrip("/")
        self.api_key = spec.get("api_key") or os.environ.get(spec.get("api_key_env", ""), "")
        self.paths = spec.get("paths", {})
        self.client = httpx.Client(transport=transport, timeout=timeout)

    def stream(self, protocol: str, model: str, prompt: str, **kw) -> StreamResult:
        path, body = build_request(protocol, model, prompt, **kw)
        if protocol in self.paths:
            path = self.paths[protocol].format(model=model)
        r = StreamResult()
        t0 = time.perf_counter()
        try:
            with self.client.stream("POST", self.base_url + path, json=body,
                                    headers=auth_headers(protocol, self.api_key)) as resp:
                r.status = resp.status_code
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", "replace").strip()
                    r.error = f"HTTP {resp.status_code}" + (f": {redact(body)}"[:200] if body else "")
                    return r
                for ev in sse_events(resp.iter_lines()):
                    if apply_event(protocol, ev, r) and r.ttft is None:
                        r.ttft = time.perf_counter() - t0
        except httpx.HTTPError as e:
            r.error = redact(f"{type(e).__name__}: {e}")[:200]
            return r
        finally:
            r.total = time.perf_counter() - t0
        r.ok = not r.error and (bool(r.text) or bool(r.tool_calls) or r.output_tokens > 0)
        if not r.ok and not r.error:
            r.error = "empty stream"
        r.error = redact(r.error)
        return r


def probe_one(gw: Gateway, model: str, protocol: str, *, thinking_levels=(), cache_repeat: int = 40,
              speed_tokens: int = 300) -> ProbeRow:
    row = ProbeRow(model=model, protocol=protocol)
    # 1. connectivity + speed
    base = gw.stream(protocol, model, "Count from 1 to 60, comma separated, no other text.",
                     max_tokens=speed_tokens)
    row.connect, row.status, row.error, row.model_echo = base.ok, base.status, base.error, base.model_echo
    if not base.ok:
        return row
    row.ttft_ms = int(base.ttft * 1000) if base.ttft is not None else None
    row.tok_s = base.tok_per_s
    # 2. tool calling: model must emit a lookup_sku call with the sku argument
    tr = gw.stream(protocol, model, "Use the lookup_sku tool to check stock for SKU AB-1234. "
                   "Do not answer from memory.", tools=True)
    row.tools = bool(tr.ok and any(tc["name"] == "lookup_sku" and "AB-1234" in tc["args"]
                                   for tc in tr.tool_calls))
    # 3. cache: identical long system prefix twice; second call should report cached tokens
    system = CACHE_FILLER * cache_repeat
    first = gw.stream(protocol, model, "Reply with OK.", system=system, cache=True, max_tokens=16)
    second = gw.stream(protocol, model, "Reply with OK.", system=system, cache=True, max_tokens=16)
    if first.ok and second.ok:
        row.cached_tokens = second.cached_tokens
        row.cache_hit = second.cached_tokens > 0
        if second.input_tokens:
            denom = second.input_tokens + (second.cached_tokens if protocol == "anthropic" else 0)
            row.cache_ratio = round(second.cached_tokens / max(denom, 1), 2)
    # 4. thinking levels: does each level change reasoning token usage?
    for lvl in thinking_levels:
        th = gw.stream(protocol, model, "Is 391 prime? Answer yes or no.", thinking=lvl, max_tokens=64)
        row.thinking[lvl] = th.reasoning_tokens if th.ok else -1
    return row


def run_probes(cfg: dict, transport: Optional[httpx.BaseTransport] = None, log=print) -> list:
    pcfg = cfg.get("probe", {})
    gws = {n: Gateway(n, s, transport=transport, timeout=pcfg.get("timeout", 60))
           for n, s in cfg["gateways"].items()}
    rows = []
    for m in cfg["models"]:
        for proto in m["protocols"]:
            row = probe_one(gws[m["gateway"]], m["id"], proto, thinking_levels=m.get("thinking", []),
                            cache_repeat=pcfg.get("cache_repeat", 40),
                            speed_tokens=pcfg.get("speed_tokens", 300))
            log(f"[probe] {m['id']:<24} {proto:<9} connect={row.connect} tools={row.tools} "
                f"cache={row.cache_hit} ttft={row.ttft_ms}ms tok/s={row.tok_s}")
            rows.append(row)
    return rows


def thinking_verdict(levels: dict) -> str:
    """Summarize whether thinking levels actually differ (e.g. xhigh vs high)."""
    vals = {k: v for k, v in levels.items() if v is not None}
    if not vals:
        return "-"
    if all(v < 0 for v in vals.values()):
        return "rejected"
    ok = {k: v for k, v in vals.items() if v >= 0}
    if ok and all(v == 0 for v in ok.values()):
        return "not reported"  # gateway accepted the levels but exposes no reasoning token count
    if len(set(ok.values())) == 1 and len(ok) == len(vals):
        return "no effect" if len(ok) > 1 else f"{next(iter(ok))}={next(iter(ok.values()))}"
    parts, seen = [], {}
    for k, v in vals.items():
        if v < 0:
            parts.append(f"{k}=rejected")
        elif v in seen:
            parts.append(f"{k}=={seen[v]}")
        else:
            seen[v] = k
            parts.append(f"{k}={v}")
    return " / ".join(parts)


def dump_rows(rows: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in rows], f, ensure_ascii=False, indent=2)
