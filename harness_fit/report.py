"""Scoring and Markdown leaderboard.

Score (0..100) per harness x model, averaged over repeats:
    50 * solve_rate          hidden checker passes and tests untouched
  + 15 * tool_success        share of tool calls that did not error
  + 15 * speed               fastest_wall / this_wall   (relative within the bench)
  + 10 * token_efficiency    fewest_tokens / this_tokens (relative, solved runs weighted)
  + 10 * cache_ratio         cached / (cached + uncached input)
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .probes import thinking_verdict

WEIGHTS = {"solve": 50, "tool": 15, "speed": 15, "tokens": 10, "cache": 10}


def aggregate(runs: list) -> list:
    groups = defaultdict(list)
    for r in runs:
        d = r if isinstance(r, dict) else r.to_dict()
        groups[(d["harness"], d["model"])].append(d)
    rows = []
    for (h, m), rs in groups.items():
        tool_calls = sum(r["tool_calls"] for r in rs)
        tool_err = sum(r["tool_errors"] for r in rs)
        rows.append({
            "harness": h, "model": m, "runs": len(rs),
            "solve_rate": mean(1.0 if r["solved"] else 0.0 for r in rs),
            "checks": mean(r["checks_passed"] / max(r["checks_total"], 1) for r in rs),
            "tool_calls": round(tool_calls / len(rs), 1),
            "tool_success": round(1 - tool_err / tool_calls, 3) if tool_calls else 0.0,
            "wall_s": round(mean(r["wall_s"] for r in rs), 2),
            "tokens": int(mean(r["input_tokens"] + r["output_tokens"] + r["cached_tokens"] for r in rs)),
            "cache_ratio": round(mean(r["cache_ratio"] for r in rs), 3),
            "errors": sorted({r["error"] for r in rs if r["error"]}),
        })
    return rows


def score(rows: list) -> list:
    if not rows:
        return []
    walls = [r["wall_s"] for r in rows if r["wall_s"] > 0 and r["solve_rate"] > 0] or [1.0]
    toks = [r["tokens"] for r in rows if r["tokens"] > 0 and r["solve_rate"] > 0] or [1]
    fastest, leanest = min(walls), min(toks)
    for r in rows:
        speed = fastest / r["wall_s"] if r["wall_s"] > 0 and r["solve_rate"] > 0 else 0.0
        eff = leanest / r["tokens"] if r["tokens"] > 0 and r["solve_rate"] > 0 else 0.0
        parts = {"solve": r["solve_rate"], "tool": r["tool_success"], "speed": speed,
                 "tokens": eff, "cache": r["cache_ratio"]}
        r["score"] = round(sum(WEIGHTS[k] * min(v, 1.0) for k, v in parts.items()), 1)
    return sorted(rows, key=lambda r: (-r["score"], r["wall_s"]))


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def probe_table(probes: list) -> str:
    lines = ["| model | protocol | connect | tools | cache | TTFT ms | tok/s | thinking | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in probes:
        d = p if isinstance(p, dict) else p.to_dict()
        yn = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
        cache = yn(d["cache_hit"]) + (f" ({_pct(d['cache_ratio'])})" if d["cache_hit"] else "")
        note = d["error"][:60].replace("|", "/") if d["error"] else (
            f"served as {d['model_echo']}" if d["model_echo"] and d["model_echo"] != d["model"] else "")
        lines.append(f"| {d['model']} | {d['protocol']} | {yn(d['connect'])} | {yn(d['tools'])} | {cache} | "
                     f"{d['ttft_ms'] if d['ttft_ms'] is not None else '-'} | {d['tok_s'] or '-'} | "
                     f"{thinking_verdict(d.get('thinking') or {})} | {note} |")
    return "\n".join(lines)


def leaderboard(rows: list) -> str:
    lines = ["| # | harness | model | score | solved | tool calls | tool ok | wall s | tokens | cache |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['harness']} | {r['model']} | {r['score']} | {_pct(r['solve_rate'])} | "
                     f"{r['tool_calls']} | {_pct(r['tool_success'])} | {r['wall_s']} | {r['tokens']} | "
                     f"{_pct(r['cache_ratio'])} |")
    return "\n".join(lines)


def verdicts(rows: list, probes: list) -> list:
    out = []
    solved = [r for r in rows if r["solve_rate"] > 0]
    if rows:
        out.append(f"Best overall: {rows[0]['harness']} + {rows[0]['model']} (score {rows[0]['score']}).")
    if solved:
        f = min(solved, key=lambda r: r["wall_s"])
        c = min(solved, key=lambda r: r["tokens"])
        out.append(f"Fastest that solved: {f['harness']} + {f['model']} ({f['wall_s']}s).")
        out.append(f"Leanest that solved: {c['harness']} + {c['model']} ({c['tokens']} tokens).")
    for r in rows:
        if r["solve_rate"] == 0 and r["errors"]:
            out.append(f"Broken: {r['harness']} + {r['model']}: {r['errors'][0]}")
    pd = [p if isinstance(p, dict) else p.to_dict() for p in probes]
    for d in pd:
        if d["connect"] and d["cache_hit"] is False:
            out.append(f"No cache: {d['model']} via {d['protocol']} returned 0 cached tokens on an identical prefix.")
        if d["connect"] and d["tools"] is False:
            out.append(f"No tool calls: {d['model']} via {d['protocol']} never emitted the probe tool call.")
        tv = thinking_verdict(d.get("thinking") or {})
        if tv == "no effect":
            out.append(f"Thinking levels have no effect: {d['model']} via {d['protocol']}.")
        elif "==" in tv:
            dup = ", ".join(p.replace("==", " behaves like ") for p in tv.split(" / ") if "==" in p)
            out.append(f"Thinking alias: {d['model']} via {d['protocol']}: {dup}.")
    return out


def render(probes: list, runs: list) -> str:
    rows = score(aggregate(runs))
    md = ["# Harness fit report", ""]
    if probes:
        md += ["## Protocol probes", "", probe_table(probes), ""]
    if rows:
        md += ["## Harness leaderboard", "", leaderboard(rows), "",
               "Score = " + " + ".join(f"{v}*{k}" for k, v in WEIGHTS.items()), ""]
    v = verdicts(rows, probes)
    if v:
        md += ["## Verdicts", ""] + [f"- {x}" for x in v] + [""]
    return "\n".join(md)
