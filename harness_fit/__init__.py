"""harness-fit-bench: probe gateway models per protocol, run harnesses on one task, rank combos.

Modules:
    protocols   request builders + SSE stream parsers for openai / anthropic / gemini
    probes      connectivity, tool-calling, cache, speed and thinking-level probes
    task        the shared "fix the bug" fixture workspace and its verifier
    parsers     event-log parsers for pi / claude / codex harness output
    runner      runs each harness x model on the fixture and collects stats
    report      scoring and Markdown leaderboard
    mockgw      in-process mock gateway (httpx.MockTransport) for demo and tests
    fakeharness scripted harness that emits pi / claude / codex style events
    cli         `python -m harness_fit ...` entry point
"""

__version__ = "0.1.0"

PROTOCOLS = ("openai", "anthropic", "gemini")
