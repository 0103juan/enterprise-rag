"""Traces: one line per question in traces.jsonl, with a span per stage (duration, model calls, tokens, cost).

    python tracing.py [traces.jsonl]   per-stage latency and cost of the questions in a trace file
"""

import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

TRACES = Path("traces.jsonl")
_open: ContextVar[dict | None] = ContextVar("open_span", default=None)


# ponytail: plain JSONL and a context manager. Swap this for OpenTelemetry spans when there is a
# collector to send them to; the stage names and attributes carry over.
@contextmanager
def span(spans: list[dict], name: str):
    """Time a stage and collect the usage and cost of every model call made inside it."""
    stage = {"name": name, "calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    token, start = _open.set(stage), time.perf_counter()
    try:
        yield stage
    finally:
        _open.reset(token)
        stage["ms"] = round((time.perf_counter() - start) * 1000)
        spans.append(stage)


def record(call: dict) -> None:
    """Charge a model call, as the gateway recorded it, to the stage that made it."""
    stage = _open.get()
    if stage is not None:
        stage["calls"] += 1
        stage["model"] = call["model"]  # the model that answered, which a fallback can change
        for kind in ("input_tokens", "output_tokens", "usd"):
            stage[kind] += call[kind]


def write(trace: dict, path: Path = TRACES) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(trace, ensure_ascii=False) + "\n")


def summarize(traces: list[dict]) -> str:
    """Per-stage means per question, then the latency and cost of a whole question."""
    stages = defaultdict(lambda: defaultdict(float))
    for trace in traces:
        for stage in trace["spans"]:
            for key in ("calls", "input_tokens", "output_tokens", "ms", "usd"):
                stages[stage["name"]][key] += stage[key] / len(traces)
    lines = [f"{'stage':<10}{'calls':>7}{'in tok':>9}{'out tok':>9}{'ms':>8}{'usd':>9}   (mean per question)"]
    lines += [f"{name:<10}{s['calls']:>7.2f}{s['input_tokens']:>9.0f}{s['output_tokens']:>9.0f}{s['ms']:>8.0f}"
              f"{s['usd']:>9.4f}" for name, s in stages.items()]
    ms, usd = sorted(t["ms"] for t in traces), [t["usd"] for t in traces]
    lines.append(f"{len(traces)} questions: ${sum(usd) / len(usd):.4f} per question (max ${max(usd):.4f}), "
                 f"latency p50 {ms[len(ms) // 2] / 1000:.1f} s, p95 {ms[int(len(ms) * 0.95)] / 1000:.1f} s")
    return "\n".join(lines)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else TRACES
    print(summarize([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]))
