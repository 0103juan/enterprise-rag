"""Evaluation against data/golden.jsonl.

    python evaluate.py               retrieval ablation; local models only, no API calls
    python evaluate.py --generation  end-to-end run with Claude (rewrite, answer, hallucination check)

Either run exits with an error when a metric falls below its floor in gates.json, which is what fails CI.
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel

import tracing
from pipeline import ABSTAIN, DOCS, answer, call_model
from retrieval import MODES, Index

GOLDEN = Path(__file__).parent / "data" / "golden.jsonl"
GATES = Path(__file__).parent / "gates.json"


def below(measured: dict[str, float], floors: dict[str, float]) -> list[str]:
    """The metrics that fell below their floor."""
    return [f"{name} {measured[name]:.3f} is below the gate {floor}"
            for name, floor in floors.items() if measured[name] < floor]


def retrieval_ablation(index: Index, golden: list[dict]) -> dict[str, dict[str, float]]:
    """Rank of the expected chunk under each retrieval mode, on the raw question (no rewriting)."""
    rows = [row for row in golden if row["expected"] and "history" not in row]
    table = {}
    for mode in MODES:
        ranks = []
        for row in rows:
            ids = [chunk.id for chunk in index.search([row["question"]], k=5, mode=mode)]
            ranks.append(ids.index(row["expected"]) + 1 if row["expected"] in ids else None)
        table[mode] = {
            "hit@1": sum(r == 1 for r in ranks) / len(rows),
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / len(rows),
            "MRR@5": sum(1 / r for r in ranks if r) / len(rows),
        }
    return table


class Grade(BaseModel):
    correct: bool


def is_correct(client, question: str, reference: str, candidate: str) -> bool:
    response = call_model(
        client, "parse", output_config={"effort": "low"}, output_format=Grade,
        system="Grade a candidate answer against the reference answer. It is correct if it states the same "
               "facts as the reference; extra detail is fine, a contradiction or a missing key fact is not.",
        messages=[{"role": "user", "content":
                   f"Question: {question}\nReference: {reference}\nCandidate: {candidate}"}])
    return bool(response.parsed_output and response.parsed_output.correct)


def generation_eval(client, index: Index, golden: list[dict]) -> tuple[dict[str, float], list[dict], list[dict]]:
    """The metrics, one trace per question, and the grader's own spans (an evaluation cost, not a serving cost)."""
    totals = {"retrieved": 0, "cited": 0, "correct": 0, "groundedness": 0.0}
    answerable = sum(1 for row in golden if row["expected"])
    traces, grading = [], []
    for row in golden:
        result = answer(client, index, row["question"], row.get("history", ""))
        if row["expected"] is None:  # out-of-scope question: the only right answer is to abstain
            correct = result.answer == ABSTAIN
        else:
            with tracing.span(grading, "grade"):
                correct = is_correct(client, row["question"], row["answer"], result.answer)
            totals["retrieved"] += row["expected"] in [c.id for c in result.retrieved]
            totals["cited"] += row["expected"] in [c.id for c in result.sources]
        totals["correct"] += correct
        totals["groundedness"] += result.groundedness
        traces.append({**result.trace, "expected": row["expected"], "correct": correct})
        tracing.write(traces[-1])
        if not correct or result.unsupported:
            print(f"  FAIL {row['question']!r}\n       -> {result.answer}\n       unsupported: {result.unsupported}")
    return {
        "expected chunk retrieved": totals["retrieved"] / answerable,
        "expected chunk cited": totals["cited"] / answerable,
        "answer correct": totals["correct"] / len(golden),
        "groundedness": totals["groundedness"] / len(golden),
    }, traces, grading


if __name__ == "__main__":
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    index = Index.from_dir(DOCS)
    gates = json.loads(GATES.read_text(encoding="utf-8"))
    if "--generation" in sys.argv:
        import anthropic
        metrics, traces, grading = generation_eval(anthropic.Anthropic(), index, golden)
        for metric, value in metrics.items():
            print(f"{metric:<26} {value:.1%}")
        print(tracing.summarize(traces))
        print(f"grading added ${sum(span['usd'] for span in grading):.2f} to this run; traces in {tracing.TRACES}")
        failed = below(metrics, gates["generation"])
        usd_per_question = sum(trace["usd"] for trace in traces) / len(traces)
        if usd_per_question > gates["max_usd_per_question"]:
            failed.append(f"cost ${usd_per_question:.4f} per question is over the budget "
                          f"${gates['max_usd_per_question']}")
    else:
        table = retrieval_ablation(index, golden)
        print(f"{len(index.chunks)} chunks\n{'mode':<16}{'hit@1':>8}{'hit@3':>8}{'MRR@5':>8}")
        for mode, metrics in table.items():
            print(f"{mode:<16}" + "".join(f"{value:>8.3f}" for value in metrics.values()))
        failed = below(table["hybrid+rerank"], gates["retrieval"])  # the mode the pipeline serves with
    if failed:
        sys.exit("REGRESSION\n  " + "\n  ".join(failed))
    print("gates passed")
