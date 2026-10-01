"""Evaluation against data/golden.jsonl.

    python evaluate.py               retrieval ablation; local models only, no API calls
    python evaluate.py --generation  end-to-end run with Claude (rewrite, answer, hallucination check)
"""

import json
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from pipeline import ABSTAIN, DOCS, LLM, answer
from retrieval import MODES, Index

GOLDEN = Path(__file__).parent / "data" / "golden.jsonl"
USD_PER_MTOK = {"input_tokens": 2.00, "output_tokens": 10.00}  # claude-sonnet-5-5; thinking bills as output


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


def metered(client) -> Counter:
    """Total the token usage of every call made through this client."""
    usage = Counter()
    for name in ("create", "parse"):
        call = getattr(client.beta.messages, name)

        def wrapper(*args, _call=call, **kwargs):
            response = _call(*args, **kwargs)
            usage.update(calls=1, input_tokens=response.usage.input_tokens,
                         output_tokens=response.usage.output_tokens)
            return response

        setattr(client.beta.messages, name, wrapper)
    return usage


class Grade(BaseModel):
    correct: bool


def is_correct(client, question: str, reference: str, candidate: str) -> bool:
    response = client.beta.messages.parse(
        **LLM, output_config={"effort": "low"}, output_format=Grade,
        system="Grade a candidate answer against the reference answer. It is correct if it states the same "
               "facts as the reference; extra detail is fine, a contradiction or a missing key fact is not.",
        messages=[{"role": "user", "content":
                   f"Question: {question}\nReference: {reference}\nCandidate: {candidate}"}])
    return bool(response.parsed_output and response.parsed_output.correct)


def generation_eval(client, index: Index, golden: list[dict]) -> dict[str, float]:
    totals = {"retrieved": 0, "cited": 0, "correct": 0, "groundedness": 0.0}
    answerable = sum(1 for row in golden if row["expected"])
    for row in golden:
        result = answer(client, index, row["question"], row.get("history", ""))
        if row["expected"] is None:  # out-of-scope question: the only right answer is to abstain
            correct = result.answer == ABSTAIN
        else:
            correct = is_correct(client, row["question"], row["answer"], result.answer)
            totals["retrieved"] += row["expected"] in [c.id for c in result.retrieved]
            totals["cited"] += row["expected"] in [c.id for c in result.sources]
        totals["correct"] += correct
        totals["groundedness"] += result.groundedness
        if not correct or result.unsupported:
            print(f"  FAIL {row['question']!r}\n       -> {result.answer}\n       unsupported: {result.unsupported}")
    return {
        "expected chunk retrieved": totals["retrieved"] / answerable,
        "expected chunk cited": totals["cited"] / answerable,
        "answer correct": totals["correct"] / len(golden),
        "groundedness": totals["groundedness"] / len(golden),
    }


if __name__ == "__main__":
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    index = Index.from_dir(DOCS)
    if "--generation" in sys.argv:
        import anthropic
        client = anthropic.Anthropic()
        usage = metered(client)
        for metric, value in generation_eval(client, index, golden).items():
            print(f"{metric:<26} {value:.1%}")
        cost = sum(usage[kind] * usd / 1e6 for kind, usd in USD_PER_MTOK.items())
        print(f"{len(golden)} questions, {dict(usage)}, ${cost:.2f} (${cost / len(golden):.4f} per question)")
    else:
        print(f"{len(index.chunks)} chunks\n{'mode':<16}{'hit@1':>8}{'hit@3':>8}{'MRR@5':>8}")
        for mode, metrics in retrieval_ablation(index, golden).items():
            print(f"{mode:<16}" + "".join(f"{value:>8.3f}" for value in metrics.values()))
