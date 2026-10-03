"""Fast checks with no model downloads and no API calls. `python evaluate.py` is the retrieval-quality check."""

from types import SimpleNamespace

import pytest
from model_gateway import Gateway

import evaluate
import pipeline
from pipeline import ABSTAIN, Claim, Rewrite, Verdict
from retrieval import BM25, Chunk, chunk_markdown, rrf, tokenize


def test_chunks_follow_headings_and_carry_their_title():
    chunks = chunk_markdown("pto.md", "# PTO Policy\n\nIntro.\n\n## Sick leave (2025)\n\nTen days.\n")
    assert [c.id for c in chunks] == ["pto.md#overview", "pto.md#sick-leave-2025"]
    assert chunks[1].text == "PTO Policy > Sick leave (2025)\n\nTen days."


def test_long_sections_split_on_paragraphs():
    body = "\n\n".join(" ".join(["word"] * 100) for _ in range(3))
    chunks = chunk_markdown("d.md", f"# D\n\n## Long\n\n{body}", max_words=220)
    assert [c.id for c in chunks] == ["d.md#long", "d.md#long-2"]


def test_bm25_finds_exact_identifiers():
    docs = ["submit form FIN-204 within 30 days", "the hotel cap is 220 dollars", "report lost devices quickly"]
    scores = BM25([tokenize(d) for d in docs]).scores(tokenize("FIN-204"))
    assert scores.index(max(scores)) == 0 and scores[1] == scores[2] == 0


def test_rrf_rewards_agreement_between_rankings():
    assert rrf([[1, 2, 3], [2, 1, 4], [2, 5, 1]])[:2] == [2, 1]


class FakeIndex:
    chunks = [Chunk("a.md#x", "A", "A\n\nThe cap is $220."), Chunk("b.md#y", "B", "B\n\nUnrelated.")]

    def search(self, queries):
        return self.chunks


def test_unsupported_claim_triggers_one_corrective_retry(monkeypatch):
    drafts = iter(["The cap is $220 [1] and breakfast is free [1].", "The cap is $220 [1]."])
    verdicts = iter([[Claim(text="cap is $220", supported=True), Claim(text="breakfast is free", supported=False)],
                     [Claim(text="cap is $220", supported=True)]])
    corrections = []
    monkeypatch.setattr(pipeline, "rewrite", lambda client, q, history: [q])
    monkeypatch.setattr(pipeline, "generate", lambda c, q, chunks, corr="": corrections.append(corr) or next(drafts))
    monkeypatch.setattr(pipeline, "judge", lambda c, text, chunks: next(verdicts))

    result = pipeline.answer(None, FakeIndex(), "hotel cap?")

    assert result.answer == "The cap is $220 [1]."
    assert "breakfast is free" in corrections[1] and corrections[0] == ""
    assert result.groundedness == 1.0 and [s.id for s in result.sources] == ["a.md#x"]


def test_abstention_is_not_judged_and_cites_nothing(monkeypatch):
    monkeypatch.setattr(pipeline, "rewrite", lambda client, q, history: [q])
    monkeypatch.setattr(pipeline, "generate", lambda *a: ABSTAIN)
    monkeypatch.setattr(pipeline, "judge", lambda *a: 1 / 0)

    result = pipeline.answer(None, FakeIndex(), "dress code?")

    assert result.answer == ABSTAIN and result.sources == [] and result.groundedness == 1.0


class FakeMessages:
    """Stands in for client.beta.messages: every call uses 1,000 input and 100 output tokens, and the model
    that was asked answers."""
    usage = SimpleNamespace(input_tokens=1000, output_tokens=100)

    def parse(self, output_format, **request):
        parsed = (Rewrite(standalone="hotel cap", variants=[]) if output_format is Rewrite
                  else Verdict(claims=[Claim(text="cap is $220", supported=True)]))
        return SimpleNamespace(parsed_output=parsed, model=request["model"], usage=self.usage)

    def create(self, **request):
        text = SimpleNamespace(type="text", text="The cap is $220 [1].")
        return SimpleNamespace(stop_reason="end_turn", content=[text], model=request["model"], usage=self.usage)


def test_trace_has_one_span_per_stage_with_its_tokens_and_cost():
    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))

    trace = pipeline.answer(Gateway(client, routes={"rewrite": "small"}), FakeIndex(), "hotel cap?").trace

    assert [span["name"] for span in trace["spans"]] == ["rewrite", "retrieve", "generate", "judge"]
    assert [span["calls"] for span in trace["spans"]] == [1, 0, 1, 1]
    assert [span.get("model") for span in trace["spans"]] == [
        "claude-haiku-4-5", None, "claude-sonnet-5-5", "claude-sonnet-5-5"]
    # one call on the small model and two on the large one, at the gateway's list prices
    assert trace["usd"] == pytest.approx((1000 * 1.00 + 100 * 5.00 + 2 * (1000 * 2.00 + 100 * 10.00)) / 1e6)
    assert trace["cited"] == ["a.md#x"] and not trace["abstained"]


def test_gate_names_only_the_metrics_below_their_floor():
    floors = {"hit@1": 0.76, "hit@3": 0.96}
    assert evaluate.below({"hit@1": 0.767, "hit@3": 0.967}, floors) == []
    assert evaluate.below({"hit@1": 0.733, "hit@3": 0.967}, floors) == ["hit@1 0.733 is below the gate 0.76"]
