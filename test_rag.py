"""Fast checks with no model downloads and no API calls. `python evaluate.py` is the retrieval-quality check."""

import pipeline
from pipeline import ABSTAIN, Claim
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
