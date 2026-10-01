# enterprise-rag

A retrieval-augmented generation pipeline for company policy documents, built to show what it takes to go from "vector search plus a prompt" to answers you can check.

```
question + history
      │
      ▼
 1. Query rewriting ──▶ standalone question + 2 variants in the documents' vocabulary   (Claude, structured output)
      │
      ▼
 2. Hybrid retrieval ─▶ dense (bge-small) + BM25 for every variant, fused with RRF       (local)
      │
      ▼
 3. Reranking ────────▶ cross-encoder rescoring of the 12 fused candidates, keep 4       (local)
      │
      ▼
 4. Generation ───────▶ answer with [n] citations, or an explicit "I don't know"         (Claude)
      │
      ▼
 5. Hallucination ────▶ answer split into atomic claims, each checked against the        (Claude, structured output)
    check                passages. An unsupported claim triggers one corrective rewrite.
```

## Why each stage exists

| Stage | The failure it fixes | Example from the test set |
|---|---|---|
| Query rewriting | Follow-up questions have no retrievable content on their own | "What about in San Francisco?" after a question about hotel limits |
| BM25 next to dense | Embeddings blur exact identifiers | "E-4417", "Is FIN-117 still valid?" |
| RRF fusion | BM25 and cosine scores are on different scales | Fusing by rank needs no calibration |
| Cross-encoder rerank | Bi-encoders rank the right topic above the right answer | "Do freelancers get holiday pay?" has to land on the clause about contractors |
| Heading-aware chunks | A fixed-size window cuts a rule in half | Every chunk is one policy section, prefixed with `Document > Section` |
| Claim-level check | A fluent answer can add a fact the sources never stated | Each claim is marked supported or not; the score is the supported fraction |
| Archived-document handling | Superseded policies are near-duplicates of current ones | The 2023 expense policy says $60 per day; the current one says $75 |

## Results

Retrieval ablation on `data/golden.jsonl`: 30 single-turn questions over 37 chunks, with the raw question and no rewriting. Reproduce it with `uv run python evaluate.py`, which uses local models only.

| Mode | hit@1 | hit@3 | MRR@5 |
|---|---|---|---|
| dense | 0.633 | 0.933 | 0.778 |
| hybrid (dense + BM25, RRF) | 0.533 | 0.900 | 0.732 |
| hybrid + rerank | **0.767** | **0.967** | **0.867** |

What these numbers say, and what they do not:

- **Hybrid alone ranks worse than dense here.** The questions are deliberately paraphrased ("bids" for "quotes", "medical certificate" for "doctor's note"), and on those BM25 mostly contributes noise to the top of the list.
- **Hybrid still earns its place, as a recall stage.** The expected chunk is in the 12-candidate pool for 100% of questions with hybrid against 96.7% with dense alone, and reranking a dense-only pool reaches hit@1 of 0.733 against 0.767 for the hybrid pool. The reranker can only promote what retrieval hands it.
- **The sample is small.** One question is 3.3 points. This is a development set that I looked at while building (the BM25 stopword filter was added after the first run), not a held-out benchmark.

### End to end

The full pipeline on all 33 questions (30 single-turn, 2 follow-ups that only make sense with their history, 1 out of scope), with `claude-sonnet-5-5`, run on 1 October 2026. Reproduce it with `uv run python evaluate.py --generation` (needs `ANTHROPIC_API_KEY`).

| Metric | Result |
|---|---|
| Expected chunk retrieved, after query rewriting | 96.9% (31 of 32) |
| Expected chunk cited in the answer | 90.6% (29 of 32) |
| Answer correct against the reference (LLM-graded) | 87.9% (29 of 33) |
| Groundedness (supported claims / all claims) | 100% |
| Cost | $0.25 for 135 calls: 78,108 input and 9,573 output tokens, $0.0076 per question |

**Every one of the four errors is the system declining to answer, not the system making something up.**

- Three are abstentions where the documents did hold the answer. Asked about the meal allowance "for a trip to Germany", the model would not connect Germany to the passage's "European Union" rate. Asked whether "freelancers" get holiday pay, it would not map them to "contractors". The third, a lost laptop, also ended in "I don't know".
- One is a hedge: on emailing a card number to a vendor, it cited the right rule and then said it could not confirm whether a vendor counts as an external address.
- The out-of-scope question (office dress code) was correctly refused, and both follow-up questions were answered correctly, which they cannot be without the rewriting step.

So this configuration buys zero unsupported claims at the price of refusing about one answerable question in ten. Whether that is the right trade depends on the product: for policy answers that people act on, a refusal is cheaper than a confident error. The next experiment is to let the generator make one-step bridges that it states explicitly ("Germany is in the EU, so...") and measure whether correctness rises without groundedness falling.

Caveats: correctness is graded by the same model family that wrote the answers, the set is the one I developed against, and this is a single run.

## Run it

```bash
uv sync
uv run pytest                                   # 6 fast tests, no downloads, no API key
uv run python evaluate.py                       # retrieval ablation (downloads ~150 MB of ONNX models once)
uv run python pipeline.py "How many vacation days roll over to next year?"   # needs ANTHROPIC_API_KEY
```

## Design decisions

- **No framework.** Retrieval is about 130 lines (`retrieval.py`) and the pipeline about 140 (`pipeline.py`). BM25 and RRF are written out because they are short and it keeps every ranking decision inspectable.
- **Local embeddings and reranker** through `fastembed` (ONNX, CPU). No embedding API, and retrieval quality can be measured without spending tokens.
- **Structured outputs** for the rewriter and the judge, parsed into Pydantic models, so there is no JSON scraping.
- **Fail closed.** If the judge cannot return a verdict, the answer counts as unverified. If the model declines, the pipeline abstains.
- **The index is rebuilt in memory on each run.** That is right for 37 chunks and wrong for a real corpus; the next step is persisting vectors in pgvector or Qdrant behind the same `Index.search` method.

## Limits, stated plainly

- The corpus is nine synthetic policy documents from a fictional company. Real documents bring PDFs, tables and conflicting versions, none of which are handled here.
- The hallucination judge is the same model family as the generator, so their blind spots correlate. It has not been validated against human labels, and a 100% groundedness score from it is weaker evidence than the same score from a human reviewer.
- The corrective loop runs once. An answer that still has unsupported claims is returned with those claims listed, and the caller decides what to do.
- English only: both local models are English models.

## Layout

```
retrieval.py   chunking, BM25, RRF, dense search, reranking
pipeline.py    rewrite, generate, judge, and answer() which ties them together
evaluate.py    retrieval ablation and end-to-end evaluation
data/docs/     the corpus          data/golden.jsonl   questions, expected chunks, reference answers
test_rag.py    unit tests for the pieces that need no model
```
