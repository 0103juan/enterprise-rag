"""Retrieval: heading-aware chunking, BM25 + dense search, RRF fusion, cross-encoder rerank."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
MODES = ("dense", "hybrid", "hybrid+rerank")


@dataclass(frozen=True)
class Chunk:
    id: str  # "pto-policy.md#carryover"
    title: str  # "Paid Time Off Policy > Carryover"
    text: str  # title + body: the title travels with the text into the embedding, BM25 and the prompt


def slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def chunk_markdown(name: str, markdown: str, max_words: int = 220) -> list[Chunk]:
    """One chunk per `##` section; a section longer than max_words is split on paragraph boundaries."""
    doc_title, sections = name, [("overview", [])]
    for line in markdown.splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)

    chunks = []
    for heading, lines in sections:
        parts, words = [[]], 0
        for paragraph in filter(None, (p.strip() for p in "\n".join(lines).split("\n\n"))):
            if parts[-1] and words + len(paragraph.split()) > max_words:
                parts.append([])
                words = 0
            parts[-1].append(paragraph)
            words += len(paragraph.split())
        title = doc_title if heading == "overview" else f"{doc_title} > {heading}"
        for n, part in enumerate(filter(None, parts)):
            suffix = f"-{n + 1}" if n else ""
            chunks.append(Chunk(f"{name}#{slug(heading)}{suffix}", title, f"{title}\n\n" + "\n\n".join(part)))
    return chunks


STOPWORDS = frozenset(
    "a an and are as at be by can do does for from get have how i if in is it me my need of on or our "
    "should that the their they to we what when which who will with you your".split())


def tokenize(text: str) -> list[str]:
    # Without the stopword filter, question words ("what do I need...") outvote the one content
    # word that matters; IDF alone is too weak for that on a small corpus.
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in docs]
        self.lengths = [len(doc) for doc in docs]
        self.avg_len = sum(self.lengths) / len(docs)
        df = Counter(term for tf in self.tf for term in tf)
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        return [
            sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * length / self.avg_len))
                for t in query if t in tf)
            for tf, length in zip(self.tf, self.lengths)
        ]


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: merges rankings using ranks only, so BM25 and cosine scores never need calibrating."""
    scores = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] += 1 / (k + rank)
    return [item for item, _ in scores.most_common()]


@lru_cache
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBED_MODEL)


@lru_cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(RERANK_MODEL)


class Index:
    # ponytail: rebuilt in memory on every run and searched by brute force; fine for thousands of
    # chunks. Persist the vectors and move to pgvector/Qdrant when the corpus outgrows that.
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.vectors = np.array(list(embedder().passage_embed([c.text for c in chunks])))

    @classmethod
    def from_dir(cls, docs_dir: Path) -> "Index":
        return cls([chunk for path in sorted(Path(docs_dir).glob("*.md"))
                    for chunk in chunk_markdown(path.name, path.read_text(encoding="utf-8"))])

    def _dense(self, query: str, n: int) -> list[int]:
        scores = self.vectors @ next(iter(embedder().query_embed(query)))  # vectors are unit length: dot = cosine
        return np.argsort(-scores)[:n].tolist()

    def _sparse(self, query: str, n: int) -> list[int]:
        scores = self.bm25.scores(tokenize(query))
        return [i for i in np.argsort(scores)[::-1][:n].tolist() if scores[i] > 0]

    def search(self, queries: list[str], k: int = 4, mode: str = "hybrid+rerank", pool: int = 12) -> list[Chunk]:
        """Retrieve for every query variant, fuse, then rerank the pool against queries[0] (the canonical question)."""
        rankings = []
        for query in queries:
            rankings.append(self._dense(query, pool))
            if mode != "dense":
                rankings.append(self._sparse(query, pool))
        order = rrf(rankings)[:pool]
        if mode == "hybrid+rerank":
            scores = list(reranker().rerank(queries[0], [self.chunks[i].text for i in order]))
            order = [i for _, i in sorted(zip(scores, order), reverse=True)]
        return [self.chunks[i] for i in order[:k]]
