"""Query rewriting -> hybrid retrieval + rerank -> cited answer -> claim-level hallucination check."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic
from pydantic import BaseModel

from retrieval import Chunk, Index

# A safety decline is re-run server-side on Anthropic's recommended fallback model.
LLM = {"model": "claude-sonnet-5-5", "max_tokens": 16000,
       "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
ABSTAIN = "I don't know based on the available documents."
DOCS = Path(__file__).parent / "data" / "docs"

REWRITE_SYSTEM = """You prepare search queries for a company policy knowledge base.

Given the conversation history and the latest question, produce:
- standalone: the question rewritten so it can be understood without the history. Resolve pronouns and \
elliptical follow-ups ("what about contractors?") from the history. If it is already self-contained, \
keep it as it is.
- variants: two alternative phrasings that a policy document would be likely to use, swapping everyday \
words for formal ones (for example "vacation" -> "paid time off", "bids" -> "quotes"). Keep identifiers \
such as form numbers and error codes exactly as written."""

GENERATE_SYSTEM = f"""You answer questions from Meridian Freight employees using only the numbered \
context passages.

Cite the passage that supports each sentence with its number in square brackets, like [2]. Passages marked \
archived or superseded describe rules that are no longer in force: use them only to say what changed, \
never as the current rule. Answer in one to three sentences. If the passages do not contain the answer, \
reply with exactly this sentence and nothing else: {ABSTAIN}"""

JUDGE_SYSTEM = """You audit an answer against the source passages it was written from.

Split the answer into atomic factual claims, ignoring citation markers. Mark a claim as supported only if \
the passages state it or it follows directly from them. Knowledge from outside the passages does not \
count, even when it is true."""


class Rewrite(BaseModel):
    standalone: str
    variants: list[str]


class Claim(BaseModel):
    text: str
    supported: bool


class Verdict(BaseModel):
    claims: list[Claim]


@dataclass
class Result:
    answer: str
    queries: list[str]
    retrieved: list[Chunk]
    sources: list[Chunk]  # the retrieved passages the answer actually cites
    claims: list[Claim]

    @property
    def unsupported(self) -> list[str]:
        return [c.text for c in self.claims if not c.supported]

    @property
    def groundedness(self) -> float:
        return 1 - len(self.unsupported) / len(self.claims) if self.claims else 1.0


def _context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{n}] {chunk.text}" for n, chunk in enumerate(chunks, 1))


def rewrite(client, question: str, history: str = "") -> list[str]:
    response = client.beta.messages.parse(
        **LLM, system=REWRITE_SYSTEM, output_config={"effort": "low"}, output_format=Rewrite,
        messages=[{"role": "user",
                   "content": f"<history>\n{history}\n</history>\n\n<question>\n{question}\n</question>"}])
    parsed = response.parsed_output
    if parsed is None:  # declined or truncated: retrieval still works on the raw question
        return [question]
    return [parsed.standalone, *parsed.variants[:2]]


def generate(client, question: str, chunks: list[Chunk], correction: str = "") -> str:
    response = client.beta.messages.create(
        **LLM, system=GENERATE_SYSTEM, output_config={"effort": "low"},
        messages=[{"role": "user", "content":
                   f"<context>\n{_context(chunks)}\n</context>\n\n<question>\n{question}\n</question>{correction}"}])
    if response.stop_reason == "refusal":
        return ABSTAIN
    return "".join(block.text for block in response.content if block.type == "text").strip()


def judge(client, answer: str, chunks: list[Chunk]) -> list[Claim]:
    response = client.beta.messages.parse(
        **LLM, system=JUDGE_SYSTEM, output_config={"effort": "medium"}, output_format=Verdict,
        messages=[{"role": "user", "content":
                   f"<passages>\n{_context(chunks)}\n</passages>\n\n<answer>\n{answer}\n</answer>"}])
    if response.parsed_output is None:  # fail closed: an answer we could not verify is not grounded
        return [Claim(text="(verification unavailable)", supported=False)]
    return response.parsed_output.claims


def answer(client, index: Index, question: str, history: str = "") -> Result:
    queries = rewrite(client, question, history)
    chunks = index.search(queries)

    def attempt(correction: str = "") -> tuple[str, list[Claim]]:
        text = generate(client, queries[0], chunks, correction)
        return text, ([] if text == ABSTAIN else judge(client, text, chunks))

    text, claims = attempt()
    unsupported = [c.text for c in claims if not c.supported]
    if unsupported:  # one corrective pass; if it still fails, the caller sees result.unsupported
        text, claims = attempt(
            "\n\nA previous draft made these claims, which the passages do not support. "
            "Leave them out:\n" + "\n".join(f"- {claim}" for claim in unsupported))

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources = [chunks[n - 1] for n in cited if 1 <= n <= len(chunks)]
    return Result(text, queries, chunks, sources, claims)


if __name__ == "__main__":
    result = answer(anthropic.Anthropic(), Index.from_dir(DOCS), " ".join(sys.argv[1:]))
    print(result.answer, "\n")
    for source in result.sources:
        print(f"  source: {source.id}")
    print(f"  queries: {result.queries}")
    print(f"  groundedness: {result.groundedness:.0%}")
    for claim in result.unsupported:
        print(f"  UNSUPPORTED: {claim}")
