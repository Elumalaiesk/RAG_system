"""Stage 6 - retrieval plus generation.

This is the "G" in RAG. The placeholder version did no generation at all: it
concatenated the system prompt and the filled-in QA template and returned that
string as the answer, so the API replied with its own prompt.

The real flow is four steps:

    1. Embed the question with the SAME model used at ingestion time.
    2. Retrieve the nearest chunks, and discard weak matches.
    3. If nothing survives, decline to answer WITHOUT calling the model.
    4. Otherwise ask the configured LLM to answer from the numbered extracts.

Step 3 is the one people skip. Handing the model five irrelevant chunks and
hoping it declines costs money on every junk query and invites the model to
stitch an answer together from whatever it was given. Deciding not to answer is
a retrieval decision, and the similarity scores already contain that decision.

Note what this module does NOT know: which provider is answering. That lives
behind LLMClient in services/llm.py, so switching between Claude and Gemini is a
configuration change, not a code change here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.prompts.templates import NO_CONTEXT_ANSWER, QA_PROMPT, SYSTEM_PROMPT, format_context
from app.services.embedder import Embedder
from app.services.llm import LLMClient
from app.services.vector_db import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    """One cited extract, numbered to match the [n] markers in the answer."""

    number: int
    citation: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    score: float
    excerpt: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    grounded: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    # Citation markers in the answer that do not correspond to any source.
    invalid_citations: list[int] = field(default_factory=list)


class RAGEngine:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        llm: LLMClient,
        min_similarity: float = 0.10,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        self.min_similarity = min_similarity

    # ---------------------------------------------------------------- retrieval

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Find relevant chunks, dropping those below the similarity floor.

        `min_similarity` is the single most important number to tune against
        your own documents. Too high and valid answers get refused; too low and
        the model is fed noise and starts inventing. It is not transferable
        between embedding models - a cosine of 0.4 means different things to
        MiniLM and to a larger model - so re-tune it if you swap the embedder.
        Note it is unaffected by the LLM provider: retrieval happens entirely
        locally, before any model is called.
        """
        query_vector = self.embedder.embed_query(query)
        hits = self.vector_store.search(query_vector, top_k=top_k, document_ids=document_ids)
        return [hit for hit in hits if hit.score >= self.min_similarity]

    @staticmethod
    def _to_sources(hits: list[RetrievedChunk]) -> list[Source]:
        return [
            Source(
                number=position,
                citation=hit.citation,
                document_id=hit.document_id,
                filename=hit.filename,
                page_start=hit.page_start,
                page_end=hit.page_end,
                score=round(hit.score, 4),
                # Trimmed so an API response stays readable; the model always
                # receives the full chunk text.
                excerpt=hit.text[:400] + ("..." if len(hit.text) > 400 else ""),
            )
            for position, hit in enumerate(hits, start=1)
        ]

    def _build_prompt(self, query: str, hits: list[RetrievedChunk]) -> str:
        return QA_PROMPT.format(
            context=format_context([(hit.citation, hit.text) for hit in hits]),
            question=query,
        )

    # Matches "[2]" and also grouped forms like "[5, 14]" that models emit.
    _CITATION = re.compile(r"\[([\d,\s]+)\]")

    @classmethod
    def _invalid_citations(cls, text: str, source_count: int) -> list[int]:
        """Find citation markers that point at no supplied extract.

        Observed live with gemini-2.5-flash: given 4 extracts it answered with
        "[5, 14]", having lifted those numbers from the policy's own clause
        numbering inside the extract text. A marker like that is worse than no
        citation - it looks verifiable and is not, and a reader clicking it
        finds nothing.

        The prompt now forbids it explicitly, but a prompt is a request, not a
        guarantee, so the result is checked. Detection only: the answer is still
        returned, with the offending markers reported so the caller can show a
        warning rather than presenting a false citation as fact.
        """
        used = {
            int(number)
            for group in cls._CITATION.findall(text)
            for number in re.findall(r"\d+", group)
        }
        return sorted(n for n in used if n < 1 or n > source_count)

    # --------------------------------------------------------------- generation

    def answer(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> Answer:
        hits = self.retrieve(query, top_k=top_k, document_ids=document_ids)
        if not hits:
            # No model call: retrieval already established there is nothing to
            # ground an answer in.
            return Answer(
                text=NO_CONTEXT_ANSWER, sources=[], grounded=False, model=self.llm.model
            )

        sources = self._to_sources(hits)
        # Every provider raises RuntimeError with an actionable message, which
        # the endpoint turns into a 502.
        result = self.llm.complete(SYSTEM_PROMPT, self._build_prompt(query, hits))

        invalid = self._invalid_citations(result.text, len(sources))
        if invalid:
            logger.warning(
                "%s cited %s but only %d extract(s) were supplied - the model is "
                "most likely echoing the document's own clause numbers.",
                self.llm.model,
                invalid,
                len(sources),
            )

        return Answer(
            text=result.text,
            sources=sources,
            grounded=not result.declined,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=self.llm.model,
            invalid_citations=invalid,
        )

    def stream_answer(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield the answer incrementally, for a responsive chat UI."""
        hits = self.retrieve(query, top_k=top_k, document_ids=document_ids)
        if not hits:
            yield NO_CONTEXT_ANSWER
            return
        yield from self.llm.stream(SYSTEM_PROMPT, self._build_prompt(query, hits))

    # ------------------------------------------------------------------ costing

    def count_prompt_tokens(self, query: str, top_k: int = 5) -> int:
        """Approximate input size for a query, without generating an answer.

        Uses the LOCAL embedding tokenizer, not the provider's. That makes it
        provider-independent and free, at the cost of being an estimate: every
        model tokenizes differently, so treat this as a way to compare the
        relative cost of top_k=5 against top_k=20 rather than as a billing
        figure. The exact count for a call is in the response's input_tokens.
        """
        hits = self.retrieve(query, top_k=top_k)
        prompt = self._build_prompt(query, hits) if hits else query
        return self.embedder.count_tokens(SYSTEM_PROMPT + "\n" + prompt)
