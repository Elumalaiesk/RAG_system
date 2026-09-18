"""Unit tests for the ingestion and retrieval pipeline.

These deliberately avoid the Anthropic API entirely. Retrieval is where RAG
usually breaks, it is testable without network access or spend, and it is the
ceiling on answer quality - so it is what gets covered here. The one test that
touches generation stubs the client out.
"""

from __future__ import annotations

import pytest

from app.services.chunker import Chunk, chunk_pages, chunk_text, split_spans
from app.services.embedder import Embedder
from app.services.parser import Page, clean_text, extract_pdf
from app.services.pipeline import IngestionPipeline
from app.services.rag_engine import RAGEngine
from app.services.vector_db import VectorStore
from tests.pdf_fixture import SAMPLE_POLICY_PAGES, write_pdf


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def policy_pdf(tmp_path_factory):
    return write_pdf(tmp_path_factory.mktemp("pdfs") / "motor-policy.pdf", SAMPLE_POLICY_PAGES)


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Session-scoped: loading the model takes seconds, so load it once."""
    return Embedder()


@pytest.fixture
def indexed_store(tmp_path, policy_pdf, embedder) -> VectorStore:
    store = VectorStore(tmp_path / "chroma", collection_name="test")
    pipeline = IngestionPipeline(store, embedder, chunk_size=220, chunk_overlap=40)
    pipeline.ingest(policy_pdf, document_id="motor", filename="motor-policy.pdf")
    return store


# ----------------------------------------------------------------------- parser


def test_clean_text_preserves_paragraph_breaks() -> None:
    # The chunker relies on blank lines to find clause boundaries, so cleaning
    # must not collapse them the way " ".join(text.split()) would.
    assert clean_text("Clause one.\n\n\n\nClause two.") == "Clause one.\n\nClause two."


def test_clean_text_rejoins_hyphenated_line_breaks() -> None:
    assert clean_text("an excl-\nusion applies") == "an exclusion applies"


def test_extract_pdf_returns_one_entry_per_page(policy_pdf) -> None:
    parsed = extract_pdf(policy_pdf)
    assert parsed.page_count == 3
    assert parsed.empty_pages == []
    assert not parsed.is_probably_scanned
    assert "OWN DAMAGE COVER" in parsed.pages[0].text
    assert "NO CLAIM BONUS" in parsed.pages[2].text


def test_scanned_pdf_is_detected() -> None:
    # Pages with no text layer must be flagged, not silently indexed as nothing.
    from app.services.parser import ParsedPdf

    parsed = ParsedPdf(
        pages=[Page(1, ""), Page(2, ""), Page(3, "a bit of text")],
        page_count=3,
        empty_pages=[1, 2],
    )
    assert parsed.is_probably_scanned


# ---------------------------------------------------------------------- chunker


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=4, overlap=4)


def test_chunker_hard_cuts_when_no_boundary_exists() -> None:
    # No whitespace anywhere, so there is nothing to snap to and the naive
    # character window is the correct behaviour.
    assert chunk_text("abcdefghij", chunk_size=5, overlap=2) == ["abcde", "defgh", "ghij"]


def test_chunker_snaps_to_sentence_boundaries() -> None:
    text = "First clause is here. Second clause follows on. Third clause ends it."
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    # Every chunk should begin at a word start, never mid-word.
    for chunk in chunks:
        assert chunk[:1] != " "
        assert not chunk.startswith(("lause", "ollows", "nds"))


def test_chunk_spans_cover_every_character() -> None:
    # Overlap must never turn into a gap - dropped text is silently unanswerable.
    text = "\n\n".join(SAMPLE_POLICY_PAGES)
    covered: set[int] = set()
    for start, end in split_spans(text, chunk_size=300, overlap=60):
        covered.update(range(start, end))
    assert covered == set(range(len(text)))


def test_chunks_carry_page_provenance(policy_pdf) -> None:
    chunks = chunk_pages(extract_pdf(policy_pdf).pages, chunk_size=600, overlap=100)
    assert chunks
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(1 <= chunk.page_start <= chunk.page_end <= 3 for chunk in chunks)
    # A chunk that straddles a page break should report a range, proving the
    # offset mapping works rather than defaulting everything to page 1.
    assert any(chunk.page_start != chunk.page_end for chunk in chunks)


# --------------------------------------------------------------------- embedder


def test_embeddings_are_normalized(embedder) -> None:
    # Unit length is what makes dot product equal cosine similarity.
    vectors = embedder.embed(["theft of the insured vehicle", "no claim bonus"])
    for vector in vectors:
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-5)


def test_embeddings_capture_meaning_not_length(embedder) -> None:
    """The property the old `[float(len(text))]` placeholder could never have.

    'car was stolen' must sit closer to the theft clause than to an unrelated
    sentence of similar length.
    """
    query, related, unrelated = embedder.embed(
        [
            "my car was stolen",
            "loss or damage caused by burglary, housebreaking or theft",
            "the quarterly premium is payable by direct debit",
        ]
    )
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    assert dot(query, related) > dot(query, unrelated)


def test_embed_handles_empty_input(embedder) -> None:
    assert embedder.embed([]) == []


# ------------------------------------------------------------------ vector store


def test_ingestion_indexes_chunks(indexed_store) -> None:
    assert indexed_store.count() > 0
    assert indexed_store.document_ids() == ["motor"]


def test_search_returns_relevant_chunk_with_citation(indexed_store, embedder) -> None:
    hits = indexed_store.search(embedder.embed_query("what is the excess on a claim?"), top_k=3)
    assert hits
    assert any("excess" in hit.text.lower() for hit in hits)
    assert all(hit.filename == "motor-policy.pdf" for hit in hits)
    assert all(hit.citation.startswith("motor-policy.pdf p") for hit in hits)


def test_search_scores_are_ordered_and_bounded(indexed_store, embedder) -> None:
    hits = indexed_store.search(embedder.embed_query("theft cover"), top_k=4)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.01 <= score <= 1.01 for score in scores)


def test_offtopic_question_scores_below_the_floor(indexed_store, embedder) -> None:
    # This is what lets the engine decline without spending a model call.
    hits = indexed_store.search(embedder.embed_query("what is the capital of France?"), top_k=3)
    assert all(hit.score < 0.10 for hit in hits)


def test_document_filter_scopes_the_search(indexed_store, embedder) -> None:
    vector = embedder.embed_query("excess")
    assert indexed_store.search(vector, top_k=3, document_ids=["motor"])
    assert indexed_store.search(vector, top_k=3, document_ids=["some-other-policy"]) == []


def test_delete_document_removes_its_vectors(indexed_store, embedder) -> None:
    removed = indexed_store.delete_document("motor")
    assert removed > 0
    assert indexed_store.count() == 0
    assert indexed_store.search(embedder.embed_query("excess"), top_k=3) == []


def test_reingesting_replaces_rather_than_duplicates(indexed_store, embedder, policy_pdf) -> None:
    before = indexed_store.count()
    pipeline = IngestionPipeline(indexed_store, embedder, chunk_size=220, chunk_overlap=40)
    pipeline.ingest(policy_pdf, document_id="motor", filename="motor-policy.pdf")
    assert indexed_store.count() == before


# ------------------------------------------------------------------- rag engine


class _ExplodingLLM:
    """An LLM that fails the test if it is ever called.

    Used to prove the retrieval guardrail: when nothing clears the similarity
    floor, no provider request should happen at all.
    """

    model = "must-not-be-called"

    def complete(self, system, user):
        raise AssertionError("the model must not be called when retrieval finds nothing")

    def stream(self, system, user):
        raise AssertionError("the model must not be called when retrieval finds nothing")



def test_engine_declines_without_calling_the_model(indexed_store, embedder) -> None:
    """The retrieval guardrail: no relevant chunks means no API call at all."""

    engine = RAGEngine(
        vector_store=indexed_store,
        embedder=embedder,
        llm=_ExplodingLLM(),
        min_similarity=0.10,
    )
    answer = engine.answer("what is the capital of France?")
    assert answer.grounded is False
    assert answer.sources == []
    assert "do not contain information" in answer.text


def test_engine_builds_a_numbered_citable_prompt(indexed_store, embedder) -> None:
    engine = RAGEngine(
        vector_store=indexed_store,
        embedder=embedder,
        llm=_ExplodingLLM(),
        min_similarity=0.10,
    )
    hits = engine.retrieve("what is the excess?", top_k=3)
    assert hits

    prompt = engine._build_prompt("what is the excess?", hits)
    assert "[1] Source: motor-policy.pdf p" in prompt
    assert "what is the excess?" in prompt

    # Source numbering returned to the caller must match the [n] markers the
    # model sees, or every citation in the answer points at the wrong document.
    sources = engine._to_sources(hits)
    assert [source.number for source in sources] == list(range(1, len(hits) + 1))
    assert sources[0].citation in prompt
