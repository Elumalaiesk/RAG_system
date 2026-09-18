"""HTTP-level tests.

Generation is stubbed throughout - these tests assert the API contract (status
codes, response shape, wiring), not model behaviour. A test that calls Claude
would be slow, non-deterministic, and would charge money on every CI run.

The interesting assertions here are the ones that would have caught the two
wiring bugs in the original scaffold: that an upload actually lands in the
index, and that a query can retrieve what a previous request indexed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_pipeline, get_rag_engine, get_vector_store
from app.api.v1.endpoints.documents import get_registry
from app.main import app
from app.services.rag_engine import Answer, Source
from tests.pdf_fixture import SAMPLE_POLICY_PAGES, write_pdf


class StubEngine:
    """Stands in for RAGEngine, recording what it was asked.

    `min_similarity` is part of the interface the endpoint reads for its
    diagnostics block, so the stub has to expose it too.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.min_similarity = 0.10

    def answer(self, query, top_k=5, document_ids=None):
        self.calls.append({"query": query, "top_k": top_k, "document_ids": document_ids})
        return Answer(
            text="A compulsory excess of 5,000 applies to every claim [1].",
            sources=[
                Source(
                    number=1,
                    citation="motor-policy.pdf p.1",
                    document_id="motor",
                    filename="motor-policy.pdf",
                    page_start=1,
                    page_end=1,
                    score=0.62,
                    excerpt="A compulsory excess of 5,000 applies...",
                )
            ],
            grounded=True,
            input_tokens=812,
            output_tokens=44,
            model="stub-model-1",
        )

    def stream_answer(self, query, top_k=5, document_ids=None):
        yield "A compulsory excess "
        yield "of 5,000 applies [1]."


@pytest.fixture
def stub_engine():
    engine = StubEngine()
    app.dependency_overrides[get_rag_engine] = lambda: engine
    yield engine
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def policy_bytes(tmp_path):
    return write_pdf(tmp_path / "motor-policy.pdf", SAMPLE_POLICY_PAGES).read_bytes()


# ------------------------------------------------------------------------ chat


def test_ask_returns_answer_with_numbered_sources(client, stub_engine) -> None:
    response = client.post("/api/v1/chat/ask", json={"query": "What is the excess?"})
    assert response.status_code == 200

    body = response.json()
    assert body["answer"].startswith("A compulsory excess")
    assert body["grounded"] is True
    assert body["diagnostics"]["input_tokens"] == 812
    assert len(body["sources"]) == 1
    assert body["sources"][0]["number"] == 1
    assert body["sources"][0]["citation"] == "motor-policy.pdf p.1"


def test_server_decides_top_k_not_the_client(client, stub_engine) -> None:
    """top_k is deliberately NOT a client input.

    A chat client has no basis for choosing 5 over 12, and accepting it would
    let any caller quadruple the cost of every request. A client that sends it
    anyway is ignored - the value comes from settings.
    """
    from app.core.config import get_settings

    client.post(
        "/api/v1/chat/ask",
        json={"query": "What is the excess?", "top_k": 12, "document_ids": ["other"]},
    )
    call = stub_engine.calls[-1]
    assert call["top_k"] == get_settings().top_k
    assert call["top_k"] != 12
    # document_ids is not read from the body either; only `document_id` is, via
    # resolve_document_scope.
    assert call["document_ids"] is None


def test_document_id_scopes_retrieval(client, stub_engine) -> None:
    client.post(
        "/api/v1/chat/ask",
        json={"query": "What is the excess?", "document_id": "motor"},
    )
    assert stub_engine.calls[-1]["document_ids"] == ["motor"]


def test_conversation_id_is_minted_then_echoed(client, stub_engine) -> None:
    """The server owns the id: a client cannot be expected to invent one."""
    first = client.post("/api/v1/chat/ask", json={"query": "What is the excess?"}).json()
    assert first["conversation_id"]

    second = client.post(
        "/api/v1/chat/ask",
        json={"query": "And the deductible?", "conversation_id": first["conversation_id"]},
    ).json()
    assert second["conversation_id"] == first["conversation_id"]


def test_diagnostics_are_grouped_not_top_level(client, stub_engine) -> None:
    """A chat UI should be able to ignore internals wholesale."""
    body = client.post("/api/v1/chat/ask", json={"query": "What is the excess?"}).json()
    for internal in ("model", "input_tokens", "output_tokens", "invalid_citations"):
        assert internal not in body, f"{internal} leaked to the top level"
    d = body["diagnostics"]
    assert d["model"] == "stub-model-1"
    assert d["input_tokens"] == 812
    assert d["top_k"] > 0
    assert "min_similarity" in d


def test_ask_rejects_empty_query(client, stub_engine) -> None:
    assert client.post("/api/v1/chat/ask", json={"query": ""}).status_code == 422


def test_unknown_fields_on_chat_request_are_ignored(client, stub_engine) -> None:
    """Extra keys must not 400 - old clients may still send top_k."""
    response = client.post("/api/v1/chat/ask", json={"query": "hi", "top_k": 99})
    assert response.status_code == 200


def test_upstream_failure_becomes_502(client) -> None:
    class FailingEngine:
        def answer(self, *args, **kwargs):
            raise RuntimeError("Anthropic API key missing or invalid.")

    app.dependency_overrides[get_rag_engine] = lambda: FailingEngine()
    try:
        response = client.post("/api/v1/chat/ask", json={"query": "What is the excess?"})
        assert response.status_code == 502
        assert "API key" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_stream_returns_the_full_answer(client, stub_engine) -> None:
    response = client.post("/api/v1/chat/ask/stream", json={"query": "What is the excess?"})
    assert response.status_code == 200
    assert response.text == "A compulsory excess of 5,000 applies [1]."


# ------------------------------------------------------------------- documents


def test_upload_rejects_non_pdf(client) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_empty_file(client) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_upload_reports_unparseable_pdf_as_422(client) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("broken.pdf", b"%PDF-1.4 this is not a real pdf", "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_indexes_the_document_and_makes_it_queryable(
    client, policy_bytes, tmp_path
) -> None:
    """The end-to-end wiring test.

    This is precisely what the original scaffold could not do: upload a PDF and
    then retrieve from it. It exercises parse -> chunk -> embed -> index -> search
    for real (only generation is stubbed), so it fails if any link is broken.
    """
    from app.services.embedder import Embedder
    from app.services.pipeline import IngestionPipeline
    from app.services.registry import DocumentRegistry
    from app.services.vector_db import VectorStore

    store = VectorStore(tmp_path / "chroma", collection_name="apitest")
    embedder = Embedder()
    registry = DocumentRegistry(tmp_path / "documents.json")

    app.dependency_overrides[get_pipeline] = lambda: IngestionPipeline(
        store, embedder, chunk_size=600, chunk_overlap=100
    )
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_registry] = lambda: registry

    try:
        upload = client.post(
            "/api/v1/documents/upload",
            files={"file": ("motor-policy.pdf", policy_bytes, "application/pdf")},
        )
        assert upload.status_code == 201, upload.text
        body = upload.json()
        assert body["status"] == "indexed"
        assert body["page_count"] == 3
        assert body["chunk_count"] > 0

        document_id = body["document_id"]

        # It is listed...
        listed = client.get("/api/v1/documents/status").json()
        assert [entry["document_id"] for entry in listed] == [document_id]
        assert listed[0]["filename"] == "motor-policy.pdf"

        # ...and, crucially, it is actually retrievable.
        hits = store.search(embedder.embed_query("what is the excess?"), top_k=3)
        assert any("excess" in hit.text.lower() for hit in hits)

        # Deleting removes the vectors too, not just the catalogue entry.
        assert client.delete(f"/api/v1/documents/{document_id}").status_code == 204
        assert store.count() == 0
        assert client.get("/api/v1/documents/status").json() == []
    finally:
        app.dependency_overrides.clear()


def test_delete_unknown_document_is_404(client) -> None:
    assert client.delete("/api/v1/documents/does-not-exist").status_code == 404


# ---------------------------------------------------------------------- health


def test_health_reports_index_size(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "indexed_chunks" in body
    assert "indexed_documents" in body


# ------------------------------------------------------------------------ cors


def test_console_is_served_at_root(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The base-URL control is what makes the page usable when it is NOT served
    # from the API's own origin.
    assert 'id="base"' in response.text


def test_cors_allows_the_file_origin(client) -> None:
    """A page opened from disk sends `Origin: null`.

    Supported on purpose so double-clicking the HTML works, but it is the
    loosest entry in the allowlist - see the note on CORS_ORIGINS in config.
    """
    response = client.get("/health", headers={"Origin": "null"})
    assert response.headers.get("access-control-allow-origin") == "null"


def test_cors_preflight_succeeds_without_an_api_key(client) -> None:
    """Preflight carries no x-api-key, so CORS must sit outside the key check.

    If the api-key middleware ran first it would answer 401 to the OPTIONS
    request and the browser would never send the real one.
    """
    response = client.options(
        "/api/v1/chat/ask",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"
    assert "x-api-key" in response.headers.get("access-control-allow-headers", "").lower()


def test_cors_rejects_an_unlisted_origin(client) -> None:
    """The allowlist must actually exclude things - never a wildcard."""
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_headers_present_on_error_responses(client) -> None:
    """Without this the browser reports an opaque network error, not the 502."""

    class FailingEngine:
        def answer(self, *args, **kwargs):
            raise RuntimeError("upstream is down")

    app.dependency_overrides[get_rag_engine] = lambda: FailingEngine()
    try:
        response = client.post(
            "/api/v1/chat/ask",
            json={"query": "What is the excess?"},
            headers={"Origin": "null"},
        )
        assert response.status_code == 502
        assert response.headers.get("access-control-allow-origin") == "null"
    finally:
        app.dependency_overrides.clear()


def test_ask_reports_which_model_answered(client, stub_engine) -> None:
    """With the provider selectable at runtime, an answer needs its attribution.

    This field was silently empty at first: added to the schema but never passed
    through from the engine, so it defaulted to "".
    """
    body = client.post("/api/v1/chat/ask", json={"query": "What is the excess?"}).json()
    assert body["diagnostics"]["model"] == "stub-model-1"


# ------------------------------------------------------------------- retrieval


def test_retrieval_search_runs_no_model(client, policy_bytes, tmp_path) -> None:
    """The debug surface must never call an LLM - that is its whole value.

    It answers "was the right clause retrieved?" for free, which is a different
    question from "did the model use it properly" and needs a different fix.
    """
    from app.services.embedder import Embedder
    from app.services.pipeline import IngestionPipeline
    from app.services.rag_engine import RAGEngine
    from app.services.vector_db import VectorStore

    embedder = Embedder()
    store = VectorStore(tmp_path / "chroma", collection_name="searchtest")
    pdf = tmp_path / "motor.pdf"
    pdf.write_bytes(policy_bytes)
    IngestionPipeline(store, embedder, 220, 40).ingest(pdf, "motor", "motor-policy.pdf")

    class ExplodingLLM:
        model = "must-not-be-called"

        def complete(self, *a, **k):
            raise AssertionError("/retrieval/search must not call the model")

        def stream(self, *a, **k):
            raise AssertionError("/retrieval/search must not call the model")

    engine = RAGEngine(store, embedder, ExplodingLLM(), min_similarity=0.10)
    app.dependency_overrides[get_rag_engine] = lambda: engine
    try:
        response = client.post(
            "/api/v1/retrieval/search",
            json={"query": "what is the excess?", "top_k": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["hits"], "expected retrieval hits"
        assert body["total_indexed"] == store.count()
        assert body["floor_applied"] is True
        assert [h["number"] for h in body["hits"]] == list(range(1, len(body["hits"]) + 1))
        assert all(h["score"] >= body["min_similarity"] for h in body["hits"])
    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_can_show_what_the_floor_rejects(client, tmp_path, policy_bytes) -> None:
    """apply_floor=false is the diagnostic that matters most.

    A correct chunk sitting just below MIN_SIMILARITY and an empty index look
    identical from /chat/ask, and they are opposite problems.
    """
    from app.services.embedder import Embedder
    from app.services.pipeline import IngestionPipeline
    from app.services.rag_engine import RAGEngine
    from app.services.vector_db import VectorStore

    embedder = Embedder()
    store = VectorStore(tmp_path / "chroma2", collection_name="floortest")
    pdf = tmp_path / "motor.pdf"
    pdf.write_bytes(policy_bytes)
    IngestionPipeline(store, embedder, 220, 40).ingest(pdf, "motor", "motor-policy.pdf")

    # A floor of 0.9 rejects everything; apply_floor=false must still show hits.
    engine = RAGEngine(store, embedder, object(), min_similarity=0.9)
    app.dependency_overrides[get_rag_engine] = lambda: engine
    try:
        strict = client.post(
            "/api/v1/retrieval/search",
            json={"query": "what is the excess?", "top_k": 3, "apply_floor": True},
        ).json()
        assert strict["hits"] == []

        unfiltered = client.post(
            "/api/v1/retrieval/search",
            json={"query": "what is the excess?", "top_k": 3, "apply_floor": False},
        ).json()
        assert unfiltered["hits"], "apply_floor=false should reveal the rejected hits"
        assert unfiltered["floor_applied"] is False
        assert unfiltered["min_similarity"] == 0.9
    finally:
        app.dependency_overrides.clear()


def test_retrieval_search_validates_top_k(client) -> None:
    assert client.post("/api/v1/retrieval/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/api/v1/retrieval/search", json={"query": "x", "top_k": 99}).status_code == 422
