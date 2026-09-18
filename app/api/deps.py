"""Stage 7 - shared, process-lifetime dependencies.

The bug this replaces was subtle and total. The old provider was:

    def get_rag_engine():
        yield RAGEngine()

`Depends` calls its provider on EVERY request, so that constructed a brand new
RAGEngine - and with it a brand new, empty VectorStore - for each incoming call.
Nothing could ever be retrieved, because the index a request searched was
created milliseconds earlier and had never been written to. Even after the
ingestion pipeline was wired up, this alone would have kept the service
returning "not enough information" forever.

Two things also make per-request construction expensive rather than merely
wrong: the embedding model takes seconds to load into memory, and ChromaDB
holds a file lock on its persistence directory. Both must be created once and
shared. `@lru_cache` gives us that: the provider still runs per request, but it
returns the same instance every time.
"""

import logging
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.embedder import Embedder
from app.services.llm import LLMClient, build_llm
from app.services.pipeline import IngestionPipeline
from app.services.rag_engine import RAGEngine
from app.services.vector_db import VectorStore

logger = logging.getLogger(__name__)


def get_app_settings() -> Settings:
    return get_settings()


@lru_cache
def get_embedder() -> Embedder:
    """One embedding model for the process. Loads lazily on first embed call."""
    return Embedder(model_name=get_settings().embedding_model)


@lru_cache
def get_vector_store() -> VectorStore:
    """One Chroma client for the process, holding the on-disk index."""
    settings = get_settings()
    return VectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
    )


@lru_cache
def get_llm() -> LLMClient:
    """The configured language model, one instance for the process.

    Constructed eagerly but CONNECTED lazily: each implementation builds its
    HTTP client on first use, so a missing key does not break startup. That
    matters because ingestion, retrieval and /health need no LLM key at all -
    embeddings run locally - and a missing key should surface as a clear 502 on
    /chat/ask rather than taking the whole service down.
    """
    settings = get_settings()
    llm = build_llm(settings)
    logger.info("LLM provider=%s model=%s", settings.llm_provider, llm.model)
    return llm


@lru_cache
def get_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        vector_store=get_vector_store(),
        embedder=get_embedder(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_unit=settings.chunk_unit,
    )


@lru_cache
def get_rag_engine() -> RAGEngine:
    return RAGEngine(
        vector_store=get_vector_store(),
        embedder=get_embedder(),
        llm=get_llm(),
        min_similarity=get_settings().min_similarity,
    )


def resolve_document_scope(document_id: str | None) -> list[str] | None:
    """Decide which documents a chat request may search.

    A deliberate seam, and the reason the chat API takes a single `document_id`
    rather than the arbitrary `document_ids` list it used to accept.

    Today this is a pass-through: the caller is a trusted first-party UI, so
    "restrict to the policy on screen" is a legitimate product feature and there
    is no one to defend against. But the shape matters. Retrieval scope is
    fundamentally a property of WHO is asking, not of what the request body says,
    so the moment this service holds more than one customer's documents the
    check belongs here - derive the permitted ids from the authenticated
    principal and ignore the request entirely. Accepting ids from the body at
    that point would be a textbook IDOR: any caller could read any policy by
    guessing an id.

    Returns None to mean "search everything".
    """
    return [document_id] if document_id else None
