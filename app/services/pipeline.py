"""Stage 5 - the ingestion pipeline.

This module is the wire that was missing. Before it, `/documents/upload` saved a
PDF to disk, returned `status: "queued"`, and nothing ever consumed that queue -
so the index stayed permanently empty and every question got "not enough
information". Parsing, chunking, embedding, and indexing all existed as separate
functions with no caller joining them up.

Ingestion is one linear pass:

    PDF file
      -> extract_pdf()   per-page text            (parser)
      -> chunk_pages()   citable, overlapping     (chunker)
      -> embed()         one vector per chunk     (embedder)
      -> add_chunks()    persisted + indexed      (vector_db)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.services.chunker import chunk_pages
from app.services.embedder import Embedder
from app.services.parser import extract_pdf
from app.services.vector_db import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    empty_pages: list[int]
    warning: str | None = None


class IngestionPipeline:
    """Parses, chunks, embeds and indexes a PDF."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        chunk_size: int = 220,
        chunk_overlap: int = 40,
        chunk_unit: str = "tokens",
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_unit = chunk_unit

    def _chunking_arguments(self) -> tuple[int, int, object]:
        """Resolve the chunk budget, clamped to what the model can actually read.

        In token mode a configured CHUNK_SIZE larger than the model's input
        limit would reintroduce exactly the silent truncation this mode exists
        to prevent, so it is clamped rather than trusted. That is worth doing
        loudly: someone who sets 512 for a 256-token model has a wrong mental
        model, and a warning corrects it where a silent clamp would not.
        """
        if self.chunk_unit != "tokens":
            return self.chunk_size, self.chunk_overlap, None

        ceiling = self.embedder.usable_input_tokens
        size = self.chunk_size
        overlap = self.chunk_overlap
        if size > ceiling:
            logger.warning(
                "CHUNK_SIZE=%d exceeds what %s can read (%d usable tokens); "
                "clamping to %d. Text beyond the limit would be discarded "
                "before embedding.",
                size,
                self.embedder._model_name,
                ceiling,
                ceiling,
            )
            size = ceiling
            overlap = min(overlap, max(0, size - 1))
        return size, overlap, self.embedder.token_offsets

    def ingest(self, path: str | Path, document_id: str, filename: str) -> IngestionResult:
        parsed = extract_pdf(path)

        # A scanned policy has no text layer, so it would index zero chunks and
        # fail silently at query time. Surface it as an error the caller can act
        # on instead - the fix is OCR, not a retry.
        if parsed.is_probably_scanned:
            raise ValueError(
                f"{filename}: {len(parsed.empty_pages)} of {parsed.page_count} pages "
                "contain no extractable text. This PDF is most likely a scan and "
                "needs OCR before it can be indexed."
            )

        size, overlap, token_offsets = self._chunking_arguments()
        chunks = chunk_pages(parsed.pages, size, overlap, token_offsets=token_offsets)
        if not chunks:
            raise ValueError(f"{filename}: no usable text found after chunking.")

        embeddings = self.embedder.embed([chunk.text for chunk in chunks])

        # Re-ingesting the same document_id should replace it, not duplicate it.
        # Chroma's add() upserts on matching IDs, but a document that got shorter
        # would leave orphaned chunks behind, so clear it first.
        self.vector_store.delete_document(document_id)
        stored = self.vector_store.add_chunks(chunks, embeddings, document_id, filename)

        warning = None
        if parsed.empty_pages:
            warning = (
                f"{len(parsed.empty_pages)} page(s) had no extractable text and were "
                f"skipped: {parsed.empty_pages[:10]}"
            )

        return IngestionResult(
            document_id=document_id,
            filename=filename,
            page_count=parsed.page_count,
            chunk_count=stored,
            empty_pages=parsed.empty_pages,
            warning=warning,
        )
