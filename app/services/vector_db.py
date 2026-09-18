"""Stage 4 - storing and searching vectors, backed by ChromaDB.

Retrieval is a nearest-neighbour search: embed the question, then find the
chunks whose vectors sit closest to it. Chroma persists those vectors to disk
and runs an approximate nearest-neighbour (HNSW) index over them, so search
stays fast as the corpus grows instead of comparing against every chunk.

Two things this replaces from the placeholder:
  - The old store lived in a plain Python list, so every restart lost the index.
  - It ranked by counting shared words, which is keyword search wearing a vector
    store's clothing - and it never called the embedder at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.chunker import Chunk

# Chroma is told to use cosine distance. Combined with the normalized vectors
# from the embedder, this makes similarity = 1 - distance, bounded in [0, 2].
_DISTANCE_SPACE = "cosine"


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit, carrying everything needed to cite and rank it."""

    text: str
    document_id: str
    filename: str
    page_start: int
    page_end: int
    chunk_index: int
    score: float  # cosine similarity, higher is better

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.filename} p.{self.page_start}"
        return f"{self.filename} pp.{self.page_start}-{self.page_end}"


class VectorStore:
    """Persistent vector index over policy chunks."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "policies") -> None:
        import chromadb
        from chromadb.config import Settings

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # embedding_function is left unset on purpose: we pass vectors in
        # ourselves so that indexing and querying provably use the same model.
        # Letting Chroma embed for us would silently use its own default model
        # and produce a corpus that cannot be searched by our query vectors.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": _DISTANCE_SPACE},
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        document_id: str,
        filename: str,
    ) -> int:
        """Index a document's chunks. Returns the number stored."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        self._collection.add(
            # Deterministic IDs, so re-ingesting the same document overwrites its
            # chunks rather than silently duplicating them in the index.
            ids=[f"{document_id}:{chunk.index}" for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": chunk.index,
                }
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Find the `top_k` chunks nearest to the query vector.

        `document_ids` restricts the search to specific policies - the metadata
        filter a real product needs so one customer cannot retrieve another
        customer's documents.
        """
        if self.count() == 0:
            return []

        where = {"document_id": {"$in": document_ids}} if document_ids else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma nests results one level per query vector; we sent exactly one.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        hits: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            hits.append(
                RetrievedChunk(
                    text=text,
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "unknown")),
                    page_start=int(metadata.get("page_start", 0)),
                    page_end=int(metadata.get("page_end", 0)),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    # Convert distance to a similarity so "higher is better"
                    # holds everywhere above this line.
                    score=1.0 - float(distance),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document. Returns the count."""
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return int(self._collection.count())

    def document_ids(self) -> list[str]:
        """Distinct document IDs currently indexed."""
        everything = self._collection.get(include=["metadatas"])
        seen = {
            str(metadata.get("document_id"))
            for metadata in everything.get("metadatas", [])
            if metadata and metadata.get("document_id")
        }
        return sorted(seen)
