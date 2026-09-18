"""Index a folder of policy PDFs from the command line.

Ingestion via the HTTP API is fine for one file, but when you are iterating on
chunk size or swapping embedding models you want to rebuild the whole index in
one command without a server running.

    python scripts/ingest.py path/to/pdfs
    python scripts/ingest.py path/to/pdfs --reset
    python scripts/ingest.py policy.pdf --chunk-size 700 --chunk-overlap 120

The document ID is derived from the filename, so re-running the command
re-indexes in place instead of creating duplicates.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Allow `python scripts/ingest.py` from the project root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.services.embedder import Embedder  # noqa: E402
from app.services.pipeline import IngestionPipeline  # noqa: E402
from app.services.registry import DocumentRegistry  # noqa: E402
from app.services.vector_db import VectorStore  # noqa: E402


def document_id_for(path: Path) -> str:
    """A stable, filesystem-safe ID derived from the filename."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem).strip("-").lower() or "document"


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Index policy PDFs into the vector store.")
    parser.add_argument("source", type=Path, help="A PDF file, or a folder of PDFs.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Chunk budget, in tokens (or characters if --chunk-unit=characters).",
    )
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument(
        "--chunk-unit",
        choices=("tokens", "characters"),
        default=settings.chunk_unit,
        help="Token mode guarantees chunks fit the embedding model's input limit.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing index first. Required when changing the embedding "
        "model, since old vectors would be incomparable with new ones.",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: {args.source} does not exist", file=sys.stderr)
        return 1

    pdfs = (
        sorted(args.source.rglob("*.pdf"))
        if args.source.is_dir()
        else [args.source]
    )
    if not pdfs:
        print(f"error: no PDF files found under {args.source}", file=sys.stderr)
        return 1

    store_path = Path(settings.upload_dir).parent / "documents.json"

    if args.reset:
        print(f"Removing existing index at {settings.chroma_dir}")
        shutil.rmtree(settings.chroma_dir, ignore_errors=True)
        # Clear the catalogue as well. Dropping only the vectors leaves the two
        # stores disagreeing: /documents/status would still list documents that
        # can no longer be retrieved from.
        forgotten = DocumentRegistry(store_path).clear()
        if forgotten:
            print(f"Cleared {forgotten} stale document record(s)")

    store = VectorStore(settings.chroma_dir, collection_name=settings.collection_name)
    registry = DocumentRegistry(store_path)
    pipeline = IngestionPipeline(
        vector_store=store,
        embedder=Embedder(model_name=settings.embedding_model),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_unit=args.chunk_unit,
    )

    print(
        f"Indexing {len(pdfs)} PDF(s) with chunk_size={args.chunk_size} "
        f"overlap={args.chunk_overlap} (unit: {args.chunk_unit})\n"
    )
    failures = 0
    for pdf in pdfs:
        document_id = document_id_for(pdf)
        try:
            result = pipeline.ingest(pdf, document_id=document_id, filename=pdf.name)
        except Exception as exc:  # noqa: BLE001 - report and continue to the next file
            failures += 1
            print(f"  FAILED  {pdf.name}: {exc}")
            continue

        registry.add(
            document_id=result.document_id,
            filename=result.filename,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            warning=result.warning,
        )
        note = f"  ({result.warning})" if result.warning else ""
        print(f"  indexed {pdf.name}: {result.page_count} pages -> {result.chunk_count} chunks{note}")

    print(f"\nIndex now holds {store.count()} chunks across {len(store.document_ids())} document(s).")
    if failures:
        print(f"{failures} file(s) failed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
