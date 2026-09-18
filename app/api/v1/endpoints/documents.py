"""Document ingestion endpoints.

The upload handler previously wrote the PDF to disk, recorded it in a dict, and
returned `status: "queued"` - but no worker ever consumed that queue, so the
document was never parsed, chunked, embedded or indexed. This version runs the
pipeline and only reports success once the chunks are actually searchable.
"""

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_pipeline, get_vector_store
from app.core.config import get_settings
from app.schemas.document import DocumentMetadata, IngestionStatus
from app.services.pipeline import IngestionPipeline
from app.services.registry import DocumentRegistry
from app.services.vector_db import VectorStore

router = APIRouter()

# 25 MB. Without a ceiling, a single large upload is read fully into memory by
# `await file.read()` and can exhaust the process.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@lru_cache
def get_registry() -> DocumentRegistry:
    settings = get_settings()
    return DocumentRegistry(Path(settings.upload_dir).parent / "documents.json")


@router.post("/upload", response_model=IngestionStatus, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    registry: DocumentRegistry = Depends(get_registry),
) -> IngestionStatus:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF documents are supported")

    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document_id = str(uuid4())
    upload_dir = Path(get_settings().upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Store under the generated ID, never under the client-supplied filename -
    # a name like "../../app/main.py" would otherwise escape the upload folder.
    stored_path = upload_dir / f"{document_id}.pdf"
    stored_path.write_bytes(payload)

    filename = file.filename or f"{document_id}.pdf"

    try:
        result = pipeline.ingest(stored_path, document_id=document_id, filename=filename)
    except ValueError as exc:
        # Unreadable, encrypted, or scanned-without-OCR: not retryable, and we
        # should not leave an orphan file behind for a document we cannot serve.
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    registry.add(
        document_id=result.document_id,
        filename=result.filename,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        warning=result.warning,
    )

    return IngestionStatus(
        document_id=result.document_id,
        status="indexed",
        filename=result.filename,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        message=result.warning,
    )


@router.get("/status", response_model=list[DocumentMetadata])
def document_status(
    registry: DocumentRegistry = Depends(get_registry),
) -> list[DocumentMetadata]:
    return registry.list()


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    registry: DocumentRegistry = Depends(get_registry),
    vector_store: VectorStore = Depends(get_vector_store),
) -> None:
    if not registry.remove(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    # Delete the vectors too. Removing only the catalogue entry would leave the
    # document invisible but still retrievable - a deletion that does not delete.
    vector_store.delete_document(document_id)
    (Path(get_settings().upload_dir) / f"{document_id}.pdf").unlink(missing_ok=True)
