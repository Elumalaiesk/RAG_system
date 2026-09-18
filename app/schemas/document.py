from datetime import datetime

from pydantic import BaseModel, Field


class IngestionStatus(BaseModel):
    """Result of an upload. `status` is 'indexed' or 'failed'.

    Note it is no longer 'queued'. The old value promised background processing
    that did not exist; ingestion now completes synchronously before the
    response is returned, so the status reflects what actually happened.
    """

    document_id: str
    status: str
    filename: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    message: str | None = None


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    content_type: str
    uploaded_at: datetime
    page_count: int | None = None
    chunk_count: int | None = None
    warning: str | None = Field(
        default=None,
        description="Non-fatal ingestion issues, e.g. pages with no extractable text.",
    )
