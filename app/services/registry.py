"""A tiny on-disk registry of ingested documents.

The endpoint layer previously tracked uploads in a module-level dict. That dict
is process state: it emptied on every restart while the Chroma index on disk
kept its chunks, so `/documents/status` would report an empty service that was
in fact still happily answering questions from indexed documents. Two stores
disagreeing about what exists is a bug generator.

This keeps the catalogue (what was uploaded, when, how many pages) next to the
index that holds the vectors, so both survive a restart together. JSON on disk
is deliberately the least sophisticated thing that removes the inconsistency -
in production this is a table in Postgres alongside per-tenant ownership.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.document import DocumentMetadata


class DocumentRegistry:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Uvicorn serves requests from a thread pool, so concurrent uploads can
        # otherwise interleave read-modify-write and lose an entry.
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A truncated file (killed mid-write) should not brick the service.
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        # Write to a temp file then replace, so a crash cannot leave a half-
        # written registry behind. os.replace is atomic on Windows and POSIX.
        temp = self._path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        temp.replace(self._path)

    def add(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        warning: str | None = None,
    ) -> DocumentMetadata:
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            content_type="application/pdf",
            uploaded_at=datetime.now(timezone.utc),
            page_count=page_count,
            chunk_count=chunk_count,
            warning=warning,
        )
        with self._lock:
            data = self._read()
            data[document_id] = json.loads(metadata.model_dump_json())
            self._write(data)
        return metadata

    def list(self) -> list[DocumentMetadata]:
        return [DocumentMetadata(**entry) for entry in self._read().values()]

    def get(self, document_id: str) -> DocumentMetadata | None:
        entry = self._read().get(document_id)
        return DocumentMetadata(**entry) if entry else None

    def clear(self) -> int:
        """Forget every document. Returns how many entries were removed.

        Needed by `scripts/ingest.py --reset`, which deletes the vector index.
        Without clearing the catalogue too, the two stores diverge: /documents/
        status keeps advertising documents whose vectors are gone, and asking
        about one returns nothing with no explanation.
        """
        with self._lock:
            data = self._read()
            count = len(data)
            if count:
                self._write({})
            return count

    def remove(self, document_id: str) -> bool:
        with self._lock:
            data = self._read()
            existed = data.pop(document_id, None) is not None
            if existed:
                self._write(data)
            return existed
