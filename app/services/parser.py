"""Stage 1 - PDF text extraction.

Insurance answers have to be traceable to a specific clause, so extraction
deliberately does NOT flatten the document into one big string. It keeps page
boundaries, because every chunk built later inherits a page number from here.
That is what lets the API answer "excluded under section 4, page 12" instead of
making an unsourceable claim the user cannot verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class Page:
    """One page of extracted text. `number` is 1-indexed to match PDF viewers."""

    number: int
    text: str


@dataclass(frozen=True)
class ParsedPdf:
    pages: list[Page]
    page_count: int
    empty_pages: list[int]

    @property
    def is_probably_scanned(self) -> bool:
        """True when most pages yielded no text.

        pypdf reads the text layer only. A policy that was scanned to image has
        no text layer, so extraction silently returns empty strings and the whole
        pipeline indexes nothing. Detecting it here turns a confusing "the bot
        knows nothing about my document" bug into an explicit, actionable error.
        Fixing it requires OCR (e.g. Tesseract), which is out of scope for now.
        """
        return self.page_count > 0 and len(self.empty_pages) > self.page_count / 2


# A word split across a line break by hyphenation: "excl-\nusion" -> "exclusion".
# Policy PDFs are full of these; left unfixed they poison both the embedding
# (the model never saw the token "excl") and any keyword search.
_HYPHEN_LINEBREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
_HORIZONTAL_SPACE = re.compile("[ \t\u00a0]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace while PRESERVING paragraph breaks.

    Note the contrast with a naive `" ".join(text.split())`: that collapses
    newlines too, which destroys the blank-line paragraph boundaries the chunker
    relies on to avoid cutting a clause in half. Clean conservatively - every
    structural signal you erase here is one the chunker cannot use later.
    """
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    # Strip trailing spaces on each line without touching the line breaks.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def extract_pdf(path: str | Path) -> ParsedPdf:
    """Extract per-page text from a PDF.

    Raises ValueError for a file that is encrypted or unreadable, so the caller
    can report a clear 4xx rather than indexing an empty document.
    """
    # pypdf raises its own exception hierarchy (PdfReadError, PdfStreamError,
    # and assorted parsing errors) for a truncated or malformed file. Converting
    # them to ValueError here keeps that library detail out of the API layer:
    # the endpoint catches ValueError and returns 422. Without this a corrupt
    # upload escapes as an unhandled exception and becomes a 500 - reporting a
    # server fault for what is really a bad input.
    try:
        reader = PdfReader(str(path))
        page_objects = list(reader.pages)
    except PdfReadError as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - pypdf is not exhaustive about its types
        raise ValueError(f"Could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        # An empty password unlocks many "protected" policy PDFs; try it once.
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf raises several types here
            raise ValueError(f"PDF is password protected: {path}") from exc

    pages: list[Page] = []
    empty_pages: list[int] = []

    for page_number, pdf_page in enumerate(page_objects, start=1):
        try:
            raw = pdf_page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one malformed page must not kill the ingest
            raw = ""
        text = clean_text(raw)
        if not text:
            empty_pages.append(page_number)
        pages.append(Page(number=page_number, text=text))

    return ParsedPdf(pages=pages, page_count=len(pages), empty_pages=empty_pages)
