"""Stage 2 - splitting a document into retrievable chunks.

Chunking is the highest-leverage and most under-appreciated step in RAG. The
chunk is the unit of retrieval: whatever the embedder sees is what can come
back. Four properties matter:

1. Boundaries. A hard cut every N units routinely severs a clause from its own
   exclusion ("...is covered" | "except where the vehicle was used for hire").
   Retrieved alone, either half is actively misleading. So we snap each cut back
   to the nearest paragraph, then sentence, then word boundary.
2. Overlap. A clause sitting exactly on a boundary would otherwise appear in
   neither neighbour with enough context to be understood. A modest overlap
   costs a little storage and buys recall.
3. Provenance. A chunk that cannot say which page it came from cannot be cited,
   and an insurance answer without a citation is not usable. So chunks are
   objects carrying page ranges, not bare strings.
4. Fit. The embedding model has a hard input limit measured in TOKENS, and
   silently discards anything beyond it. This is the subtle one, and it is why
   this module counts tokens rather than characters - see below.

Why tokens, not characters
--------------------------
Sizing chunks in characters cannot be made safe. The characters-per-token ratio
depends entirely on the text: ordinary policy prose runs about 3.7 characters
per token, but a passage dense with identifiers and addresses ("UIN 101N186V09",
"Jeevan Nidhi II Bldg., Gr. Floor") drops to 2.5, because the tokenizer shreds
each identifier into many pieces. Measured on a real 50-page policy bond, a
1000-character budget put 10 chunks over a 256-token limit; 800 still left 5
over. Any character budget safe for one document overflows on another.

Counting tokens directly removes the guesswork: the budget is expressed in the
same unit the model's limit is measured in, so it is correct by construction on
any document. Character mode is retained for comparison and for callers with no
tokenizer to hand.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass

from app.services.parser import Page

# A function mapping text to one (start, end) character span per token.
# Supplied by Embedder.token_offsets; kept as a plain callable so this module
# stays free of any dependency on transformers or a specific model.
TokenOffsets = Callable[[str], list[tuple[int, int]]]

# Separator inserted between pages when they are concatenated into one stream.
_PAGE_SEPARATOR = "\n\n"

# Fragments shorter than this are dropped: a 20-character sliver ("SECTION 4")
# embeds to near-noise and only crowds out real content in the top-k results.
_MIN_CHUNK_CHARS = 60

# How far forward we will look to escape a half-word at the start of a chunk.
_START_SNAP_LIMIT = 50

# Preferred cut points, best first. We snap backwards to the last one of these
# inside a search window at the end of the chunk.
_PARAGRAPH_BREAK = "\n\n"
# Matches a sentence terminator, an optional closing bracket or quote of any
# kind (straight or curly - [^\w\s] covers both without needing Unicode
# literals in this file), then whitespace.
_SENTENCE_END = re.compile(r"[.!?;:][^\w\s]?\s")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text, with everything needed to cite it."""

    text: str
    index: int  # position within the document, 0-based
    page_start: int
    page_end: int
    char_start: int
    char_end: int


# ------------------------------------------------------------------- boundaries


def _snap_end(text: str, start: int, end: int) -> int:
    """Move `end` backwards to the nearest natural boundary, if one is close.

    The search window is the final 30% of the chunk. Limiting the window matters:
    without it, a chunk in a table-heavy policy schedule (no sentence enders for
    thousands of characters) would snap back almost to `start` and produce a
    stream of tiny chunks. If no boundary is found in the window we accept the
    hard cut - a slightly awkward split beats a pathologically small chunk.

    Snapping only ever moves the cut EARLIER, which can only reduce the token
    count. That direction is what keeps the token budget a genuine ceiling.
    """
    window_start = max(start + 1, end - int((end - start) * 0.3))
    window = text[window_start:end]

    paragraph = window.rfind(_PARAGRAPH_BREAK)
    if paragraph != -1:
        return window_start + paragraph + len(_PARAGRAPH_BREAK)

    sentence_matches = list(_SENTENCE_END.finditer(window))
    if sentence_matches:
        return window_start + sentence_matches[-1].end()

    newline = window.rfind("\n")
    if newline != -1:
        return window_start + newline + 1

    space = window.rfind(" ")
    if space != -1:
        return window_start + space + 1

    return end


def _snap_start(text: str, start: int) -> int:
    """Move `start` forward off a half-word left behind by the overlap step.

    Snapping only the end is not enough. The next chunk begins partway back into
    the previous one, at an offset that usually lands mid-word, so chunks open
    with fragments like "ciation, wear and tear...". Those leading fragments are
    not real tokens: they shift the embedding slightly and they look like
    corruption when the chunk is shown to a user as a cited excerpt.

    If no whitespace appears within `_START_SNAP_LIMIT` characters we leave the
    offset alone rather than skipping content - text with no spaces at all (a
    long identifier, a table row) must not be silently dropped.
    """
    if start <= 0 or start >= len(text):
        return start
    if not text[start - 1].isalnum() or not text[start].isalnum():
        return start

    limit = min(len(text), start + _START_SNAP_LIMIT)
    for position in range(start, limit):
        if text[position].isspace():
            return position + 1
    return start


# ----------------------------------------------------------------- span finding


def split_spans(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[tuple[int, int]]:
    """Split `text` into overlapping (start, end) spans, measured in CHARACTERS.

    The original approach, kept for comparison and for callers with no tokenizer.
    Prefer `split_spans_by_tokens` - see the module docstring for why a character
    budget cannot guarantee the model's token limit is respected.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _snap_end(text, start, end)
        spans.append((start, end))
        if end >= len(text):
            break
        # max(..., start + 1) guarantees forward progress. Without it a snap that
        # lands within `overlap` of `start` would loop forever.
        next_start = max(end - overlap, start + 1)
        snapped = _snap_start(text, next_start)
        start = snapped if snapped < end else next_start
    return spans


def split_spans_by_tokens(
    text: str,
    token_offsets: TokenOffsets,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[tuple[int, int]]:
    """Split `text` into spans of at most `chunk_size` TOKENS.

    The document is tokenized once, giving a character span per token. We then
    walk that list in windows, convert each window back to a character span,
    snap the cut to a sentence or word boundary, and re-derive how many tokens
    actually fell inside. Because snapping only ever moves the cut earlier, the
    token budget is never exceeded.

    `chunk_size` and `overlap` are both in tokens.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    # Drop zero-width tokens; some tokenizers emit them for control characters
    # and they would make the offset arithmetic below ambiguous.
    offsets = [(s, e) for s, e in token_offsets(text) if e > s]
    if not offsets:
        return []

    token_starts = [s for s, _ in offsets]
    total = len(offsets)

    spans: list[tuple[int, int]] = []
    first = 0
    while first < total:
        last = min(first + chunk_size, total)  # exclusive
        char_start = offsets[first][0]
        char_end = offsets[last - 1][1]

        if last < total:
            snapped_end = _snap_end(text, char_start, char_end)
            if snapped_end > char_start:
                char_end = snapped_end
                # Re-derive the token index: the first token that starts at or
                # after the new cut. Everything before it is inside this chunk.
                last = max(bisect_right(token_starts, char_end - 1), first + 1)

        snapped_start = _snap_start(text, char_start)
        if snapped_start < char_end:
            char_start = snapped_start

        spans.append((char_start, char_end))
        if last >= total:
            break
        first = max(last - overlap, first + 1)

    return spans


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Convenience wrapper returning plain strings (character mode, no provenance)."""
    return [text[s:e] for s, e in split_spans(text, chunk_size, overlap)]


# ---------------------------------------------------------------------- pages


def _build_stream(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages into one stream, remembering where each page landed.

    Chunking page-by-page would be simpler, but a clause that straddles a page
    break would then be split every single time - and page breaks in policy
    documents fall in arbitrary places. Chunking the joined stream lets a chunk
    span pages naturally; the offset map below recovers which pages it touched.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0

    for page in pages:
        if not page.text:
            continue
        if parts:
            parts.append(_PAGE_SEPARATOR)
            cursor += len(_PAGE_SEPARATOR)
        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        spans.append((start, cursor, page.number))

    return "".join(parts), spans


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 220,
    overlap: int = 40,
    token_offsets: TokenOffsets | None = None,
) -> list[Chunk]:
    """Turn parsed pages into citable chunks.

    When `token_offsets` is supplied, `chunk_size` and `overlap` are TOKEN counts
    and the result is guaranteed to fit the embedding model's limit. Without it,
    they fall back to being character counts - convenient for tests, but unsafe
    for a real corpus.
    """
    text, page_spans = _build_stream(pages)
    if not text.strip():
        return []

    if token_offsets is not None:
        raw_spans = split_spans_by_tokens(text, token_offsets, chunk_size, overlap)
    else:
        raw_spans = split_spans(text, chunk_size, overlap)

    chunks: list[Chunk] = []
    for char_start, char_end in raw_spans:
        body = text[char_start:char_end].strip()
        if len(body) < _MIN_CHUNK_CHARS:
            continue

        # Every page whose span overlaps this chunk's span.
        touched = [
            page_number
            for page_from, page_to, page_number in page_spans
            if page_from < char_end and page_to > char_start
        ]
        if not touched:
            touched = [page_spans[0][2]] if page_spans else [1]

        chunks.append(
            Chunk(
                text=body,
                index=len(chunks),
                page_start=min(touched),
                page_end=max(touched),
                char_start=char_start,
                char_end=char_end,
            )
        )

    return chunks
