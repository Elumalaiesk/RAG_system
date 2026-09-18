"""Stage 3 - turning text into vectors.

An embedding maps a piece of text to a fixed-length list of floats positioned so
that semantically similar texts land near each other. That is the whole trick
behind RAG: it lets "am I covered if someone steals my car?" retrieve a clause
headed "Theft of the Insured Vehicle" even though the two share almost no words.

The placeholder this replaced returned `[float(len(text))]` - a 1-dimensional
vector of the string's length. It is worth understanding why that is not merely
weak but meaningless: it encodes only how long the text is, so "theft is
covered" and "theft is excluded" (same length) embed to the exact same point.

Model: all-MiniLM-L6-v2, which runs locally on CPU, produces 384-dimensional
vectors, and is downloaded once (~90 MB) to the HuggingFace cache on first use.
Its input ceiling is 256 TOKENS - the constraint that drives token-aware
chunking; see `max_input_tokens` and `token_offsets` below.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# [CLS] and [SEP] are added around every input, so the usable budget for real
# content is two tokens less than the model's advertised maximum.
_SPECIAL_TOKEN_ALLOWANCE = 2


class Embedder:
    """Wraps a sentence-transformers model. Load once, reuse for the process."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        # Reentrant: _measuring_tokenizer() holds this lock and then reads
        # self.model, which locks again. A plain Lock would deadlock whenever the
        # tokenizer is requested before the model has been loaded.
        self._lock = threading.RLock()

    @property
    def model(self):
        """Load the model on first use.

        Deliberately lazy. Loading takes seconds and the first run downloads
        ~90 MB, so doing it at import time would make every `pytest` collection
        and every `--reload` restart pay that cost. The double-checked lock keeps
        two concurrent requests from loading two copies into memory.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Vector length, e.g. 384 for MiniLM.

        sentence-transformers 5.x renamed this method; support both so the code
        works across versions rather than emitting a FutureWarning on every call.
        """
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    @property
    def max_input_tokens(self) -> int:
        """The model's hard input ceiling, in TOKENS (256 for MiniLM).

        Not characters. Text beyond this is dropped by the model before a vector
        is produced, with no error raised.
        """
        return int(self.model.max_seq_length)

    @property
    def usable_input_tokens(self) -> int:
        """Tokens available for actual content, after [CLS]/[SEP]."""
        return max(1, self.max_input_tokens - _SPECIAL_TOKEN_ALLOWANCE)

    # ------------------------------------------------------------- tokenization

    def _measuring_tokenizer(self):
        """An independent tokenizer copy with truncation and padding disabled.

        This exists because of a trap that silently broke token-aware chunking.
        The model's own tokenizer is pre-configured to TRUNCATE at the sequence
        length - which is correct for embedding, but catastrophic for measuring:
        asked how many tokens a 50-page document is, it answered 128, because it
        had already thrown the rest away. Chunking on that answer produced one
        single chunk for the entire document.

        A tokenizer used to decide how to split text must never truncate. We
        therefore serialize the tokenizer and rebuild it, then explicitly clear
        both settings. Rebuilding rather than mutating matters: the original
        object is shared with the model, and disabling truncation on it would
        change how embeddings are produced.
        """
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    backend = getattr(self.model.tokenizer, "backend_tokenizer", None)
                    if backend is None:
                        self._tokenizer = False  # no fast tokenizer; use fallback
                    else:
                        from tokenizers import Tokenizer

                        copy = Tokenizer.from_str(backend.to_str())
                        copy.no_truncation()
                        copy.no_padding()
                        self._tokenizer = copy
        return self._tokenizer or None

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Map every token to its (start, end) character span in `text`.

        This is what makes token-aware chunking possible. Chunk size has to be
        expressed in tokens - that is the unit the model's limit is measured in -
        but chunks must still be *cut* at character positions that fall on
        sentence and word boundaries. The offset map is the bridge between the
        two coordinate systems.
        """
        tokenizer = self._measuring_tokenizer()
        if tokenizer is not None:
            return list(tokenizer.encode(text, add_special_tokens=False).offsets)
        # Fallback for a slow (pure-Python) tokenizer.
        encoded = self.model.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        return [tuple(pair) for pair in encoded["offset_mapping"]]

    def count_tokens(self, text: str) -> int:
        """Token count for `text`, excluding special tokens."""
        tokenizer = self._measuring_tokenizer()
        if tokenizer is not None:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        return len(self.model.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    # ---------------------------------------------------------------- embedding

    def _warn_if_truncated(self, texts: Sequence[str]) -> None:
        """Report chunks the model will silently truncate.

        This is one of the quietest failure modes in RAG. The transformer drops
        anything past `max_seq_length` before producing a vector, so the tail of
        an over-long chunk contributes nothing to whether that chunk is ever
        retrieved - while still being stored, and still being shown to the LLM
        if it is retrieved for some other reason. Nothing errors; recall just
        quietly degrades in a way no amount of prompt tuning can explain.

        With token-aware chunking this should never fire. It stays as a tripwire
        for the cases that bypass the chunker: a very long single query, or a
        corpus chunked in character mode.
        """
        limit = self.max_input_tokens
        over = [count for count in (self.count_tokens(text) for text in texts) if count > limit]
        if over:
            logger.warning(
                "%d of %d input(s) exceed the embedding model's %d-token limit "
                "(largest: %d tokens); %d token(s) will be truncated before "
                "embedding and cannot influence retrieval. Lower CHUNK_SIZE, or "
                "switch to a model with a longer input limit.",
                len(over),
                len(texts),
                limit,
                max(over),
                sum(count - limit for count in over),
            )

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of documents.

        `normalize_embeddings=True` scales every vector to unit length, making
        the dot product of two vectors equal to their cosine similarity - what
        the vector store compares on. Note this is a no-op for MiniLM, whose
        pipeline already ends in a Normalize module; it becomes load-bearing the
        moment EMBEDDING_MODEL changes, since most models do not normalize and
        long chunks would then score higher purely for being long.

        Batching matters too: one `encode` call over 200 chunks is far faster
        than 200 calls, because the model runs them through as a single tensor.
        """
        if not texts:
            return []
        self._warn_if_truncated(texts)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Split out from `embed` because asymmetric models (e5, BGE, ...) require a
        different prefix for queries than for documents. MiniLM is symmetric so
        the two are identical today, but keeping the seam means swapping in such
        a model later is a one-line change here rather than a hunt through
        every call site.
        """
        return self.embed([text])[0]
