"""Request and response shapes for the chat API.

The earlier version put `query`, `top_k`, `conversation_id` and `document_ids`
in one flat request, which conflated three audiences that have little to do with
each other:

  - the END USER, who supplies exactly one thing: a question;
  - the CLIENT APP, which knows which policy is on screen and which
    conversation a turn belongs to;
  - the DEVELOPER, who wants to sweep `top_k` and inspect similarity scores.

A chatbot has no basis for choosing `top_k=5` over `top_k=12`, so exposing it as
a client input is a retrieval detail leaking through the API. And accepting a
list of `document_ids` from the request body lets any caller name any document -
acceptable for a trusted first-party UI, unsafe as soon as more than one
customer's documents share the index (see `resolve_document_scope` in
api/deps.py, which is the single place to add that check).

So the tuning knobs moved to a debug surface (`/retrieval/search`), and what the
client cannot meaningfully supply is now decided server-side.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """What a chatbot client sends: a question, and which thread it belongs to."""

    query: str = Field(min_length=1, description="The user's question.")
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Omit on the first turn - the server mints one and returns it. Send it "
            "back on later turns to keep them grouped. NOTE: it does not yet affect "
            "retrieval; each question is answered independently, so a follow-up like "
            "'what about quarterly?' retrieves poorly. Correlation only for now."
        ),
    )
    document_id: str | None = Field(
        default=None,
        description=(
            "Restrict the answer to one policy - the 'asking about this document' "
            "case. Omit to search everything."
        ),
    )


class SourceRef(BaseModel):
    """One retrieved extract, numbered to match the [n] markers in the answer.

    User-facing and load-bearing: an insurance answer the reader cannot trace
    back to a page is not actionable. `score` is included because it drives the
    confidence bars in the test console; a product UI can ignore it.
    """

    number: int
    citation: str = Field(description="Human-readable location, e.g. 'policy.pdf p.12'.")
    document_id: str
    filename: str
    page_start: int
    page_end: int
    score: float = Field(description="Cosine similarity to the question; higher is better.")
    excerpt: str


class Diagnostics(BaseModel):
    """Internals, grouped so a client can ignore them wholesale.

    Kept out of the top level deliberately: a chat UI should not need to know
    what a cosine floor is in order to render an answer. Everything here exists
    for the test console, the eval scripts, and debugging.
    """

    model: str = Field(default="", description="Which model produced the answer.")
    input_tokens: int = 0
    output_tokens: int = 0
    top_k: int = Field(default=0, description="Chunks requested, as decided by the server.")
    min_similarity: float = Field(default=0.0, description="The relevance floor applied.")
    invalid_citations: list[int] = Field(
        default_factory=list,
        description=(
            "Citation markers matching no source. Non-empty means the model invented "
            "a reference - usually by echoing the document's own clause numbers - so "
            "those markers are unverified."
        ),
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    conversation_id: str = Field(
        description="Always returned. Send it back on the next turn."
    )
    grounded: bool = Field(
        default=True,
        description="False when nothing relevant was retrieved, or the model declined.",
    )
    diagnostics: Diagnostics | None = None


# --------------------------------------------------------------- debug surface


class SearchRequest(BaseModel):
    """Retrieval-only request, for tuning and debugging.

    This is where the knobs belong. It runs no model, so it costs nothing and it
    answers "why did I get that answer?" without spending tokens - the question
    you ask most often while tuning chunk size or the similarity floor.
    """

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_ids: list[str] | None = None
    apply_floor: bool = Field(
        default=True,
        description=(
            "Whether to drop hits below MIN_SIMILARITY. Set false to see what the "
            "floor is rejecting - the fastest way to tell a floor that is too high "
            "from retrieval that genuinely found nothing."
        ),
    )


class SearchResponse(BaseModel):
    query: str
    hits: list[SourceRef] = Field(default_factory=list)
    total_indexed: int = Field(description="Chunks in the collection, for context.")
    min_similarity: float
    floor_applied: bool


class StreamChunk(BaseModel):
    content: str
    done: bool = False
