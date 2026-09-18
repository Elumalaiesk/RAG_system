"""Chat endpoints.

Two surfaces, deliberately separated by audience:

  POST /chat/ask          - what a chatbot calls. Question in, cited answer out.
                            No retrieval knobs: the server picks top_k from
                            config, because a chat client has no basis to choose.
  POST /chat/ask/stream   - the same, streamed for a responsive UI.

The retrieval knobs live on /retrieval/search (see retrieval.py), which runs no
model and therefore costs nothing.
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_rag_engine, resolve_document_scope
from app.core.config import Settings, get_settings
from app.schemas.chat import ChatRequest, ChatResponse, Diagnostics, SourceRef
from app.services.rag_engine import RAGEngine

router = APIRouter()


@router.post("/ask", response_model=ChatResponse)
def ask(
    request: ChatRequest,
    engine: RAGEngine = Depends(get_rag_engine),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    # The server owns retrieval breadth. Exposing it would put a number on the
    # client that only the operator can reason about, and would let a caller
    # quadruple the cost of every request.
    top_k = settings.top_k
    document_ids = resolve_document_scope(request.document_id)

    try:
        result = engine.answer(request.query, top_k=top_k, document_ids=document_ids)
    except RuntimeError as exc:
        # RAGEngine converts provider failures (auth, quota, connectivity) into
        # RuntimeError with an actionable message. 502 because the failure is
        # upstream of us, not caused by the caller's request.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.text,
        sources=[SourceRef(**vars(source)) for source in result.sources],
        # Minted here when the client did not supply one, so a first turn always
        # comes back with a handle the client can send on the next turn.
        conversation_id=request.conversation_id or str(uuid4()),
        grounded=result.grounded,
        diagnostics=Diagnostics(
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            top_k=top_k,
            min_similarity=engine.min_similarity,
            invalid_citations=result.invalid_citations,
        ),
    )


@router.post("/ask/stream")
def ask_stream(
    request: ChatRequest,
    engine: RAGEngine = Depends(get_rag_engine),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream the answer token by token.

    Two limitations worth knowing, both inherent to streaming rather than
    oversights. HTTP status is committed the moment the first byte is sent, so a
    failure partway through cannot become a 502 - it surfaces as a truncated
    body. And sources cannot be attached, because the response is a plain text
    stream; the conversation id is returned in a header instead. Use /ask when
    you need citations or reliable error semantics.
    """
    document_ids = resolve_document_scope(request.document_id)
    conversation_id = request.conversation_id or str(uuid4())

    return StreamingResponse(
        engine.stream_answer(request.query, top_k=settings.top_k, document_ids=document_ids),
        media_type="text/plain",
        headers={"x-conversation-id": conversation_id},
    )
