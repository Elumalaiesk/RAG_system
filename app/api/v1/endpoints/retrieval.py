"""Retrieval-only endpoint: the developer's surface.

This is where the knobs the chat API no longer exposes actually belong. It runs
NO language model, which is the point twice over: it costs nothing, and it
isolates the half of RAG that usually breaks.

When an answer looks wrong there are only two possibilities - the right clause
was never retrieved, or it was retrieved and the model mishandled it. Those need
completely different fixes (chunking and the similarity floor versus the prompt),
and this endpoint tells you which one you have, for free, in one request.
"""

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_engine
from app.core.config import Settings, get_settings
from app.schemas.chat import SearchRequest, SearchResponse, SourceRef
from app.services.rag_engine import RAGEngine

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    engine: RAGEngine = Depends(get_rag_engine),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    query_vector = engine.embedder.embed_query(request.query)
    hits = engine.vector_store.search(
        query_vector, top_k=request.top_k, document_ids=request.document_ids
    )

    # apply_floor=false is the diagnostic that matters most: it shows what the
    # floor is throwing away. A correct chunk sitting just below MIN_SIMILARITY
    # and an empty index look identical from /chat/ask, and they are opposite
    # problems.
    if request.apply_floor:
        hits = [hit for hit in hits if hit.score >= engine.min_similarity]

    return SearchResponse(
        query=request.query,
        hits=[
            SourceRef(
                number=position,
                citation=hit.citation,
                document_id=hit.document_id,
                filename=hit.filename,
                page_start=hit.page_start,
                page_end=hit.page_end,
                score=round(hit.score, 4),
                excerpt=hit.text[:400] + ("..." if len(hit.text) > 400 else ""),
            )
            for position, hit in enumerate(hits, start=1)
        ],
        total_indexed=engine.vector_store.count(),
        min_similarity=engine.min_similarity,
        floor_applied=request.apply_floor,
    )
