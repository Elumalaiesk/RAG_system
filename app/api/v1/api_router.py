from fastapi import APIRouter

from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.retrieval import router as retrieval_router

api_router = APIRouter()
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
# Retrieval-only surface: the tuning knobs live here, not on /chat.
api_router.include_router(retrieval_router, prefix="/retrieval", tags=["retrieval"])
