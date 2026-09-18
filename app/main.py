import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.deps import get_vector_store
from app.api.v1.api_router import api_router
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    # Open the Chroma index at startup rather than on the first request, so a
    # broken persistence directory fails loudly here instead of as a 500 later.
    store = get_vector_store()
    logger.info(
        "%s ready - model=%s, embeddings=%s, indexed chunks=%d",
        settings.app_name,
        settings.llm_model,
        settings.embedding_model,
        store.count(),
    )
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    current = get_settings()
    if current.api_key and request.url.path.startswith(current.api_v1_prefix):
        if request.headers.get("x-api-key") != current.api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


# Added AFTER the api-key middleware so it sits OUTERMOST in the stack.
# Order matters twice over: a CORS preflight (OPTIONS) carries no x-api-key
# header, so if the api-key middleware ran first it would answer 401 and the
# real request would never be sent; and error responses need CORS headers too,
# or the browser reports an opaque "network error" instead of the actual 502.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,  # no cookies; keeps us off the "*" + credentials trap
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key"],
)


@app.get("/health")
def health() -> dict:
    """Liveness plus a quick look at whether anything is actually indexed.

    `indexed_chunks: 0` is the first thing to check when every question comes
    back with "not enough information" - it means ingestion, not retrieval, is
    where the problem is.
    """
    store = get_vector_store()
    return {
        "status": "ok",
        "indexed_chunks": store.count(),
        "indexed_documents": len(store.document_ids()),
    }


app.include_router(api_router, prefix=settings.api_v1_prefix)


# Serving the console from this app is the recommended route: at `/` it is
# same-origin with the API, so relative fetches just work and no CORS entry is
# involved. Opening the HTML from disk is also supported (the console lets you
# point it at a base URL, and CORS_ORIGINS includes the file:// "null" origin),
# but same-origin is one less moving part.
_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")
