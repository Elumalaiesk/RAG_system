from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable from `.env` or the environment.

    Every knob that changes retrieval or generation quality lives here rather
    than being buried as a default deep in a service, so you can tune the system
    without editing code - and so a change is visible in one place.
    """

    # --- Application ---
    app_name: str = "Insurance RAG Service"
    api_v1_prefix: str = "/api/v1"
    api_key: str | None = None
    upload_dir: str = "storage/uploads"
    chroma_dir: str = "storage/chroma"

    # Browser origins allowed to call this API, comma-separated.
    #
    # Needed only when the test console is NOT served by this app. Served from
    # `/` it is same-origin and no CORS entry is required; opened as a file, or
    # from a separate dev server, the browser sends an Origin header that must
    # appear here or every request is blocked.
    #
    # "null" is the literal Origin a browser sends for a file:// page. It is
    # included so double-clicking app/static/index.html works, but note it
    # matches ANY file:// page on the machine - drop it once you stop needing
    # that, and replace this whole list with your real frontend origin before
    # this service goes anywhere near production. Never use "*" here: with real
    # policy documents behind it, that lets any website read them.
    cors_origins: str = (
        "http://localhost:8000,http://127.0.0.1:8000,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:3000,http://127.0.0.1:3000,"
        "null"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # --- Retrieval ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "policies"
    # Chunk budget, in TOKENS (not characters) when chunk_unit is "tokens".
    #
    # Tokens are the unit the embedding model's limit is measured in. A
    # character budget cannot be made safe: the characters-per-token ratio
    # swings from ~3.7 on policy prose to ~2.5 on passages dense with
    # identifiers and addresses, so any character size that fits one document
    # overflows on another and the excess is silently discarded before
    # embedding. See the chunker module docstring.
    #
    # 220 leaves headroom under MiniLM's 254 usable tokens (256 minus
    # [CLS]/[SEP]) and is roughly one or two policy clauses - big enough to
    # carry an exclusion together with what it modifies, small enough that a
    # retrieved chunk is mostly signal. Raise EMBEDDING_MODEL's limit first if
    # you want larger chunks; the pipeline clamps to what the model can read.
    chunk_size: int = 220
    chunk_overlap: int = 40
    # "tokens" (recommended) or "characters" (the older, unsafe behaviour,
    # retained so you can measure the difference with scripts/evaluate.py).
    chunk_unit: str = "tokens"
    top_k: int = 5
    # Cosine-similarity floor below which a hit is treated as irrelevant.
    #
    # TUNE THIS against your own documents - it is the single most consequential
    # number here, and it is NOT portable between embedding models. Run
    # scripts/evaluate.py: it prints the lowest similarity among correct hits,
    # and this value must sit comfortably below that or you will refuse
    # questions the index can actually answer.
    #
    # 0.10 is a deliberately permissive starting point. On the sample policy a
    # correct chunk scored as low as 0.153, while a completely unrelated
    # question ("what is the capital of France?") scored -0.07. A floor of 0.15
    # would have cleared that correct hit by 0.003 - a margin that is noise, not
    # a decision. 0.10 keeps real answers safe and still rejects nonsense.
    # Raise it if you see irrelevant chunks reaching the model.
    min_similarity: float = 0.10

    # --- Generation ---
    # "anthropic" (Claude) or "gemini" (Google, via its OpenAI-compatible
    # endpoint). Retrieval is identical either way and runs locally, so this
    # only changes who writes the final answer - which makes it easy to compare
    # two models against the same index and the same prompt.
    llm_provider: str = "gemini"

    # Model id. Must match the provider:
    #   anthropic -> claude-opus-5, claude-sonnet-5, claude-haiku-4-5, ...
    #   gemini    -> gemini-2.5-flash, gemini-flash-latest, ...
    # A "models/" prefix is accepted and normalized away.
    #
    # Verified against the live free-tier endpoint: the Pro models
    # (gemini-3.1-pro-preview, gemini-pro-latest) return 429 with zero quota,
    # and gemini-2.5-pro returns 404 "no longer available to new users" while
    # still appearing in the model list. Flash is what works without billing.
    llm_model: str = "gemini-2.5-flash"

    # Base URL for the OpenAI-compatible provider. Ignored for anthropic.
    # Also works for a local runtime (Ollama, vLLM) if you point it there.
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    # Keys. Only the one for the active provider is needed. Both are read from
    # .env, which is gitignored - never commit a key, and rotate any key that
    # has been pasted into a chat, an issue, or a screenshot.
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # Anthropic-only cost/thoroughness dial: low | medium | high | xhigh | max.
    # Has no effect on Gemini - the OpenAI-compatible layer exposes no
    # equivalent, and Gemini 2.5's thinking budget is not configurable through
    # it either.
    llm_effort: str = "high"
    # A ceiling, not a target - you are billed for tokens produced, not for this.
    # Generous on purpose: on BOTH providers, reasoning tokens are drawn from
    # this same allowance, and a limit too small shows up as an empty or
    # truncated answer rather than an error.
    llm_max_tokens: int = 16000
    # Anthropic-only: on a policy refusal, retry server-side on a fallback model.
    llm_refusal_fallback: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
