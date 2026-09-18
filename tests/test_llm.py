"""Tests for the provider seam.

No network calls: each provider's SDK client is stubbed, so these verify the
things that actually break when swapping providers - model-id handling, where
the system prompt goes, how usage is read back, and whether failures arrive as
the RuntimeError the API layer knows how to turn into a 502.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.llm import AnthropicLLM, LLMResult, OpenAICompatibleLLM, build_llm

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ------------------------------------------------------------------- fake client


def fake_openai_client(text="Answer [1].", prompt_tokens=812, completion_tokens=44,
                       finish_reason="stop", captured=None):
    """Minimal stand-in for openai.OpenAI, recording the request."""

    def create(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# ------------------------------------------------------------ model id handling


@pytest.mark.parametrize(
    "given,expected",
    [
        ("models/gemini-2.5-flash", "gemini-2.5-flash"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
        ("models/gemini-flash-latest", "gemini-flash-latest"),
    ],
)
def test_models_prefix_is_stripped(given, expected) -> None:
    """A "models/" prefix is normalized away.

    Gemini's ListModels returns ids in that form, so it is what you get by
    copying one out. The endpoint accepts either form, but normalizing keeps
    `model` - and the API response's `model` field - consistent.
    """
    llm = OpenAICompatibleLLM(model=given, api_key="k", base_url=GEMINI_BASE)
    assert llm.model == expected


# -------------------------------------------------------------- request shaping


def test_system_prompt_is_sent_as_a_message() -> None:
    """The key structural difference from Anthropic.

    Anthropic takes `system` as a top-level parameter; the OpenAI shape needs it
    as the first message. Getting this wrong silently drops the grounding rules
    and the model starts answering from general knowledge.
    """
    captured: dict = {}
    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE,
        client=fake_openai_client(captured=captured),
    )
    llm.complete("GROUNDING RULES", "the question")

    assert captured["messages"][0] == {"role": "system", "content": "GROUNDING RULES"}
    assert captured["messages"][1] == {"role": "user", "content": "the question"}
    assert captured["model"] == "gemini-2.5-flash"


def test_temperature_defaults_to_zero() -> None:
    """Grounded extraction wants the clause as written, not a paraphrase."""
    captured: dict = {}
    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE,
        client=fake_openai_client(captured=captured),
    )
    llm.complete("s", "u")
    assert captured["temperature"] == 0.0


def test_usage_is_mapped_to_the_common_shape() -> None:
    """OpenAI calls them prompt/completion; our Answer reports input/output."""
    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE,
        client=fake_openai_client(prompt_tokens=1234, completion_tokens=56),
    )
    result = llm.complete("s", "u")
    assert isinstance(result, LLMResult)
    assert (result.input_tokens, result.output_tokens) == (1234, 56)
    assert result.declined is False


def test_content_filter_is_reported_as_declined() -> None:
    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE,
        client=fake_openai_client(text="", finish_reason="content_filter"),
    )
    assert llm.complete("s", "u").declined is True


def test_empty_answer_from_exhausted_budget_raises_clearly() -> None:
    """Gemini 2.5 spends reasoning tokens from the same max_tokens allowance.

    If it runs out before writing anything, the API returns HTTP 200 with an
    empty string - a genuinely baffling result unless the cause is named.
    """
    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE, max_tokens=32,
        client=fake_openai_client(text="", finish_reason="length"),
    )
    with pytest.raises(RuntimeError, match="consumed before any visible output"):
        llm.complete("s", "u")


def test_missing_key_raises_before_any_request() -> None:
    llm = OpenAICompatibleLLM(model="gemini-2.5-flash", api_key=None, base_url=GEMINI_BASE)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.complete("s", "u")


# --------------------------------------------------------- error translation


def test_provider_errors_become_runtime_errors() -> None:
    """The API layer catches exactly RuntimeError, so SDK errors must convert.

    Anything that escapes as its own exception type becomes a 500 instead of an
    actionable 502.
    """
    import openai

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise openai.AuthenticationError(
                        "bad key",
                        response=SimpleNamespace(status_code=401, headers={}, request=None),
                        body=None,
                    )

    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE, client=Boom()
    )
    with pytest.raises(RuntimeError, match="rejected"):
        llm.complete("s", "u")


# ------------------------------------------------------------------- streaming


def test_streaming_yields_only_non_empty_deltas() -> None:
    """Real streams include role-only and empty deltas that must not be yielded."""
    events = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="The excess "))]),
        SimpleNamespace(choices=[]),  # keep-alive with no choices at all
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="is 5,000 [1]."))]),
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(events)))
    )
    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE, client=client
    )
    assert "".join(llm.stream("s", "u")) == "The excess is 5,000 [1]."


# --------------------------------------------------------------------- factory


def _settings(**overrides):
    base = dict(
        llm_provider="gemini",
        llm_model="models/gemini-2.5-flash",
        llm_base_url=GEMINI_BASE,
        anthropic_api_key=None,
        gemini_api_key="k",
        llm_effort="high",
        llm_max_tokens=16000,
        llm_refusal_fallback=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_factory_builds_gemini() -> None:
    llm = build_llm(_settings())
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == "gemini-2.5-flash"


def test_factory_builds_anthropic() -> None:
    llm = build_llm(_settings(llm_provider="anthropic", llm_model="claude-opus-5"))
    assert isinstance(llm, AnthropicLLM)
    assert llm.model == "claude-opus-5"


def test_factory_is_case_and_whitespace_tolerant() -> None:
    assert isinstance(build_llm(_settings(llm_provider="  Gemini ")), OpenAICompatibleLLM)


def test_factory_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm(_settings(llm_provider="llama"))


# -------------------------------------------------- engine is provider-agnostic


def test_engine_reports_which_model_answered(tmp_path) -> None:
    """Switching providers must not require touching RAGEngine."""
    from app.services.embedder import Embedder
    from app.services.pipeline import IngestionPipeline
    from app.services.rag_engine import RAGEngine
    from app.services.vector_db import VectorStore
    from tests.pdf_fixture import SAMPLE_POLICY_PAGES, write_pdf

    embedder = Embedder()
    store = VectorStore(tmp_path / "chroma", collection_name="prov")
    pdf = write_pdf(tmp_path / "p.pdf", SAMPLE_POLICY_PAGES)
    IngestionPipeline(store, embedder, 220, 40).ingest(pdf, "p", "p.pdf")

    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE,
        client=fake_openai_client(text="The excess is 5,000 [1]."),
    )
    answer = RAGEngine(store, embedder, llm, min_similarity=0.10).answer("what is the excess?")

    assert answer.text == "The excess is 5,000 [1]."
    assert answer.model == "gemini-2.5-flash"
    assert answer.grounded is True
    assert answer.sources


def test_no_model_call_when_retrieval_finds_nothing(tmp_path) -> None:
    """The cost guard must hold for every provider, not just Anthropic."""
    from app.services.embedder import Embedder
    from app.services.pipeline import IngestionPipeline
    from app.services.rag_engine import RAGEngine
    from app.services.vector_db import VectorStore
    from tests.pdf_fixture import SAMPLE_POLICY_PAGES, write_pdf

    embedder = Embedder()
    store = VectorStore(tmp_path / "chroma2", collection_name="prov2")
    pdf = write_pdf(tmp_path / "p.pdf", SAMPLE_POLICY_PAGES)
    IngestionPipeline(store, embedder, 220, 40).ingest(pdf, "p", "p.pdf")

    class Exploding:
        model = "gemini-2.5-flash"

        def complete(self, *a, **k):
            raise AssertionError("the model must not be called with no context")

        def stream(self, *a, **k):
            raise AssertionError("the model must not be called with no context")

    answer = RAGEngine(store, embedder, Exploding(), min_similarity=0.10).answer(
        "what is the capital of France?"
    )
    assert answer.grounded is False
    assert answer.sources == []


# ------------------------------------------------ Anthropic credential handling


def test_anthropic_missing_credentials_becomes_actionable() -> None:
    """The SDK resolves credentials at REQUEST time, signalling with TypeError.

    Left unhandled that escapes as a 500. It must become the RuntimeError the
    API layer renders as a 502 naming the variable to set.
    """

    class NoCredentials:
        class beta:  # noqa: N801 - mirrors the SDK attribute path
            class messages:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise TypeError(
                        "Could not resolve authentication method. Expected one of "
                        "api_key, auth_token, or credentials to be set."
                    )

    llm = AnthropicLLM(client=NoCredentials())
    with pytest.raises(RuntimeError, match="No Anthropic credentials found"):
        llm.complete("s", "u")


def test_anthropic_unrelated_type_errors_are_not_swallowed() -> None:
    """Blanket-catching TypeError would hide real bugs behind "set your key"."""

    class Buggy:
        class beta:  # noqa: N801
            class messages:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise TypeError("create() got an unexpected keyword argument 'foo'")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AnthropicLLM(client=Buggy()).complete("s", "u")


# ----------------------------------------------------------- citation validity


@pytest.mark.parametrize(
    "text,count,expected",
    [
        ("The excess is 5,000 [1].", 4, []),
        ("Grace is 15 days [3, 4].", 4, []),
        # Observed live: gemini-2.5-flash echoed the policy's own clause numbers.
        ("Not applicable for Single Premium [5, 14].", 4, [5, 14]),
        ("Mixed [2] and bogus [9].", 4, [9]),
        ("Zero is not a source [0].", 4, [0]),
        ("No citations at all.", 4, []),
        # Bracketed non-numbers must not be treated as citations.
        ("A range [see Part D] and [1].", 2, []),
    ],
)
def test_out_of_range_citations_are_detected(text, count, expected) -> None:
    """A citation pointing at no source is worse than no citation.

    It looks verifiable and is not. The prompt forbids it, but a prompt is a
    request rather than a guarantee, so the output is checked.
    """
    from app.services.rag_engine import RAGEngine

    assert RAGEngine._invalid_citations(text, count) == expected


def test_answer_reports_invalid_citations(tmp_path) -> None:
    from app.services.embedder import Embedder
    from app.services.pipeline import IngestionPipeline
    from app.services.rag_engine import RAGEngine
    from app.services.vector_db import VectorStore
    from tests.pdf_fixture import SAMPLE_POLICY_PAGES, write_pdf

    embedder = Embedder()
    store = VectorStore(tmp_path / "cite", collection_name="cite")
    pdf = write_pdf(tmp_path / "p.pdf", SAMPLE_POLICY_PAGES)
    IngestionPipeline(store, embedder, 220, 40).ingest(pdf, "p", "p.pdf")

    llm = OpenAICompatibleLLM(
        model="gemini-2.5-flash", api_key="k", base_url=GEMINI_BASE,
        client=fake_openai_client(text="The excess is 5,000 [7]."),
    )
    answer = RAGEngine(store, embedder, llm, min_similarity=0.10).answer("excess?", top_k=2)
    assert answer.invalid_citations == [7]
