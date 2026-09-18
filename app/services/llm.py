"""The generation half of RAG, behind a provider-agnostic seam.

Retrieval and generation are independent concerns: everything upstream of this
module (parsing, chunking, embedding, search) is unchanged by which model writes
the final answer. Putting the provider behind one small interface keeps that
true, and has a concrete payoff for a learning project - you can point the same
index and the same prompt at two different models and compare the answers, which
is the only honest way to decide whether a model upgrade is worth paying for.

Two implementations:

  AnthropicLLM          - Claude, via the official `anthropic` SDK.
  OpenAICompatibleLLM   - anything speaking the OpenAI chat-completions API.
                          Google exposes Gemini through such an endpoint, so the
                          same class serves Gemini, and also local runtimes like
                          Ollama or vLLM if you point base_url at them.

Both raise RuntimeError with an actionable message on failure, so the API layer
has exactly one exception type to translate into a 502.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

# Beta flag enabling Anthropic's server-side refusal fallback.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    declined: bool = False  # the provider refused on policy grounds


class LLMClient(Protocol):
    """What RAGEngine needs from a language model, and nothing more.

    Deliberately narrow. Provider-specific concepts - Anthropic's adaptive
    thinking and effort levels, Gemini's safety settings - are configured inside
    each implementation rather than appearing in this signature, because a seam
    that leaks one provider's vocabulary is not a seam.
    """

    model: str

    def complete(self, system: str, user: str) -> LLMResult: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...


# --------------------------------------------------------------------- Anthropic


class AnthropicLLM:
    """Claude, via the official SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        effort: str = "high",
        max_tokens: int = 16000,
        refusal_fallback: bool = True,
        client=None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client = client
        self.effort = effort
        self.max_tokens = max_tokens
        self.refusal_fallback = refusal_fallback

    @property
    def client(self):
        if self._client is None:
            import anthropic

            # A bare Anthropic() resolves credentials from ANTHROPIC_API_KEY or
            # an `ant auth login` profile. Never hardcode a key.
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _kwargs(self, system: str, user: str) -> dict:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            # Policy questions often turn on interacting clauses (a sub-limit
            # inside a waiting period inside an exclusion), which is the kind of
            # multi-step reading adaptive thinking helps with.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if self.refusal_fallback:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def complete(self, system: str, user: str) -> LLMResult:
        import anthropic

        try:
            response = self.client.beta.messages.create(**self._kwargs(system, user))
        except TypeError as exc:
            # The SDK signals "no credential resolved" with a TypeError on the
            # request, not at construction. Left unhandled it becomes a 500.
            if "authentication" not in str(exc).lower():
                raise
            raise RuntimeError(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env, "
                "or set LLM_PROVIDER=gemini to use Gemini instead."
            ) from exc
        except anthropic.AuthenticationError:
            raise RuntimeError("Anthropic API key is invalid or revoked.") from None
        except anthropic.RateLimitError as exc:
            retry = exc.response.headers.get("retry-after", "60")
            raise RuntimeError(f"Anthropic rate limit hit; retry in {retry}s.") from exc
        except anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError("Could not reach the Anthropic API.") from exc

        # A refusal arrives as HTTP 200, so it must be checked explicitly.
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            logger.warning("Claude declined the request (category=%s)", category)
            return LLMResult(
                text="The assistant declined to answer this question.",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                declined=True,
            )

        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        return LLMResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self.client.beta.messages.stream(**self._kwargs(system, user)) as stream:
            yield from stream.text_stream


# ------------------------------------------------- OpenAI-compatible (Gemini &c)


class OpenAICompatibleLLM:
    """Any endpoint speaking the OpenAI chat-completions API.

    Used here for Gemini via Google's compatibility layer:

        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model    = "gemini-2.5-flash"   # Pro has no free-tier quota

    Note what this layer does NOT carry across. There is no equivalent of
    Anthropic's `effort` dial, and Gemini 2.5's own thinking budget is not
    exposed through the OpenAI shape - so `max_tokens` has to be generous,
    because reasoning tokens are consumed from the same allowance as the visible
    answer. A tight limit shows up as an empty or truncated response rather than
    an error, which is a confusing failure the first time you meet it.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        client=None,
    ) -> None:
        # Google's OpenAI layer accepts the model id with or without the
        # "models/" prefix that its own ListModels response returns (verified
        # against the live endpoint - both forms behave identically). Normalized
        # anyway so `model` reads back consistently, whichever form was
        # configured, and so logs and the API's `model` field agree.
        self.model = model.removeprefix("models/")
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self.max_tokens = max_tokens
        # 0.0 for grounded extraction: we want the clause as written, not a
        # creative paraphrase of it.
        self.temperature = temperature

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not self._api_key:
                raise RuntimeError(
                    "No API key for the OpenAI-compatible provider. Set "
                    "GEMINI_API_KEY in rag-insurance-service/.env"
                )
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _messages(self, system: str, user: str) -> list[dict]:
        # Unlike Anthropic, the system prompt is a message rather than a
        # top-level parameter.
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def complete(self, system: str, user: str) -> LLMResult:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, user),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except openai.AuthenticationError:
            raise RuntimeError(
                f"API key rejected by {self._base_url}. Check GEMINI_API_KEY in .env."
            ) from None
        except openai.RateLimitError as exc:
            # A 429 here often is not a transient rate limit. On Gemini's free
            # tier the Pro models carry ZERO quota, so every request to
            # gemini-3.1-pro-preview or gemini-pro-latest returns 429 no matter
            # how long you wait. Flash models do work.
            raise RuntimeError(
                f"Quota exceeded for '{self.model}'. On Gemini's free tier the "
                "Pro models have no quota at all - retrying will not help. Set "
                "LLM_MODEL=gemini-2.5-flash in .env, or enable billing. If you "
                "are already on a Flash model, this is the per-minute limit: "
                f"wait and retry. ({exc})"
            ) from exc
        except openai.NotFoundError as exc:
            # Usually not a typo. Google retires models while still listing them,
            # so a name copied from ListModels can 404 with "no longer available
            # to new users" - which is what happens with gemini-2.5-pro.
            raise RuntimeError(
                f"Model '{self.model}' is not available to this key: {exc}. "
                "Retired or tier-restricted models still appear in the model "
                "list, so check the message above rather than the spelling. "
                "gemini-2.5-flash works on the free tier."
            ) from exc
        except openai.APIStatusError as exc:
            raise RuntimeError(f"Provider error ({exc.status_code}): {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach {self._base_url}.") from exc

        choice = response.choices[0] if response.choices else None
        text = ((choice.message.content if choice else None) or "").strip()
        usage = response.usage

        # An empty body with a length finish_reason means the whole allowance
        # went on reasoning tokens. Silent otherwise - so name it.
        if not text and choice is not None and choice.finish_reason == "length":
            raise RuntimeError(
                f"{self.model} produced no answer: the {self.max_tokens}-token "
                "limit was consumed before any visible output. Raise "
                "LLM_MAX_TOKENS - reasoning tokens draw on the same allowance."
            )

        return LLMResult(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            # Gemini reports a blocked prompt as a "content_filter" finish.
            declined=bool(choice is not None and choice.finish_reason == "content_filter"),
        )

    def stream(self, system: str, user: str) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for event in stream:
            if not event.choices:
                continue
            piece = event.choices[0].delta.content
            if piece:
                yield piece


# ------------------------------------------------------------------------ factory


def build_llm(settings) -> LLMClient:
    """Construct the configured provider. Raises ValueError for an unknown one."""
    provider = (settings.llm_provider or "anthropic").strip().lower()

    if provider == "anthropic":
        return AnthropicLLM(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            effort=settings.llm_effort,
            max_tokens=settings.llm_max_tokens,
            refusal_fallback=settings.llm_refusal_fallback,
        )

    if provider in {"gemini", "openai_compatible"}:
        return OpenAICompatibleLLM(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
            base_url=settings.llm_base_url,
            max_tokens=settings.llm_max_tokens,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER {provider!r}. Use 'anthropic' or 'gemini'."
    )
