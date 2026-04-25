"""Async wrapper around the google-genai SDK.

Single canonical client used by every other module that talks to Gemini.

Capabilities:
    - ``generate``        — one-shot text generation (returns text + usage).
    - ``generate_stream`` — token-by-token streaming via async generator.
    - ``generate_json``   — JSON output with defensive parsing (used by guards).
    - ``embed``           — embedding vectors with TokenUsage.
    - ``cumulative_usage`` — total tokens / cost across this client's lifetime.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import get_logger
from src.models import TokenUsage

logger = get_logger(__name__)

_RETRYABLE = (genai_errors.APIError, genai_errors.ServerError, TimeoutError)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

# Sentinel for the streaming queue producer signalling end-of-stream.
_STREAM_DONE = object()

ContentLike = str | list[genai_types.Content]


@dataclass(frozen=True)
class GenerateResult:
    """One LLM call's text plus token accounting."""

    text: str
    usage: TokenUsage


@dataclass(frozen=True)
class GenerateJSONResult:
    """One LLM JSON call's parsed payload plus token accounting."""

    data: dict
    raw_text: str
    usage: TokenUsage
    parse_ok: bool


@dataclass(frozen=True)
class StreamChunk:
    """One incremental piece of streamed output."""

    text: str
    is_final: bool
    usage: TokenUsage | None  # populated only on the final chunk


class _CumulativeUsage:
    """Mutable running total of tokens + cost for one GeminiClient."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0

    def add(self, u: TokenUsage) -> None:
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.cost_usd += u.estimated_cost_usd

    def snapshot(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            estimated_cost_usd=self.cost_usd,
        )


class GeminiClient:
    """Single async client for the entire process."""

    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key or settings.gemini_api_key)
        self._cumulative = _CumulativeUsage()

    @property
    def cumulative_usage(self) -> TokenUsage:
        """Tokens + cost for every call this instance has made."""
        return self._cumulative.snapshot()

    # -- text -----------------------------------------------------------------

    async def generate(
        self,
        prompt: ContentLike,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> GenerateResult:
        """Generate plain text. ``prompt`` may be a string or a Content list.

        Args:
            prompt: user message or full conversation as Content list.
            model: Gemini model id (defaults to ``settings.llm_model``).
            system_instruction: optional system prompt.
            temperature: sampling temperature.
        """
        model = model or settings.llm_model
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        text, usage = await self._call_text(model, prompt, config)
        return GenerateResult(text=text, usage=usage)

    async def generate_stream(
        self,
        prompt: ContentLike,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[StreamChunk]:
        """Stream output token-by-token. Yields ``StreamChunk`` objects.

        The final chunk has ``is_final=True`` and a populated ``usage`` field;
        intermediate chunks carry only delta text.
        """
        model = model or settings.llm_model
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        queue: asyncio.Queue = asyncio.Queue()

        def _producer() -> None:
            try:
                final_meta = None
                for chunk in self.client.models.generate_content_stream(
                    model=model, contents=prompt, config=config
                ):
                    text = chunk.text or ""
                    if text:
                        queue.put_nowait(StreamChunk(text=text, is_final=False, usage=None))
                    if getattr(chunk, "usage_metadata", None):
                        final_meta = chunk.usage_metadata
                p = getattr(final_meta, "prompt_token_count", 0) or 0
                c = getattr(final_meta, "candidates_token_count", 0) or 0
                usage = TokenUsage(
                    prompt_tokens=p,
                    completion_tokens=c,
                    total_tokens=p + c,
                    estimated_cost_usd=_estimate_text_cost(p, c),
                )
                self._cumulative.add(usage)
                queue.put_nowait(StreamChunk(text="", is_final=True, usage=usage))
            except Exception as exc:  # noqa: BLE001 - propagate to consumer
                queue.put_nowait(exc)
            finally:
                queue.put_nowait(_STREAM_DONE)

        producer_task = asyncio.create_task(asyncio.to_thread(_producer))
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            await producer_task

    async def generate_json(
        self,
        prompt: str,
        model: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.0,
    ) -> GenerateJSONResult:
        """Generate JSON output with defensive parsing.

        Returns the parsed dict, the raw model output, token usage, and a
        ``parse_ok`` flag. Callers should treat ``parse_ok=False`` as a guard
        failure and fall back to a safe default decision.
        """
        model = model or settings.judge_model
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
        text, usage = await self._call_text(model, prompt, config)

        parsed: dict
        ok = True
        try:
            parsed = _parse_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("json_parse_failed", error=str(exc), raw=text[:300])
            parsed = {}
            ok = False

        return GenerateJSONResult(data=parsed, raw_text=text, usage=usage, parse_ok=ok)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _call_text(
        self,
        model: str,
        prompt: ContentLike,
        config: genai_types.GenerateContentConfig,
    ) -> tuple[str, TokenUsage]:
        def _do() -> tuple[str, TokenUsage]:
            resp = self.client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = resp.text or ""
            meta = resp.usage_metadata
            p = getattr(meta, "prompt_token_count", 0) or 0
            c = getattr(meta, "candidates_token_count", 0) or 0
            usage = TokenUsage(
                prompt_tokens=p,
                completion_tokens=c,
                total_tokens=p + c,
                estimated_cost_usd=_estimate_text_cost(p, c),
            )
            return text, usage

        text, usage = await asyncio.to_thread(_do)
        self._cumulative.add(usage)
        return text, usage

    # -- embedding -----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def embed(
        self,
        texts: list[str],
        task_type: str | None = None,
        model: str | None = None,
    ) -> tuple[list[list[float]], TokenUsage]:
        """Embed up to 100 strings in one call. For large lists, use Embedder."""
        if not texts:
            return [], TokenUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated_cost_usd=0.0
            )
        m = model or settings.embedding_model
        cfg = genai_types.EmbedContentConfig(task_type=task_type) if task_type else None

        def _do() -> list[list[float]]:
            resp = self.client.models.embed_content(model=m, contents=texts, config=cfg)
            return [list(e.values) for e in resp.embeddings]

        vectors = await asyncio.to_thread(_do)
        approx = sum(max(1, len(t) // 4) for t in texts)
        usage = TokenUsage(
            prompt_tokens=approx,
            completion_tokens=0,
            total_tokens=approx,
            estimated_cost_usd=(approx / 1_000_000) * settings.cost_per_1m_embedding_tokens_usd,
        )
        self._cumulative.add(usage)
        return vectors, usage


# -- helpers -----------------------------------------------------------------


def build_contents(history: list[dict], query: str) -> list[genai_types.Content]:
    """Convert ``[{"role": "user|assistant", "content": "..."}]`` to Gemini Contents."""
    out: list[genai_types.Content] = []
    for msg in history or []:
        role = "model" if msg.get("role") in ("assistant", "model") else "user"
        text = str(msg.get("content", ""))
        out.append(genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=text)]))
    out.append(
        genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=query)])
    )
    return out


def _parse_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_RE.sub("", cleaned).strip()
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
    return json.loads(cleaned)


def _estimate_text_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens / 1_000_000) * settings.cost_per_1m_input_tokens_usd
        + (completion_tokens / 1_000_000) * settings.cost_per_1m_output_tokens_usd
    )
