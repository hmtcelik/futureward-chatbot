"""Naive (Chatbase-style) chatbot baseline.

Single Gemini call, ``SYSTEM_PROMPT_NAIVE``, no retrieval, no guards. Exists
purely to demonstrate what an out-of-the-box LLM deployment looks like and
why the guarded pipeline is worth building.
"""

from __future__ import annotations

import time
import uuid

import structlog

from src.guards.prompts import SYSTEM_PROMPT_NAIVE
from src.llm.gemini_client import GeminiClient, build_contents
from src.logger import get_logger
from src.models import ChatResponse

logger = get_logger(__name__)


class NaiveChatbot:
    """No retrieval. No guards. Just a thin LLM call."""

    variant = "naive"

    def __init__(self, client: GeminiClient | None = None):
        self.client = client or GeminiClient()

    async def chat(self, query: str, history: list[dict] | None = None) -> ChatResponse:
        """Send query (and optional history) to Gemini, return ChatResponse."""
        cid = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(correlation_id=cid, variant=self.variant)
        log = logger.bind(correlation_id=cid, variant=self.variant)
        started = time.perf_counter()
        log.info("chat_request", query_length=len(query), history_turns=len(history or []))

        try:
            contents = build_contents(history or [], query)
            result = await self.client.generate(
                contents, system_instruction=SYSTEM_PROMPT_NAIVE
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            log.info(
                "chat_response",
                total_latency_ms=latency_ms,
                total_tokens=result.usage.total_tokens,
                total_cost=round(result.usage.estimated_cost_usd, 6),
                final_decision_path="naive_direct",
            )
            return ChatResponse(
                answer=result.text,
                retrieved_chunks=[],
                input_guard=None,
                output_guard=None,
                token_usage=result.usage,
                latency_ms=latency_ms,
                correlation_id=cid,
                chatbot_variant=self.variant,
            )
        finally:
            structlog.contextvars.clear_contextvars()
