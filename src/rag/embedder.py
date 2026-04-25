"""Batched embedding wrapper around ``GeminiClient.embed``.

The client method handles a single API call and returns vectors + TokenUsage.
This module fans large input lists out into ``batch_size`` sub-calls and sums
the per-call usage into one aggregate.
"""

from __future__ import annotations

from src.config import settings
from src.llm.gemini_client import GeminiClient
from src.logger import get_logger
from src.models import TokenUsage

logger = get_logger(__name__)


class Embedder:
    """Batched + retry-aware Gemini embedder."""

    def __init__(
        self,
        client: GeminiClient | None = None,
        model: str | None = None,
        batch_size: int = 100,
    ):
        self.client = client or GeminiClient()
        self.model = model or settings.embedding_model
        self.batch_size = batch_size

    async def embed(
        self, texts: list[str], task_type: str | None = None
    ) -> tuple[list[list[float]], TokenUsage]:
        """Embed ``texts``, batching internally. Returns vectors + summed usage."""
        if not texts:
            empty = TokenUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated_cost_usd=0.0
            )
            return [], empty

        all_vectors: list[list[float]] = []
        total_prompt = 0
        total_cost = 0.0
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors, usage = await self.client.embed(
                batch, task_type=task_type, model=self.model
            )
            all_vectors.extend(vectors)
            total_prompt += usage.prompt_tokens
            total_cost += usage.estimated_cost_usd
            logger.debug(
                "embed_batch", offset=start, size=len(batch), total_so_far=len(all_vectors)
            )

        summary = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=0,
            total_tokens=total_prompt,
            estimated_cost_usd=total_cost,
        )
        return all_vectors, summary
