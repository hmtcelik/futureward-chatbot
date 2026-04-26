"""Guarded RAG chatbot pipeline.

Stages, in order:
    1. ``input_guard`` (LLM judge) AND query embedding — run in parallel.
    2. Vector store query for top-k chunks.
    3. Generation with ``SYSTEM_PROMPT_GUARDED`` + retrieved context.
    4. ``output_guard`` (faithfulness judge) on the answer.

Refusal short-circuits:
    - Input guard refuses     → return REFUSAL_RESPONSE (chunks discarded).
    - Top similarity < threshold → return LOW_CONFIDENCE_RESPONSE.
    - Output guard refuses    → return NOT_GROUNDED_RESPONSE; the original
      answer is logged at WARNING level for manual review.

Telemetry: ``correlation_id`` is bound via structlog contextvars at the start
of every chat call so any log line emitted during the request — including
those from the guards, embedder, retriever — carries the same trace id.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import structlog

from src.config import settings
from src.guards.input_guard import check_on_topic
from src.guards.output_guard import check_grounded
from src.guards.prompts import (
    LOW_CONFIDENCE_RESPONSE,
    NOT_GROUNDED_RESPONSE,
    REFUSAL_RESPONSE,
    SYSTEM_PROMPT_GUARDED,
)
from src.llm.gemini_client import GeminiClient, build_contents
from src.logger import get_logger
from src.models import (
    ChatResponse,
    Chunk,
    GuardDecision,
    GuardDecisionType,
    RetrievedChunk,
    StageTimings,
    TokenUsage,
)
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever

logger = get_logger(__name__)


def _zero_usage() -> TokenUsage:
    return TokenUsage(
        prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated_cost_usd=0.0
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as a labeled block ready for the answer model."""
    if not chunks:
        return "<context>(no relevant context retrieved)</context>"
    parts: list[str] = []
    for rc in chunks:
        parts.append(
            f"[Source: {rc.chunk.document_title}] (URL: {rc.chunk.document_url})\n"
            f"{rc.chunk.text}"
        )
    body = "\n\n---\n\n".join(parts)
    return f"<context>\n{body}\n</context>"


class GuardedChatbot:
    """Input guard → retrieve → guarded generate → output guard."""

    variant = "guarded"

    def __init__(
        self,
        client: GeminiClient | None = None,
        retriever: Retriever | None = None,
    ):
        self.client = client or GeminiClient()
        if retriever is None:
            embedder = Embedder(client=self.client)
            retriever = Retriever(embedder=embedder)
        self.retriever = retriever

    async def chat(self, query: str, history: list[dict] | None = None) -> ChatResponse:
        """Run the full guarded pipeline. Always returns a ChatResponse."""
        cid = str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(correlation_id=cid, variant=self.variant)
        log = logger.bind(correlation_id=cid, variant=self.variant)
        started = time.perf_counter()
        usage_baseline = self.client.cumulative_usage
        log.info("chat_request", query_length=len(query), history_turns=len(history or []))

        try:
            return await self._chat_inner(
                query, history or [], cid, log, started, usage_baseline
            )
        finally:
            structlog.contextvars.clear_contextvars()

    async def _chat_inner(
        self,
        query: str,
        history: list[dict],
        cid: str,
        log,
        started: float,
        usage_baseline: TokenUsage,
    ) -> ChatResponse:
        # --- Stage 1+2: input guard + embedding in parallel --------------
        embed_started = time.perf_counter()
        input_task = asyncio.create_task(check_on_topic(query, client=self.client))
        embed_task = asyncio.create_task(
            self.retriever.embedder.embed([query], task_type="RETRIEVAL_QUERY")
        )
        input_decision, embed_result = await asyncio.gather(input_task, embed_task)
        embedding_ms = int((time.perf_counter() - embed_started) * 1000)
        query_vectors, embed_usage = embed_result
        embedding_dimensions = len(query_vectors[0]) if query_vectors else 0
        embedding_preview = (
            [round(float(v), 4) for v in query_vectors[0][:8]] if query_vectors else []
        )

        log.info(
            "input_guard_decision",
            decision=input_decision.decision.value,
            confidence=input_decision.confidence,
            latency_ms=input_decision.latency_ms,
        )

        if input_decision.decision == GuardDecisionType.REFUSE_OFF_TOPIC:
            return self._respond_refusal(
                cid=cid,
                started=started,
                input_decision=input_decision,
                usage_baseline=usage_baseline,
                stage_timings=StageTimings(
                    input_guard_ms=input_decision.latency_ms,
                    embedding_ms=embedding_ms,
                ),
                embedding_dimensions=embedding_dimensions,
                embedding_input_tokens=embed_usage.prompt_tokens,
                embedding_preview=embedding_preview,
                log=log,
            )

        # --- Stage 3: vector store query --------------------------------
        retrieval_started = time.perf_counter()
        if not query_vectors or self.retriever.store.count() == 0:
            log.warning("retrieval_no_vectors_or_empty_store")
            chunks: list[RetrievedChunk] = []
        else:
            raw_hits = self.retriever.store.query(query_vectors[0], top_k=settings.top_k)
            chunks = []
            for rank, hit in enumerate(raw_hits):
                similarity = max(0.0, min(1.0, 1.0 - hit["distance"]))
                meta = hit["metadata"] or {}
                chunks.append(
                    RetrievedChunk(
                        chunk=Chunk(
                            chunk_id=hit["id"],
                            document_url=meta.get("document_url", ""),
                            document_title=meta.get("document_title", ""),
                            text=hit["document"] or "",
                            chunk_hash=meta.get("chunk_hash", ""),
                            position=int(meta.get("position", 0)),
                            metadata={"tokens": int(meta.get("tokens", 0))},
                        ),
                        similarity_score=similarity,
                        rank=rank,
                    )
                )
        retrieval_ms = int((time.perf_counter() - retrieval_started) * 1000)

        top_score = chunks[0].similarity_score if chunks else 0.0
        log.info("retrieval_complete", results=len(chunks), top_score=top_score)

        if not chunks or top_score < settings.similarity_threshold:
            return self._respond_low_confidence(
                cid=cid,
                started=started,
                input_decision=input_decision,
                top_score=top_score,
                usage_baseline=usage_baseline,
                stage_timings=StageTimings(
                    input_guard_ms=input_decision.latency_ms,
                    embedding_ms=embedding_ms,
                    retrieval_ms=retrieval_ms,
                ),
                embedding_dimensions=embedding_dimensions,
                embedding_input_tokens=embed_usage.prompt_tokens,
                embedding_preview=embedding_preview,
                log=log,
            )

        # --- Stage 4: generation ----------------------------------------
        contents = build_contents(history, query)
        context_block = _format_context(chunks)
        gen_prompt_text = f"{context_block}\n\nUser question: {query}"
        contents[-1].parts[0].text = gen_prompt_text

        gen_started = time.perf_counter()
        accumulated_text = ""
        gen_usage: TokenUsage | None = None
        async for chunk_evt in self.client.generate_stream(
            contents, system_instruction=SYSTEM_PROMPT_GUARDED
        ):
            if chunk_evt.text:
                accumulated_text += chunk_evt.text
            if chunk_evt.is_final and chunk_evt.usage is not None:
                gen_usage = chunk_evt.usage
        generation_ms = int((time.perf_counter() - gen_started) * 1000)
        if gen_usage is None:
            gen_usage = _zero_usage()
        log.info(
            "generation_complete",
            latency_ms=generation_ms,
            prompt_tokens=gen_usage.prompt_tokens,
            completion_tokens=gen_usage.completion_tokens,
            answer_chars=len(accumulated_text),
        )

        # --- Stage 5: output guard --------------------------------------
        output_decision = await check_grounded(
            accumulated_text, chunks, client=self.client
        )
        log.info(
            "output_guard_decision",
            decision=output_decision.decision.value,
            confidence=output_decision.confidence,
            latency_ms=output_decision.latency_ms,
        )

        final_answer = accumulated_text
        suppressed_answer: str | None = None
        decision_path = "answer"
        if output_decision.decision == GuardDecisionType.REFUSE_NOT_GROUNDED:
            log.warning(
                "ungrounded_answer_replaced",
                original_answer=accumulated_text[:500],
                reason=output_decision.reason,
            )
            suppressed_answer = accumulated_text
            final_answer = NOT_GROUNDED_RESPONSE
            decision_path = "not_grounded"

        latency_ms = int((time.perf_counter() - started) * 1000)
        total_usage = _diff_usage(usage_baseline, self.client.cumulative_usage)
        log.info(
            "chat_response",
            total_latency_ms=latency_ms,
            total_tokens=total_usage.total_tokens,
            total_cost=round(total_usage.estimated_cost_usd, 6),
            final_decision_path=decision_path,
        )
        return ChatResponse(
            answer=final_answer,
            retrieved_chunks=chunks,
            input_guard=input_decision,
            output_guard=output_decision,
            token_usage=total_usage,
            latency_ms=latency_ms,
            correlation_id=cid,
            chatbot_variant=self.variant,
            stage_timings=StageTimings(
                input_guard_ms=input_decision.latency_ms,
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                output_guard_ms=output_decision.latency_ms,
            ),
            embedding_dimensions=embedding_dimensions,
            embedding_input_tokens=embed_usage.prompt_tokens,
            embedding_preview=embedding_preview,
            similarity_threshold=settings.similarity_threshold,
            system_prompt_used=SYSTEM_PROMPT_GUARDED,
            suppressed_answer=suppressed_answer,
        )

    def _respond_refusal(
        self,
        cid: str,
        started: float,
        input_decision: GuardDecision,
        usage_baseline: TokenUsage,
        stage_timings: StageTimings,
        embedding_dimensions: int,
        embedding_input_tokens: int,
        embedding_preview: list[float],
        log,
    ) -> ChatResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        total_usage = _diff_usage(usage_baseline, self.client.cumulative_usage)
        log.info(
            "chat_response",
            total_latency_ms=latency_ms,
            total_tokens=total_usage.total_tokens,
            total_cost=round(total_usage.estimated_cost_usd, 6),
            final_decision_path="refused_off_topic",
        )
        return ChatResponse(
            answer=REFUSAL_RESPONSE,
            retrieved_chunks=[],
            input_guard=input_decision,
            output_guard=None,
            token_usage=total_usage,
            latency_ms=latency_ms,
            correlation_id=cid,
            chatbot_variant=self.variant,
            stage_timings=stage_timings,
            embedding_dimensions=embedding_dimensions,
            embedding_input_tokens=embedding_input_tokens,
            embedding_preview=embedding_preview,
            similarity_threshold=settings.similarity_threshold,
        )

    def _respond_low_confidence(
        self,
        cid: str,
        started: float,
        input_decision: GuardDecision,
        top_score: float,
        usage_baseline: TokenUsage,
        stage_timings: StageTimings,
        embedding_dimensions: int,
        embedding_input_tokens: int,
        embedding_preview: list[float],
        log,
    ) -> ChatResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        # Synthetic guard decision so the UI can surface "we didn't have
        # confident coverage" without a separate field.
        synthetic = GuardDecision(
            decision=GuardDecisionType.REFUSE_LOW_CONFIDENCE,
            reason=(
                f"top similarity {top_score:.3f} below threshold "
                f"{settings.similarity_threshold:.2f}"
            ),
            confidence=1.0 - top_score,
            judge_model="threshold:" + str(settings.similarity_threshold),
            latency_ms=0,
        )
        total_usage = _diff_usage(usage_baseline, self.client.cumulative_usage)
        log.info(
            "chat_response",
            total_latency_ms=latency_ms,
            total_tokens=total_usage.total_tokens,
            total_cost=round(total_usage.estimated_cost_usd, 6),
            final_decision_path="low_confidence",
            top_score=top_score,
        )
        return ChatResponse(
            answer=LOW_CONFIDENCE_RESPONSE,
            retrieved_chunks=[],
            input_guard=synthetic,
            output_guard=None,
            token_usage=total_usage,
            latency_ms=latency_ms,
            correlation_id=cid,
            chatbot_variant=self.variant,
            stage_timings=stage_timings,
            embedding_dimensions=embedding_dimensions,
            embedding_input_tokens=embedding_input_tokens,
            embedding_preview=embedding_preview,
            similarity_threshold=settings.similarity_threshold,
        )


def _diff_usage(before: TokenUsage, after: TokenUsage) -> TokenUsage:
    """Return ``after - before``, clamped to non-negative."""
    p = max(0, after.prompt_tokens - before.prompt_tokens)
    c = max(0, after.completion_tokens - before.completion_tokens)
    cost = max(0.0, after.estimated_cost_usd - before.estimated_cost_usd)
    return TokenUsage(
        prompt_tokens=p, completion_tokens=c, total_tokens=p + c, estimated_cost_usd=cost
    )
