"""Output guard: judge whether an answer is grounded in retrieved context.

Fails OPEN: a parse error or judge call failure returns PASS so a single
flaky judge call doesn't block a reasonable answer. The trade-off is
accepted because the input guard already fails closed; if the answer is
visibly wrong the eval suite catches it offline rather than blocking the
user mid-conversation.
"""

from __future__ import annotations

import time

from src.config import settings
from src.guards.prompts import OUTPUT_GUARD_PROMPT
from src.llm.gemini_client import GeminiClient
from src.logger import get_logger
from src.models import GuardDecision, GuardDecisionType, RetrievedChunk

logger = get_logger(__name__)

_VALID_DECISIONS = {"pass", "refuse_not_grounded"}


async def check_grounded(
    answer: str,
    chunks: list[RetrievedChunk],
    client: GeminiClient | None = None,
) -> GuardDecision:
    """Decide whether ``answer`` is supported by the retrieved chunks.

    Args:
        answer: model-generated answer text.
        chunks: chunks that were passed to the answer model as context.
        client: optional GeminiClient (testability / reuse).

    Returns:
        GuardDecision with ``decision`` ∈ {PASS, REFUSE_NOT_GROUNDED}.
        Fails open (PASS) on any error.
    """
    client = client or GeminiClient()
    context = _format_context(chunks)
    prompt = OUTPUT_GUARD_PROMPT.format(context=context, answer=answer)
    started = time.perf_counter()

    try:
        result = await client.generate_json(prompt, model=settings.judge_model)
    except Exception as exc:  # noqa: BLE001 - guard must never raise
        logger.warning("output_guard_call_failed", error=str(exc))
        return _fail_open("guard call failed", started)

    if not result.parse_ok:
        return _fail_open("malformed JSON", started)

    raw_decision = str(result.data.get("decision", "")).lower().strip()
    reason = str(result.data.get("reason", "") or "")
    try:
        confidence = float(result.data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if raw_decision not in _VALID_DECISIONS:
        return _fail_open(f"invalid decision: {raw_decision!r}", started)

    decision_type = (
        GuardDecisionType.PASS
        if raw_decision == "pass"
        else GuardDecisionType.REFUSE_NOT_GROUNDED
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return GuardDecision(
        decision=decision_type,
        reason=reason or raw_decision,
        confidence=confidence,
        judge_model=settings.judge_model,
        latency_ms=elapsed_ms,
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context provided)"
    blocks: list[str] = []
    for rc in chunks:
        blocks.append(
            f"[{rc.chunk.document_title}] (score={rc.similarity_score:.2f})\n{rc.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _fail_open(reason: str, started: float) -> GuardDecision:
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return GuardDecision(
        decision=GuardDecisionType.PASS,
        reason=f"fail-open: {reason}",
        confidence=0.0,
        judge_model=settings.judge_model,
        latency_ms=elapsed_ms,
    )
