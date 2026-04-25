"""Input guard: classify a user query as on-topic or off-topic.

The judge model receives the query and the topic specification. The guard
fails closed: any parse failure, missing decision, or judge error results
in ``REFUSE_OFF_TOPIC`` so a malicious or malformed query never slips through.
"""

from __future__ import annotations

import time

from src.config import settings
from src.guards.prompts import INPUT_GUARD_PROMPT
from src.llm.gemini_client import GeminiClient
from src.logger import get_logger
from src.models import GuardDecision, GuardDecisionType

logger = get_logger(__name__)

_VALID_DECISIONS = {"pass", "refuse_off_topic"}


async def check_on_topic(
    query: str, client: GeminiClient | None = None
) -> GuardDecision:
    """Decide whether a query is in-scope for Talent Taiwan.

    Args:
        query: raw user input.
        client: optional GeminiClient (for testability / reuse).

    Returns:
        GuardDecision with ``decision`` ∈ {PASS, REFUSE_OFF_TOPIC}.
        Fails closed (REFUSE_OFF_TOPIC) on any error.
    """
    client = client or GeminiClient()
    prompt = INPUT_GUARD_PROMPT.format(query=query)
    started = time.perf_counter()

    try:
        result = await client.generate_json(prompt, model=settings.judge_model)
    except Exception as exc:  # noqa: BLE001 - guard must never raise
        logger.warning("input_guard_call_failed", error=str(exc))
        return _fail_closed("guard call failed", started)

    if not result.parse_ok:
        return _fail_closed("malformed JSON", started)

    raw_decision = str(result.data.get("decision", "")).lower().strip()
    reason = str(result.data.get("reason", "") or "")
    try:
        confidence = float(result.data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if raw_decision not in _VALID_DECISIONS:
        return _fail_closed(f"invalid decision: {raw_decision!r}", started)

    decision_type = (
        GuardDecisionType.PASS
        if raw_decision == "pass"
        else GuardDecisionType.REFUSE_OFF_TOPIC
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return GuardDecision(
        decision=decision_type,
        reason=reason or raw_decision,
        confidence=confidence,
        judge_model=settings.judge_model,
        latency_ms=elapsed_ms,
    )


def _fail_closed(reason: str, started: float) -> GuardDecision:
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return GuardDecision(
        decision=GuardDecisionType.REFUSE_OFF_TOPIC,
        reason=f"fail-closed: {reason}",
        confidence=0.0,
        judge_model=settings.judge_model,
        latency_ms=elapsed_ms,
    )
