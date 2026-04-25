"""Live chatbot tests against real Gemini.

Slow (each guarded test makes 3+ API calls: input guard + generation +
output guard, plus an embedding). Total per-run cost is ~$0.005. Skip in
CI with ``SKIP_LIVE_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest
from structlog.testing import capture_logs

from src.guards.prompts import REFUSAL_RESPONSE
from src.llm.guarded_chatbot import GuardedChatbot
from src.llm.naive_chatbot import NaiveChatbot
from src.models import GuardDecisionType

_SKIP = os.getenv("SKIP_LIVE_TESTS") == "1"
_pytestmark_live = pytest.mark.skipif(_SKIP, reason="SKIP_LIVE_TESTS=1")

# Keep the boilerplate-marker comparisons case-insensitive.
_REFUSAL_MARKER = "outside what i can help with"


@_pytestmark_live
async def test_naive_answers_off_topic():
    """Naive baseline takes the bait — no built-in refusal logic."""
    n = NaiveChatbot()
    r = await n.chat("Should I invest in Bitcoin?")
    assert r.chatbot_variant == "naive"
    assert r.input_guard is None
    assert r.output_guard is None
    assert r.retrieved_chunks == []
    assert _REFUSAL_MARKER not in r.answer.lower(), (
        f"naive answered the off-topic prompt with our refusal language: {r.answer!r}"
    )
    print(
        f"\n[naive/off_topic] latency={r.latency_ms}ms tokens={r.token_usage.total_tokens} "
        f"answer_first_100={r.answer[:100]!r}"
    )


@_pytestmark_live
async def test_guarded_refuses_off_topic():
    """Guarded variant refuses Bitcoin, discards retrieved chunks."""
    g = GuardedChatbot()
    r = await g.chat("Should I invest in Bitcoin?")
    assert r.chatbot_variant == "guarded"
    assert r.input_guard is not None
    assert r.input_guard.decision == GuardDecisionType.REFUSE_OFF_TOPIC
    assert r.retrieved_chunks == [], (
        "off-topic refusal should discard retrieved chunks before returning"
    )
    assert r.answer.strip() == REFUSAL_RESPONSE.strip()
    print(
        f"\n[guarded/off_topic] latency={r.latency_ms}ms tokens={r.token_usage.total_tokens} "
        f"input_guard={r.input_guard.decision.value}"
    )


@_pytestmark_live
async def test_guarded_answers_with_citations():
    """Tax exemption query yields grounded answer with NT$3 + 50% + citation."""
    g = GuardedChatbot()
    r = await g.chat("What is the gold card tax exemption?")
    assert r.chatbot_variant == "guarded"
    assert r.input_guard is not None and r.input_guard.decision == GuardDecisionType.PASS
    assert r.output_guard is not None and r.output_guard.decision == GuardDecisionType.PASS
    assert len(r.retrieved_chunks) >= 1
    # Content checks — the corpus contains "50% tax exemption ... NT$3 million".
    assert "NT$3" in r.answer or "NT$ 3" in r.answer or "3 million" in r.answer, (
        f"answer missing NT$3 reference: {r.answer!r}"
    )
    assert "50%" in r.answer, f"answer missing 50% reference: {r.answer!r}"
    assert "[Source:" in r.answer, f"answer missing inline citation: {r.answer!r}"
    print(
        f"\n[guarded/answer] latency={r.latency_ms}ms tokens={r.token_usage.total_tokens} "
        f"chunks={len(r.retrieved_chunks)} top_score={r.retrieved_chunks[0].similarity_score:.3f} "
        f"output_guard={r.output_guard.decision.value}"
    )


@_pytestmark_live
async def test_guarded_low_confidence():
    """Big Mac query has no coverage — accept off-topic OR low-confidence refusal."""
    g = GuardedChatbot()
    r = await g.chat("What is the price of a Big Mac in Taipei?")
    assert r.chatbot_variant == "guarded"
    assert r.input_guard is not None
    assert r.input_guard.decision in {
        GuardDecisionType.REFUSE_OFF_TOPIC,
        GuardDecisionType.REFUSE_LOW_CONFIDENCE,
    }, f"unexpected decision: {r.input_guard.decision}"
    print(
        f"\n[guarded/low_conf] latency={r.latency_ms}ms decision={r.input_guard.decision.value} "
        f"reason={r.input_guard.reason!r}"
    )


@_pytestmark_live
async def test_guarded_correlation_id():
    """Every log line emitted by the chatbot during a chat turn carries the
    same correlation_id. Sub-module logs rely on structlog contextvars and
    are not captured here; the chatbot's own bound-logger lines are.
    """
    g = GuardedChatbot()
    with capture_logs() as cap:
        r1 = await g.chat("How long is the Employment Gold Card valid?")
        r2 = await g.chat("What is the gold card tax exemption?")

    assert r1.correlation_id and r2.correlation_id
    assert r1.correlation_id != r2.correlation_id, "each chat should mint a new id"

    cids = [e.get("correlation_id") for e in cap if "correlation_id" in e]
    assert len(cids) >= 4, f"expected >=4 logs with correlation_id, got {len(cids)}"
    distinct = set(cids)
    assert distinct == {r1.correlation_id, r2.correlation_id}, (
        f"unexpected ids: {distinct} vs {{r1, r2}}={set([r1.correlation_id, r2.correlation_id])}"
    )
    # At least one event per correlation_id (otherwise the trace is broken).
    for cid in (r1.correlation_id, r2.correlation_id):
        count = sum(1 for c in cids if c == cid)
        assert count >= 2, f"correlation_id {cid} only appeared {count} times"
    print(
        f"\n[guarded/cid] r1={r1.correlation_id[:8]} ({sum(1 for c in cids if c == r1.correlation_id)} logs) "
        f"r2={r2.correlation_id[:8]} ({sum(1 for c in cids if c == r2.correlation_id)} logs)"
    )
