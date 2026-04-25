"""Streamlit ↔ asyncio bridge plus per-session rate limiting.

Streamlit reruns the script top-to-bottom for every interaction, so anything
asyncio-related has to wrap a coroutine in a fresh event loop on each call.
``run_async`` does that. ``RateLimiter`` lives in ``st.session_state`` and
caps how often / how many times a single browser session can hit the live
API — important for any public deploy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

import streamlit as st

T = TypeVar("T")

# Public-deploy guardrails. Tweak in one place rather than three.
MAX_QUERIES_PER_SESSION = 30
MIN_SECONDS_BETWEEN_QUERIES = 3.0


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


@dataclass
class RateLimitDecision:
    """Outcome of a rate-limit check."""

    allowed: bool
    reason: str = ""
    seconds_until_next: float = 0.0
    remaining_in_session: int = 0


class RateLimiter:
    """Tracks query count + last-call timestamp in ``st.session_state``."""

    def __init__(
        self,
        max_per_session: int = MAX_QUERIES_PER_SESSION,
        min_seconds_between: float = MIN_SECONDS_BETWEEN_QUERIES,
    ):
        self.max_per_session = max_per_session
        self.min_seconds_between = min_seconds_between
        st.session_state.setdefault("_rate_count", 0)
        st.session_state.setdefault("_rate_last_at", 0.0)

    @property
    def count(self) -> int:
        return int(st.session_state.get("_rate_count", 0))

    def check(self) -> RateLimitDecision:
        now = time.time()
        last = float(st.session_state.get("_rate_last_at", 0.0))
        used = self.count

        if used >= self.max_per_session:
            return RateLimitDecision(
                allowed=False,
                reason=(
                    f"Session limit reached ({self.max_per_session} queries). "
                    "Refresh the page to start a new session."
                ),
                remaining_in_session=0,
            )

        wait = (last + self.min_seconds_between) - now
        if wait > 0:
            return RateLimitDecision(
                allowed=False,
                reason=f"Slow down — please wait {wait:.1f}s between queries.",
                seconds_until_next=wait,
                remaining_in_session=self.max_per_session - used,
            )

        return RateLimitDecision(
            allowed=True, remaining_in_session=self.max_per_session - used
        )

    def commit(self) -> None:
        """Record that a query was just executed."""
        st.session_state["_rate_count"] = self.count + 1
        st.session_state["_rate_last_at"] = time.time()
