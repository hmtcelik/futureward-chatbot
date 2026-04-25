"""Shared Streamlit components for the demo pages.

Render-only helpers — they take pydantic models and produce Streamlit widgets.
No business logic except the small ``determine_verdict`` rule, which is a
deterministic mapping over existing ChatResponse fields.
"""

from __future__ import annotations

import html
import re
import uuid
from enum import Enum

import streamlit as st

from src.guards.prompts import (
    LOW_CONFIDENCE_RESPONSE,
    NOT_GROUNDED_RESPONSE,
    REFUSAL_RESPONSE,
)
from src.models import (
    ChatResponse,
    GuardDecision,
    GuardDecisionType,
    RetrievedChunk,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Guard pills.
# ---------------------------------------------------------------------------
# Friendlier labels for non-engineer reviewers. The raw decision name is
# preserved in the title= tooltip for technical readers.
_GUARD_PILL_STYLE: dict[GuardDecisionType, tuple[str, str, str]] = {
    GuardDecisionType.PASS: (
        "✓",
        "On-topic",
        "background:#0f3a1f;color:#7ee6a3;",
    ),
    GuardDecisionType.REFUSE_OFF_TOPIC: (
        "⊘",
        "Off-topic — refused",
        "background:#3a1f1f;color:#ff8b8b;",
    ),
    GuardDecisionType.REFUSE_LOW_CONFIDENCE: (
        "→",
        "Low confidence — escalated",
        "background:#3a341a;color:#ffd97a;",
    ),
    GuardDecisionType.REFUSE_NOT_GROUNDED: (
        "⊘",
        "Unsafe answer — caught",
        "background:#3a1f1f;color:#ff8b8b;",
    ),
}

_OUTPUT_PILL_OVERRIDE: dict[GuardDecisionType, tuple[str, str, str]] = {
    GuardDecisionType.PASS: (
        "✓",
        "Verified",
        "background:#0f3a1f;color:#7ee6a3;",
    ),
}


def render_guard_pill(decision: GuardDecision | None, slot: str = "input") -> None:
    """Render a small inline pill summarizing a guard decision.

    Args:
        decision: the GuardDecision to render. ``None`` is a no-op.
        slot: ``"input"`` or ``"output"`` — only changes the PASS label
            (Verified vs. On-topic).
    """
    if decision is None:
        return
    style_map = (
        _OUTPUT_PILL_OVERRIDE | _GUARD_PILL_STYLE
        if slot == "output"
        else _GUARD_PILL_STYLE
    )
    icon, label, style = style_map.get(
        decision.decision, ("⚪", "Unknown", "background:#222;color:#aaa;")
    )
    tooltip = f"{decision.decision.value} — {decision.reason or '(no reason)'}"
    st.html(
        f"<span title='{html.escape(tooltip)}' "
        f"style='display:inline-block;padding:2px 10px;border-radius:999px;"
        f"font-family:\"JetBrains Mono\",monospace;font-size:0.72rem;font-weight:500;"
        f"{style}margin-right:6px;letter-spacing:0.02em'>"
        f"{icon} {html.escape(label)}</span>"
    )


# ---------------------------------------------------------------------------
# Citations.
# ---------------------------------------------------------------------------
def render_citations(chunks: list[RetrievedChunk]) -> None:
    """Render retrieved chunks as a friendly expander.

    Similarity scores are intentionally omitted here — they belong on the
    Inside-the-RAG page. This page is for clarity, not depth.
    """
    if not chunks:
        return
    label = f"Documents this answer is based on ({len(chunks)})"
    with st.expander(label, expanded=False):
        seen: set[str] = set()
        for rc in chunks:
            url = rc.chunk.document_url
            if url in seen:
                continue
            seen.add(url)
            title = html.escape(rc.chunk.document_title)
            url_safe = html.escape(url)
            st.html(
                f"<div style='padding:0.35rem 0'>"
                f"<span style='color:#d63d3d;margin-right:0.4rem'>→</span>"
                f"<a href='{url_safe}' target='_blank' rel='noopener' "
                f"style='color:#f5f1e8;text-decoration:underline;text-underline-offset:3px'>"
                f"{title}</a>"
                f"<div style='font-family:\"JetBrains Mono\",monospace;"
                f"font-size:0.7rem;color:#666;padding-left:1.1rem;margin-top:0.15rem'>"
                f"{url_safe}</div>"
                f"</div>"
            )


# ---------------------------------------------------------------------------
# Cost badge.
# ---------------------------------------------------------------------------
def render_cost_badge(usage: TokenUsage, latency_ms: int) -> None:
    """Tiny grey caption with cost + latency for a single response."""
    cost = f"${usage.estimated_cost_usd:.4f}" if usage.estimated_cost_usd else "$0.0000"
    st.html(
        f"<div style='font-family:\"JetBrains Mono\",monospace;font-size:0.7rem;"
        f"color:#666;margin-top:0.5rem;letter-spacing:0.02em'>"
        f"⏱ {latency_ms / 1000:.1f}s · 💰 {cost} · 🔢 {usage.total_tokens} tokens"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Verdict banner.
# ---------------------------------------------------------------------------
class Verdict(str, Enum):
    """Trust verdict shown below an assistant answer."""

    HALLUCINATION_RISK = "hallucination_risk"
    GROUNDED_CITED = "grounded_cited"
    CORRECTLY_ESCALATED = "correctly_escalated"
    ANSWERED_UNVERIFIED = "answered_unverified"


# query_category values used by the comparison page when the user picks a
# suggestion from the sidebar; free-typed queries pass None.
_HALLUCINATION_TRIGGER_CATEGORIES = {"hallucination", "sparse_coverage"}
_REFUSAL_BODIES = {
    REFUSAL_RESPONSE.strip(),
    LOW_CONFIDENCE_RESPONSE.strip(),
    NOT_GROUNDED_RESPONSE.strip(),
}
_NUMERIC_RE = re.compile(r"\b(?:NT\$|\$)?\d{2,}(?:[.,]\d+)?\b")


def determine_verdict(response: ChatResponse, query_category: str | None) -> Verdict:
    """Map ``ChatResponse`` + sidebar category → trust verdict."""
    if response.chatbot_variant == "guarded":
        if response.answer.strip() in _REFUSAL_BODIES:
            return Verdict.CORRECTLY_ESCALATED
        if (
            response.input_guard is not None
            and response.input_guard.decision != GuardDecisionType.PASS
        ):
            return Verdict.CORRECTLY_ESCALATED
        if (
            response.output_guard is not None
            and response.output_guard.decision != GuardDecisionType.PASS
        ):
            return Verdict.CORRECTLY_ESCALATED
        if (
            response.output_guard is not None
            and response.output_guard.decision == GuardDecisionType.PASS
            and len(response.retrieved_chunks) > 0
        ):
            return Verdict.GROUNDED_CITED
        return Verdict.CORRECTLY_ESCALATED

    # naive variant
    if query_category in _HALLUCINATION_TRIGGER_CATEGORIES:
        return Verdict.HALLUCINATION_RISK
    # Heuristic for free-typed: long answer with concrete numerics → risk.
    if (
        query_category is None
        and response.token_usage.completion_tokens > 50
        and _NUMERIC_RE.search(response.answer or "")
    ):
        return Verdict.HALLUCINATION_RISK
    return Verdict.ANSWERED_UNVERIFIED


_VERDICT_VISUALS = {
    Verdict.HALLUCINATION_RISK: {
        "icon": "⚠",
        "label": "HALLUCINATION RISK",
        "explainer": (
            "This answer contains specific numbers, dates, or rules that aren't "
            "in Talent Taiwan's source documents. The model fabricated "
            "authoritative-sounding details."
        ),
        "border": "#c73e3e",
        "bg": "rgba(199,62,62,0.08)",
        "icon_color": "#ff8b8b",
    },
    Verdict.GROUNDED_CITED: {
        "icon": "✓",
        "label": "GROUNDED & CITED",
        "explainer": (
            "Every claim in this answer is backed by retrieved Talent Taiwan "
            "documents. Output guard verified groundedness before showing this "
            "to you."
        ),
        "border": "#5dba6f",
        "bg": "rgba(93,186,111,0.08)",
        "icon_color": "#7ee6a3",
    },
    Verdict.CORRECTLY_ESCALATED: {
        "icon": "→",
        "label": "CORRECTLY ESCALATED",
        "explainer": (
            "The system recognized it couldn't safely answer and routed the "
            "user to Talent Taiwan staff. This is the correct outcome — "
            "escalation is a feature, not a failure."
        ),
        "border": "#d4aa50",
        "bg": "rgba(212,170,80,0.08)",
        "icon_color": "#ffd97a",
    },
    Verdict.ANSWERED_UNVERIFIED: {
        "icon": "○",
        "label": "ANSWERED (UNVERIFIED)",
        "explainer": (
            "The naive setup answered, but with no retrieval grounding and no "
            "verification. It might be right, it might be wrong — there's no "
            "way to tell from the response alone."
        ),
        "border": "#444",
        "bg": "rgba(136,136,136,0.06)",
        "icon_color": "#888",
    },
}


def render_verdict_banner(verdict: Verdict) -> None:
    """Render the colored verdict bar that appears below an assistant answer."""
    v = _VERDICT_VISUALS[verdict]
    st.html(
        f"""
        <div style="
          margin-top: 0.75rem;
          padding: 0.85rem 1rem;
          background: {v["bg"]};
          border-top: 1px solid {v["border"]};
          display: flex;
          align-items: flex-start;
          gap: 0.7rem;
        ">
          <span style="
            font-family: 'JetBrains Mono', monospace;
            color: {v["icon_color"]};
            font-size: 1rem;
            line-height: 1.3;
          ">{v["icon"]}</span>
          <div style="flex:1;min-width:0">
            <div style="
              font-family: 'JetBrains Mono', monospace;
              font-size: 0.78rem;
              font-weight: 600;
              letter-spacing: 0.12em;
              color: {v["icon_color"]};
              margin-bottom: 0.25rem;
            ">{v["label"]}</div>
            <div style="
              font-family: 'Inter Tight', system-ui, sans-serif;
              font-size: 0.88rem;
              line-height: 1.5;
              color: #b9b3a4;
            ">{v["explainer"]}</div>
          </div>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Truncatable long-answer renderer (naive only).
# ---------------------------------------------------------------------------
def render_truncatable_answer(content: str, char_limit: int = 800, line_limit: int = 12) -> None:
    """Render long content with a CSS-only "Show full answer ↓" toggle.

    Streamlit's chat_message normally renders markdown via ``st.markdown``;
    long fabricated responses can dominate the page. We render the markdown
    once into HTML with markdown-it-py (already a Streamlit dep) and wrap it
    with a checkbox-driven CSS toggle so a user can expand on demand.
    """
    is_long = len(content) > char_limit or content.count("\n") > line_limit
    if not is_long:
        st.markdown(content)
        return

    try:
        from markdown_it import MarkdownIt

        body_html = MarkdownIt("commonmark").render(content)
    except Exception:  # noqa: BLE001 - fall back if md-it unavailable
        body_html = "<pre>" + html.escape(content) + "</pre>"

    cb_id = f"long-{uuid.uuid4().hex[:8]}"
    st.html(
        f"""
        <style>
        .long-cb-{cb_id} {{ display: none; }}
        .long-wrap-{cb_id} {{
          max-height: 380px;
          overflow: hidden;
          position: relative;
          transition: max-height 240ms ease-out;
        }}
        .long-wrap-{cb_id}::after {{
          content: '';
          position: absolute;
          left: 0;
          right: 0;
          bottom: 0;
          height: 90px;
          background: linear-gradient(to bottom, transparent, var(--bg-soft));
          pointer-events: none;
        }}
        .long-cb-{cb_id}:checked ~ .long-wrap-{cb_id} {{
          max-height: 4000px;
        }}
        .long-cb-{cb_id}:checked ~ .long-wrap-{cb_id}::after {{ display: none; }}
        .long-toggle-{cb_id} {{
          display: inline-block;
          cursor: pointer;
          font-family: "JetBrains Mono", monospace;
          font-size: 0.78rem;
          color: var(--muted);
          margin-top: 0.6rem;
          text-decoration: underline;
          text-underline-offset: 3px;
          user-select: none;
        }}
        .long-toggle-{cb_id}:hover {{ color: var(--ink); }}
        .long-cb-{cb_id}:not(:checked) ~ .long-toggle-{cb_id}::before {{ content: 'Show full answer ↓'; }}
        .long-cb-{cb_id}:checked ~ .long-toggle-{cb_id}::before {{ content: 'Hide ↑'; }}
        .long-wrap-{cb_id} p:first-child {{ margin-top: 0; }}
        .long-wrap-{cb_id} p:last-child {{ margin-bottom: 0; }}
        </style>
        <input type="checkbox" id="cb-{cb_id}" class="long-cb-{cb_id}">
        <div class="long-wrap-{cb_id}">{body_html}</div>
        <label for="cb-{cb_id}" class="long-toggle-{cb_id}"></label>
        """
    )


# ---------------------------------------------------------------------------
# Status row (used by landing setup banner).
# ---------------------------------------------------------------------------
def render_status_row(label: str, ok: bool, detail: str = "") -> None:
    """One-line status indicator used by the landing page system check."""
    icon = "✅" if ok else "❌"
    color = "#7ee6a3" if ok else "#ff8b8b"
    st.html(
        f"<div style='padding:6px 0'>"
        f"<span style='font-size:1.05rem'>{icon}</span> "
        f"<span style='font-weight:600'>{html.escape(label)}</span>"
        f"<span style='color:{color};margin-left:10px'>{html.escape(detail)}</span>"
        f"</div>"
    )


# Backwards-compat shim: old chat_message helper. Comparison page now renders
# inline; kept here for any future module that wants a one-liner.
def render_chat_message(role: str, content: str, response: ChatResponse | None = None) -> None:
    """Unified message bubble. ``response`` triggers guard/citation/cost extras."""
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content if content else "_(empty response)_")
        if response is not None and role == "assistant":
            cols = st.columns([1, 1, 2])
            with cols[0]:
                render_guard_pill(response.input_guard, slot="input")
            with cols[1]:
                render_guard_pill(response.output_guard, slot="output")
            render_citations(response.retrieved_chunks)
            render_cost_badge(response.token_usage, response.latency_ms)
