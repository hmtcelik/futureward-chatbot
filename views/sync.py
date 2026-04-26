"""Pipeline Sync — operational story page (Task 2 evidence).

Five sections:
    A. Current index state (stats strip + inventory table)
    B. Simulate a content change (selectbox + editable text area + save)
    C. Run incremental sync (real backend operation, animated 5-stage timeline)
    D. Cost projection at scale (live sliders + Altair line chart)
    E. Closing argument (prose)

Reuses sidebar nav, fonts, palette, fade-up animations from sibling pages.
No chat input.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.config import settings
from src.pipeline.sync import (
    StageReport,
    SyncResult,
    list_indexed_documents,
    project_cost,
    read_extracted,
    run_real_sync,
)
from src.ui.runtime import run_async
from src.ui.theme import render_sidebar_meta


# ---------------------------------------------------------------------------
# Session state.
# ---------------------------------------------------------------------------
st.session_state.setdefault("sn_selected_url", None)
st.session_state.setdefault("sn_editor", "")
st.session_state.setdefault("sn_editor_loaded_for", None)
st.session_state.setdefault("sn_save_toast_at", 0.0)
st.session_state.setdefault("sn_show_full_inv", False)
st.session_state.setdefault("sn_sync_running", False)
st.session_state.setdefault("sn_sync_result", None)


def fmt_cost(value: float) -> str:
    """Editorial cost formatting: 4-dec under 1¢, 2-dec at/above 1¢."""
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:,.2f}"


_qp = st.query_params
if "reset" in _qp:
    st.session_state["sn_selected_url"] = None
    st.session_state["sn_editor"] = ""
    st.session_state["sn_editor_loaded_for"] = None
    st.session_state["sn_sync_result"] = None
    st.session_state["sn_sync_running"] = False
    _qp.clear()


# ---------------------------------------------------------------------------
# Stylesheet.
# ---------------------------------------------------------------------------
st.html(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
      max-width: 1280px !important;
      margin: 0 auto !important;
      padding: 0 3rem !important;
    }
    @media (max-width: 768px) {
      [data-testid="stMainBlockContainer"] { padding: 0 1.25rem !important; }
    }

    /* Header. */
    .sn-shell { padding: 2.5rem 0 1.5rem 0; }
    .sn-eyebrow {
      font-family: "JetBrains Mono", monospace;
      font-size: 0.7rem;
      letter-spacing: 0.2em;
      color: var(--accent);
      text-transform: uppercase;
      margin-bottom: 1rem;
    }
    .sn-title {
      font-family: "Fraunces", Georgia, serif;
      font-weight: 400;
      font-size: clamp(34px, 4.6vw, 52px);
      line-height: 1.05;
      letter-spacing: -0.015em;
      color: var(--ink);
      margin: 0 0 1rem 0;
      max-width: 18ch;
    }
    .sn-lead {
      font-family: "Inter Tight", system-ui, sans-serif;
      font-size: 1.05rem;
      line-height: 1.55;
      color: #aaa;
      max-width: 720px;
      margin: 0;
    }

    /* Section header. */
    .sn-section { margin: 5rem 0 0 0; }

    /* Editorial expander (used for the inventory table). */
    [data-testid="stExpander"] {
      background: transparent;
      border: 1px solid #1a1a1a;
      border-radius: 0;
      margin-top: 1rem;
    }
    [data-testid="stExpander"] details > summary {
      font-family: "JetBrains Mono", monospace !important;
      font-size: 11px !important;
      letter-spacing: 0.15em !important;
      text-transform: uppercase !important;
      color: #888 !important;
      padding: 0.85rem 1.25rem !important;
      list-style: none !important;
      cursor: pointer !important;
    }
    [data-testid="stExpander"] details > summary:hover {
      color: var(--ink) !important;
      background: #0e0e0e !important;
    }
    [data-testid="stExpander"] details[open] > summary {
      color: var(--accent) !important;
      border-bottom: 1px solid #1a1a1a !important;
    }
    [data-testid="stExpander"] details > summary svg {
      stroke: currentColor !important;
      width: 14px !important;
      height: 14px !important;
    }
    [data-testid="stExpander"] details > div { padding: 1rem 1.25rem !important; }
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] svg {
      stroke: currentColor !important;
    }
    .sn-section-eye {
      font-family: "JetBrains Mono", monospace;
      font-size: 0.62rem;
      letter-spacing: 0.2em;
      color: #666;
      text-transform: uppercase;
      margin-bottom: 1.25rem;
    }
    .sn-section-intro {
      font-family: "Inter Tight", sans-serif;
      font-size: 0.95rem;
      line-height: 1.6;
      color: #aaa;
      max-width: 720px;
      margin: 0 0 2rem 0;
    }

    /* --- Section A — Stats strip + inventory ------------------------- */
    .sn-stats {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 3rem;
      margin: 3rem 0;
    }
    @media (max-width: 768px) {
      .sn-stats { grid-template-columns: 1fr; gap: 2rem; }
    }
    .sn-stat-num {
      font-family: "Fraunces", Georgia, serif;
      font-size: clamp(40px, 5vw, 56px);
      font-weight: 300;
      color: var(--ink);
      line-height: 1;
      letter-spacing: -0.02em;
      margin: 0 0 0.6rem 0;
    }
    .sn-stat-label {
      font-family: "JetBrains Mono", monospace;
      font-size: 0.68rem;
      letter-spacing: 0.18em;
      color: #666;
      text-transform: uppercase;
    }

    .sn-inv-head {
      font-family: "JetBrains Mono", monospace;
      font-size: 0.62rem;
      letter-spacing: 0.2em;
      color: #555;
      text-transform: uppercase;
      margin: 2rem 0 0.75rem 0;
    }
    .sn-inv-table {
      width: 100%;
      border-collapse: collapse;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
    }
    .sn-inv-table thead th {
      text-align: left;
      font-weight: 500;
      color: #555;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-size: 10px;
      padding: 0.6rem 0.6rem;
      border-bottom: 1px solid #1a1a1a;
    }
    .sn-inv-table tbody td {
      padding: 0.55rem 0.6rem;
      border-bottom: 1px solid #141414;
      color: #aaa;
      vertical-align: top;
    }
    .sn-inv-table tbody tr:hover td { background: #0e0e0e; }
    .sn-inv-table td.title {
      font-family: "Inter Tight", sans-serif;
      font-size: 13px;
      color: #d8d2c5;
    }
    .sn-inv-toggle {
      display: inline-block;
      margin-top: 0.75rem;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      color: var(--accent);
      cursor: pointer;
      text-decoration: none;
      letter-spacing: 0.04em;
    }
    .sn-inv-toggle:hover { color: var(--accent-hot); }

    /* --- Section B — read-only source viewer ------------------------- */
    .sn-source-content {
      max-height: 360px;
      overflow-y: auto;
      background: #0e0e0e;
      border: 1px solid #1a1a1a;
      padding: 1.25rem 1.5rem;
      margin-top: 0.5rem;
    }
    .sn-source-content pre {
      margin: 0;
      font-family: "Inter Tight", system-ui, sans-serif;
      font-size: 13px;
      line-height: 1.65;
      color: #d8d2c5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    /* Skipped stage in the timeline. */
    .sn-stage[data-skipped="true"] .sn-stage-num,
    .sn-stage[data-skipped="true"] .sn-stage-label { color: #555; }
    .sn-stage[data-skipped="true"] .sn-stage-summary { color: #555; font-style: italic; }
    .sn-stage-node[data-state="skipped"] {
      border-color: #1a1a1a;
      background: #0a0a0a;
    }

    /* --- Section B — selectbox -------------------------------------- */
    .sn-form-label {
      font-family: "JetBrains Mono", monospace;
      font-size: 10px;
      letter-spacing: 0.15em;
      color: #666;
      text-transform: uppercase;
      margin: 1.25rem 0 0.5rem 0;
    }
    [data-testid="stTextArea"] textarea {
      background: #0e0e0e !important;
      border: 1px solid #1a1a1a !important;
      border-radius: 0 !important;
      font-family: "JetBrains Mono", monospace !important;
      font-size: 12px !important;
      color: #d8d2c5 !important;
      line-height: 1.55 !important;
      padding: 1.25rem !important;
    }
    [data-testid="stTextArea"] textarea:focus {
      border-color: var(--accent) !important;
      outline: none !important;
      box-shadow: none !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      background: #0e0e0e !important;
      border: 1px solid #1a1a1a !important;
      border-radius: 0 !important;
      font-family: "Inter Tight", sans-serif !important;
      color: #d8d2c5 !important;
    }

    .sn-mod-line {
      display: flex;
      align-items: center;
      gap: 1.5rem;
      margin: 0.85rem 0 1.25rem 0;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      letter-spacing: 0.04em;
    }
    .sn-mod-changed { color: #d4aa50; }
    .sn-mod-unchanged { color: #555; }

    .st-key-sn-actions [data-testid="stButton"] button {
      font-family: "JetBrains Mono", monospace !important;
      font-size: 12px !important;
      font-weight: 600 !important;
      padding: 0.7rem 1.4rem !important;
      border: 1px solid var(--accent) !important;
      background: transparent !important;
      color: var(--accent) !important;
      border-radius: 0 !important;
      box-shadow: none !important;
      transition: all 200ms ease-out !important;
      letter-spacing: 0.04em !important;
    }
    .st-key-sn-actions [data-testid="stButton"] button:hover {
      background: var(--accent) !important;
      color: #0a0a0a !important;
    }
    .st-key-sn-actions [data-testid="stButton"] button p {
      margin: 0 !important; font-size: inherit !important;
    }
    .sn-reset-link {
      display: inline-block;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      color: #555;
      cursor: pointer;
      letter-spacing: 0.04em;
      margin-left: 1rem;
      text-decoration: underline;
      text-underline-offset: 3px;
    }
    .sn-reset-link:hover { color: #888; }

    .sn-toast {
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      color: #5dba6f;
      margin-top: 0.85rem;
      letter-spacing: 0.04em;
    }

    /* --- Section C — Run sync big button + animated timeline -------- */
    .st-key-sn-run [data-testid="stButton"] button {
      width: 100% !important;
      font-family: "Fraunces", Georgia, serif !important;
      font-style: italic !important;
      font-size: 20px !important;
      font-weight: 400 !important;
      padding: 1.5rem 2rem !important;
      border: 1px solid var(--accent) !important;
      background: transparent !important;
      color: var(--ink) !important;
      border-radius: 0 !important;
      box-shadow: none !important;
      letter-spacing: 0 !important;
      transition: all 200ms ease-out !important;
    }
    .st-key-sn-run [data-testid="stButton"] button:hover {
      background: var(--accent) !important;
      color: #0a0a0a !important;
    }
    .st-key-sn-run [data-testid="stButton"] button:disabled {
      border-color: #2a2a2a !important;
      color: #444 !important;
      cursor: not-allowed !important;
    }

    /* Sync timeline (reuses pipeline-timeline shape). */
    .sn-timeline {
      position: relative;
      padding-left: 3rem;
      margin: 2rem 0 0 0;
    }
    .sn-timeline::before {
      content: "";
      position: absolute;
      left: calc(0.95rem - 0.5px);
      top: 1rem;
      bottom: 1rem;
      width: 1px;
      background: #1a1a1a;
      z-index: 0;
    }
    .sn-timeline[data-state="complete"]::before { background: #5dba6f; }
    .sn-stage { position: relative; padding-bottom: 1.5rem; }
    .sn-stage:last-child { padding-bottom: 0; }
    .sn-stage-node {
      position: absolute;
      left: -3rem;
      top: 0;
      width: 1.9rem;
      height: 1.9rem;
      border-radius: 50%;
      background: #0a0a0a;
      border: 2px solid #1a1a1a;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      font-weight: 700;
      color: transparent;
      transition: all 200ms ease-out;
    }
    .sn-stage-node[data-state="pass"] {
      border-color: #5dba6f; background: #5dba6f; color: #0a0a0a;
    }
    .sn-stage-node[data-state="running"] {
      border-color: var(--accent); background: #0a0a0a;
      animation: sn-pulse 1.5s ease-in-out infinite;
    }
    @keyframes sn-pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.15); opacity: 0.7; }
    }
    .sn-stage-head {
      display: flex;
      align-items: baseline;
      gap: 1rem;
      padding-top: 0.25rem;
    }
    .sn-stage-num {
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      font-weight: 600;
      color: var(--accent);
    }
    .sn-stage-label {
      font-family: "JetBrains Mono", monospace;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.15em;
      color: var(--ink);
      text-transform: uppercase;
    }
    .sn-stage-summary {
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      color: #888;
      margin-left: auto;
      letter-spacing: 0.02em;
    }
    .sn-stage-detail {
      margin-top: 0.4rem;
      margin-left: 0;
      font-family: "Inter Tight", sans-serif;
      font-size: 12px;
      color: #888;
    }
    /* Loading-state stage progressive reveal. */
    @keyframes sn-step-reveal {
      from { opacity: 0; transform: translateY(-2px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .sn-stage.loading {
      opacity: 0;
      animation: sn-step-reveal 400ms ease-out forwards;
    }
    .sn-stage.loading.s1 { animation-delay: 0ms; }
    .sn-stage.loading.s2 { animation-delay: 400ms; }
    .sn-stage.loading.s3 { animation-delay: 800ms; }
    .sn-stage.loading.s4 { animation-delay: 1300ms; }
    .sn-stage.loading.s5 { animation-delay: 2400ms; }

    /* Sync summary card. */
    .sn-summary {
      margin: 3rem 0 0 0;
      padding: 2.5rem 0;
      border-top: 1px solid #1a1a1a;
      border-bottom: 1px solid #1a1a1a;
    }
    .sn-summary-eye {
      font-family: "JetBrains Mono", monospace;
      font-size: 12px;
      letter-spacing: 0.15em;
      color: #5dba6f;
      text-transform: uppercase;
      margin-bottom: 1rem;
    }
    .sn-summary-headline {
      font-family: "Fraunces", Georgia, serif;
      font-size: 24px;
      line-height: 1.3;
      color: var(--ink);
      margin: 0;
      max-width: 36ch;
    }
    .sn-summary-detail {
      margin-top: 2rem;
      font-family: "JetBrains Mono", monospace;
      font-size: 12px;
    }
    .sn-summary-detail-head {
      font-size: 10px;
      letter-spacing: 0.2em;
      color: #666;
      text-transform: uppercase;
      margin-bottom: 0.85rem;
    }
    .sn-summary-row {
      display: flex;
      justify-content: space-between;
      padding: 0.5rem 0;
      border-bottom: 1px solid #141414;
      color: #d8d2c5;
    }
    .sn-summary-row:last-child { border-bottom: none; }
    .sn-summary-row.savings { color: #5dba6f; font-weight: 500; }

    /* --- Section D — Sliders + chart --------------------------------- */
    [data-testid="stSlider"] { margin-bottom: 1.25rem; }
    [data-testid="stSlider"] label {
      font-family: "JetBrains Mono", monospace !important;
      font-size: 10px !important;
      letter-spacing: 0.15em !important;
      color: #666 !important;
      text-transform: uppercase !important;
    }
    [data-testid="stSlider"] [role="slider"] {
      background: var(--accent) !important;
      border: none !important;
      box-shadow: none !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: #1a1a1a !important; }
    [data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }
    [data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] + div {
      color: var(--ink) !important;
      font-family: "JetBrains Mono", monospace !important;
    }

    .sn-cost-row {
      display: flex;
      justify-content: space-between;
      padding: 0.6rem 0;
      font-family: "JetBrains Mono", monospace;
      font-size: 14px;
      color: #d8d2c5;
      border-bottom: 1px solid #141414;
    }
    .sn-cost-row.green { color: #5dba6f; }
    .sn-cost-row .label {
      font-size: 11px;
      color: #666;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }
    .sn-math {
      font-family: "JetBrains Mono", monospace;
      font-size: 10px;
      color: #555;
      line-height: 1.6;
      margin-top: 1rem;
      max-width: 720px;
    }

    /* --- Section E — Closing prose ----------------------------------- */
    .sn-prose {
      font-family: "Inter Tight", sans-serif;
      font-size: 16px;
      line-height: 1.65;
      color: #d8d2c5;
      max-width: 720px;
    }
    .sn-prose p { margin: 0 0 1.2rem 0; }
    .sn-prose p:last-child { margin: 0; }

    .sn-end-rule {
      border: 0;
      border-top: 1px solid #1a1a1a;
      margin: 4rem 0 3rem 0;
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Sidebar.
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_meta()
    st.html(
        '<a class="sb-reset-link" href="?reset=1" target="_self">↺ Clear edits</a>'
    )


# ---------------------------------------------------------------------------
# Header.
# ---------------------------------------------------------------------------
st.html(
    """
    <div class="sn-shell">
      <div class="sn-eyebrow reveal d0">DEMO 03 — KEEPING CONTENT FRESH</div>
      <h1 class="sn-title reveal d1">Keeping the index up to date.</h1>
      <p class="sn-lead reveal d2">
        Talent Taiwan's regulatory content changes constantly — visa rules,
        gold card terms, tax thresholds. Re-embedding every document on
        every update is wasteful. Here's the incremental pipeline: detect
        what changed, re-embed only that, update the vector store. Try it
        live.
      </p>
    </div>
    """
)


# ---------------------------------------------------------------------------
# Section A — Current state.
# ---------------------------------------------------------------------------
st.html('<div class="sn-section"><div class="sn-section-eye">CURRENT STATE</div></div>')

inventory = list_indexed_documents()
docs_count = len(inventory)
chunks_count = sum(r["chunks"] for r in inventory)
INDEXING_COST_USD = 0.0018  # measured Phase 2

st.html(
    f"""
    <div class="sn-stats">
      <div>
        <div class="sn-stat-num">{docs_count}</div>
        <div class="sn-stat-label">Documents indexed</div>
      </div>
      <div>
        <div class="sn-stat-num">{chunks_count}</div>
        <div class="sn-stat-label">Chunks embedded</div>
      </div>
      <div>
        <div class="sn-stat-num">${INDEXING_COST_USD:.4f}</div>
        <div class="sn-stat-label">Initial indexing cost</div>
      </div>
    </div>
    """
)

# Inventory table — collapsed behind an expander so it doesn't dominate
# first scroll. Reviewers who want to drill in click. URL column is
# truncated to the last segment; full path lives in the title= tooltip.
table_rows: list[str] = []
for row in inventory:
    full_path = row["url"].replace("https://goldcard.nat.gov.tw", "")
    short = full_path.rstrip("/").rsplit("/", 1)[-1] or full_path
    table_rows.append(
        f"<tr>"
        f'<td title="{html.escape(full_path)}">{html.escape(short)}</td>'
        f'<td class="title">{html.escape(row["title"])}</td>'
        f"<td>{row['chunks']}</td>"
        f"</tr>"
    )

with st.expander(f"Show indexed documents ({docs_count})", expanded=False):
    st.html(
        '<table class="sn-inv-table"><thead><tr>'
        "<th>URL</th><th>TITLE</th><th>CHUNKS</th>"
        "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Section B — Source content (read-only viewer).
# ---------------------------------------------------------------------------
st.html(
    '<div class="sn-section">'
    '<div class="sn-section-eye">SOURCE CONTENT</div>'
    '<p class="sn-section-intro">'
    "Every page below was scraped automatically from "
    "<a href=\"https://goldcard.nat.gov.tw/en/\" target=\"_blank\" "
    'style="color:#9ecbff;text-decoration:underline">'
    "goldcard.nat.gov.tw"
    "</a>"
    " by our crawler — no manual uploads. Pick a document to inspect what "
    "we have on file, then run the incremental sync below to re-fetch it "
    "from the live site and verify it's still in sync."
    "</p>"
    "</div>"
)

st.html('<div class="sn-form-label">SELECT A DOCUMENT</div>')

url_options = [r["url"] for r in inventory]
url_to_title = {r["url"]: r["title"] for r in inventory}
preferred_url = "https://goldcard.nat.gov.tw/en/about"
substantial = [r["url"] for r in inventory if r.get("chunks", 0) >= 3]
if preferred_url in url_options:
    default_url = preferred_url
elif substantial:
    default_url = substantial[0]
elif url_options:
    default_url = url_options[0]
else:
    default_url = None

if st.session_state["sn_selected_url"] not in url_options:
    st.session_state["sn_selected_url"] = default_url

selected_url = st.selectbox(
    "Select",
    options=url_options,
    format_func=lambda u: url_to_title.get(u, u),
    key="sn_selected_url",
    label_visibility="collapsed",
)

if selected_url:
    try:
        content, title = read_extracted(selected_url)
    except FileNotFoundError:
        content, title = "", url_to_title.get(selected_url, "")

    chunk_count = next(
        (r["chunks"] for r in inventory if r["url"] == selected_url), 0
    )
    st.html(
        f'<div style="margin: 1rem 0 0.6rem 0">'
        f'<a href="{html.escape(selected_url)}" target="_blank" rel="noopener" '
        f'style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'color:#9ecbff;text-decoration:underline;text-underline-offset:3px;'
        f'letter-spacing:0.02em">'
        f"{html.escape(selected_url)} ↗</a>"
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
        f'color:#555;margin-left:1rem;letter-spacing:0.04em">'
        f"{chunk_count} chunks · {len(content):,} chars"
        f"</span>"
        f"</div>"
    )

    preview = content if len(content) <= 1500 else content[:1500] + "\n\n…"
    st.html(
        f'<div class="sn-source-content">'
        f'<pre>{html.escape(preview)}</pre>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Section C — Run sync (real fetch + extract + compare + re-embed if changed).
# ---------------------------------------------------------------------------
st.html(
    '<div class="sn-section">'
    '<div class="sn-section-eye">RUN THE SYNC</div>'
    '<p class="sn-section-intro">'
    "Re-fetch the selected page from the live site, extract clean text, "
    "compare its hash against what we already have, and re-embed only if "
    "the content drifted. This is exactly what would run on a cron schedule "
    "in production."
    "</p>"
    "</div>"
)


def _run_sync() -> None:
    st.session_state["sn_sync_running"] = True
    st.session_state["sn_sync_result"] = None


with st.container(key="sn-run"):
    st.button(
        "Run sync now  →",
        key="sn-run-btn",
        on_click=_run_sync,
        disabled=selected_url is None,
        width="stretch",
    )


# Loading placeholder — real fetch + extract + compare + (maybe) re-embed.
if st.session_state["sn_sync_running"]:
    st.html(
        '<div class="sn-timeline" data-state="loading">'
        '<div class="sn-stage loading s1"><div class="sn-stage-node" data-state="running"></div>'
        '<div class="sn-stage-head"><span class="sn-stage-num">01</span>'
        '<span class="sn-stage-label">FETCHING FROM SOURCE</span>'
        '<span class="sn-stage-summary">running…</span></div></div>'
        '<div class="sn-stage loading s2"><div class="sn-stage-node" data-state="running"></div>'
        '<div class="sn-stage-head"><span class="sn-stage-num">02</span>'
        '<span class="sn-stage-label">EXTRACTING TEXT</span>'
        '<span class="sn-stage-summary">running…</span></div></div>'
        '<div class="sn-stage loading s3"><div class="sn-stage-node" data-state="running"></div>'
        '<div class="sn-stage-head"><span class="sn-stage-num">03</span>'
        '<span class="sn-stage-label">COMPARING HASHES</span>'
        '<span class="sn-stage-summary">running…</span></div></div>'
        '<div class="sn-stage loading s4"><div class="sn-stage-node" data-state="running"></div>'
        '<div class="sn-stage-head"><span class="sn-stage-num">04</span>'
        '<span class="sn-stage-label">RE-CHUNKING + EMBEDDING</span>'
        '<span class="sn-stage-summary">running…</span></div></div>'
        '<div class="sn-stage loading s5"><div class="sn-stage-node" data-state="running"></div>'
        '<div class="sn-stage-head"><span class="sn-stage-num">05</span>'
        '<span class="sn-stage-label">UPSERTING TO VECTOR STORE</span>'
        '<span class="sn-stage-summary">running…</span></div></div>'
        "</div>"
    )
    try:
        result: SyncResult = run_async(run_real_sync(selected_url))
        st.session_state["sn_sync_result"] = result.model_dump()
    except Exception as exc:  # noqa: BLE001
        st.session_state["sn_sync_result"] = None
        st.html(
            f'<div class="sn-toast" style="color:#d8a77a">'
            f"Sync failed: {type(exc).__name__}: {exc}"
            f"</div>"
        )
    st.session_state["sn_sync_running"] = False
    st.rerun()


# Render result. If no_changes, the result has only 3 stages — synthesize
# stages 04/05 as "skipped" so the timeline always reads as 5 rows.
result_data = st.session_state.get("sn_sync_result")
if result_data:
    result = SyncResult(**result_data)
    timeline_state = "complete" if not result.no_changes else "complete"
    stages_to_render = list(result.stages)
    if result.no_changes:
        stages_to_render.append(
            StageReport(
                name="embedding",
                label="RE-CHUNKING + EMBEDDING",
                summary="skipped (no changes)",
            )
        )
        stages_to_render.append(
            StageReport(
                name="upserting",
                label="UPSERTING TO VECTOR STORE",
                summary="skipped (no changes)",
            )
        )

    rows: list[str] = []
    for i, stage in enumerate(stages_to_render):
        is_skipped = "skipped" in stage.summary
        node_state = "skipped" if is_skipped else "pass"
        skipped_attr = ' data-skipped="true"' if is_skipped else ""
        rows.append(
            f'<div class="sn-stage"{skipped_attr}>'
            f'<div class="sn-stage-node" data-state="{node_state}"></div>'
            f'<div class="sn-stage-head">'
            f'<span class="sn-stage-num">{i + 1:02d}</span>'
            f'<span class="sn-stage-label">{html.escape(stage.label)}</span>'
            f'<span class="sn-stage-summary">{html.escape(stage.summary)}</span>'
            f"</div></div>"
        )
    st.html(
        f'<div class="sn-timeline" data-state="{timeline_state}">'
        + "".join(rows)
        + "</div>"
    )

    if result.no_changes:
        st.html(
            f"""
            <div class="sn-summary">
              <div class="sn-summary-eye" style="color:#5dba6f">NO CHANGES · INDEX IN SYNC</div>
              <p class="sn-summary-headline">
                Live page hash matches our index.<br>
                No re-embedding needed — saved 100% of the re-index cost.
              </p>
              <div class="sn-summary-detail">
                <div class="sn-summary-detail-head">DETAIL BREAKDOWN</div>
                <div class="sn-summary-row">
                  <span>What ran (incremental)</span>
                  <span>$0.0000 (zero embedding calls)</span>
                </div>
                <div class="sn-summary-row">
                  <span>What full re-index would have cost</span>
                  <span>{fmt_cost(result.full_reindex_cost_usd)}</span>
                </div>
                <div class="sn-summary-row savings">
                  <span>Savings on this run</span>
                  <span>{fmt_cost(result.savings_usd)} (100%)</span>
                </div>
              </div>
            </div>
            """
        )
    else:
        st.html(
            f"""
            <div class="sn-summary">
              <div class="sn-summary-eye">SYNC COMPLETE · CONTENT UPDATED</div>
              <p class="sn-summary-headline">
                Re-embedded {result.delta_chunk_count} of
                {result.total_chunks_in_index} chunks.<br>
                Saved {result.savings_pct:.1f}% versus full re-index.
              </p>
              <div class="sn-summary-detail">
                <div class="sn-summary-detail-head">DETAIL BREAKDOWN</div>
                <div class="sn-summary-row">
                  <span>What ran (incremental)</span>
                  <span>{fmt_cost(result.incremental_cost_usd)}</span>
                </div>
                <div class="sn-summary-row">
                  <span>What full re-index would have cost</span>
                  <span>{fmt_cost(result.full_reindex_cost_usd)}</span>
                </div>
                <div class="sn-summary-row savings">
                  <span>Savings</span>
                  <span>{fmt_cost(result.savings_usd)} ({result.savings_pct:.1f}%)</span>
                </div>
              </div>
            </div>
            """
        )

    def _reset_sync_state() -> None:
        st.session_state["sn_sync_result"] = None
        st.session_state["sn_sync_running"] = False

    st.button("Run again", key="sn-reset-sync-btn", on_click=_reset_sync_state)


# ---------------------------------------------------------------------------
# Section D — Cost projection.
# ---------------------------------------------------------------------------
st.html(
    '<div class="sn-section">'
    '<div class="sn-section-eye">AT PRODUCTION SCALE</div>'
    '<p class="sn-section-intro">'
    "At our current scale (50 docs, 134 chunks), incremental sync saves "
    "cents. But Talent Taiwan's full content footprint includes thousands "
    "of regulatory pages, FAQs, and policy updates that change weekly. "
    "Here's what the same incremental logic looks like at production scale."
    "</p>"
    "</div>"
)

corpus_size = st.slider(
    "CORPUS SIZE (DOCUMENTS)",
    min_value=100,
    max_value=100_000,
    value=5_000,
    step=100,
    key="sn-corpus",
)
chunks_per_doc = st.slider(
    "AVG CHUNKS PER DOC",
    min_value=2,
    max_value=30,
    value=8,
    step=1,
    key="sn-chunks-per-doc",
)
change_rate = st.slider(
    "WEEKLY CHANGE RATE (%)",
    min_value=1,
    max_value=25,
    value=5,
    step=1,
    key="sn-change-rate",
)

proj = project_cost(corpus_size, chunks_per_doc, float(change_rate))

st.html(
    f"""
    <div style="margin: 1rem 0">
      <div class="sn-cost-row">
        <span class="label">Weekly cost · naive full re-index</span>
        <span>{fmt_cost(proj["naive_weekly"])}</span>
      </div>
      <div class="sn-cost-row green">
        <span class="label">Weekly cost · incremental sync</span>
        <span>{fmt_cost(proj["incremental_weekly"])}</span>
      </div>
    </div>
    """
)

# Altair chart — 12-month cumulative.
df = pd.DataFrame(
    {
        "month": proj["months"] * 2,
        "cost": proj["naive_cum"] + proj["incr_cum"],
        "Strategy": ["Naive full re-index"] * 12 + ["Incremental sync"] * 12,
    }
)
chart = (
    alt.Chart(df)
    .mark_line(strokeWidth=3)
    .encode(
        x=alt.X(
            "month:O",
            title="MONTH",
            axis=alt.Axis(
                grid=True,
                gridColor="#1a1a1a",
                gridOpacity=0.6,
                labelColor="#666",
                titleColor="#666",
                titleFontSize=10,
                titleFont="JetBrains Mono",
                labelFont="JetBrains Mono",
                labelFontSize=10,
                domain=False,
                tickColor="#1a1a1a",
            ),
        ),
        y=alt.Y(
            "cost:Q",
            title="CUMULATIVE COST (USD)",
            axis=alt.Axis(
                format="$,.0f",
                grid=True,
                gridColor="#1a1a1a",
                gridOpacity=0.6,
                labelColor="#666",
                titleColor="#666",
                titleFontSize=10,
                titleFont="JetBrains Mono",
                labelFont="JetBrains Mono",
                labelFontSize=10,
                domain=False,
                tickColor="#1a1a1a",
            ),
        ),
        color=alt.Color(
            "Strategy:N",
            scale=alt.Scale(
                domain=["Naive full re-index", "Incremental sync"],
                range=["#d63d3d", "#5dba6f"],
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
                labelColor="#aaa",
                labelFont="JetBrains Mono",
                labelFontSize=11,
                symbolStrokeWidth=3,
            ),
        ),
    )
    .properties(height=320, background="transparent")
    .configure_view(strokeWidth=0)
)
st.altair_chart(chart, width="stretch")

st.html(
    '<div class="sn-math">'
    "Based on Gemini embedding pricing ($0.025 per 1M tokens) and an "
    "estimate of 550 tokens per chunk. Chart shows cumulative cost over "
    "12 months."
    "</div>"
)


# ---------------------------------------------------------------------------
# Section E — Closing.
# ---------------------------------------------------------------------------
st.html(
    '<div class="sn-section">'
    '<div class="sn-section-eye">THE OPERATIONAL ARGUMENT</div>'
    '<div class="sn-prose">'
    "<p>Chatbase requires manual re-uploads. Every visa rule update means "
    "a staff member exporting docs, uploading to Chatbase's UI, and "
    "waiting. At Talent Taiwan's scale, this is operational debt that "
    "compounds: more pages, more updates, more manual work, more chances "
    "for the index to drift out of sync with the actual website.</p>"
    "<p>The incremental pipeline above runs in minutes on a cron schedule. "
    "New pages are detected by hash. Modified pages re-embed only their "
    "changed chunks. Deleted pages remove their entries cleanly. The "
    "ChromaDB index always reflects the current state of "
    "goldcard.nat.gov.tw — automatically.</p>"
    "<p>Same model. Same content. Same answers. Different operational "
    "profile. This is the kind of difference that doesn't show up in a "
    "chatbot demo, but shows up every Monday morning in the staff's "
    "calendar.</p>"
    "</div>"
    "</div>"
    '<hr class="sn-end-rule">'
)
