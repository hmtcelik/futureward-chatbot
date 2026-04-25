"""Landing page (overview).

Renders the editorial hero, three numbered demo entries, live stats strip,
and footer. Hides the sidebar entirely so the landing surface is
distraction-free; sibling pages restore it via the global theme.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import settings
from src.rag.vector_store import VectorStore
from src.ui.theme import render_sidebar_meta

with st.sidebar:
    render_sidebar_meta()


# ---------------------------------------------------------------------------
# Live state — read once, render as case-study figures.
# ---------------------------------------------------------------------------
manifest_path = Path(settings.manifest_path)
docs_indexed = 0
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        docs_indexed = sum(
            1 for v in manifest.values() if v.get("status", "active") == "active"
        )
    except (OSError, json.JSONDecodeError):
        docs_indexed = 0

chunks_indexed = 0
chroma_error = ""
try:
    chunks_indexed = VectorStore().count()
except Exception as exc:  # noqa: BLE001 - landing page must always render
    chroma_error = str(exc)

api_key_ok = bool(settings.gemini_api_key)
INDEXING_COST_USD = 0.0018  # Phase 2 measurement, presented as case-study figure.

state_warnings: list[str] = []
if not api_key_ok:
    state_warnings.append("Gemini API key missing — set GEMINI_API_KEY in .env.")
if docs_indexed == 0 or chunks_indexed == 0:
    state_warnings.append(
        "Index not built — run <code>python -m scripts.initial_crawl</code>."
    )
elif chroma_error:
    state_warnings.append(f"Vector store error: {chroma_error}")

warning_html = ""
if state_warnings:
    items = "<br>".join(f"· {w}" for w in state_warnings)
    warning_html = f'<div class="state-warn reveal d7">{items}</div>'


def _fmt_dollars(v: float) -> str:
    return f"${v:.4f}"


PAGE = f"""
<style>
.editorial {{
  max-width: 1100px;
  margin: 0 auto;
  padding: 8rem 4rem 5rem 4rem;
}}
@media (max-width: 768px) {{
  .editorial {{ padding: 4rem 1.5rem 3rem 1.5rem; }}
}}
.eyebrow {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.2em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 2rem;
}}
.hero {{
  font-family: "Fraunces", Georgia, serif;
  font-weight: 400;
  font-size: clamp(48px, 7.5vw, 88px);
  line-height: 1.05;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0 0 1.5rem 0;
  max-width: 18ch;
}}
.hero em {{
  font-style: italic;
  color: var(--accent);
  font-weight: 500;
}}
.subtitle {{
  font-family: "Inter Tight", system-ui, sans-serif;
  font-size: 1.32rem;
  line-height: 1.5;
  font-weight: 400;
  color: #b9b3a4;
  max-width: 720px;
  margin: 0 0 2rem 0;
}}
.meta {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.78rem;
  color: var(--muted);
  letter-spacing: 0.02em;
}}
.rule {{ border: 0; border-top: 1px solid var(--rule); margin: 5rem 0; }}
.rule.tight {{ margin: 3rem 0 2rem 0; }}

.demos {{ display: flex; flex-direction: column; }}
.demo-row {{
  display: grid;
  grid-template-columns: 80px 1fr 160px;
  gap: 2.5rem;
  align-items: baseline;
  padding: 3rem 1.25rem;
  margin: 0 -1.25rem;
  border-top: 1px solid var(--rule);
  text-decoration: none;
  color: inherit;
  cursor: pointer;
  transition: background 200ms ease-out;
}}
.demo-row:last-child {{ border-bottom: 1px solid var(--rule); }}
.demo-row:hover {{ background: var(--bg-soft); }}
.demo-row:hover .demo-num {{ color: var(--accent-hot); }}
.demo-row:hover .demo-link {{ text-decoration-thickness: 2px; }}
.demo-row:hover .demo-arrow {{ transform: translateX(4px); }}
.demo-num {{
  font-family: "JetBrains Mono", monospace;
  font-size: 1.1rem;
  font-weight: 400;
  color: var(--accent);
  letter-spacing: 0.05em;
  transition: color 200ms ease-out;
}}
.demo-title {{
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.7rem;
  font-weight: 400;
  line-height: 1.2;
  margin: 0 0 0.7rem 0;
  color: var(--ink);
  letter-spacing: -0.01em;
}}
.demo-desc {{
  font-family: "Inter Tight", system-ui, sans-serif;
  font-size: 0.98rem;
  line-height: 1.55;
  color: #a59f93;
  max-width: 60ch;
  margin: 0;
}}
.demo-link {{
  font-family: "Inter Tight", system-ui, sans-serif;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--accent);
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 4px;
  transition: text-decoration-thickness 180ms ease-out;
  white-space: nowrap;
  justify-self: end;
}}
.demo-arrow {{
  font-family: "Inter Tight", system-ui, sans-serif;
  display: inline-block;
  margin-left: 0.35em;
  transition: transform 200ms ease-out;
}}
@media (max-width: 768px) {{
  .demo-row {{ grid-template-columns: 1fr; gap: 0.75rem; padding: 2rem 1.25rem; }}
  .demo-link {{ justify-self: start; }}
}}
.stats {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 3rem;
  padding-left: calc(80px + 2.5rem);
}}
@media (max-width: 768px) {{ .stats {{ grid-template-columns: 1fr; gap: 2rem; padding-left: 0; }} }}
.stat-num {{
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(40px, 5vw, 56px);
  font-weight: 300;
  color: var(--ink);
  line-height: 1;
  letter-spacing: -0.02em;
  margin: 0 0 0.6rem 0;
}}
.stat-label {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
}}
.state-warn {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.78rem;
  color: #d8a77a;
  margin-bottom: 2.5rem;
  padding: 0.75rem 1rem;
  border-left: 2px solid var(--accent);
  background: #14100d;
}}
.state-warn code {{ background: #1d1813; padding: 1px 6px; border-radius: 3px; font-size: 0.78rem; color: #f5d8a4; }}
.footer {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72rem;
  color: var(--muted);
  line-height: 1.6;
  letter-spacing: 0.02em;
}}
</style>

<div class="editorial">
<div class="eyebrow reveal d0">SKILL TEST · TALENT TAIWAN · APRIL 2026</div>
<h1 class="hero reveal d1">Building a <em>guarded</em> RAG for a<br>government chatbot.</h1>
<p class="subtitle reveal d2">Talent Taiwan currently runs Chatbase on Gemini 3 Flash and is dealing with hallucinations, off-topic drift, and manual content sync. This demo shows what a custom architecture buys, side by side with the baseline.</p>
<div class="meta reveal d3">By Hamit Çelik <span style="color:#444;margin:0 0.5rem">·</span> Submission for Jerry @ Talent Taiwan</div>

<hr class="rule tight reveal d4">

<div class="demos">
<a class="demo-row reveal d4" href="/comparison" target="_self"><div class="demo-num">01</div><div><div class="demo-title">The Comparison</div><p class="demo-desc">Same query, two pipelines. Naive Chatbase-style on the left, guarded RAG on the right. Hallucinations, off-topic drift, and missing citations all fail in the same place — visibly.</p></div><div class="demo-link">Open demo<span class="demo-arrow">→</span></div></a>
<a class="demo-row reveal d5" href="/inside" target="_self"><div class="demo-num">02</div><div><div class="demo-title">Inside the Pipeline</div><p class="demo-desc">Every response with its full trace: input guard, retrieval scores, output guard, token costs, correlation ID. The black box, opened.</p></div><div class="demo-link">Open demo<span class="demo-arrow">→</span></div></a>
<a class="demo-row reveal d6" href="/sync" target="_self"><div class="demo-num">03</div><div><div class="demo-title">Sync at Scale</div><p class="demo-desc">Edit a document. Run incremental sync. Watch only the changed chunks re-embed. Cost projection at 10× and 100× corpus size.</p></div><div class="demo-link">Open demo<span class="demo-arrow">→</span></div></a>
</div>

<hr class="rule reveal d7">

{warning_html}

<div class="stats reveal d8">
<div><div class="stat-num">{docs_indexed if docs_indexed else "—"}</div><div class="stat-label">Documents indexed</div></div>
<div><div class="stat-num">{chunks_indexed if chunks_indexed else "—"}</div><div class="stat-label">Chunks embedded</div></div>
<div><div class="stat-num">{_fmt_dollars(INDEXING_COST_USD)}</div><div class="stat-label">Indexing cost</div></div>
</div>

<hr class="rule reveal d8">

<div class="footer reveal d8">All Gemini calls are real <span style="color:#444;margin:0 0.5rem">·</span> No mocks <span style="color:#444;margin:0 0.5rem">·</span> Source content from goldcard.nat.gov.tw English pages <span style="color:#444;margin:0 0.5rem">·</span> See <code style="background:#141414;padding:1px 6px;border-radius:3px">data/decisions.md</code> for design rationale</div>
</div>
"""

st.html(PAGE)
