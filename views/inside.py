"""Inside the pipeline (stub) — full implementation lands later."""

import streamlit as st

from src.ui.theme import render_sidebar_meta

with st.sidebar:
    render_sidebar_meta()

st.html(
    """
    <div class="subpage">
      <div class="subpage-eyebrow reveal d0">DEMO 02 · COMING NEXT</div>
      <h1 class="subpage-title reveal d1">Inside the Pipeline</h1>
      <p class="subpage-lead reveal d2">Per-response trace: input guard, retrieval scores, output guard, token costs, correlation ID. Full implementation lands in a later iteration.</p>
    </div>
    """
)
