"""Pipeline sync (stub) — full implementation lands later."""

import streamlit as st

from src.ui.theme import render_sidebar_meta

with st.sidebar:
    render_sidebar_meta()

st.html(
    """
    <div class="subpage">
      <div class="subpage-eyebrow reveal d0">DEMO 03 · COMING NEXT</div>
      <h1 class="subpage-title reveal d1">Sync at Scale</h1>
      <p class="subpage-lead reveal d2">Edit a doc, run incremental sync, project cost at 10× and 100× corpus size. Full implementation lands in a later iteration.</p>
    </div>
    """
)
