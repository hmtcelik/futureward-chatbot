"""Shared editorial theme — one place, used by every page.

Streamlit reruns the entry script on every interaction, so we inject the
global CSS once per run from ``app.py``. Individual views call
``inject_hide_sidebar_css`` if they want a chrome-free landing surface.
"""

from __future__ import annotations

import streamlit as st


_GLOBAL_THEME = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,500;1,9..144,400;1,9..144,500&family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0a0a;
  --bg-soft: #0e0e0e;
  --ink: #f5f1e8;
  --muted: #888;
  --rule: #1a1a1a;
  --accent: #d63d3d;
  --accent-hot: #e85555;
}

/* Hide Streamlit chrome we never want — but leave the toolbar container
   alone so the sidebar reopen button stays mounted/visible. */
header[data-testid="stHeader"] {
  background: transparent;
  height: auto;
}
[data-testid="stToolbar"] [data-testid="stMainMenu"],
[data-testid="stToolbar"] [data-testid="stDeployButton"],
[data-testid="stToolbar"] [data-testid="stStatusWidget"],
[data-testid="stStatusWidget"],
[data-testid="stConnectionStatus"],
div[role="status"]:not(.cmp-generating):not([class*="cmp-"]) { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* Sidebar reopen button — force visible across Streamlit versions. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
button[data-testid="stExpandSidebarButton"],
button[data-testid="stBaseButton-headerNoPadding"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  color: var(--muted) !important;
}
[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover,
button[data-testid="stExpandSidebarButton"]:hover {
  color: var(--ink) !important;
}

/* Body / app shell. */
body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--ink);
  font-family: "Inter Tight", system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] > .main { padding-top: 0; }
[data-testid="stMainBlockContainer"] {
  padding: 0 !important;
  max-width: 100% !important;
}

/* Sidebar editorial skin (active when not hidden). Width only forced
   while expanded so Streamlit's collapse/reopen toggle stays functional. */
section[data-testid="stSidebar"] {
  background: var(--bg) !important;
  border-right: 1px solid var(--rule);
}
section[data-testid="stSidebar"][aria-expanded="true"] {
  width: 240px !important;
  min-width: 240px !important;
}
/* Reopen-arrow chrome: keep visible + tone it down. */
[data-testid="stSidebarCollapsedControl"] button,
button[kind="header"][data-testid="stBaseButton-headerNoPadding"] {
  color: var(--muted) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  background: transparent;
  padding: 1rem 1rem 0.5rem 1rem;
  border-bottom: none;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  padding: 1rem 1rem 1.5rem 1rem;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
  padding: 0.25rem 0.5rem 1rem 0.5rem;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul {
  padding: 0;
  margin: 0;
  gap: 2px;
  display: flex;
  flex-direction: column;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a {
  font-family: "JetBrains Mono", monospace !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.03em !important;
  color: var(--muted) !important;
  border-radius: 6px;
  padding: 0.55rem 0.75rem !important;
  margin: 0 !important;
  background: transparent;
  transition: background 160ms ease, color 160ms ease;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a span {
  color: inherit !important;
  font-family: inherit !important;
  font-size: inherit !important;
  letter-spacing: inherit !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a:hover {
  color: var(--ink) !important;
  background: var(--bg-soft);
  text-decoration: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[aria-current="page"] {
  color: var(--ink) !important;
  background: #1a1a1a !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[aria-current="page"] span {
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[aria-current="page"]:hover {
  background: #1f1f1f !important;
}
/* Mute the icon column (we keep it minimal). */
section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
  color: inherit !important;
  opacity: 0.55;
  font-size: 1rem !important;
}

/* Sidebar reset-chat link — same on every page. */
.sb-reset-link {
  display: block;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.78rem;
  color: #9ecbff;
  padding: 0.55rem 0.75rem;
  margin: 1.25rem 0.5rem 0 0.5rem;
  text-decoration: none;
  letter-spacing: 0.03em;
  border-top: 1px solid #1a1a1a;
  padding-top: 1rem;
  transition: color 160ms ease;
}
.sb-reset-link:hover { color: #cfe6ff; }

/* Sidebar footer eyebrow + meta block. */
.sidebar-eyebrow {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.62rem;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  padding: 0 0.75rem;
  margin: 1.25rem 0 0.6rem 0;
}
.sidebar-meta {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  color: var(--muted);
  padding: 0 0.75rem;
  line-height: 1.5;
}

/* Reusable reveal animations. */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
.reveal { opacity: 0; animation: fadeUp 400ms ease-out forwards; }
.d0 { animation-delay: 0ms; }
.d1 { animation-delay: 80ms; }
.d2 { animation-delay: 160ms; }
.d3 { animation-delay: 240ms; }
.d4 { animation-delay: 320ms; }
.d5 { animation-delay: 400ms; }
.d6 { animation-delay: 480ms; }
.d7 { animation-delay: 560ms; }
.d8 { animation-delay: 640ms; }

/* Reusable subpage chrome. */
.subpage {
  max-width: 1100px;
  margin: 0 auto;
  padding: 3.5rem 3rem 4rem 3rem;
}
@media (max-width: 768px) {
  .subpage { padding: 2rem 1.25rem 3rem 1.25rem; }
}
.subpage-eyebrow {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.2em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 1.25rem;
}
.subpage-title {
  font-family: "Fraunces", Georgia, serif;
  font-weight: 400;
  font-size: clamp(34px, 4.5vw, 52px);
  line-height: 1.1;
  letter-spacing: -0.01em;
  color: var(--ink);
  margin: 0 0 1rem 0;
}
.subpage-lead {
  font-family: "Inter Tight", system-ui, sans-serif;
  font-size: 1.08rem;
  line-height: 1.55;
  color: #b9b3a4;
  max-width: 720px;
  margin: 0 0 2.5rem 0;
}
</style>
"""


_HIDE_SIDEBAR = """
<style>
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
"""


def inject_global_theme() -> None:
    """Inject the editorial palette + sidebar skin. Call once per script run."""
    st.html(_GLOBAL_THEME)


def hide_sidebar() -> None:
    """Per-page CSS to hide the sidebar entirely (used by the landing page)."""
    st.html(_HIDE_SIDEBAR)


def render_sidebar_meta() -> None:
    """Editorial meta block under the auto-rendered nav."""
    st.html(
        """
        <div class="sidebar-eyebrow">SKILL TEST</div>
        <div class="sidebar-meta">
          Talent Taiwan<br>
          April 2026
        </div>
        """
    )
