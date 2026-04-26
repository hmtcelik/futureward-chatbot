"""Streamlit entry point — navigation router + global theme.

Uses ``st.navigation`` (Streamlit ≥ 1.36) so we can give each page a
human label instead of letting Streamlit auto-derive them from filenames
(which produced the "app" entry the previous build had). Page content
lives in ``views/`` and is imported by Streamlit per route.
"""

from __future__ import annotations

import os

import streamlit as st

# Streamlit Cloud bridges secrets via st.secrets, not via .env. Pipe known
# keys into os.environ BEFORE pydantic-settings reads them in src.config.
try:
    if hasattr(st, "secrets"):
        for _k in ("GEMINI_API_KEY",):
            if _k in st.secrets and not os.environ.get(_k):
                os.environ[_k] = str(st.secrets[_k])
except Exception:  # noqa: BLE001 - secrets file may not exist locally
    pass

from src.config import settings  # noqa: E402
from src.logger import configure_logging  # noqa: E402
from src.ui.theme import inject_global_theme  # noqa: E402

st.set_page_config(
    page_title="Talent Taiwan AI Demo",
    layout="wide",
    page_icon="🇹🇼",
    initial_sidebar_state="expanded",
)
configure_logging(settings.log_level, settings.log_format)
inject_global_theme()

home = st.Page("views/home.py", title="Overview", url_path="", default=True)
comparison = st.Page(
    "views/comparison.py", title="Side-by-side test", url_path="comparison"
)
inside = st.Page("views/inside.py", title="How it works", url_path="inside")
sync = st.Page("views/sync.py", title="Content sync", url_path="sync")

nav = st.navigation([home, comparison, inside, sync])
nav.run()
