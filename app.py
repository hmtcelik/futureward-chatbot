"""Streamlit entry point — navigation router + global theme.

Uses ``st.navigation`` (Streamlit ≥ 1.36) so we can give each page a
human label instead of letting Streamlit auto-derive them from filenames
(which produced the "app" entry the previous build had). Page content
lives in ``views/`` and is imported by Streamlit per route.
"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.logger import configure_logging
from src.ui.theme import inject_global_theme

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
    "views/comparison.py", title="The Comparison", url_path="comparison"
)
inside = st.Page(
    "views/inside.py", title="Inside the Pipeline", url_path="inside"
)
sync = st.Page("views/sync.py", title="Sync at Scale", url_path="sync")

nav = st.navigation([home, comparison, inside, sync])
nav.run()
