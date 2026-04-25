"""HTML → clean text extraction.

Strips navigation, footers, scripts, styles, and other boilerplate before
returning the visible text content of a page. Conservative on purpose —
losing a few sentences is better than feeding nav menus into the embedder.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

# Tags that almost never carry primary content.
_BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
    "svg",
    "button",
)

# Substrings in `class` / `id` that signal navigation, sidebars, banners.
_BOILERPLATE_SELECTORS = (
    "nav",
    "menu",
    "footer",
    "sidebar",
    "breadcrumb",
    "cookie",
    "banner",
    "social",
    "share",
    "newsletter",
    "subscribe",
    "advert",
)

_WS_RE = re.compile(r"[ \t\f\v]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ExtractedPage:
    """Result of extracting one HTML document."""

    title: str
    text: str
    content_hash: str
    links: list[str]


def extract(html: str, base_url: str) -> ExtractedPage:
    """Parse HTML and return clean title, text, content hash, and outbound links.

    Args:
        html: raw HTML string.
        base_url: URL the HTML came from; used to resolve relative links.

    Returns:
        An ``ExtractedPage`` with normalized text and SHA-256 content hash.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()

    for el in list(soup.find_all(attrs={"class": True})):
        if el.parent is None or not el.attrs:
            continue
        cls = el.attrs.get("class") or []
        if isinstance(cls, str):
            cls = [cls]
        if _looks_boilerplate(" ".join(cls)):
            el.decompose()
    for el in list(soup.find_all(attrs={"id": True})):
        if el.parent is None or not el.attrs:
            continue
        if _looks_boilerplate(el.attrs.get("id") or ""):
            el.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else base_url

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = _normalize_whitespace(text)

    links = _collect_links(soup, base_url)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return ExtractedPage(title=title, text=text, content_hash=content_hash, links=links)


def _looks_boilerplate(token: str) -> bool:
    token_l = token.lower()
    return any(s in token_l for s in _BOILERPLATE_SELECTORS)


def _normalize_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    return _NEWLINE_RE.sub("\n\n", text).strip()


def _collect_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    from urllib.parse import urljoin, urlparse

    out: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        # Drop fragment.
        clean = parsed._replace(fragment="").geturl()
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out
