"""Polite async BFS crawler for Talent Taiwan / Gold Card sites.

Constraints (all driven by ``src.config.settings``):
    - Respects each domain's ``robots.txt``.
    - Caps total pages and depth.
    - Inserts a per-domain delay between requests (default 1 req/sec).
    - Restricted to an allow-list of domains.

Returns ``Document`` objects ready for chunking. Raw HTML is also written to
disk (under ``settings.raw_html_dir``) for offline debugging; it is excluded
from the committed repo via ``.gitignore``.
"""

from __future__ import annotations

import asyncio
import hashlib
import urllib.robotparser
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

from src.config import settings
from src.logger import get_logger
from src.models import Document, DocumentStatus
from src.scraper.extractor import extract

logger = get_logger(__name__)


class Crawler:
    """Async BFS crawler. One instance per crawl session."""

    def __init__(
        self,
        seed_urls: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        request_delay_seconds: float | None = None,
        user_agent: str | None = None,
        path_prefix: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.seed_urls = seed_urls or settings.crawl_seed_urls
        self.allowed_domains = set(allowed_domains or settings.crawl_allowed_domains)
        self.max_pages = max_pages or settings.crawl_max_pages
        self.max_depth = max_depth or settings.crawl_max_depth
        self.request_delay_seconds = (
            request_delay_seconds
            if request_delay_seconds is not None
            else settings.crawl_request_delay_seconds
        )
        self.user_agent = user_agent or settings.crawl_user_agent
        self.path_prefix = (
            path_prefix if path_prefix is not None else settings.crawl_path_prefix
        )
        self.timeout_seconds = timeout_seconds

        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_fetch_at: dict[str, float] = defaultdict(float)

    async def crawl(self) -> list[Document]:
        """Run the full crawl. Returns one Document per successfully fetched URL."""
        documents: list[Document] = []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((u, 0) for u in self.seed_urls)

        # raw_html dir for debug snapshots
        raw_dir = Path(settings.raw_html_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_seconds,
            follow_redirects=True,
        ) as client:
            while queue and len(documents) < self.max_pages:
                url, depth = queue.popleft()

                if url in visited:
                    continue
                visited.add(url)

                if not self._domain_allowed(url):
                    logger.debug("domain_skip", url=url)
                    continue

                if not self._path_allowed(url):
                    logger.debug("path_skip", url=url)
                    continue

                if not await self._robots_allow(client, url):
                    logger.info("robots_disallow", url=url)
                    continue

                await self._respect_rate_limit(url)

                try:
                    resp = await client.get(url)
                except httpx.HTTPError as exc:
                    logger.warning("fetch_error", url=url, error=str(exc))
                    continue

                if resp.status_code != 200:
                    logger.info("non_200", url=url, status=resp.status_code)
                    continue

                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype.lower():
                    logger.debug("non_html_skip", url=url, content_type=ctype)
                    continue

                html = resp.text
                self._write_raw(raw_dir, url, html)

                try:
                    page = extract(html, url)
                except Exception as exc:  # noqa: BLE001 - log + skip is intentional
                    logger.warning("extract_error", url=url, error=str(exc))
                    continue

                if len(page.text) < 200:
                    logger.debug("text_too_short", url=url, length=len(page.text))
                    continue

                now = datetime.now(timezone.utc)
                documents.append(
                    Document(
                        url=url,
                        title=page.title,
                        content=page.text,
                        content_hash=page.content_hash,
                        crawled_at=now,
                        last_modified=now,
                        status=DocumentStatus.ACTIVE,
                        metadata={"depth": depth, "fetch_status": resp.status_code},
                    )
                )
                logger.info(
                    "page_indexed",
                    url=url,
                    depth=depth,
                    title=page.title[:80],
                    chars=len(page.text),
                    indexed=len(documents),
                )

                if depth < self.max_depth:
                    for link in page.links:
                        if (
                            link not in visited
                            and self._domain_allowed(link)
                            and self._path_allowed(link)
                        ):
                            queue.append((link, depth + 1))

        logger.info(
            "crawl_complete", indexed=len(documents), visited=len(visited), queued=len(queue)
        )
        return documents

    def _domain_allowed(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

    def _path_allowed(self, url: str) -> bool:
        """Restrict crawl to the configured locale prefix (e.g. ``/en/``).

        Accepts both the bare prefix without trailing slash and any path
        starting with the prefix. Empty prefix disables the filter.
        """
        if not self.path_prefix:
            return True
        path = urlparse(url).path or "/"
        bare = self.path_prefix.rstrip("/")
        return path == bare or path.startswith(self.path_prefix)

    async def _robots_allow(self, client: httpx.AsyncClient, url: str) -> bool:
        host = urlparse(url).netloc
        scheme = urlparse(url).scheme or "https"
        base = f"{scheme}://{host}"

        if host not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                resp = await client.get(f"{base}/robots.txt")
                if resp.status_code == 200:
                    rp.parse(resp.text.splitlines())
                else:
                    # No robots.txt → permissive default per RFC.
                    rp.parse([])
            except httpx.HTTPError as exc:
                logger.warning("robots_fetch_error", host=host, error=str(exc))
                rp.parse([])
            self._robots_cache[host] = rp

        return self._robots_cache[host].can_fetch(self.user_agent, url)

    async def _respect_rate_limit(self, url: str) -> None:
        host = urlparse(url).netloc
        loop = asyncio.get_event_loop()
        now = loop.time()
        last = self._last_fetch_at[host]
        wait = (last + self.request_delay_seconds) - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_fetch_at[host] = loop.time()

    @staticmethod
    def _write_raw(raw_dir: Path, url: str, html: str) -> None:
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        (raw_dir / f"{digest}.html").write_text(html, encoding="utf-8")
