"""Manifest of crawled URLs → content hash + timestamps.

The manifest is a single JSON file (path from settings.manifest_path) with
the shape::

    {
        "https://...": {
            "content_hash": "<sha256>",
            "last_crawled": "<iso datetime>",
            "last_modified": "<iso datetime>",
            "title": "<page title>",
            "status": "active" | "deleted"
        },
        ...
    }

Hash-based change detection: if the SHA-256 of the cleaned text differs
between crawls, the document is considered changed and its chunks must be
re-embedded. This is the core of the incremental-sync story (Task 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.logger import get_logger
from src.models import DocumentStatus

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChangeReport:
    """Summary of a manifest reconciliation pass."""

    new_urls: list[str]
    changed_urls: list[str]
    unchanged_urls: list[str]
    deleted_urls: list[str]

    @property
    def has_changes(self) -> bool:
        """True if anything was added, modified, or removed."""
        return bool(self.new_urls or self.changed_urls or self.deleted_urls)


class Manifest:
    """JSON-backed URL → metadata table.

    Reads on init, writes on ``save()``. Not thread-safe; callers must
    serialize writes (the crawler is single-process so this is fine).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("manifest_corrupt", path=str(self.path), error=str(exc))
                self.data = {}
        else:
            self.data = {}

    def save(self) -> None:
        """Write the manifest atomically to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, url: str) -> dict | None:
        """Return metadata for ``url`` or None if not indexed."""
        return self.data.get(url)

    def upsert(
        self,
        url: str,
        content_hash: str,
        title: str,
        status: DocumentStatus = DocumentStatus.ACTIVE,
    ) -> bool:
        """Insert or update an entry. Returns True if content actually changed."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self.data.get(url)
        changed = (existing is None) or (existing.get("content_hash") != content_hash)
        last_modified = now if changed else existing.get("last_modified", now)
        self.data[url] = {
            "content_hash": content_hash,
            "last_crawled": now,
            "last_modified": last_modified,
            "title": title,
            "status": status.value,
        }
        return changed

    def mark_deleted(self, url: str) -> None:
        """Mark a URL as deleted without removing its history."""
        if url in self.data:
            self.data[url]["status"] = DocumentStatus.DELETED.value
            self.data[url]["last_modified"] = datetime.now(timezone.utc).isoformat()

    def active_urls(self) -> set[str]:
        """All URLs currently in ACTIVE status."""
        return {
            u for u, meta in self.data.items() if meta.get("status") == DocumentStatus.ACTIVE.value
        }

    def reconcile(self, observed_hashes: dict[str, str]) -> ChangeReport:
        """Compare a fresh crawl's URL→hash map against the manifest.

        Args:
            observed_hashes: ``{url: content_hash}`` from the latest crawl.

        Returns:
            ChangeReport classifying each URL as new/changed/unchanged/deleted.
            Caller is responsible for actually re-embedding changed/new chunks
            and calling ``mark_deleted`` for vanished URLs.
        """
        new_urls: list[str] = []
        changed_urls: list[str] = []
        unchanged_urls: list[str] = []

        for url, h in observed_hashes.items():
            existing = self.data.get(url)
            if existing is None or existing.get("status") == DocumentStatus.DELETED.value:
                new_urls.append(url)
            elif existing.get("content_hash") != h:
                changed_urls.append(url)
            else:
                unchanged_urls.append(url)

        deleted_urls = [u for u in self.active_urls() if u not in observed_hashes]

        return ChangeReport(
            new_urls=new_urls,
            changed_urls=changed_urls,
            unchanged_urls=unchanged_urls,
            deleted_urls=deleted_urls,
        )
