"""Incremental sync logic for the Pipeline Sync demo page.

Two flavours:

- ``run_incremental_sync(url)`` — re-reads the extracted file from disk,
  re-chunks if its hash drifted from the manifest, re-embeds. Used during
  the no-network testing phase.
- ``run_real_sync(url)`` — fetches the live URL from ``goldcard.nat.gov.tw``,
  runs the BS4 extractor, compares hashes, re-chunks + re-embeds only on
  diff. The page uses this so the demo is genuinely auditable: real HTTP,
  real extraction, real cost.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import get_logger
from src.models import Document, DocumentStatus, TokenUsage
from src.rag.chunker import chunk_document
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.scraper.change_detector import Manifest
from src.scraper.extractor import extract

logger = get_logger(__name__)

PRICE_PER_1M_EMBED_TOKENS = settings.cost_per_1m_embedding_tokens_usd
TOKENS_PER_CHUNK_ESTIMATE = 550


class StageReport(BaseModel):
    """One sync-pipeline stage output for the UI."""

    name: str
    label: str
    summary: str


class SyncResult(BaseModel):
    """Outcome of one incremental sync run."""

    url: str
    title: str
    no_changes: bool = False
    chunks_before: int = 0
    chunks_after: int = 0
    total_chunks_in_index: int = 0
    delta_chunk_count: int = 0
    incremental_cost_usd: float = 0.0
    full_reindex_cost_usd: float = 0.0
    savings_usd: float = 0.0
    savings_pct: float = 0.0
    usage: TokenUsage = Field(
        default_factory=lambda: TokenUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated_cost_usd=0.0
        )
    )
    stages: list[StageReport] = Field(default_factory=list)


def _filename_for(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16] + ".txt"


def list_indexed_documents() -> list[dict]:
    """Return ``[{url, title, content_hash, last_indexed, chunks}, ...]``
    from the live manifest + ChromaDB inventory.
    """
    manifest_path = Path(settings.manifest_path)
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store_inventory = {row["url"]: row["chunks"] for row in VectorStore().inventory()}
    out = []
    for url, meta in manifest.items():
        if meta.get("status", "active") != "active":
            continue
        out.append(
            {
                "url": url,
                "title": meta.get("title", "(untitled)"),
                "content_hash": meta.get("content_hash", ""),
                "last_indexed": meta.get("last_crawled", ""),
                "chunks": store_inventory.get(url, 0),
            }
        )
    return sorted(out, key=lambda r: r["url"])


def read_extracted(url: str) -> tuple[str, str]:
    """Return ``(content, title)`` for an indexed URL from local disk."""
    path = Path(settings.scraped_dir) / _filename_for(url)
    if not path.exists():
        raise FileNotFoundError(f"No extracted file for {url}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("content", ""), payload.get("title", "")


def read_original(url: str) -> str:
    """Return the snapshotted original content for an indexed URL."""
    path = Path(settings.scraped_dir).parent / "originals" / _filename_for(url)
    if not path.exists():
        # Fall back to current extracted content if no snapshot exists.
        content, _ = read_extracted(url)
        return content
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("content", "")


def write_extracted(url: str, new_content: str, title: str) -> str:
    """Write ``new_content`` back to the extracted file. Returns the new hash."""
    path = Path(settings.scraped_dir) / _filename_for(url)
    new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    payload = {
        "url": url,
        "title": title,
        "content_hash": new_hash,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "content": new_content,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return new_hash


async def run_incremental_sync(url: str) -> SyncResult:
    """Re-embed only the chunks for ``url`` if its hash changed."""
    manifest = Manifest(settings.manifest_path)
    store = VectorStore()
    embedder = Embedder()

    stages: list[StageReport] = []
    total_chunks_in_index = store.count()

    # ---- Stage 01 ----
    stages.append(
        StageReport(
            name="scanning",
            label="SCANNING MANIFEST",
            summary=f"{len(manifest.data)} documents",
        )
    )

    # ---- Stage 02 ----
    content, title = read_extracted(url)
    new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    existing = manifest.get(url) or {}
    old_hash = existing.get("content_hash", "")
    if new_hash == old_hash:
        stages.append(
            StageReport(name="comparing", label="COMPARING HASHES", summary="no changes")
        )
        return SyncResult(
            url=url,
            title=title,
            no_changes=True,
            total_chunks_in_index=total_chunks_in_index,
            stages=stages,
        )

    stages.append(
        StageReport(
            name="comparing",
            label="COMPARING HASHES",
            summary="1 changed · 0 new · 0 deleted",
        )
    )

    # ---- Stage 03 ----
    inventory = {row["url"]: row["chunks"] for row in store.inventory()}
    chunks_before = inventory.get(url, 0)
    doc = Document(
        url=url,
        title=title,
        content=content,
        content_hash=new_hash,
        crawled_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc),
        status=DocumentStatus.ACTIVE,
    )
    new_chunks = chunk_document(doc)
    stages.append(
        StageReport(
            name="rechunking",
            label="RE-CHUNKING MODIFIED DOC",
            summary=(
                f"old chunks: {chunks_before} → new chunks: {len(new_chunks)}"
            ),
        )
    )

    # ---- Stage 04 ----
    if not new_chunks:
        return SyncResult(
            url=url,
            title=title,
            chunks_before=chunks_before,
            total_chunks_in_index=total_chunks_in_index,
            stages=stages,
        )
    vectors, usage = await embedder.embed(
        [c.text for c in new_chunks], task_type="RETRIEVAL_DOCUMENT"
    )
    pct = (
        len(new_chunks) / total_chunks_in_index * 100
        if total_chunks_in_index
        else 0.0
    )
    stages.append(
        StageReport(
            name="embedding",
            label="EMBEDDING DELTA",
            summary=(
                f"{len(new_chunks)} of {total_chunks_in_index} chunks "
                f"({pct:.1f}%) · ${usage.estimated_cost_usd:.6f}"
            ),
        )
    )

    # ---- Stage 05 ----
    store.delete_by_url(url)
    store.upsert(new_chunks, vectors)
    manifest.upsert(url, new_hash, title)
    manifest.save()
    new_total = store.count()
    stages.append(
        StageReport(
            name="upserting",
            label="UPSERTING TO VECTOR STORE",
            summary=(
                f"removed {chunks_before} stale · added {len(new_chunks)} new "
                f"· total {new_total}"
            ),
        )
    )

    # Cost comparison.
    incremental_cost = usage.estimated_cost_usd
    # Naive baseline = cost of re-embedding the whole index instead of just
    # the changed document.
    full_reindex_tokens = total_chunks_in_index * TOKENS_PER_CHUNK_ESTIMATE
    full_reindex_cost = (full_reindex_tokens / 1_000_000) * PRICE_PER_1M_EMBED_TOKENS
    savings = max(0.0, full_reindex_cost - incremental_cost)
    savings_pct = (savings / full_reindex_cost * 100) if full_reindex_cost > 0 else 0.0

    return SyncResult(
        url=url,
        title=title,
        chunks_before=chunks_before,
        chunks_after=len(new_chunks),
        total_chunks_in_index=new_total,
        delta_chunk_count=len(new_chunks),
        incremental_cost_usd=incremental_cost,
        full_reindex_cost_usd=full_reindex_cost,
        savings_usd=savings,
        savings_pct=savings_pct,
        usage=usage,
        stages=stages,
    )


async def fetch_and_extract(url: str) -> dict:
    """HTTP fetch + BS4 extract for a single URL. Real network call."""
    async with httpx.AsyncClient(
        headers={"User-Agent": settings.crawl_user_agent},
        timeout=20.0,
        follow_redirects=True,
    ) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    page = extract(resp.text, url)
    return {
        "content": page.text,
        "title": page.title,
        "content_hash": page.content_hash,
        "status_code": resp.status_code,
        "byte_count": len(resp.text),
    }


async def run_real_sync(url: str) -> SyncResult:
    """Fetch the live URL, extract, compare hash, re-embed only on diff."""
    manifest = Manifest(settings.manifest_path)
    store = VectorStore()
    embedder = Embedder()

    inv = {row["url"]: row["chunks"] for row in store.inventory()}
    chunks_before = inv.get(url, 0)
    total_chunks_in_index = store.count()

    stages: list[StageReport] = []

    # ---- 01 Fetch ----
    fetched = await fetch_and_extract(url)
    stages.append(
        StageReport(
            name="fetching",
            label="FETCHING FROM SOURCE",
            summary=(
                f"HTTP {fetched['status_code']} · "
                f"{fetched['byte_count']:,} bytes"
            ),
        )
    )

    # ---- 02 Extract ----
    text_chars = len(fetched["content"])
    stages.append(
        StageReport(
            name="extracting",
            label="EXTRACTING TEXT",
            summary=(
                f"{text_chars:,} chars · sha256 "
                f"{fetched['content_hash'][:8]}…"
            ),
        )
    )

    # ---- 03 Compare ----
    new_hash = fetched["content_hash"]
    existing = manifest.get(url) or {}
    old_hash = existing.get("content_hash", "")

    full_cost = (
        total_chunks_in_index * TOKENS_PER_CHUNK_ESTIMATE / 1_000_000
    ) * PRICE_PER_1M_EMBED_TOKENS

    if new_hash == old_hash:
        stages.append(
            StageReport(
                name="comparing",
                label="COMPARING HASHES",
                summary="✓ unchanged · skip re-embed",
            )
        )
        # Stages 04 + 05 reported but marked as skipped in the UI.
        return SyncResult(
            url=url,
            title=fetched["title"],
            no_changes=True,
            chunks_before=chunks_before,
            chunks_after=chunks_before,
            total_chunks_in_index=total_chunks_in_index,
            incremental_cost_usd=0.0,
            full_reindex_cost_usd=full_cost,
            savings_usd=full_cost,
            savings_pct=100.0 if full_cost > 0 else 0.0,
            stages=stages,
        )

    stages.append(
        StageReport(
            name="comparing",
            label="COMPARING HASHES",
            summary=f"old {old_hash[:8]}… → new {new_hash[:8]}… · changed",
        )
    )

    # ---- 04 Re-chunk + embed ----
    # Persist the freshly fetched content so future sync runs see it.
    write_extracted(url, fetched["content"], fetched["title"])
    doc = Document(
        url=url,
        title=fetched["title"],
        content=fetched["content"],
        content_hash=new_hash,
        crawled_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc),
        status=DocumentStatus.ACTIVE,
    )
    new_chunks = chunk_document(doc)
    vectors, usage = await embedder.embed(
        [c.text for c in new_chunks], task_type="RETRIEVAL_DOCUMENT"
    )
    stages.append(
        StageReport(
            name="embedding",
            label="RE-CHUNKING + EMBEDDING",
            summary=(
                f"{len(new_chunks)} chunks · "
                f"${usage.estimated_cost_usd:.6f}"
            ),
        )
    )

    # ---- 05 Upsert ----
    store.delete_by_url(url)
    store.upsert(new_chunks, vectors)
    manifest.upsert(url, new_hash, fetched["title"])
    manifest.save()
    new_total = store.count()
    stages.append(
        StageReport(
            name="upserting",
            label="UPSERTING TO VECTOR STORE",
            summary=(
                f"removed {chunks_before} stale · "
                f"added {len(new_chunks)} · total {new_total}"
            ),
        )
    )

    incremental_cost = usage.estimated_cost_usd
    savings = max(0.0, full_cost - incremental_cost)
    savings_pct = (savings / full_cost * 100) if full_cost > 0 else 0.0
    return SyncResult(
        url=url,
        title=fetched["title"],
        chunks_before=chunks_before,
        chunks_after=len(new_chunks),
        total_chunks_in_index=new_total,
        delta_chunk_count=len(new_chunks),
        incremental_cost_usd=incremental_cost,
        full_reindex_cost_usd=full_cost,
        savings_usd=savings,
        savings_pct=savings_pct,
        usage=usage,
        stages=stages,
    )


def project_cost(
    corpus_size: int, chunks_per_doc: int, change_rate_pct: float
) -> dict:
    """Project weekly + 12-month cumulative costs for naive vs incremental sync."""
    total_chunks = corpus_size * chunks_per_doc
    changed_chunks_per_week = total_chunks * (change_rate_pct / 100.0)

    naive_weekly_tokens = total_chunks * TOKENS_PER_CHUNK_ESTIMATE
    incremental_weekly_tokens = changed_chunks_per_week * TOKENS_PER_CHUNK_ESTIMATE

    naive_weekly_cost = (naive_weekly_tokens / 1_000_000) * PRICE_PER_1M_EMBED_TOKENS
    incremental_weekly_cost = (
        incremental_weekly_tokens / 1_000_000
    ) * PRICE_PER_1M_EMBED_TOKENS

    months = list(range(1, 13))
    naive_cum = [naive_weekly_cost * 4 * m for m in months]
    incr_cum = [incremental_weekly_cost * 4 * m for m in months]
    return {
        "naive_weekly": naive_weekly_cost,
        "incremental_weekly": incremental_weekly_cost,
        "naive_annual": naive_weekly_cost * 52,
        "incremental_annual": incremental_weekly_cost * 52,
        "months": months,
        "naive_cum": naive_cum,
        "incr_cum": incr_cum,
    }
