"""End-to-end indexing: crawl → extract → chunk → embed → upsert → manifest.

Idempotent: re-running uses content hashes from the manifest to skip
unchanged documents. URLs that vanish from the crawl are marked deleted in
the manifest and removed from the vector store.

Usage::

    python -m scripts.initial_crawl
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from src.config import settings
from src.logger import configure_logging, get_logger
from src.models import Document
from src.rag.chunker import chunk_document
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.scraper.change_detector import Manifest
from src.scraper.crawler import Crawler

logger = get_logger(__name__)


def _filename_for(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16] + ".txt"


def _write_extracted(extracted_dir: Path, doc: Document) -> None:
    payload = {
        "url": doc.url,
        "title": doc.title,
        "content_hash": doc.content_hash,
        "crawled_at": doc.crawled_at.isoformat(),
        "content": doc.content,
    }
    path = extracted_dir / _filename_for(doc.url)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def _index_document(
    doc: Document, embedder: Embedder, store: VectorStore
) -> int:
    """Chunk, embed, and upsert one document. Returns chunk count."""
    chunks = chunk_document(doc)
    if not chunks:
        logger.warning("no_chunks", url=doc.url)
        return 0
    vectors, _ = await embedder.embed(
        [c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT"
    )
    # Stale chunks (e.g. content shrank) are removed before upsert.
    store.delete_by_url(doc.url)
    store.upsert(chunks, vectors)
    return len(chunks)


async def main() -> None:
    """Run the full pipeline once."""
    configure_logging(settings.log_level, settings.log_format)
    logger.info(
        "crawl_start",
        seeds=settings.crawl_seed_urls,
        max_pages=settings.crawl_max_pages,
        max_depth=settings.crawl_max_depth,
    )

    extracted_dir = Path(settings.scraped_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    crawler = Crawler()
    documents = await crawler.crawl()

    manifest = Manifest(settings.manifest_path)
    embedder = Embedder()
    store = VectorStore()

    new_count = changed_count = unchanged_count = 0
    total_chunks = 0
    observed_hashes: dict[str, str] = {}

    # Map each indexed url -> chunk count so we can detect manifest/store drift
    # (e.g. fresh chroma_db but populated manifest).
    indexed_url_counts = {row["url"]: row["chunks"] for row in store.inventory()}

    for doc in documents:
        observed_hashes[doc.url] = doc.content_hash
        existing = manifest.get(doc.url)
        is_new = existing is None or existing.get("status") == "deleted"
        is_changed = (not is_new) and (existing.get("content_hash") != doc.content_hash)
        # If the manifest says we have it but the vector store doesn't, treat
        # it as new — manifest and store have drifted apart.
        missing_in_store = indexed_url_counts.get(doc.url, 0) == 0

        _write_extracted(extracted_dir, doc)

        if is_new or is_changed or missing_in_store:
            n = await _index_document(doc, embedder, store)
            total_chunks += n
            if is_new:
                new_count += 1
            else:
                changed_count += 1
            logger.info(
                "indexed",
                url=doc.url,
                kind="new" if is_new else "changed",
                chunks=n,
            )
        else:
            unchanged_count += 1
            logger.debug("skip_unchanged", url=doc.url)

        manifest.upsert(doc.url, doc.content_hash, doc.title)

    # Reconcile deletions only if crawl produced something — guards against
    # accidental wipes on a totally failed run.
    deleted_count = 0
    if observed_hashes:
        for url in manifest.active_urls() - observed_hashes.keys():
            removed = store.delete_by_url(url)
            manifest.mark_deleted(url)
            deleted_count += 1
            logger.info("marked_deleted", url=url, removed_chunks=removed)

    manifest.save()

    cumulative = embedder.client.cumulative_usage
    logger.info(
        "crawl_done",
        crawled=len(documents),
        new=new_count,
        changed=changed_count,
        unchanged=unchanged_count,
        deleted=deleted_count,
        chunks_indexed=total_chunks,
        vector_store_total=store.count(),
        embedding_tokens=cumulative.total_tokens,
        estimated_cost_usd=round(cumulative.estimated_cost_usd, 6),
    )


if __name__ == "__main__":
    asyncio.run(main())
