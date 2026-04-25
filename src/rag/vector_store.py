"""ChromaDB persistent client wrapper.

Wraps a single Chroma collection and exposes ``upsert``, ``delete_by_url``,
``query``, and inventory helpers. Embeddings are provided explicitly by the
caller; the collection is configured with ``embedding_function=None`` because
we use Gemini, not Chroma's bundled model.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.logger import get_logger
from src.models import Chunk

logger = get_logger(__name__)

# Chroma metadata can hold scalars only; we keep the chunk text in the
# `documents` field and store these scalar fields per chunk.
_METADATA_FIELDS = ("document_url", "document_title", "position", "chunk_hash", "tokens")


class VectorStore:
    """Thin async-friendly wrapper around a Chroma collection."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ):
        path = Path(persist_dir or settings.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        logger.debug(
            "vector_store_ready",
            path=str(path),
            collection=self._collection.name,
            count=self._collection.count(),
        )

    @property
    def name(self) -> str:
        """Underlying Chroma collection name."""
        return self._collection.name

    def count(self) -> int:
        """Total number of stored chunks."""
        return self._collection.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks with their embeddings.

        Args:
            chunks: Chunk objects to store.
            embeddings: parallel list of embedding vectors.
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must align in length")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_url": c.document_url,
                "document_title": c.document_title,
                "position": c.position,
                "chunk_hash": c.chunk_hash,
                "tokens": int(c.metadata.get("tokens", 0)),
            }
            for c in chunks
        ]
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("vector_upsert", n=len(chunks), collection=self._collection.name)

    def delete_by_url(self, url: str) -> int:
        """Delete every chunk whose ``document_url`` equals ``url``.

        Returns:
            Number of chunks removed.
        """
        existing = self._collection.get(where={"document_url": url})
        ids = existing.get("ids", []) if existing else []
        if ids:
            self._collection.delete(ids=ids)
            logger.info("vector_delete", url=url, removed=len(ids))
        return len(ids)

    def query(
        self,
        embedding: list[float],
        top_k: int | None = None,
    ) -> list[dict]:
        """Return the ``top_k`` most similar chunks for a query embedding.

        Each result is a raw dict with keys ``id``, ``document``, ``metadata``,
        and ``distance``. Distance is cosine distance (lower = more similar).
        Conversion to similarity score is the retriever's job.
        """
        k = top_k or settings.top_k
        result = self._collection.query(query_embeddings=[embedding], n_results=k)

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        out: list[dict] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=False):
            out.append(
                {
                    "id": chunk_id,
                    "document": doc,
                    "metadata": meta,
                    "distance": float(dist),
                }
            )
        return out

    def inventory(self) -> list[dict]:
        """Return one row per (document_url, chunk_count) for diagnostics."""
        all_meta = self._collection.get(include=["metadatas"])
        metas = all_meta.get("metadatas", []) or []
        per_url: dict[str, dict] = {}
        for m in metas:
            url = m.get("document_url")
            if not url:
                continue
            entry = per_url.setdefault(
                url, {"url": url, "title": m.get("document_title", ""), "chunks": 0}
            )
            entry["chunks"] += 1
        return sorted(per_url.values(), key=lambda r: r["url"])
