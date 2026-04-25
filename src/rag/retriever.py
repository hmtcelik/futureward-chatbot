"""Top-k retriever combining the embedder and the vector store.

Embeds a query, queries the store, and returns ``RetrievedChunk`` objects
with cosine similarity in [0, 1]. A reranking hook is exposed for future
work (e.g. a cross-encoder pass) but is a no-op by default.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from src.config import settings
from src.logger import get_logger
from src.models import Chunk, RetrievedChunk
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

logger = get_logger(__name__)

RerankFn = Callable[[str, list[RetrievedChunk]], Awaitable[list[RetrievedChunk]]]


class Retriever:
    """Embed-then-search retriever with an optional reranker."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: RerankFn | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.reranker = reranker

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to ``query``.

        Args:
            query: user question.
            top_k: number of chunks to return (defaults to settings).

        Returns:
            Ordered list of RetrievedChunk objects, highest similarity first.
            Empty list when the store is empty.
        """
        k = top_k or settings.top_k
        if self.store.count() == 0:
            logger.warning("vector_store_empty")
            return []

        vectors, _ = await self.embedder.embed([query], task_type="RETRIEVAL_QUERY")
        if not vectors:
            return []

        raw_hits = self.store.query(vectors[0], top_k=k)
        out: list[RetrievedChunk] = []
        for rank, hit in enumerate(raw_hits):
            similarity = max(0.0, min(1.0, 1.0 - hit["distance"]))
            meta = hit["metadata"] or {}
            chunk = Chunk(
                chunk_id=hit["id"],
                document_url=meta.get("document_url", ""),
                document_title=meta.get("document_title", ""),
                text=hit["document"] or "",
                chunk_hash=meta.get("chunk_hash", ""),
                position=int(meta.get("position", 0)),
                metadata={"tokens": int(meta.get("tokens", 0))},
            )
            out.append(RetrievedChunk(chunk=chunk, similarity_score=similarity, rank=rank))

        if self.reranker is not None:
            out = await self.reranker(query, out)

        logger.info(
            "retrieve",
            query_len=len(query),
            results=len(out),
            top_score=(out[0].similarity_score if out else 0.0),
        )
        return out
