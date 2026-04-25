"""Paragraph-aware token chunker.

Splits a Document's cleaned text into roughly fixed-size chunks (token-based,
not character-based) with a configurable overlap between consecutive chunks.
Paragraph boundaries are preserved when possible so retrieved chunks read as
coherent passages, which materially helps the output guard's grounding check.

Token counting uses tiktoken's ``cl100k_base`` encoding. It is not the exact
Gemini tokenizer but is close enough for chunking; the ~5–10% drift would not
change the retrieval quality of 500-token chunks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import tiktoken

from src.config import settings
from src.logger import get_logger
from src.models import Chunk, Document

logger = get_logger(__name__)

_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_oversized_paragraph(paragraph: str, max_tokens: int) -> list[str]:
    """Split a paragraph that exceeds the chunk budget into sentence groups."""
    sentences = _SENTENCE_SPLIT.split(paragraph) if paragraph else []
    if not sentences:
        return []

    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for s in sentences:
        s_tokens = _count_tokens(s)
        if buf and buf_tokens + s_tokens > max_tokens:
            out.append(" ".join(buf).strip())
            buf, buf_tokens = [s], s_tokens
        else:
            buf.append(s)
            buf_tokens += s_tokens
    if buf:
        out.append(" ".join(buf).strip())
    return out


def _tail_tokens(text: str, n_tokens: int) -> str:
    """Return the trailing ``n_tokens`` tokens of ``text`` as a string."""
    if n_tokens <= 0 or not text:
        return ""
    ids = _ENCODING.encode(text)
    if len(ids) <= n_tokens:
        return text
    return _ENCODING.decode(ids[-n_tokens:])


def chunk_document(
    doc: Document,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Split ``doc`` into Chunk objects.

    Args:
        doc: source Document with cleaned text.
        chunk_size_tokens: target tokens per chunk (defaults to settings).
        overlap_tokens: tokens carried from the end of one chunk into the
            start of the next (defaults to settings).

    Returns:
        Ordered list of Chunk objects. Each chunk has a stable ``chunk_id``
        of the form ``"<sha256(url)[:16]>::<position>"`` so re-running on the
        same document with the same settings yields the same ids.
    """
    chunk_size = chunk_size_tokens or settings.chunk_size_tokens
    overlap = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(doc.content) if p.strip()]

    pieces: list[str] = []
    for p in paragraphs:
        if _count_tokens(p) > chunk_size:
            pieces.extend(_split_oversized_paragraph(p, chunk_size))
        else:
            pieces.append(p)

    chunks_text: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for piece in pieces:
        p_tokens = _count_tokens(piece)
        if buf and buf_tokens + p_tokens > chunk_size:
            chunks_text.append("\n\n".join(buf).strip())
            tail = _tail_tokens(chunks_text[-1], overlap)
            buf = [tail, piece] if tail else [piece]
            buf_tokens = _count_tokens("\n\n".join(buf))
        else:
            buf.append(piece)
            buf_tokens += p_tokens
    if buf:
        chunks_text.append("\n\n".join(buf).strip())

    url_digest = hashlib.sha256(doc.url.encode("utf-8")).hexdigest()[:16]
    out: list[Chunk] = []
    for idx, text in enumerate(chunks_text):
        if not text:
            continue
        out.append(
            Chunk(
                chunk_id=f"{url_digest}::{idx}",
                document_url=doc.url,
                document_title=doc.title,
                text=text,
                chunk_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                position=idx,
                metadata={
                    "tokens": _count_tokens(text),
                    "document_hash": doc.content_hash,
                },
            )
        )

    logger.debug(
        "chunked",
        url=doc.url,
        chunks=len(out),
        avg_tokens=(sum(c.metadata["tokens"] for c in out) // max(len(out), 1)) if out else 0,
    )
    return out


def chunk_documents(docs: Iterable[Document]) -> list[Chunk]:
    """Apply ``chunk_document`` over an iterable, flattening results."""
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out
