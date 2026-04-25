"""Pydantic data contracts shared across the codebase.

Every value passed between modules is one of these models. No raw dicts.
Touching any field here is a breaking change — review carefully.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle state of a scraped document in the index."""

    ACTIVE = "active"
    DELETED = "deleted"


class Document(BaseModel):
    """A single scraped page after HTML cleanup."""

    url: str
    title: str
    content: str
    content_hash: str
    crawled_at: datetime
    last_modified: datetime
    status: DocumentStatus = DocumentStatus.ACTIVE
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrieval-sized slice of a Document, ready for embedding."""

    chunk_id: str
    document_url: str
    document_title: str
    text: str
    chunk_hash: str
    position: int
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with its similarity score."""

    chunk: Chunk
    similarity_score: float
    rank: int


class GuardDecisionType(str, Enum):
    """Outcomes a guard can emit."""

    PASS = "pass"
    REFUSE_OFF_TOPIC = "refuse_off_topic"
    REFUSE_LOW_CONFIDENCE = "refuse_low_confidence"
    REFUSE_NOT_GROUNDED = "refuse_not_grounded"


class GuardDecision(BaseModel):
    """Result of a single guard invocation (input or output)."""

    decision: GuardDecisionType
    reason: str
    confidence: float
    judge_model: str
    latency_ms: int


class TokenUsage(BaseModel):
    """Token counts and estimated cost for one LLM call (or aggregate)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class ChatResponse(BaseModel):
    """Full response object returned by either chatbot variant."""

    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    input_guard: GuardDecision | None = None
    output_guard: GuardDecision | None = None
    token_usage: TokenUsage
    latency_ms: int
    correlation_id: str
    chatbot_variant: str  # "naive" | "guarded"


class EvalQuestion(BaseModel):
    """One ground-truth Q&A entry from the eval set."""

    id: str
    question: str
    category: str  # "visa" | "gold_card" | "tax" | "off_topic" | "edge_case"
    expected_behavior: str  # "answer" | "refuse" | "escalate"
    ground_truth_urls: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Outcome of running one EvalQuestion through one chatbot variant."""

    question_id: str
    chatbot_variant: str
    response: ChatResponse
    recall_at_5: float | None = None
    faithfulness: float | None = None
    refusal_correct: bool | None = None
