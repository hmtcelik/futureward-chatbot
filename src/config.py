"""Typed application settings loaded from env vars / .env file."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    All configuration is loaded from environment variables (or a .env file).
    No magic numbers anywhere else in the codebase — every tunable lives here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API
    gemini_api_key: str = Field(
        default="",
        description="Gemini API key. Required at runtime; empty default allows import-time access.",
    )

    # Models
    # Spec target is "Gemini 3 Flash" — the real published id is
    # `gemini-3-flash-preview` as of 2026-04. `gemini-flash-latest` is a
    # safe fallback (alias to the current stable flash).
    llm_model: str = "gemini-3-flash-preview"
    judge_model: str = "gemini-3-flash-preview"
    embedding_model: str = "gemini-embedding-001"

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.55

    # Chunking
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Crawler
    crawl_max_pages: int = 50
    crawl_max_depth: int = 3
    crawl_request_delay_seconds: float = 1.0
    crawl_user_agent: str = "TalentTaiwanDemoBot/0.1 (educational, polite)"
    crawl_seed_urls: list[str] = [
        "https://goldcard.nat.gov.tw/en/",
    ]
    crawl_allowed_domains: list[str] = [
        "goldcard.nat.gov.tw",
    ]
    # Only follow URLs whose path starts with this prefix (or equals it
    # without trailing slash). Keeps the index English-only and avoids
    # crawling /zh/, /zh-tw/, /ja/ localized siblings.
    crawl_path_prefix: str = "/en/"

    # Storage
    chroma_persist_dir: str = "data/chroma_db"
    chroma_collection_name: str = "talent_taiwan"
    scraped_dir: str = "data/scraped/extracted"
    raw_html_dir: str = "data/scraped/raw_html"
    manifest_path: str = "data/manifest.json"

    # Costs (per 1M tokens; used for telemetry display only)
    cost_per_1m_input_tokens_usd: float = 0.075
    cost_per_1m_output_tokens_usd: float = 0.30
    cost_per_1m_embedding_tokens_usd: float = 0.025

    # Logging
    log_level: str = "INFO"
    log_format: str = "console"  # "json" | "console"


settings = Settings()
