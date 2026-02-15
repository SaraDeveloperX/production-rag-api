"""Centralised configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file before anything reads os.environ
load_dotenv()


class Settings:
    """Application settings sourced from environment variables."""

    # API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "")

    # Models
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TOP_K: int = int(os.getenv("TOP_K", "4"))

    # Safety
    BLOCKLIST: list[str] = [
        t.strip().lower()
        for t in os.getenv("BLOCKLIST", "").split(",")
        if t.strip()
    ]

    # Rate limiting
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "5/minute")

    # Auth
    API_KEY: str = os.getenv("API_KEY", "change_me")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"

    # Pinecone
    NAMESPACE: str = os.getenv("NAMESPACE", "dev")

    # Data
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))

    # Evaluation
    EVAL_MODE: bool = os.getenv("EVAL_MODE", "false").lower() == "true"

    # Cache
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "0"))  # 0 = disabled
    CACHE_MAX_ITEMS: int = int(os.getenv("CACHE_MAX_ITEMS", "256"))


settings = Settings()
