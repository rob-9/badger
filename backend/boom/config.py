"""
Centralized configuration from environment variables.

All env vars are read once at import time. Call validate_keys() on
startup to fail fast if required API keys are missing.
"""

import os
import sys

# API keys (required)
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")

# Model defaults
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_HAIKU_MODEL: str = os.getenv("CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")
VOYAGE_MODEL: str = os.getenv("VOYAGE_MODEL", "voyage-3")
VOYAGE_CONTEXT_MODEL: str = os.getenv("VOYAGE_CONTEXT_MODEL", "voyage-context-3")
VOYAGE_RERANK_MODEL: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")

# Feature flags
RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() in ("true", "1", "yes")

# RAG pipeline
_relevance_raw = os.getenv("RELEVANCE_THRESHOLD", "0.3")
try:
    RELEVANCE_THRESHOLD: float = float(_relevance_raw)
except ValueError:
    print(f"ERROR: RELEVANCE_THRESHOLD must be a number, got: {_relevance_raw}", file=sys.stderr)
    sys.exit(1)

# Infrastructure
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
VECTOR_STORAGE_DIR: str = os.getenv("VECTOR_STORAGE_DIR", ".data/vectors")

# Allowed directories for EPUB import (comma-separated paths)
EPUB_IMPORT_ALLOWED_DIRS: list[str] = [
    d.strip() for d in os.getenv("EPUB_IMPORT_ALLOWED_DIRS", "~/").split(",") if d.strip()
]

# Maximum input size for book indexing (bytes)
MAX_INDEX_INPUT_SIZE: int = int(os.getenv("MAX_INDEX_INPUT_SIZE", str(10 * 1024 * 1024)))  # 10MB default


def validate_keys() -> None:
    """Exit with a clear error if required API keys are missing."""
    missing = []
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        print("Set them in backend/.env or export them before starting the server.", file=sys.stderr)
        sys.exit(1)
