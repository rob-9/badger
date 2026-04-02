# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Reading app with RAG-powered Q&A. Python (FastAPI) backend + Next.js 14 (App Router) frontend. Users upload EPUB/PDF, select text, ask questions answered by Claude using Voyage AI embeddings.

## Commands

```bash
./dev.sh                         # Run both backend + frontend
cd frontend && npm run dev       # Frontend only (port 3000)
cd backend && uvicorn badger.api.server:app --reload --port 8000  # Backend only
./smoke-test.sh                  # Quick integration test (~$0.002, real APIs with Haiku)
```

## Environment

Set in `backend/.env`:
```bash
ANTHROPIC_API_KEY=sk-ant-...  # Claude API (required)
VOYAGE_API_KEY=pa-...         # Voyage AI embeddings (required)
CLAUDE_MODEL=claude-sonnet-4-20250514  # optional override
VOYAGE_MODEL=voyage-3                   # optional override
DEV_MODE=lite                          # optional: cheap Haiku + reduced agent for testing
CORS_ORIGINS=http://localhost:3000      # optional override
VECTOR_STORAGE_DIR=.data/vectors        # legacy JSON path (used for auto-migration)
QDRANT_URL=                             # empty = embedded mode (no server needed)
QDRANT_API_KEY=                         # for Qdrant Cloud (optional)
QDRANT_STORAGE_DIR=.data/qdrant         # embedded Qdrant data directory
```

## Architecture

**Backend** (`backend/badger/`):
- `config.py` — Centralized env var config with `validate_keys()` startup check
- `api/server.py` — FastAPI app with lifespan init, CORS, 3 endpoints + health checks
- `core/rag.py` — RAG service: Voyage embeddings + Claude generation, position-aware context
- `core/vector_store.py` — Qdrant-backed vector storage (embedded mode for dev, remote for prod)
- `core/chunker.py` — Text chunking (2000 char chunks, 200 overlap, sentence-aware boundaries)

**Frontend** (`frontend/src/`):
- `lib/api.ts` — API client for Python backend (all fetch calls go to localhost:8000)
- `app/` — Next.js App Router pages
- `components/` — EpubReader, ChatPanel, QuestionPopup, FileUploader, BookHistory

**RAG Pipeline**: `chunker.py` (2000 char chunks) → `rag.py` (Voyage embeddings) → `vector_store.py` (Qdrant cosine search) → `rag.py` (Claude `claude-sonnet-4-20250514`)

**API Endpoints** (Python backend on port 8000):
- `POST /api/rag/index` — `{book_id, text}` → chunks, embeds, stores
- `POST /api/rag/query` — `{book_id?, question, selected_text?, use_rag?, reader_position?}` → `{answer, sources[]}`
- `POST /api/agent` — `{selected_text, surrounding_text?, document_title?}` → structured explanation

**Key Details**:
- Voyage embeddings use `input_type: 'query'` for questions, `'document'` for chunks
- Top 5 chunks (cosine similarity) sent to Claude with position-aware context (PAST vs AHEAD)
- Vectors stored in Qdrant (`.data/qdrant/` in embedded mode), query logs at `.data/logs/queries.jsonl`
- Legacy `.data/vectors/*.json` auto-migrated to Qdrant on first run
- Books stored client-side only: IndexedDB `badger-books` (ArrayBuffer) + localStorage (metadata)
- Path alias: `@/*` → `./src/*`

## Git

- Do NOT add `Co-Authored-By` lines to commit messages
- One feature/fix per commit — do not bundle multiple issues into a single commit
