`# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Next.js 14 (App Router) reading app with RAG-powered Q&A. Users upload EPUB/PDF, select text, ask questions answered by Claude using Voyage AI embeddings.

## Commands

```bash
npm run dev        # Development server
npm run build      # Production build
npm run typecheck  # TypeScript check
```

## Environment

```bash
ANTHROPIC_API_KEY=sk-ant-...  # Claude API
VOYAGE_API_KEY=pa-...         # Voyage AI embeddings
```

## Architecture

**RAG Pipeline**: `chunker.ts` (2000 char chunks, 200 overlap) → `rag.ts` (Voyage `voyage-3` embeddings) → `vectorStore.ts` (cosine similarity) → `rag.ts` (Claude `claude-sonnet-4-20250514`)

**Dual Persistence**:
- Vectors: Server-side at `.next/cache/vectors/[bookId].json`, client-side IndexedDB `boom-vectors`
- Books: Client-side only, IndexedDB `boom-books` (ArrayBuffer) + localStorage (metadata)

**API Routes**:
- `POST /api/rag/index`: `{bookId, text}` → chunks, embeds, stores
- `POST /api/rag/query`: `{bookId?, question, selectedText?, useRag?}` → `{answer, sources[]}`

**Key Details**:
- Voyage embeddings use `inputType: 'query'` for questions, `'document'` for chunks
- Top 5 chunks (cosine similarity) sent to Claude with system prompt to answer only from context
- Lazy client initialization in server code to avoid runtime errors
- Path alias: `@/*` → `./src/*`
