# Boom Core - RAG for Reading

Simple, focused implementation of RAG for your personal reading assistant.

## What's Here

```
boom/core/
├── chunker.py        # Split text into ~500 token chunks
├── vector_store.py   # File-based vector storage + cosine similarity
├── rag.py           # Main RAG service (Voyage + Claude)
├── chat.py          # (Not yet implemented)
├── documents.py     # (Not yet implemented)
└── storage.py       # (Not yet implemented)
```

## How It Works

**1. Index a book:**
```python
from boom.core.rag import RAGService

rag = RAGService()
await rag.index_book("my-book-id", full_text)
```

This:
- Chunks the text (~2000 chars per chunk, 200 char overlap)
- Embeds each chunk using Voyage AI (voyage-3 model)
- Stores vectors in `.data/vectors/my-book-id.json`

**2. Query a book:**
```python
response = await rag.query_book("my-book-id", "What is the main theme?")
print(response.answer)
print(response.sources)  # Where the answer came from
```

This:
- Embeds your question
- Finds top 5 similar chunks (cosine similarity)
- Sends chunks + question to Claude
- Returns answer with sources

## Key Design Decisions

**Simple, not enterprise:**
- File-based storage (not a vector database)
- In-memory caching for speed
- Single user (no auth, permissions, etc.)
- Focused on reading, not general-purpose RAG

**What we migrated from TypeScript:**
- ✅ Chunking logic (sentence-aware boundaries)
- ✅ Voyage AI embeddings (voyage-3 model)
- ✅ Cosine similarity search
- ✅ Claude RAG workflow
- ✅ File persistence

**What's different:**
- Simpler! No IndexedDB (backend-only now)
- Python's better async/await
- Native type hints (not TypeScript)
