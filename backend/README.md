# Badger Backend

Python backend for Badger reading assistant.

## Quick Start

**1. Install dependencies:**
```bash
cd backend
pip install -e .
```

**2. Set up environment:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

**3. Run the server:**
```bash
uvicorn badger.api.server:app --reload --port 8000
```

Server runs on `http://localhost:8000`

## API Endpoints

### POST /api/rag/index
Index a book for RAG search.

**Request:**
```json
{
  "book_id": "my-book",
  "text": "Full book text..."
}
```

### POST /api/rag/query
Query a book with RAG or simple Claude query.

**Request:**
```json
{
  "book_id": "my-book",
  "question": "What is the main theme?",
  "selected_text": "optional selected text",
  "use_rag": true
}
```

**Response:**
```json
{
  "answer": "The main theme is...",
  "sources": [
    {
      "text": "Relevant chunk...",
      "score": 0.89,
      "chunk_index": 5
    }
  ]
}
```

### POST /api/agent
Get AI assistance for selected text (explanations, definitions, etc.)

**Request:**
```json
{
  "selected_text": "Text the user selected",
  "surrounding_text": "Context around it",
  "document_title": "Book name"
}
```

## Development

The backend is simple and focused:
- `badger/core/` - RAG logic (chunking, embeddings, search)
- `badger/api/` - FastAPI server with 3 endpoints
- `.data/vectors/` - Where vector embeddings are stored

No database, no auth, no complexity. Just RAG for your books.
