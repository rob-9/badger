"""
FastAPI server for Boom reading assistant.

Simple API with three endpoints:
1. POST /api/rag/index - Index a book
2. POST /api/rag/query - Query a book with RAG
3. POST /api/agent - Get AI assistance for selected text
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from anthropic import Anthropic

from boom.core.rag import RAGService

# Global RAG service instance
rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global rag_service
    print("[Server] Initializing RAG service...")
    rag_service = RAGService()
    print("[Server] Server ready!")
    yield
    print("[Server] Shutting down...")


app = FastAPI(
    title="Boom Reading Assistant API",
    description="Simple API for AI-powered reading assistance",
    version="0.1.0",
    lifespan=lifespan
)

# CORS - allow Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === RAG Endpoints ===

class IndexBookRequest(BaseModel):
    """Request to index a book."""
    book_id: str
    text: str


class QueryBookRequest(BaseModel):
    """Request to query a book."""
    book_id: Optional[str] = None
    question: str
    selected_text: Optional[str] = None
    use_rag: bool = True
    reader_position: Optional[float] = None  # 0-1, where reader currently is


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    sources: Optional[list] = None


@app.post("/api/rag/index")
async def index_book(request: IndexBookRequest):
    """
    Index a book for RAG.

    This chunks the text, generates embeddings, and stores them for search.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        await rag_service.index_book(request.book_id, request.text)
        return {"success": True, "book_id": request.book_id}
    except Exception as e:
        print(f"[Server] Error indexing book: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/query", response_model=QueryResponse)
async def query_book(request: QueryBookRequest):
    """
    Query a book with RAG or simple Claude query.

    If use_rag=True and book_id provided, uses RAG (retrieval + generation).
    Otherwise, just sends question + selected text to Claude.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        if request.use_rag and request.book_id:
            # Use RAG (retrieval + generation)
            response = await rag_service.query_book(
                book_id=request.book_id,
                question=request.question,
                selected_text=request.selected_text,
                reader_position=request.reader_position
            )
            return QueryResponse(
                answer=response.answer,
                sources=response.sources
            )
        elif request.selected_text:
            # Simple query with selected text only
            answer = await rag_service.query_simple(
                question=request.question,
                selected_text=request.selected_text
            )
            return QueryResponse(answer=answer)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either book_id (for RAG) or selected_text (for simple query)"
            )
    except Exception as e:
        print(f"[Server] Error querying book: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Agent Assistant Endpoint ===

class AgentRequest(BaseModel):
    """Request for AI agent assistance."""
    selected_text: str
    surrounding_text: Optional[str] = None
    document_title: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from AI agent."""
    explanation: str
    definitions: list[str]
    related_concepts: list[str]
    suggestions: list[str]


@app.post("/api/agent", response_model=AgentResponse)
async def agent_assist(request: AgentRequest):
    """
    Get AI assistance for selected text.

    Provides explanation, definitions, related concepts, and suggestions.
    """
    try:
        anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        prompt = f"""You are a reading assistant AI. A user is reading "{request.document_title or 'a document'}" and has selected the following text: "{request.selected_text}".

Here is the surrounding context:
"{request.surrounding_text or 'No surrounding context provided.'}"

Please provide:
1. A clear explanation of the selected text
2. Any key definitions if the text contains technical terms
3. Related concepts that might help understanding
4. Suggestions for deeper comprehension

Respond in JSON format with the following structure:
{{
  "explanation": "string",
  "definitions": ["string array"],
  "relatedConcepts": ["string array"],
  "suggestions": ["string array"]
}}"""

        message = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        content = message.content[0]
        if content.type != 'text':
            raise HTTPException(status_code=500, detail="No text response from AI")

        import json
        response_data = json.loads(content.text)

        return AgentResponse(
            explanation=response_data.get("explanation", ""),
            definitions=response_data.get("definitions", []),
            related_concepts=response_data.get("relatedConcepts", []),
            suggestions=response_data.get("suggestions", [])
        )
    except Exception as e:
        print(f"[Server] Error in agent assist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Health Check ===

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Boom API is running"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "ok",
        "rag_service": "initialized" if rag_service else "not initialized"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
