"""
FastAPI server for Boom reading assistant.

Simple API with three endpoints:
1. POST /api/rag/index - Index a book
2. POST /api/rag/query - Query a book with RAG
3. POST /api/agent - Get AI assistance for selected text
"""

import json
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from anthropic import Anthropic

from boom import config
from boom.core.rag import RAGService
from boom.core.graph import build_qa_graph

logger = logging.getLogger(__name__)

# Global services
rag_service: Optional[RAGService] = None
anthropic_client: Optional[Anthropic] = None
qa_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global rag_service, anthropic_client, qa_graph

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config.validate_keys()

    logger.info("Initializing services...")
    anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    rag_service = RAGService(storage_dir=config.VECTOR_STORAGE_DIR)
    qa_graph = build_qa_graph(
        anthropic=anthropic_client,
        vector_store=rag_service.vector_store,
        voyage_client=rag_service.voyage,
    )
    logger.info("Server ready (LangGraph pipeline active)")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Boom Reading Assistant API",
    description="Simple API for AI-powered reading assistance",
    version="0.1.0",
    lifespan=lifespan
)

# CORS - allow Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
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
        logger.error("Error indexing book: %s", e)
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
            # Use LangGraph pipeline (classify → retrieve → generate → log)
            result = await qa_graph.ainvoke({
                "question": request.question,
                "selected_text": request.selected_text,
                "reader_position": request.reader_position or 0.0,
                "book_id": request.book_id,
            })
            return QueryResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
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
        logger.error("Error querying book: %s", e)
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
    if not anthropic_client:
        raise HTTPException(status_code=503, detail="Anthropic client not initialized")

    try:
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

        message = anthropic_client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        content = message.content[0]
        if content.type != 'text':
            raise HTTPException(status_code=500, detail="No text response from AI")

        try:
            response_data = json.loads(content.text)
        except json.JSONDecodeError:
            logger.error("Failed to parse Claude response as JSON: %s", content.text[:200])
            raise HTTPException(status_code=500, detail="Invalid JSON response from AI")

        return AgentResponse(
            explanation=response_data.get("explanation", ""),
            definitions=response_data.get("definitions", []),
            related_concepts=response_data.get("relatedConcepts", []),
            suggestions=response_data.get("suggestions", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in agent assist: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# === Health Check ===

@app.get("/api/rag/indexed/{book_id}")
async def is_book_indexed(book_id: str):
    """Check if a book is already indexed."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    return {"indexed": rag_service.vector_store.has_book(book_id)}


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
