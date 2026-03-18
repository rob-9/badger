"""
FastAPI server for Badger reading assistant.

Simple API with three endpoints:
1. POST /api/rag/index - Index a book
2. POST /api/rag/query - Query a book with RAG
3. POST /api/agent - Get AI assistance for selected text
"""

import asyncio
import io
import json
import logging
import re
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, model_validator
from typing import Optional
from anthropic import Anthropic, AsyncAnthropic

from badger import config
from badger.core.rag import RAGService
from badger.core.agent import build_agent, log_agent_query

logger = logging.getLogger(__name__)

# Global services
rag_service: Optional[RAGService] = None
anthropic_client: Optional[Anthropic] = None
async_anthropic_client: Optional[AsyncAnthropic] = None
agent = None

BOOK_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
MAX_BOOK_ID_LENGTH = 200


def validate_book_id(book_id: str) -> None:
    """Validate book_id is safe for use in file paths."""
    if not book_id or len(book_id) > MAX_BOOK_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid book ID")
    if not BOOK_ID_PATTERN.match(book_id):
        raise HTTPException(status_code=400, detail="Invalid book ID")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global rag_service, anthropic_client, async_anthropic_client, agent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config.validate_keys()

    logger.info("Initializing services...")
    anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    async_anthropic_client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    rag_service = RAGService()
    await rag_service.vector_store.initialize()
    agent = build_agent(
        anthropic=anthropic_client,
        async_anthropic=async_anthropic_client,
        vector_store=rag_service.vector_store,
        voyage_client=rag_service.voyage,
    )
    logger.info("Server ready (tool-calling agent active)")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Badger Reading Assistant API",
    description="Simple API for AI-powered reading assistance",
    version="0.1.0",
    lifespan=lifespan
)

# CORS - allow Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# === RAG Endpoints ===

class IndexBookRequest(BaseModel):
    """Request to index a book."""
    book_id: str
    text: Optional[str] = None
    structured_content: Optional[dict] = None

    @model_validator(mode="after")
    def check_content(self):
        if not self.text and not self.structured_content:
            raise ValueError("Must provide either 'text' or 'structured_content'")
        max_size = config.MAX_INDEX_INPUT_SIZE
        if self.text and len(self.text) > max_size:
            raise ValueError(f"Text exceeds maximum size of {max_size // (1024 * 1024)}MB")
        if self.structured_content:
            content_size = len(json.dumps(self.structured_content))
            if content_size > max_size:
                raise ValueError(f"Structured content exceeds maximum size of {max_size // (1024 * 1024)}MB")
        return self


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

    validate_book_id(request.book_id)

    try:
        if request.structured_content:
            await rag_service.index_book_structured(request.book_id, request.structured_content)
        else:
            await rag_service.index_book(request.book_id, request.text)
        return {"success": True, "book_id": request.book_id}
    except Exception as e:
        logger.error("Error indexing book: %s", e)
        raise HTTPException(status_code=500, detail="Failed to index book")


@app.post("/api/rag/query", response_model=QueryResponse)
async def query_book(request: QueryBookRequest):
    """
    Query a book with RAG or simple Claude query.

    If use_rag=True and book_id provided, uses RAG (retrieval + generation).
    Otherwise, just sends question + selected text to Claude.
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if request.book_id:
        validate_book_id(request.book_id)

    try:
        if request.use_rag and request.book_id:
            logger.info("RAG query: %s | book=%s | position=%.0f%%",
                        request.question[:80], request.book_id[:20],
                        (request.reader_position or 0) * 100)
            result = await agent["run_agent"](
                book_id=request.book_id,
                question=request.question,
                selected_text=request.selected_text,
                reader_position=request.reader_position or 0.0,
            )
            tool_calls = result.get("tool_calls", [])
            logger.info("RAG done: %d tool calls, %d sources | %s",
                        len(tool_calls), len(result.get("sources", [])),
                        ", ".join(f"{tc['tool']}→{tc['chunks_returned']}ch" for tc in tool_calls) or "no tools")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error querying book: %s", e)
        raise HTTPException(status_code=500, detail="Failed to process query")


@app.post("/api/rag/query/stream")
async def query_book_stream(request: QueryBookRequest):
    """Stream a RAG query response via Server-Sent Events."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if request.book_id:
        validate_book_id(request.book_id)

    logger.info("RAG stream: %s | book=%s | position=%.0f%%",
                request.question[:80], request.book_id[:20] if request.book_id else "none",
                (request.reader_position or 0) * 100)

    async def event_generator():
        try:
            if request.use_rag and request.book_id:
                state = None
                async for event in agent["run_agent_streaming"](
                    book_id=request.book_id,
                    question=request.question,
                    selected_text=request.selected_text,
                    reader_position=request.reader_position or 0.0,
                ):
                    if event["type"] == "status":
                        yield f"event: status\ndata: {json.dumps({'stage': event['stage']})}\n\n"
                    elif event["type"] == "sources":
                        yield f"event: sources\ndata: {json.dumps(event['sources'])}\n\n"
                    elif event["type"] == "token":
                        yield f"event: token\ndata: {json.dumps({'text': event['text']})}\n\n"
                    elif event["type"] == "done":
                        yield "event: done\ndata: {}\n\n"
                    elif event["type"] == "result":
                        state = event["state"]

                # Evaluate + log after streaming completes
                if state:
                    tool_calls = state.get("tool_calls", [])
                    logger.info("RAG stream done: %d tool calls, %d sources | %s",
                                len(tool_calls), len(state.get("sources", [])),
                                ", ".join(f"{tc['tool']}→{tc['chunks_returned']}ch" for tc in tool_calls) or "no tools")
                    try:
                        eval_result = await agent["evaluate"](state)
                        state.update(eval_result)
                        log_agent_query(state)
                    except Exception:
                        logger.warning("Failed to evaluate/log streaming query", exc_info=True)

            elif request.selected_text:
                # Simple path: stream without RAG pipeline
                yield f"event: status\ndata: {json.dumps({'stage': 'generating'})}\n\n"

                system = 'You are a reading companion. Be direct and concise — answer the question, then stop. Address the reader as "you" in conversation. When discussing book events, use actual character names — never "the protagonist" or second-person narration.'
                user_prompt = f'Selected text: "{request.selected_text}"\n\nQuestion: {request.question}'

                async with async_anthropic_client.messages.stream(
                    model=config.CLAUDE_MODEL,
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        yield f"event: token\ndata: {json.dumps({'text': text})}\n\n"

                yield "event: done\ndata: {}\n\n"

            else:
                yield f"event: error\ndata: {json.dumps({'message': 'Must provide either book_id (for RAG) or selected_text'})}\n\n"

        except Exception as e:
            logger.error("Error in streaming query: %s", e)
            yield f"event: error\ndata: {json.dumps({'message': 'An internal error occurred'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
        prompt = f"""You are a reading assistant AI. A user is reading a book and has selected the following text: "{request.selected_text}".

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

        message = await asyncio.to_thread(
            anthropic_client.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
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
        raise HTTPException(status_code=500, detail="Failed to process request")


# === Local EPUB Import ===

class ImportLocalRequest(BaseModel):
    """Request to import an EPUB from a local filesystem path."""
    path: str


@app.post("/api/epub/import-local")
async def import_local_epub(request: ImportLocalRequest):
    """
    Read an EPUB from a local path, re-zipping if it's an exploded directory
    (common with Apple Books). Returns the raw EPUB bytes.
    """
    p = Path(request.path).expanduser().resolve()

    # Validate path is within allowed directories
    allowed = False
    for allowed_dir in config.EPUB_IMPORT_ALLOWED_DIRS:
        allowed_path = Path(allowed_dir).expanduser().resolve()
        try:
            p.relative_to(allowed_path)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(status_code=403, detail="Path not in allowed directories")

    if not p.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    if p.is_file() and p.suffix.lower() == ".epub":
        data = p.read_bytes()
        filename = p.name
    elif p.is_dir() and (p / "META-INF" / "container.xml").exists():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # mimetype must be first and uncompressed per EPUB spec
            mimetype_path = p / "mimetype"
            if mimetype_path.exists():
                zf.writestr("mimetype", mimetype_path.read_text(), compress_type=zipfile.ZIP_STORED)
            for file in sorted(p.rglob("*")):
                if file.is_file() and file.name != "mimetype":
                    arcname = str(file.relative_to(p))
                    zf.write(file, arcname)
        data = buf.getvalue()
        filename = p.name if p.suffix.lower() == ".epub" else p.name + ".epub"
    else:
        raise HTTPException(
            status_code=400,
            detail="Path must be an .epub file or an exploded EPUB directory"
        )

    return Response(
        content=data,
        media_type="application/epub+zip",
        headers={"X-Filename": filename},
    )


# === Health Check ===

@app.get("/api/rag/indexed/{book_id}")
async def is_book_indexed(book_id: str):
    """Check if a book is already indexed."""
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")
    validate_book_id(book_id)
    return {"indexed": rag_service.vector_store.has_book(book_id)}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Badger API is running"}


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
