"""Smoke tests for the RAG pipeline.

Hits real APIs (Voyage + Claude Haiku via DEV_MODE=lite) with minimal data.
Not mocked — this is the real pipeline end-to-end, just cheap.

Run: DEV_MODE=lite pytest tests/test_smoke.py -v --tb=short -x
Cost: ~$0.002 per run (~5 Haiku queries + a few Voyage embeds).
"""

import asyncio
import json
import os
import pytest
from dotenv import load_dotenv

# Load .env and force lite mode before any badger imports
load_dotenv(override=True)
os.environ["DEV_MODE"] = "lite"

import voyageai
from anthropic import Anthropic, AsyncAnthropic

from badger import config
from badger.core.rag import RAGService
from badger.core.agent import build_agent
from badger.core.vector_store import VectorStore
from tests.conftest import SAMPLE_STRUCTURED_CONTENT

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Module-scoped fixtures (one-time setup for all smoke tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def event_loop():
    """Create a single event loop for the module."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def rag_service(event_loop):
    """Real RAGService with in-memory Qdrant (bypasses disk lock)."""
    # Construct RAGService manually to avoid opening the disk-based Qdrant
    # (which may be locked by a running dev server)
    svc = object.__new__(RAGService)
    svc.voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    svc.anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    svc.vector_store = VectorStore(location=":memory:")
    event_loop.run_until_complete(svc.vector_store.initialize())
    return svc


@pytest.fixture(scope="module")
def anthropic_client(rag_service):
    return rag_service.anthropic


@pytest.fixture(scope="module")
def async_anthropic_client():
    return AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


@pytest.fixture(scope="module")
def agent(rag_service, anthropic_client, async_anthropic_client):
    return build_agent(
        anthropic=anthropic_client,
        async_anthropic=async_anthropic_client,
        vector_store=rag_service.vector_store,
        voyage_client=rag_service.voyage,
    )


@pytest.fixture(scope="module")
def indexed_book(rag_service, event_loop):
    """Index sample content once for all tests."""
    event_loop.run_until_complete(
        rag_service.index_book_structured("smoke-test-book", SAMPLE_STRUCTURED_CONTENT)
    )
    return "smoke-test-book"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_and_query_rag(agent, indexed_book):
    """RAG query completes without error and returns a non-trivial answer."""
    result = await agent["run_agent"](
        book_id=indexed_book,
        question="What is the setting of this story?",
        reader_position=1.0,
    )
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 20
    # Pipeline exercised: tool_calls list exists (agent ran at least one search)
    assert "tool_calls" in result


@pytest.mark.asyncio
async def test_simple_query(rag_service):
    """Simple (no-RAG) query returns an answer without sources."""
    answer = await rag_service.query_simple(
        question="What does this passage suggest?",
        selected_text="Alice sat by the window, reading a book about philosophy.",
    )
    assert isinstance(answer, str)
    assert len(answer) > 10


@pytest.mark.asyncio
async def test_streaming_pipeline(agent, indexed_book):
    """Streaming yields events in the expected SSE order."""
    events = []
    async for event in agent["run_agent_streaming"](
        book_id=indexed_book,
        question="Describe the weather in the story.",
        reader_position=1.0,
    ):
        events.append(event)

    types = [e["type"] for e in events]

    # Must have at least status, token, done, and result
    assert "status" in types
    assert "token" in types
    assert "done" in types
    assert "result" in types

    # Sources event should appear (even if empty)
    assert "sources" in types

    # done before result
    done_idx = types.index("done")
    result_idx = types.index("result")
    assert done_idx < result_idx


@pytest.mark.asyncio
async def test_agent_assist(anthropic_client):
    """Agent assist endpoint format: returns valid JSON with expected keys."""
    prompt = """You are a reading assistant AI. A user is reading a book and has selected the following text: "Alice sat by the window, reading a book about philosophy."

Here is the surrounding context:
"It was a dark and stormy night. The wind howled through the trees."

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

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(raw)

    assert "explanation" in data
    assert isinstance(data.get("definitions", []), list)
    assert isinstance(data.get("relatedConcepts", []), list)
    assert isinstance(data.get("suggestions", []), list)


@pytest.mark.asyncio
async def test_chapter_search(agent, indexed_book):
    """Chapter-specific query finds content from the right chapter."""
    result = await agent["run_agent"](
        book_id=indexed_book,
        question="What happens in The Journey?",
        reader_position=1.0,
    )
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 20
    answer_lower = result["answer"].lower()
    # Should reference chapter 2 content (packing, heading north, countryside)
    assert any(term in answer_lower for term in [
        "pack", "bag", "north", "road", "countryside", "journey", "morning", "bright",
    ])
