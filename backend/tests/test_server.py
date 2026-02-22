"""Tests for boom.api.server — FastAPI endpoints."""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from fastapi.testclient import TestClient


# ── Mocks ─────────────────────────────────────────────────────────────


@dataclass
class FakeContentBlock:
    text: str
    type: str = "text"


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeMessageResponse:
    content: list
    model: str = "claude-sonnet-4-20250514"
    usage: FakeUsage = None
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.usage is None:
            self.usage = FakeUsage()


class FakeAsyncStream:
    """Mock for AsyncAnthropic messages.stream() context manager."""

    def __init__(self, texts):
        self._texts = texts
        self._final = FakeMessageResponse(
            content=[FakeContentBlock(text="".join(texts))]
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def text_stream(self):
        return self._async_iter()

    async def _async_iter(self):
        for text in self._texts:
            yield text

    async def get_final_message(self):
        return self._final


def parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts."""
    events = []
    current_event = ""
    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: ") and current_event:
            data = json.loads(line[6:])
            events.append({"event": current_event, "data": data})
            current_event = ""
    return events


# ── Test client setup ─────────────────────────────────────────────────


@pytest.fixture
def mock_services():
    """Set up module-level service mocks before importing the app."""
    import boom.api.server as server_mod

    mock_rag = MagicMock()
    mock_rag.vector_store = MagicMock()
    mock_rag.query_simple = AsyncMock(return_value="Simple answer")
    mock_rag.index_book = AsyncMock()
    mock_rag.index_book_structured = AsyncMock()

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = FakeMessageResponse(
        content=[FakeContentBlock(
            text=json.dumps({
                "explanation": "Test explanation",
                "definitions": ["def1"],
                "relatedConcepts": ["concept1"],
                "suggestions": ["suggestion1"],
            })
        )]
    )

    mock_async_anthropic = MagicMock()
    mock_async_anthropic.messages.stream.return_value = FakeAsyncStream(
        ["Test ", "streaming ", "answer"]
    )

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "answer": "RAG answer",
        "sources": [{"text": "source...", "score": 0.9, "chunk_index": 0}],
    }

    async def mock_pre_generate(params):
        yield "classifying"
        yield "retrieving"
        yield "reranking"
        yield "filtering"
        yield "sanitizing"
        yield {
            "question": params["question"],
            "selected_text": params.get("selected_text"),
            "reader_position": params.get("reader_position", 0),
            "book_id": params["book_id"],
            "question_type": "context",
            "chunks": [
                {"text": "chunk text from book", "label": "PAST", "chunk_index": 0, "score": 0.9},
            ],
            "total_chunks": 100,
            "classify_tokens_in": 50,
            "classify_tokens_out": 10,
        }

    async def mock_evaluate(state):
        return {
            "eval_relevance": 4,
            "eval_grounding": 5,
            "eval_flags": [],
            "eval_low_confidence": False,
            "eval_tokens_in": 80,
            "eval_tokens_out": 15,
        }

    # Inject mocks
    server_mod.rag_service = mock_rag
    server_mod.anthropic_client = mock_anthropic
    server_mod.async_anthropic_client = mock_async_anthropic
    server_mod.qa_graph = mock_graph
    server_mod.qa_run_pre_generate = mock_pre_generate
    server_mod.qa_evaluate = mock_evaluate

    yield {
        "rag": mock_rag,
        "anthropic": mock_anthropic,
        "async_anthropic": mock_async_anthropic,
        "graph": mock_graph,
    }

    # Cleanup
    server_mod.rag_service = None
    server_mod.anthropic_client = None
    server_mod.async_anthropic_client = None
    server_mod.qa_graph = None
    server_mod.qa_run_pre_generate = None
    server_mod.qa_evaluate = None


@pytest.fixture
def client(mock_services):
    """TestClient with mocked services."""
    from boom.api.server import app

    return TestClient(app, raise_server_exceptions=False)


# ── Health checks ─────────────────────────────────────────────────────


class TestHealthChecks:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health(self, client, mock_services):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["rag_service"] == "initialized"

    def test_is_indexed_true(self, client, mock_services):
        mock_services["rag"].vector_store.has_book.return_value = True
        resp = client.get("/api/rag/indexed/book1")
        assert resp.status_code == 200
        assert resp.json()["indexed"] is True

    def test_is_indexed_false(self, client, mock_services):
        mock_services["rag"].vector_store.has_book.return_value = False
        resp = client.get("/api/rag/indexed/book1")
        assert resp.status_code == 200
        assert resp.json()["indexed"] is False


# ── Index endpoint ────────────────────────────────────────────────────


class TestIndexEndpoint:
    def test_index_plain_text(self, client, mock_services):
        resp = client.post("/api/rag/index", json={"book_id": "b1", "text": "Hello world."})
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_services["rag"].index_book.assert_called_once_with("b1", "Hello world.")

    def test_index_structured(self, client, mock_services):
        content = {"chapters": [{"title": "Ch1", "index": 0, "sections": []}]}
        resp = client.post(
            "/api/rag/index",
            json={"book_id": "b1", "structured_content": content},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        mock_services["rag"].index_book_structured.assert_called_once()

    def test_index_no_content_returns_422(self, client, mock_services):
        resp = client.post("/api/rag/index", json={"book_id": "b1"})
        assert resp.status_code == 422

    def test_index_missing_book_id_returns_422(self, client, mock_services):
        resp = client.post("/api/rag/index", json={"text": "Hello"})
        assert resp.status_code == 422

    def test_index_structured_preferred_over_text(self, client, mock_services):
        """When both text and structured_content are given, structured wins."""
        content = {"chapters": []}
        resp = client.post(
            "/api/rag/index",
            json={"book_id": "b1", "text": "Hello", "structured_content": content},
        )
        assert resp.status_code == 200
        mock_services["rag"].index_book_structured.assert_called_once()
        mock_services["rag"].index_book.assert_not_called()

    def test_index_error_returns_500(self, client, mock_services):
        mock_services["rag"].index_book.side_effect = RuntimeError("Embed failed")
        resp = client.post("/api/rag/index", json={"book_id": "b1", "text": "text"})
        assert resp.status_code == 500


# ── Query endpoint ────────────────────────────────────────────────────


class TestQueryEndpoint:
    def test_rag_query(self, client, mock_services):
        resp = client.post(
            "/api/rag/query",
            json={"book_id": "b1", "question": "What happens?", "use_rag": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "RAG answer"
        assert len(data["sources"]) == 1

    def test_rag_query_passes_state(self, client, mock_services):
        client.post(
            "/api/rag/query",
            json={
                "book_id": "b1",
                "question": "What?",
                "selected_text": "text",
                "reader_position": 0.5,
                "use_rag": True,
            },
        )
        call_args = mock_services["graph"].ainvoke.call_args[0][0]
        assert call_args["question"] == "What?"
        assert call_args["selected_text"] == "text"
        assert call_args["reader_position"] == 0.5
        assert call_args["book_id"] == "b1"

    def test_simple_query_with_selected_text(self, client, mock_services):
        resp = client.post(
            "/api/rag/query",
            json={"question": "What does this mean?", "selected_text": "passage", "use_rag": False},
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Simple answer"

    def test_query_no_book_no_text_returns_400(self, client, mock_services):
        resp = client.post(
            "/api/rag/query",
            json={"question": "What?", "use_rag": False},
        )
        assert resp.status_code == 400

    def test_query_missing_question_returns_422(self, client, mock_services):
        resp = client.post("/api/rag/query", json={"book_id": "b1"})
        assert resp.status_code == 422

    def test_query_default_use_rag_true(self, client, mock_services):
        resp = client.post(
            "/api/rag/query",
            json={"book_id": "b1", "question": "What?"},
        )
        assert resp.status_code == 200
        mock_services["graph"].ainvoke.assert_called()

    def test_query_error_returns_500(self, client, mock_services):
        mock_services["graph"].ainvoke.side_effect = RuntimeError("LLM error")
        resp = client.post(
            "/api/rag/query",
            json={"book_id": "b1", "question": "What?"},
        )
        assert resp.status_code == 500


# ── Stream endpoint ───────────────────────────────────────────────────


class TestStreamEndpoint:
    def test_rag_stream_returns_sse(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"book_id": "b1", "question": "What happens?", "use_rag": True},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_rag_stream_has_all_events(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"book_id": "b1", "question": "What happens?", "use_rag": True},
        )
        events = parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "sources" in event_types
        assert "token" in event_types
        assert "done" in event_types

    def test_rag_stream_status_stages(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"book_id": "b1", "question": "What?", "use_rag": True},
        )
        events = parse_sse(resp.text)
        stages = [e["data"]["stage"] for e in events if e["event"] == "status"]
        assert stages == ["classifying", "retrieving", "reranking", "filtering", "sanitizing"]

    def test_rag_stream_tokens_concatenate(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"book_id": "b1", "question": "What?", "use_rag": True},
        )
        events = parse_sse(resp.text)
        tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
        assert "".join(tokens) == "Test streaming answer"

    def test_rag_stream_sources_present(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"book_id": "b1", "question": "What?", "use_rag": True},
        )
        events = parse_sse(resp.text)
        sources_events = [e for e in events if e["event"] == "sources"]
        assert len(sources_events) == 1
        assert len(sources_events[0]["data"]) == 1  # 1 chunk from mock

    def test_simple_stream(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"question": "What does this mean?", "selected_text": "passage", "use_rag": False},
        )
        assert resp.status_code == 200
        events = parse_sse(resp.text)
        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "token" in event_types
        assert "done" in event_types
        # Simple path should not have sources or classifying/retrieving stages
        assert "sources" not in event_types
        stages = [e["data"]["stage"] for e in events if e["event"] == "status"]
        assert stages == ["generating"]

    def test_stream_no_book_no_text_returns_error(self, client, mock_services):
        resp = client.post(
            "/api/rag/query/stream",
            json={"question": "What?", "use_rag": False},
        )
        events = parse_sse(resp.text)
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "message" in error_events[0]["data"]

    def test_stream_503_when_not_initialized(self):
        import boom.api.server as server_mod

        server_mod.rag_service = None
        from boom.api.server import app

        c = TestClient(app, raise_server_exceptions=False)
        resp = c.post(
            "/api/rag/query/stream",
            json={"book_id": "b1", "question": "What?"},
        )
        assert resp.status_code == 503


# ── Agent endpoint ────────────────────────────────────────────────────


class TestAgentEndpoint:
    def test_agent_assist(self, client, mock_services):
        resp = client.post(
            "/api/agent",
            json={"selected_text": "Cogito ergo sum", "document_title": "Meditations"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation"] == "Test explanation"
        assert data["definitions"] == ["def1"]
        assert data["related_concepts"] == ["concept1"]
        assert data["suggestions"] == ["suggestion1"]

    def test_agent_with_surrounding_text(self, client, mock_services):
        resp = client.post(
            "/api/agent",
            json={
                "selected_text": "test",
                "surrounding_text": "context around test",
                "document_title": "Doc",
            },
        )
        assert resp.status_code == 200

    def test_agent_missing_selected_text_returns_422(self, client, mock_services):
        resp = client.post("/api/agent", json={})
        assert resp.status_code == 422

    def test_agent_invalid_json_from_llm_returns_500(self, client, mock_services):
        mock_services["anthropic"].messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="Not valid JSON")]
        )
        resp = client.post("/api/agent", json={"selected_text": "test"})
        assert resp.status_code == 500

    def test_agent_non_text_response_returns_500(self, client, mock_services):
        mock_services["anthropic"].messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="hi", type="image")]
        )
        resp = client.post("/api/agent", json={"selected_text": "test"})
        assert resp.status_code == 500


# ── Service not initialized ───────────────────────────────────────────


# ── Import local EPUB ────────────────────────────────────────────────


class TestImportLocalEpub:
    def test_import_epub_file(self, client, mock_services, tmp_path):
        epub_path = tmp_path / "test.epub"
        epub_path.write_bytes(b"PK\x03\x04fake epub content")
        resp = client.post("/api/epub/import-local", json={"path": str(epub_path)})
        assert resp.status_code == 200
        assert resp.headers["x-filename"] == "test.epub"
        assert resp.content == b"PK\x03\x04fake epub content"

    def test_import_exploded_directory(self, client, mock_services, tmp_path):
        epub_dir = tmp_path / "MyBook.epub"
        epub_dir.mkdir()
        (epub_dir / "mimetype").write_text("application/epub+zip")
        meta_inf = epub_dir / "META-INF"
        meta_inf.mkdir()
        (meta_inf / "container.xml").write_text("<container/>")
        (epub_dir / "content.opf").write_text("<package/>")

        resp = client.post("/api/epub/import-local", json={"path": str(epub_dir)})
        assert resp.status_code == 200
        assert resp.headers["x-filename"] == "MyBook.epub"
        # Should be a valid zip
        import zipfile, io
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        assert "mimetype" in names
        assert "META-INF/container.xml" in names

    def test_import_nonexistent_path(self, client, mock_services):
        resp = client.post("/api/epub/import-local", json={"path": "/nonexistent/path.epub"})
        assert resp.status_code == 404

    def test_import_non_epub_file(self, client, mock_services, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("not an epub")
        resp = client.post("/api/epub/import-local", json={"path": str(txt)})
        assert resp.status_code == 400

    def test_import_directory_without_container(self, client, mock_services, tmp_path):
        plain_dir = tmp_path / "not_epub"
        plain_dir.mkdir()
        (plain_dir / "random.txt").write_text("hello")
        resp = client.post("/api/epub/import-local", json={"path": str(plain_dir)})
        assert resp.status_code == 400


# ── Service not initialized ───────────────────────────────────────────


class TestServiceNotInitialized:
    def test_index_503(self):
        import boom.api.server as server_mod

        server_mod.rag_service = None
        from boom.api.server import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/rag/index", json={"book_id": "b1", "text": "hello"})
        assert resp.status_code == 503

    def test_query_503(self):
        import boom.api.server as server_mod

        server_mod.rag_service = None
        from boom.api.server import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/rag/query",
            json={"book_id": "b1", "question": "What?"},
        )
        assert resp.status_code == 503

    def test_agent_503(self):
        import boom.api.server as server_mod

        server_mod.anthropic_client = None
        from boom.api.server import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/agent", json={"selected_text": "test"})
        assert resp.status_code == 503
