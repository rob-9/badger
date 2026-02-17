"""Tests for boom.core.rag — RAG service with mocked external APIs."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass

from boom.core.rag import RAGService, RAGResponse
from boom.core.chunker import TextChunk
from boom.core.vector_store import VectorEntry, SearchResult


# ── Mocks ─────────────────────────────────────────────────────────────


@dataclass
class FakeEmbedResponse:
    embeddings: list[list[float]]


@dataclass
class FakeContextResult:
    embeddings: list[list[float]]


@dataclass
class FakeContextEmbedResponse:
    results: list[FakeContextResult]


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class FakeContentBlock:
    text: str
    type: str = "text"


@dataclass
class FakeMessageResponse:
    content: list[FakeContentBlock]
    model: str = "claude-sonnet-4-20250514"
    usage: FakeUsage = None
    stop_reason: str = "end_turn"

    def __post_init__(self):
        if self.usage is None:
            self.usage = FakeUsage()


@pytest.fixture
def mock_voyage():
    voyage = MagicMock()
    voyage.embed.return_value = FakeEmbedResponse(embeddings=[[0.1] * 1024])
    voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
        results=[FakeContextResult(embeddings=[[0.1] * 1024])]
    )
    return voyage


@pytest.fixture
def mock_anthropic():
    client = MagicMock()
    client.messages.create.return_value = FakeMessageResponse(
        content=[FakeContentBlock(text="Test answer")]
    )
    return client


@pytest.fixture
def rag_service(tmp_path, mock_voyage, mock_anthropic):
    """RAG service with mocked external clients."""
    with patch("boom.core.rag.voyageai") as mock_vmod, \
         patch("boom.core.rag.Anthropic") as mock_amod:
        mock_vmod.Client.return_value = mock_voyage
        mock_amod.return_value = mock_anthropic
        service = RAGService(storage_dir=str(tmp_path / "vectors"))
        # Replace clients with our mocks
        service.voyage = mock_voyage
        service.anthropic = mock_anthropic
        yield service


# ── RAGService.get_embedding ──────────────────────────────────────────


class TestGetEmbedding:
    @pytest.mark.asyncio
    async def test_returns_embedding(self, rag_service, mock_voyage):
        embedding = await rag_service.get_embedding("test query")
        assert len(embedding) == 1024
        mock_voyage.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_input_type(self, rag_service, mock_voyage):
        await rag_service.get_embedding("test", input_type="document")
        call_kwargs = mock_voyage.embed.call_args
        assert call_kwargs.kwargs.get("input_type") == "document" or call_kwargs[1].get("input_type") == "document"

    @pytest.mark.asyncio
    async def test_raises_on_empty_response(self, rag_service, mock_voyage):
        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=[])
        with pytest.raises(ValueError, match="No embedding"):
            await rag_service.get_embedding("test")


# ── RAGService.get_embeddings (batch) ─────────────────────────────────


class TestGetEmbeddings:
    @pytest.mark.asyncio
    async def test_single_batch(self, rag_service, mock_voyage):
        mock_voyage.embed.return_value = FakeEmbedResponse(
            embeddings=[[0.1] * 1024 for _ in range(3)]
        )
        result = await rag_service.get_embeddings(["text1", "text2", "text3"])
        assert len(result) == 3
        assert mock_voyage.embed.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_batches(self, rag_service, mock_voyage):
        """Large texts should be split into multiple API batches."""
        big_text = "x" * 100_001  # Over half the batch limit
        texts = [big_text, big_text, big_text]

        mock_voyage.embed.side_effect = [
            FakeEmbedResponse(embeddings=[[0.1] * 1024]),
            FakeEmbedResponse(embeddings=[[0.2] * 1024]),
            FakeEmbedResponse(embeddings=[[0.3] * 1024]),
        ]
        result = await rag_service.get_embeddings(texts)
        assert len(result) == 3
        assert mock_voyage.embed.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_on_empty_batch(self, rag_service, mock_voyage):
        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=None)
        with pytest.raises(ValueError, match="No embeddings"):
            await rag_service.get_embeddings(["text"])


# ── RAGService.get_contextualized_embeddings ──────────────────────────


def _make_chunks(texts: list[str], chapter_index: int = 0) -> list[TextChunk]:
    return [
        TextChunk(
            id=f"test-chunk-{i}",
            text=t,
            metadata={"book_id": "test", "chunk_index": i, "chapter_index": chapter_index},
        )
        for i, t in enumerate(texts)
    ]


class TestContextualizedEmbeddings:
    @pytest.mark.asyncio
    async def test_single_chapter(self, rag_service, mock_voyage):
        mock_voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
            results=[FakeContextResult(embeddings=[[0.1] * 1024, [0.2] * 1024])]
        )
        result = await rag_service.get_contextualized_embeddings(
            _make_chunks(["chunk1", "chunk2"], chapter_index=0)
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_uses_document_input_type(self, rag_service, mock_voyage):
        mock_voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
            results=[FakeContextResult(embeddings=[[0.1] * 1024])]
        )
        await rag_service.get_contextualized_embeddings(_make_chunks(["text"]))
        call_kwargs = mock_voyage.contextualized_embed.call_args
        assert call_kwargs.kwargs.get("input_type") == "document" or call_kwargs[1].get("input_type") == "document"

    @pytest.mark.asyncio
    async def test_groups_by_chapter(self, rag_service, mock_voyage):
        """Chunks from different chapters become separate documents in one request."""
        chunks = (
            _make_chunks(["ch0a", "ch0b"], chapter_index=0)
            + _make_chunks(["ch1a"], chapter_index=1)
        )
        mock_voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
            results=[
                FakeContextResult(embeddings=[[0.1] * 1024, [0.2] * 1024]),
                FakeContextResult(embeddings=[[0.3] * 1024]),
            ]
        )
        result = await rag_service.get_contextualized_embeddings(chunks)
        assert len(result) == 3
        # Should send 2 documents (one per chapter) in a single call
        assert mock_voyage.contextualized_embed.call_count == 1
        call_inputs = mock_voyage.contextualized_embed.call_args.kwargs.get("inputs") \
            or mock_voyage.contextualized_embed.call_args[0][0]
        assert len(call_inputs) == 2  # 2 chapter docs

    @pytest.mark.asyncio
    async def test_oversized_chapter_split(self, rag_service, mock_voyage):
        """A chapter exceeding MAX_CHARS_PER_DOC should be split into sub-groups."""
        # Create chunks that exceed the per-doc limit when combined
        big_text = "x" * 70_000  # Each chunk is 70K chars
        chunks = [
            TextChunk(
                id=f"test-chunk-{i}",
                text=big_text,
                metadata={"book_id": "test", "chunk_index": i, "chapter_index": 0},
            )
            for i in range(3)  # 3 × 70K = 210K chars, well over 120K limit
        ]
        # Each sub-doc gets its own result
        mock_voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
            results=[
                FakeContextResult(embeddings=[[0.1] * 1024]),
                FakeContextResult(embeddings=[[0.2] * 1024]),
                FakeContextResult(embeddings=[[0.3] * 1024]),
            ]
        )
        result = await rag_service.get_contextualized_embeddings(chunks)
        assert len(result) == 3
        # Should split the single chapter into multiple documents
        call_inputs = mock_voyage.contextualized_embed.call_args.kwargs.get("inputs") \
            or mock_voyage.contextualized_embed.call_args[0][0]
        assert len(call_inputs) == 3  # Each chunk too big to pair with another


# ── RAGService.index_book ─────────────────────────────────────────────


class TestIndexBook:
    @pytest.mark.asyncio
    async def test_indexes_plain_text(self, rag_service, mock_voyage):
        mock_voyage.embed.return_value = FakeEmbedResponse(
            embeddings=[[0.1] * 1024]
        )
        await rag_service.index_book("book1", "Short test text for indexing.")
        assert rag_service.vector_store.has_book("book1")

    @pytest.mark.asyncio
    async def test_skips_if_already_indexed(self, rag_service, mock_voyage):
        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=[[0.1] * 1024])
        await rag_service.index_book("book1", "Text.")
        mock_voyage.embed.reset_mock()
        await rag_service.index_book("book1", "Text.")
        mock_voyage.embed.assert_not_called()


# ── RAGService.index_book_structured ──────────────────────────────────


class TestIndexBookStructured:
    @pytest.mark.asyncio
    async def test_indexes_structured(self, rag_service, mock_voyage, mock_anthropic):
        content = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "index": 0,
                    "sections": [
                        {
                            "heading": "Intro",
                            "paragraphs": ["First paragraph of reasonable length." * 5],
                        }
                    ],
                }
            ]
        }
        mock_voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
            results=[FakeContextResult(embeddings=[[0.1] * 1024])]
        )
        mock_anthropic.messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="Summary of chapter themes.")]
        )
        await rag_service.index_book_structured("book1", content)
        assert rag_service.vector_store.has_book("book1")

    @pytest.mark.asyncio
    async def test_skips_empty_structured_content(self, rag_service, mock_voyage):
        mock_voyage.contextualized_embed.return_value = FakeContextEmbedResponse(
            results=[FakeContextResult(embeddings=[])]
        )
        await rag_service.index_book_structured("book1", {"chapters": []})
        # No chunks produced, so book should not be in store
        assert not rag_service.vector_store.has_book("book1")


# ── RAGService.query_simple ───────────────────────────────────────────


class TestQuerySimple:
    @pytest.mark.asyncio
    async def test_returns_answer(self, rag_service, mock_anthropic):
        mock_anthropic.messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="The meaning is...")]
        )
        answer = await rag_service.query_simple("What does this mean?", "some text")
        assert answer == "The meaning is..."

    @pytest.mark.asyncio
    async def test_uses_selected_text(self, rag_service, mock_anthropic):
        await rag_service.query_simple("Question?", "selected passage")
        call_args = mock_anthropic.messages.create.call_args
        user_content = call_args.kwargs.get("messages", call_args[1].get("messages", []))[0]["content"]
        assert "selected passage" in user_content


# ── RAGService.query_book ─────────────────────────────────────────────


class TestQueryBook:
    @pytest.mark.asyncio
    async def test_returns_rag_response(self, rag_service, mock_voyage, mock_anthropic, tmp_path):
        # First index the book
        entry = VectorEntry(
            chunk=TextChunk(
                id="book1-chunk-0",
                text="The answer is in the text.",
                metadata={"book_id": "book1", "chunk_index": 0},
            ),
            embedding=[0.1] * 1024,
        )
        await rag_service.vector_store.add_book("book1", [entry])

        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=[[0.1] * 1024])
        mock_anthropic.messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="The answer.")]
        )

        response = await rag_service.query_book(
            "book1", "What is this?", selected_text="the text", reader_position=0.0
        )
        assert isinstance(response, RAGResponse)
        assert response.answer == "The answer."
        assert len(response.sources) == 1

    @pytest.mark.asyncio
    async def test_query_book_none_selected_text(self, rag_service, mock_voyage, mock_anthropic, tmp_path):
        """query_book should handle selected_text=None without crashing."""
        entry = VectorEntry(
            chunk=TextChunk(
                id="book1-chunk-0",
                text="The answer is in the text.",
                metadata={"book_id": "book1", "chunk_index": 0},
            ),
            embedding=[0.1] * 1024,
        )
        await rag_service.vector_store.add_book("book1", [entry])

        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=[[0.1] * 1024])
        mock_anthropic.messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="The answer.")]
        )

        response = await rag_service.query_book("book1", "What is this?")
        assert isinstance(response, RAGResponse)
        assert response.answer == "The answer."

    @pytest.mark.asyncio
    async def test_no_results_returns_fallback(self, rag_service, mock_voyage):
        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=[[0.1] * 1024])
        response = await rag_service.query_book("nonexistent", "What?", reader_position=0.0)
        assert "couldn't find" in response.answer.lower()
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_position_labeling(self, rag_service, mock_voyage, mock_anthropic):
        entries = [
            VectorEntry(
                chunk=TextChunk(
                    id=f"b-chunk-{i}",
                    text=f"Chunk {i} text.",
                    metadata={"book_id": "b", "chunk_index": i},
                ),
                embedding=[float(i == 0), float(i == 1), float(i == 2)],
            )
            for i in range(3)
        ]
        await rag_service.vector_store.add_book("b", entries)

        mock_voyage.embed.return_value = FakeEmbedResponse(embeddings=[[1.0, 0.0, 0.0]])
        mock_anthropic.messages.create.return_value = FakeMessageResponse(
            content=[FakeContentBlock(text="Answer.")]
        )

        response = await rag_service.query_book(
            "b", "Question?", selected_text="Chunk 0", reader_position=0.5
        )
        assert isinstance(response, RAGResponse)


# ── RAGResponse ───────────────────────────────────────────────────────


class TestRAGResponse:
    def test_creation(self):
        r = RAGResponse(answer="Hello", sources=[{"text": "src"}])
        assert r.answer == "Hello"
        assert len(r.sources) == 1
