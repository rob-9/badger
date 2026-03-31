"""Shared fixtures for badger tests."""

import pytest
from badger.core.chunker import TextChunk
from badger.core.vector_store import VectorEntry, VectorStore


# --- Sample data ---

SAMPLE_STRUCTURED_CONTENT = {
    "chapters": [
        {
            "title": "The Beginning",
            "index": 0,
            "sections": [
                {
                    "heading": "Opening",
                    "paragraphs": [
                        "It was a dark and stormy night.",
                        "The wind howled through the trees.",
                        "Alice sat by the window, reading a book about philosophy.",
                    ],
                },
                {
                    "heading": "Reflection",
                    "paragraphs": [
                        "She wondered about the nature of consciousness.",
                        "The rain pattered against the glass.",
                    ],
                },
            ],
        },
        {
            "title": "The Journey",
            "index": 1,
            "sections": [
                {
                    "heading": None,
                    "paragraphs": [
                        "The next morning dawned bright and clear.",
                        "Alice packed her bags and set out on the road heading north.",
                        "The countryside was beautiful. Fields of wheat stretched to the horizon.",
                    ],
                },
            ],
        },
    ]
}


def make_chunk(book_id: str, index: int, text: str, **extra_meta) -> TextChunk:
    """Helper to create a TextChunk with minimal boilerplate."""
    meta = {"book_id": book_id, "chunk_index": index, **extra_meta}
    return TextChunk(id=f"{book_id}-chunk-{index}", text=text, metadata=meta)


def make_entry(book_id: str, index: int, text: str, embedding: list[float], **extra_meta) -> VectorEntry:
    """Helper to create a VectorEntry."""
    chunk = make_chunk(book_id, index, text, **extra_meta)
    return VectorEntry(chunk=chunk, embedding=embedding)


@pytest.fixture
def sample_entries():
    """5 entries with simple 3D embeddings for a test book."""
    return [
        make_entry("book1", 0, "The beginning of the story.", [1.0, 0.0, 0.0]),
        make_entry("book1", 1, "Alice went on a journey.", [0.0, 1.0, 0.0]),
        make_entry("book1", 2, "She found a mysterious artifact.", [0.0, 0.0, 1.0]),
        make_entry("book1", 3, "The villain appeared unexpectedly.", [0.7, 0.7, 0.0]),
        make_entry("book1", 4, "The story reached its climax.", [0.0, 0.7, 0.7]),
    ]


@pytest.fixture
def sample_entries_with_chapters():
    """7 entries across 2 chapters with chapter metadata."""
    return [
        make_entry("book1", 0, "The beginning of the story.", [1.0, 0.0, 0.0],
                    chapter_title="The Beginning", chapter_index=0),
        make_entry("book1", 1, "Alice went on a journey.", [0.0, 1.0, 0.0],
                    chapter_title="The Beginning", chapter_index=0),
        make_entry("book1", 2, "She found a mysterious artifact.", [0.0, 0.0, 1.0],
                    chapter_title="The Beginning", chapter_index=0),
        make_entry("book1", 3, "The villain appeared unexpectedly.", [0.7, 0.7, 0.0],
                    chapter_title="The Journey", chapter_index=1),
        make_entry("book1", 4, "The story reached its climax.", [0.0, 0.7, 0.7],
                    chapter_title="The Journey", chapter_index=1),
        make_entry("book1", 5, "Alice confronted the villain.", [0.5, 0.5, 0.0],
                    chapter_title="The Journey", chapter_index=1),
        make_entry("book1", 6, "The journey continued north.", [0.3, 0.3, 0.3],
                    chapter_title="The Journey", chapter_index=1),
    ]


@pytest.fixture
async def vector_store(tmp_path):
    """VectorStore backed by in-memory Qdrant (3-dimensional for test speed)."""
    store = VectorStore(location=":memory:", embedding_dim=3)
    store._old_json_dir = tmp_path / "no_legacy"  # prevent real .data/vectors from corrupting tests
    await store.initialize()
    return store
