"""Shared fixtures for boom tests."""

import pytest
from boom.core.chunker import TextChunk
from boom.core.vector_store import VectorEntry, VectorStore, SearchResult


# --- Sample data ---

SAMPLE_BOOK_TEXT = (
    "Chapter 1: The Beginning. "
    "It was a dark and stormy night. The wind howled through the trees. "
    "Alice sat by the window, reading a book about philosophy. "
    "She wondered about the nature of consciousness. "
    "The rain pattered against the glass. "
    "Chapter 2: The Journey. "
    "The next morning dawned bright and clear. Alice packed her bags. "
    "She set out on the road heading north. The countryside was beautiful. "
    "Fields of wheat stretched to the horizon. Birds sang in the hedgerows. "
    "She walked for hours before reaching the village."
)

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


def make_entry(book_id: str, index: int, text: str, embedding: list[float]) -> VectorEntry:
    """Helper to create a VectorEntry."""
    chunk = make_chunk(book_id, index, text)
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
def vector_store(tmp_path):
    """VectorStore backed by a temp directory."""
    return VectorStore(storage_dir=str(tmp_path / "vectors"))
