"""
Vector store with file system persistence.

What is a vector store?
- A database optimized for storing and searching vectors (embeddings)
- Uses similarity metrics (cosine, dot product) to find "close" vectors

This implementation:
- In-memory for fast queries
- File system persistence (survives restarts)

Production alternatives: Pinecone, Chroma, Weaviate, pgvector

Learn more:
- https://www.pinecone.io/learn/vector-database/
- https://docs.trychroma.com/
"""

import logging
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Optional
from .chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A chunk with its embedding."""
    chunk: TextChunk
    embedding: list[float]


@dataclass
class SearchResult:
    """A search result with similarity score."""
    chunk: TextChunk
    score: float  # Cosine similarity: 1 = identical, 0 = orthogonal, -1 = opposite


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Measures the angle between two vectors (ignores magnitude).
    Perfect for comparing embeddings because we care about direction, not length.

    Formula: cos(θ) = (A · B) / (||A|| × ||B||)

    Where:
    - A · B = dot product (sum of element-wise multiplication)
    - ||A|| = magnitude (sqrt of sum of squares)

    Returns: -1 to 1 (1 = most similar)
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")

    dot_product = sum(a_i * b_i for a_i, b_i in zip(a, b))
    magnitude_a = math.sqrt(sum(a_i * a_i for a_i in a))
    magnitude_b = math.sqrt(sum(b_i * b_i for b_i in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class VectorStore:
    """
    Simple file-based vector store.

    For production, replace with:
    - Pinecone (hosted, scalable)
    - Chroma (local, Python-based)
    - Qdrant (local or hosted)
    """

    def __init__(self, storage_dir: str = ".data/vectors"):
        """Initialize vector store with storage directory."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, list[VectorEntry]] = {}
        logger.info("Initialized with storage: %s", self.storage_dir)

    def _get_file_path(self, book_id: str) -> Path:
        """Get file path for a book's vectors."""
        return self.storage_dir / f"{book_id}.json"

    def _serialize_entry(self, entry: VectorEntry) -> dict:
        """Serialize VectorEntry to JSON-compatible dict."""
        return {
            'chunk': {
                'id': entry.chunk.id,
                'text': entry.chunk.text,
                'metadata': entry.chunk.metadata
            },
            'embedding': entry.embedding
        }

    def _deserialize_entry(self, data: dict) -> VectorEntry:
        """Deserialize dict to VectorEntry."""
        chunk = TextChunk(
            id=data['chunk']['id'],
            text=data['chunk']['text'],
            metadata=data['chunk']['metadata']
        )
        return VectorEntry(chunk=chunk, embedding=data['embedding'])

    async def save_to_file(self, book_id: str, entries: list[VectorEntry]) -> None:
        """Save vectors to file."""
        file_path = self._get_file_path(book_id)

        data = {
            'book_id': book_id,
            'entry_count': len(entries),
            'entries': [self._serialize_entry(e) for e in entries]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f)

        logger.info("Saved %d entries to %s", len(entries), file_path)

    async def load_from_file(self, book_id: str) -> Optional[list[VectorEntry]]:
        """Load vectors from file."""
        file_path = self._get_file_path(book_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            entries = [self._deserialize_entry(e) for e in data['entries']]
            logger.info("Loaded %d entries from %s", len(entries), file_path)
            return entries
        except Exception as e:
            logger.error("Error loading from file: %s", e)
            return None

    async def add_book(self, book_id: str, entries: list[VectorEntry]) -> None:
        """
        Add chunks with their embeddings for a specific book.

        Args:
            book_id: Unique identifier for the book
            entries: List of VectorEntry objects
        """
        self.entries[book_id] = entries
        logger.info("Added %d chunks for book %s", len(entries), book_id)

        # Persist to file system
        await self.save_to_file(book_id, entries)

    async def search(
        self,
        book_id: str,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[SearchResult]:
        """
        Search for similar chunks within a book.

        Args:
            book_id: The book to search in
            query_embedding: The embedded question
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by similarity
        """
        logger.debug("Searching book %s (top %d)", book_id, top_k)

        # Try to get from memory first
        book_entries = self.entries.get(book_id)

        # If not in memory, try loading from file
        if not book_entries:
            logger.debug("Not in memory, loading from disk")
            book_entries = await self.load_from_file(book_id)

            if book_entries:
                # Cache in memory for future queries
                self.entries[book_id] = book_entries

        if not book_entries:
            logger.info("No entries found for book: %s", book_id)
            return []

        logger.debug("Comparing against %d chunks", len(book_entries))

        # Calculate similarity for each chunk
        results = [
            SearchResult(chunk=entry.chunk, score=cosine_similarity(query_embedding, entry.embedding))
            for entry in book_entries
        ]

        # Sort by similarity (highest first) and take top K
        top_results = sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

        if top_results:
            logger.debug("Best match similarity: %.3f", top_results[0].score)

        return top_results

    def has_book(self, book_id: str) -> bool:
        """Check if a book has been indexed."""
        return book_id in self.entries or self._get_file_path(book_id).exists()

    async def remove_book(self, book_id: str) -> None:
        """Remove a book from the store."""
        # Remove from memory
        if book_id in self.entries:
            del self.entries[book_id]

        # Remove from file system
        file_path = self._get_file_path(book_id)
        if file_path.exists():
            file_path.unlink()
            logger.info("Deleted from disk: %s", book_id)

    async def get_total_chunks(self, book_id: str) -> int:
        """Return total number of indexed chunks for a book."""
        entries = self.entries.get(book_id) or await self.load_from_file(book_id)
        return len(entries) if entries else 0

    def get_stats(self) -> dict:
        """Get statistics about the store."""
        total_chunks = sum(len(entries) for entries in self.entries.values())
        return {
            'book_count': len(self.entries),
            'total_chunks': total_chunks
        }
