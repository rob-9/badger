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
import re
from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Optional
from rank_bm25 import BM25Okapi
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


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase words for BM25 matching (e.g. 'Hello World!' → ['hello', 'world'])."""
    return re.findall(r'\w+', text.lower())


class BM25Index:
    """
    BM25 (Best Match 25) keyword index over document chunks.

    BM25 is a classical text retrieval algorithm that ranks documents by
    term frequency (how often a word appears in a chunk) and inverse document
    frequency (how rare a word is across all chunks). Unlike semantic/vector
    search, BM25 excels at exact keyword matching — it will find "photosynthesis"
    even if the embedding model doesn't place it near the query vector.

    Used alongside vector search in hybrid retrieval for better recall.
    """

    def __init__(self, entries: list[VectorEntry]):
        self.entries = entries
        corpus = [_tokenize(e.chunk.text) for e in entries]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=self.entries[idx].chunk, score=float(score))
            for idx, score in ranked
            if score > 0
        ]


def reciprocal_rank_fusion(*result_lists: list[SearchResult], k: int = 60) -> list[SearchResult]:
    """
    Merge multiple ranked result lists into one using Reciprocal Rank Fusion (RRF).

    RRF is a simple, robust way to combine results from different retrieval methods
    (e.g. semantic search + BM25). Each result gets a score of 1/(k + rank + 1) from
    each list it appears in, and scores are summed. The constant k=60 dampens the
    influence of high ranks so that a result appearing in multiple lists is boosted
    more than one that ranks #1 in only one list.

    Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
    outperforms Condorcet and individual Rank Learning Methods"
    """
    scores: dict[str, float] = {}
    chunks: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
            if chunk_id not in chunks:
                chunks[chunk_id] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        SearchResult(chunk=chunks[cid].chunk, score=score)
        for cid, score in ranked
    ]


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
        self.bm25_indices: dict[str, BM25Index] = {}
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
        self.bm25_indices[book_id] = BM25Index(entries)
        logger.info("Added %d chunks for book %s (BM25 index built)", len(entries), book_id)

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

    async def get_chunks_by_range(
        self, book_id: str, start_idx: int, end_idx: int
    ) -> list[SearchResult]:
        """Return chunks by index range (for proximity retrieval). No embedding needed."""
        entries = self.entries.get(book_id) or await self.load_from_file(book_id)
        if not entries:
            return []
        return [
            SearchResult(chunk=e.chunk, score=1.0)
            for e in entries
            if start_idx <= e.chunk.metadata['chunk_index'] <= end_idx
        ]

    async def find_chunk_containing(self, book_id: str, text: str) -> int:
        """Find the chunk index that contains the given text."""
        entries = self.entries.get(book_id)
        if not entries:
            entries = await self.load_from_file(book_id)
            if entries:
                self.entries[book_id] = entries
        if not entries or not text:
            return 0
        # Normalize whitespace to match chunker output
        normalized = re.sub(r'\s+', ' ', text).strip()
        snippet = normalized[:50]
        for entry in entries:
            if snippet in entry.chunk.text:
                return entry.chunk.metadata['chunk_index']
        return 0

    async def keyword_search(self, book_id: str, text: str) -> list[SearchResult]:
        """Find all chunks that contain the given text (case-insensitive)."""
        entries = self.entries.get(book_id)
        if not entries:
            entries = await self.load_from_file(book_id)
            if entries:
                self.entries[book_id] = entries
        if not entries or not text:
            return []
        needle = text.lower()
        return [
            SearchResult(chunk=e.chunk, score=1.0)
            for e in entries
            if needle in e.chunk.text.lower()
        ]

    async def _get_bm25(self, book_id: str) -> Optional[BM25Index]:
        """
        Get or lazily build the BM25 index for a book.

        The index is built on first access (e.g. when a book was loaded from disk
        rather than freshly indexed) and cached in memory for subsequent queries.
        """
        if book_id in self.bm25_indices:
            return self.bm25_indices[book_id]

        entries = self.entries.get(book_id)
        if not entries:
            entries = await self.load_from_file(book_id)
            if entries:
                self.entries[book_id] = entries

        if not entries:
            return None

        self.bm25_indices[book_id] = BM25Index(entries)
        return self.bm25_indices[book_id]

    async def hybrid_search(
        self,
        book_id: str,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 20,
    ) -> list[SearchResult]:
        """
        Combine semantic (vector) search with BM25 (keyword) search using
        Reciprocal Rank Fusion for better recall.

        Semantic search finds chunks with similar meaning (good for paraphrases,
        conceptual matches). BM25 finds chunks with matching keywords (good for
        exact terms, names, technical vocabulary). Fusing both via RRF gives
        better retrieval than either alone — chunks that score well in both
        methods get boosted to the top.

        Returns top_k results (typically 20) for downstream reranking to narrow
        down to the final 5.
        """
        logger.debug("Hybrid search for book %s (top %d)", book_id, top_k)

        # Semantic search (vector cosine similarity)
        semantic_results = await self.search(book_id, query_embedding, top_k=top_k)

        # BM25 keyword search
        bm25 = await self._get_bm25(book_id)
        bm25_results = bm25.search(query_text, top_k=top_k) if bm25 else []

        logger.debug("Semantic: %d results, BM25: %d results", len(semantic_results), len(bm25_results))

        # Fuse both ranked lists — chunks appearing in both get higher scores
        fused = reciprocal_rank_fusion(semantic_results, bm25_results)[:top_k]

        logger.debug("Fused: %d results", len(fused))
        return fused

    def get_stats(self) -> dict:
        """Get statistics about the store."""
        total_chunks = sum(len(entries) for entries in self.entries.values())
        return {
            'book_count': len(self.entries),
            'total_chunks': total_chunks
        }
