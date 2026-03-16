"""
Vector store backed by Qdrant for persistent, scalable vector search.

What is a vector store?
- A database optimized for storing and searching vectors (embeddings)
- Uses similarity metrics (cosine, dot product) to find "close" vectors

This implementation:
- Qdrant-backed for persistent storage and fast approximate nearest-neighbor search
- Supports embedded mode (local file path), in-memory mode (for tests), and remote mode (URL)
- Hybrid search: Qdrant dense vectors + in-memory BM25 keyword index fused with RRF

Production alternatives: Pinecone, Chroma, Weaviate, pgvector

Learn more:
- https://qdrant.tech/documentation/
- https://www.pinecone.io/learn/vector-database/
- https://docs.trychroma.com/
"""

import asyncio
import uuid
import logging
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    FilterSelector,
    PayloadSchemaType,
)
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


# Bump this when the indexing pipeline changes (chunking strategy, embedding model, etc.)
# so that books indexed with an older version get automatically re-indexed.
CURRENT_INDEX_VERSION = 2

CHUNKS_COLLECTION = "chunks"
SUMMARIES_COLLECTION = "summaries"
EMBEDDING_DIM = 1024


class _SyncAsyncFacade:
    """
    Thin async facade over a synchronous QdrantClient.

    Used in embedded (file-path) mode so that only one QdrantClient instance
    holds the portalocker write-lock on the storage directory.  Every method
    call is forwarded to the underlying sync client via asyncio.to_thread so
    that the event loop is not blocked.
    """

    def __init__(self, client: "QdrantClient") -> None:
        self._client = client

    def __getattr__(self, name: str):
        sync_method = getattr(self._client, name)

        async def wrapper(*args, **kwargs):
            return await asyncio.to_thread(sync_method, *args, **kwargs)

        return wrapper


class QdrantVectorStore:
    """
    Qdrant-backed vector store.

    Supports three modes:
    - location=":memory:" for in-memory (tests)
    - url set for remote Qdrant instance
    - neither for embedded mode with local file storage
    """

    def __init__(
        self,
        *,
        storage_dir: str = ".data/vectors",
        url: str = "",
        api_key: str = "",
        location: str = "",
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            storage_dir: Directory for embedded Qdrant data (used when url and location are empty)
            url: Remote Qdrant URL (e.g. "http://localhost:6333")
            api_key: API key for remote Qdrant cloud instance
            location: Special location string (":memory:" for in-memory mode)
            embedding_dim: Dimensionality of embedding vectors (default 1024)
        """
        self._old_json_dir = Path(".data/vectors")
        # Whether to use asyncio.to_thread for async calls (embedded mode uses a
        # single sync client to avoid the file-lock conflict that arises when both
        # QdrantClient and AsyncQdrantClient try to lock the same embedded path).
        self._use_sync_wrapped: bool = False

        if location == ":memory:":
            self._sync_client = QdrantClient(location=":memory:")
            self._async_client = AsyncQdrantClient(location=":memory:")
            logger.info("Initialized in-memory Qdrant store")
        elif url:
            kwargs: dict = {"url": url}
            if api_key:
                kwargs["api_key"] = api_key
            self._sync_client = QdrantClient(**kwargs)
            self._async_client = AsyncQdrantClient(**kwargs)
            logger.info("Initialized remote Qdrant store at %s", url)
        else:
            # Embedded mode: use a single sync client to avoid the portalocker
            # conflict that occurs when both QdrantClient and AsyncQdrantClient
            # acquire a write-lock on the same directory concurrently.
            path = str(Path(storage_dir))
            self._sync_client = QdrantClient(path=path)
            self._async_client = None  # type: ignore[assignment]
            self._use_sync_wrapped = True
            logger.info("Initialized embedded Qdrant store at %s", path)

        self._embedding_dim = embedding_dim
        self._initialized = False
        self.bm25_indices: dict[str, BM25Index] = {}
        self._entry_cache: dict[str, list[VectorEntry]] = {}

    @property
    def _ac(self):
        """Async Qdrant client, or a thread-wrapped facade of the sync client (embedded mode)."""
        if not self._use_sync_wrapped:
            return self._async_client
        return _SyncAsyncFacade(self._sync_client)

    async def initialize(self) -> None:
        """Create collections and payload indexes if they don't exist."""
        existing_response = await self._ac.get_collections()
        existing = {c.name for c in existing_response.collections}

        if CHUNKS_COLLECTION not in existing:
            await self._ac.create_collection(
                collection_name=CHUNKS_COLLECTION,
                vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s", CHUNKS_COLLECTION)

        if SUMMARIES_COLLECTION not in existing:
            await self._ac.create_collection(
                collection_name=SUMMARIES_COLLECTION,
                vectors_config=VectorParams(size=self._embedding_dim, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s", SUMMARIES_COLLECTION)

        # Payload indexes for efficient filtering
        await self._ac.create_payload_index(
            collection_name=CHUNKS_COLLECTION,
            field_name="book_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self._ac.create_payload_index(
            collection_name=CHUNKS_COLLECTION,
            field_name="chunk_index",
            field_schema=PayloadSchemaType.INTEGER,
        )
        await self._ac.create_payload_index(
            collection_name=SUMMARIES_COLLECTION,
            field_name="book_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        await self._auto_migrate()
        self._initialized = True
        logger.info("QdrantVectorStore initialized")

    async def _ensure_initialized(self) -> None:
        """Initialize on first use."""
        if not self._initialized:
            await self.initialize()

    async def _auto_migrate(self) -> None:
        """Migrate legacy JSON vector files into Qdrant on first run."""
        try:
            # Use _ac so the count query goes to the same store as all writes
            count_result = await self._ac.count(CHUNKS_COLLECTION)
            chunk_count = count_result.count
        except Exception:
            chunk_count = 0

        if chunk_count > 0:
            return  # Already have data, skip migration

        json_dir = self._old_json_dir
        if not json_dir.exists():
            return

        # Only migrate chunk files (not *_summaries.json)
        json_files = [
            f for f in json_dir.glob("*.json")
            if not f.name.endswith("_summaries.json")
        ]

        if not json_files:
            return

        logger.info("Auto-migrating %d JSON vector file(s) to Qdrant", len(json_files))

        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                file_version = data.get("version", 1)
                if file_version < CURRENT_INDEX_VERSION:
                    logger.info(
                        "Skipping outdated index v%d (current v%d): %s",
                        file_version, CURRENT_INDEX_VERSION, json_file,
                    )
                    continue

                entries = [self._deserialize_entry(e) for e in data.get("entries", [])]
                if entries:
                    points = [self._make_point(e) for e in entries]
                    for i in range(0, len(points), 100):
                        await self._ac.upsert(
                            collection_name=CHUNKS_COLLECTION,
                            points=points[i:i + 100],
                        )
                    book_id = data.get("book_id", json_file.stem)
                    logger.info("Migrated %d chunks for book %s", len(entries), book_id)

                # Also migrate summaries if present
                summaries_file = json_file.parent / f"{json_file.stem}_summaries.json"
                if summaries_file.exists():
                    with open(summaries_file, "r") as f:
                        summary_data = json.load(f)
                    summary_entries = [
                        self._deserialize_entry(e)
                        for e in summary_data.get("entries", [])
                    ]
                    if summary_entries:
                        summary_points = [self._make_point(e) for e in summary_entries]
                        for i in range(0, len(summary_points), 100):
                            await self._ac.upsert(
                                collection_name=SUMMARIES_COLLECTION,
                                points=summary_points[i:i + 100],
                            )
                        logger.info(
                            "Migrated %d summaries for book %s",
                            len(summary_entries), json_file.stem,
                        )

            except Exception as e:
                logger.error("Error migrating %s: %s", json_file, e)

    # ── Point serialization helpers ──────────────────────────────────────

    def _make_point(self, entry: VectorEntry) -> PointStruct:
        """Convert a VectorEntry to a Qdrant PointStruct."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, entry.chunk.id))
        payload: dict = {
            "book_id": entry.chunk.metadata["book_id"],
            "chunk_index": entry.chunk.metadata["chunk_index"],
            "text": entry.chunk.text,
            "chunk_id": entry.chunk.id,
        }
        for key in ("chapter_title", "chapter_index", "section_heading"):
            if key in entry.chunk.metadata:
                payload[key] = entry.chunk.metadata[key]
        return PointStruct(id=point_id, vector=entry.embedding, payload=payload)

    def _point_to_entry(self, point) -> VectorEntry:
        """Convert a Qdrant point (ScoredPoint or Record) back to a VectorEntry."""
        metadata = {
            "book_id": point.payload["book_id"],
            "chunk_index": point.payload["chunk_index"],
        }
        for key in ("chapter_title", "chapter_index", "section_heading"):
            if key in point.payload:
                metadata[key] = point.payload[key]
        chunk = TextChunk(
            id=point.payload["chunk_id"],
            text=point.payload["text"],
            metadata=metadata,
        )
        return VectorEntry(chunk=chunk, embedding=point.vector or [])

    def _point_to_result(self, point, score: float) -> SearchResult:
        """Convert a Qdrant point to a SearchResult."""
        metadata = {
            "book_id": point.payload["book_id"],
            "chunk_index": point.payload["chunk_index"],
        }
        for key in ("chapter_title", "chapter_index", "section_heading"):
            if key in point.payload:
                metadata[key] = point.payload[key]
        chunk = TextChunk(
            id=point.payload["chunk_id"],
            text=point.payload["text"],
            metadata=metadata,
        )
        return SearchResult(chunk=chunk, score=score)

    # ── Legacy JSON helpers (kept for migration) ─────────────────────────

    def _deserialize_entry(self, data: dict) -> VectorEntry:
        """Deserialize a legacy JSON dict to VectorEntry."""
        chunk = TextChunk(
            id=data["chunk"]["id"],
            text=data["chunk"]["text"],
            metadata=data["chunk"]["metadata"],
        )
        return VectorEntry(chunk=chunk, embedding=data["embedding"])

    # ── Internal scroll helper ───────────────────────────────────────────

    async def _scroll_book_entries(
        self,
        book_id: str,
        collection: str = CHUNKS_COLLECTION,
    ) -> list[VectorEntry]:
        """Fetch all entries for a book from Qdrant, using cache for chunks collection."""
        if collection == CHUNKS_COLLECTION and book_id in self._entry_cache:
            return self._entry_cache[book_id]

        book_filter = Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
        )

        points: list = []
        offset = None
        while True:
            result = await self._ac.scroll(
                collection_name=collection,
                scroll_filter=book_filter,
                limit=256,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            batch, next_offset = result
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        entries = [self._point_to_entry(p) for p in points]
        entries.sort(key=lambda e: e.chunk.metadata["chunk_index"])

        if collection == CHUNKS_COLLECTION:
            self._entry_cache[book_id] = entries

        return entries

    # ── Public API ───────────────────────────────────────────────────────

    async def add_book(self, book_id: str, entries: list[VectorEntry]) -> None:
        """
        Add chunks with their embeddings for a specific book.

        Args:
            book_id: Unique identifier for the book
            entries: List of VectorEntry objects
        """
        await self._ensure_initialized()

        points = [self._make_point(e) for e in entries]
        for i in range(0, len(points), 100):
            await self._ac.upsert(
                collection_name=CHUNKS_COLLECTION,
                points=points[i:i + 100],
            )

        self.bm25_indices[book_id] = BM25Index(entries)
        self._entry_cache[book_id] = sorted(
            entries, key=lambda e: e.chunk.metadata["chunk_index"]
        )
        logger.info("Added %d chunks for book %s (BM25 index built)", len(entries), book_id)

    async def remove_book(self, book_id: str) -> None:
        """Remove a book from the store (chunks, summaries, BM25 cache, entry cache)."""
        await self._ensure_initialized()

        book_filter = Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
        )

        await self._ac.delete(
            collection_name=CHUNKS_COLLECTION,
            points_selector=FilterSelector(filter=book_filter),
        )
        await self._ac.delete(
            collection_name=SUMMARIES_COLLECTION,
            points_selector=FilterSelector(filter=book_filter),
        )

        self.bm25_indices.pop(book_id, None)
        self._entry_cache.pop(book_id, None)
        logger.info("Removed book %s from Qdrant", book_id)

    def has_book(self, book_id: str) -> bool:
        """Check if a book has been indexed."""
        # Fast path: check in-memory cache first (works for in-memory Qdrant mode
        # where the sync and async clients are separate instances)
        if book_id in self._entry_cache:
            return True
        try:
            result = self._sync_client.count(
                collection_name=CHUNKS_COLLECTION,
                count_filter=Filter(
                    must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
                ),
            )
            return result.count > 0
        except Exception:
            return False

    async def get_total_chunks(self, book_id: str) -> int:
        """Return total number of indexed chunks for a book."""
        await self._ensure_initialized()
        try:
            result = await self._ac.count(
                collection_name=CHUNKS_COLLECTION,
                count_filter=Filter(
                    must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
                ),
            )
            return result.count
        except Exception:
            return 0

    async def search(
        self,
        book_id: str,
        query_embedding: list[float],
        top_k: int = 5,
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
        await self._ensure_initialized()
        logger.debug("Searching book %s (top %d)", book_id, top_k)

        book_filter = Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
        )

        results = await self._ac.query_points(
            collection_name=CHUNKS_COLLECTION,
            query=query_embedding,
            query_filter=book_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._point_to_result(p, p.score) for p in results.points]

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

        # Semantic search (Qdrant ANN)
        semantic_results = await self.search(book_id, query_embedding, top_k=top_k)

        # BM25 keyword search
        bm25 = await self._get_bm25(book_id)
        bm25_results = bm25.search(query_text, top_k=top_k) if bm25 else []

        logger.debug("Semantic: %d results, BM25: %d results", len(semantic_results), len(bm25_results))

        # Fuse both ranked lists — chunks appearing in both get higher scores
        fused = reciprocal_rank_fusion(semantic_results, bm25_results)[:top_k]

        logger.debug("Fused: %d results", len(fused))
        return fused

    async def keyword_search(self, book_id: str, text: str) -> list[SearchResult]:
        """Find all chunks that contain the given text (case-insensitive)."""
        await self._ensure_initialized()
        if not text:
            return []
        entries = await self._scroll_book_entries(book_id)
        needle = text.lower()
        return [
            SearchResult(chunk=e.chunk, score=1.0)
            for e in entries
            if needle in e.chunk.text.lower()
        ]

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        """Normalize smart quotes, dashes, and whitespace to ASCII equivalents."""
        replacements = {
            '\u2018': "'", '\u2019': "'",  # smart single quotes
            '\u201c': '"', '\u201d': '"',  # smart double quotes
            '\u2014': '--', '\u2013': '-',  # em/en dashes
            '\u2026': '...',               # ellipsis
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return re.sub(r'\s+', ' ', text).strip()

    async def find_chunk_containing(self, book_id: str, text: str) -> Optional[int]:
        """Find the chunk index that contains the given text.

        Uses progressive matching: tries full text, then 200, 100, 50 char
        snippets with normalized quotes and case-insensitive comparison.

        Returns the chunk index if found, or None if no match.
        """
        await self._ensure_initialized()
        if not text:
            return None

        entries = await self._scroll_book_entries(book_id)
        if not entries:
            return None

        normalized = self._normalize_quotes(text)

        # Precompute normalized chunk texts once
        normalized_chunks = [
            (self._normalize_quotes(e.chunk.text).lower(), e.chunk.metadata["chunk_index"])
            for e in entries
        ]

        # Progressive snippet lengths — try longest first for best accuracy
        # Deduplicate and skip lengths longer than the text
        snippet_lengths = list(dict.fromkeys(
            l for l in [len(normalized), 200, 100, 50] if l <= len(normalized)
        ))

        for length in snippet_lengths:
            snippet = normalized[:length].lower()
            if not snippet:
                continue
            for chunk_text, chunk_index in normalized_chunks:
                if snippet in chunk_text:
                    return chunk_index

        return None

    async def get_chunks_by_range(
        self, book_id: str, start_idx: int, end_idx: int
    ) -> list[SearchResult]:
        """Return chunks by index range (for proximity retrieval). No embedding needed."""
        await self._ensure_initialized()

        book_filter = Filter(
            must=[
                FieldCondition(key="book_id", match=MatchValue(value=book_id)),
                FieldCondition(
                    key="chunk_index",
                    range=Range(gte=start_idx, lte=end_idx),
                ),
            ]
        )

        points: list = []
        offset = None
        while True:
            result = await self._ac.scroll(
                collection_name=CHUNKS_COLLECTION,
                scroll_filter=book_filter,
                limit=256,
                offset=offset,
                with_vectors=False,
                with_payload=True,
            )
            batch, next_offset = result
            points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        return [self._point_to_result(p, 1.0) for p in points]

    async def save_summaries(self, book_id: str, entries: list[VectorEntry]) -> None:
        """Save chapter summary vectors to the summaries collection."""
        await self._ensure_initialized()

        points = [self._make_point(e) for e in entries]
        for i in range(0, len(points), 100):
            await self._ac.upsert(
                collection_name=SUMMARIES_COLLECTION,
                points=points[i:i + 100],
            )
        logger.info("Saved %d chapter summaries for book %s", len(entries), book_id)

    async def search_summaries(
        self,
        book_id: str,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """
        Search chapter-level summaries for broad thematic matches.

        Chapter summaries provide high-level context that passage-level chunks
        miss — useful for "what is the book's stance on X" type questions.
        """
        await self._ensure_initialized()

        book_filter = Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
        )

        results = await self._ac.query_points(
            collection_name=SUMMARIES_COLLECTION,
            query=query_embedding,
            query_filter=book_filter,
            limit=top_k,
            with_payload=True,
        )
        return [self._point_to_result(p, p.score) for p in results.points]

    async def _get_bm25(self, book_id: str) -> Optional[BM25Index]:
        """
        Get or lazily build the BM25 index for a book.

        The index is built on first access and cached in memory for subsequent queries.
        """
        if book_id in self.bm25_indices:
            return self.bm25_indices[book_id]

        entries = await self._scroll_book_entries(book_id)
        if not entries:
            return None

        self.bm25_indices[book_id] = BM25Index(entries)
        return self.bm25_indices[book_id]

    def get_stats(self) -> dict:
        """Get statistics about the store (sync)."""
        try:
            total = self._sync_client.count(collection_name=CHUNKS_COLLECTION).count
            if total == 0 and self._entry_cache:
                # Sync client may be a separate in-memory instance; fall back to cache
                total = sum(len(v) for v in self._entry_cache.values())
            return {"book_count": 0, "total_chunks": total}
        except Exception:
            # Sync client doesn't have the collection (in-memory mode) — use cache
            total = sum(len(v) for v in self._entry_cache.values())
            return {"book_count": 0, "total_chunks": total}


VectorStore = QdrantVectorStore
