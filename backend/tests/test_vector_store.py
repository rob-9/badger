"""Tests for badger.core.vector_store — vector storage, search, BM25, RRF."""

import json
import pytest

from badger.core.chunker import TextChunk
from badger.core.vector_store import (
    VectorEntry,
    VectorStore,
    SearchResult,
    BM25Index,
    reciprocal_rank_fusion,
    _tokenize,
    CURRENT_INDEX_VERSION,
)
from tests.conftest import make_entry, make_chunk


# ── _tokenize ─────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World!") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert _tokenize("it's a test.") == ["it", "s", "a", "test"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_included(self):
        assert _tokenize("Chapter 3") == ["chapter", "3"]


# ── BM25Index ─────────────────────────────────────────────────────────


class TestBM25Index:
    def test_basic_search(self):
        entries = [
            make_entry("b", 0, "The cat sat on the mat.", [0.0]),
            make_entry("b", 1, "Dogs are loyal animals.", [0.0]),
            make_entry("b", 2, "The cat chased a mouse.", [0.0]),
        ]
        index = BM25Index(entries)
        results = index.search("cat", top_k=2)
        assert len(results) == 2
        texts = [r.chunk.text for r in results]
        assert "The cat sat on the mat." in texts
        assert "The cat chased a mouse." in texts

    def test_no_match(self):
        entries = [make_entry("b", 0, "Hello world.", [0.0])]
        index = BM25Index(entries)
        results = index.search("xyz", top_k=5)
        assert results == []

    def test_score_ordering(self):
        # BM25 IDF requires the term to appear in less than half the docs
        entries = [
            make_entry("b", 0, "cat cat cat", [0.0]),
            make_entry("b", 1, "cat dog", [0.0]),
            make_entry("b", 2, "dog bird fish", [0.0]),
            make_entry("b", 3, "elephant giraffe", [0.0]),
            make_entry("b", 4, "rabbit fox deer", [0.0]),
            make_entry("b", 5, "whale dolphin", [0.0]),
        ]
        index = BM25Index(entries)
        results = index.search("cat", top_k=2)
        assert len(results) == 2
        # "cat cat cat" should score higher than "cat dog"
        assert results[0].chunk.text == "cat cat cat"
        assert results[0].score >= results[1].score

    def test_top_k_limit(self):
        entries = [make_entry("b", i, f"word{i} common term", [0.0]) for i in range(10)]
        index = BM25Index(entries)
        results = index.search("common", top_k=3)
        assert len(results) == 3


# ── reciprocal_rank_fusion ────────────────────────────────────────────


class TestReciprocalRankFusion:
    def _make_result(self, chunk_id: str, score: float = 1.0) -> SearchResult:
        chunk = make_chunk("b", 0, f"text for {chunk_id}")
        chunk.id = chunk_id
        return SearchResult(chunk=chunk, score=score)

    def test_single_list(self):
        results = [self._make_result("a"), self._make_result("b")]
        fused = reciprocal_rank_fusion(results)
        assert len(fused) == 2
        assert fused[0].chunk.id == "a"  # higher rank in input

    def test_two_lists_boost_overlap(self):
        list1 = [self._make_result("a"), self._make_result("b")]
        list2 = [self._make_result("b"), self._make_result("c")]
        fused = reciprocal_rank_fusion(list1, list2)
        # "b" appears in both lists, so it should rank higher
        ids = [r.chunk.id for r in fused]
        assert "b" in ids[:2]  # b should be near the top

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []

    def test_single_item(self):
        fused = reciprocal_rank_fusion([self._make_result("only")])
        assert len(fused) == 1

    def test_deduplication(self):
        """Same chunk in both lists should appear once in output."""
        list1 = [self._make_result("x")]
        list2 = [self._make_result("x")]
        fused = reciprocal_rank_fusion(list1, list2)
        assert len(fused) == 1


# ── VectorStore ───────────────────────────────────────────────────────


class TestVectorStore:

    @pytest.mark.asyncio
    async def test_add_and_search(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.search("book1", [1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # Closest to [1,0,0] should be entry 0
        assert results[0].chunk.metadata["chunk_index"] == 0
        assert results[0].score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_search_nonexistent_book(self, vector_store):
        results = await vector_store.search("nobook", [1.0, 0.0], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_has_book(self, vector_store, sample_entries):
        assert not vector_store.has_book("book1")
        await vector_store.add_book("book1", sample_entries)
        assert vector_store.has_book("book1")

    @pytest.mark.asyncio
    async def test_remove_book(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        assert vector_store.has_book("book1")
        await vector_store.remove_book("book1")
        assert not vector_store.has_book("book1")
        # Search should return empty
        results = await vector_store.search("book1", [1.0, 0.0, 0.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_total_chunks(self, vector_store, sample_entries):
        assert await vector_store.get_total_chunks("book1") == 0
        await vector_store.add_book("book1", sample_entries)
        assert await vector_store.get_total_chunks("book1") == 5

    @pytest.mark.asyncio
    async def test_get_chunks_by_range(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.get_chunks_by_range("book1", 1, 3)
        indices = [r.chunk.metadata["chunk_index"] for r in results]
        assert sorted(indices) == [1, 2, 3]
        # All scores should be 1.0 (proximity, not similarity)
        for r in results:
            assert r.score == 1.0

    @pytest.mark.asyncio
    async def test_get_chunks_by_range_empty(self, vector_store):
        results = await vector_store.get_chunks_by_range("nope", 0, 5)
        assert results == []

    @pytest.mark.asyncio
    async def test_find_chunk_containing(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        idx = await vector_store.find_chunk_containing("book1", "mysterious artifact")
        assert idx == 2

    @pytest.mark.asyncio
    async def test_find_chunk_containing_not_found(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        idx = await vector_store.find_chunk_containing("book1", "nonexistent text xyz")
        assert idx == 0  # fallback to 0

    @pytest.mark.asyncio
    async def test_find_chunk_containing_empty_text(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        idx = await vector_store.find_chunk_containing("book1", "")
        assert idx == 0

    @pytest.mark.asyncio
    async def test_keyword_search(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.keyword_search("book1", "journey")
        assert len(results) == 1
        assert "journey" in results[0].chunk.text.lower()

    @pytest.mark.asyncio
    async def test_keyword_search_case_insensitive(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.keyword_search("book1", "ALICE")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_keyword_search_no_match(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.keyword_search("book1", "xyznonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_empty_text(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.keyword_search("book1", "")
        assert results == []

    @pytest.mark.asyncio
    async def test_hybrid_search(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        results = await vector_store.hybrid_search(
            "book1", [1.0, 0.0, 0.0], "beginning story", top_k=3
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_book(self, vector_store):
        results = await vector_store.hybrid_search("nobook", [1.0], "test", top_k=5)
        assert results == []


class TestVectorStorePersistence:

    @pytest.mark.asyncio
    async def test_save_and_load(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)

        # Create a fresh store pointing at the same directory
        store2 = VectorStore(storage_dir=str(vector_store.storage_dir))
        loaded = await store2.load_from_file("book1")
        assert loaded is not None
        assert len(loaded) == 5
        assert loaded[0].chunk.text == sample_entries[0].chunk.text
        assert loaded[0].embedding == sample_entries[0].embedding

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, vector_store):
        result = await vector_store.load_from_file("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_version_check_outdated(self, vector_store, sample_entries, tmp_path):
        """Outdated index version should return None on load."""
        await vector_store.add_book("book1", sample_entries)
        # Manually downgrade the version in the file
        file_path = vector_store._get_file_path("book1")
        with open(file_path, "r") as f:
            data = json.load(f)
        data["version"] = 0
        with open(file_path, "w") as f:
            json.dump(data, f)
        # Clear memory cache
        vector_store.entries.clear()
        loaded = await vector_store.load_from_file("book1")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_has_book_from_disk(self, tmp_path, sample_entries):
        store1 = VectorStore(storage_dir=str(tmp_path / "vectors"))
        await store1.add_book("book1", sample_entries)

        # Fresh store with no memory cache
        store2 = VectorStore(storage_dir=str(tmp_path / "vectors"))
        assert store2.has_book("book1")

    @pytest.mark.asyncio
    async def test_has_book_outdated_version_on_disk(self, tmp_path, sample_entries):
        store = VectorStore(storage_dir=str(tmp_path / "vectors"))
        await store.add_book("book1", sample_entries)
        # Downgrade version on disk
        file_path = store._get_file_path("book1")
        with open(file_path, "r") as f:
            data = json.load(f)
        data["version"] = 0
        with open(file_path, "w") as f:
            json.dump(data, f)
        # Clear memory
        store.entries.clear()
        assert not store.has_book("book1")

    @pytest.mark.asyncio
    async def test_search_loads_from_disk(self, tmp_path, sample_entries):
        store1 = VectorStore(storage_dir=str(tmp_path / "vectors"))
        await store1.add_book("book1", sample_entries)

        store2 = VectorStore(storage_dir=str(tmp_path / "vectors"))
        results = await store2.search("book1", [1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.metadata["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_remove_deletes_files(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        file_path = vector_store._get_file_path("book1")
        assert file_path.exists()
        await vector_store.remove_book("book1")
        assert not file_path.exists()


class TestVectorStoreSummaries:

    @pytest.mark.asyncio
    async def test_save_and_load_summaries(self, vector_store):
        entries = [
            make_entry("book1", 0, "Chapter 1 summary about themes.", [1.0, 0.0, 0.0]),
            make_entry("book1", 1, "Chapter 2 summary about conflict.", [0.0, 1.0, 0.0]),
        ]
        await vector_store.save_summaries("book1", entries)

        loaded = await vector_store.load_summaries("book1")
        assert loaded is not None
        assert len(loaded) == 2

    @pytest.mark.asyncio
    async def test_search_summaries(self, vector_store):
        entries = [
            make_entry("book1", 0, "Themes of love and loss.", [1.0, 0.0, 0.0]),
            make_entry("book1", 1, "Conflict and resolution.", [0.0, 1.0, 0.0]),
        ]
        await vector_store.save_summaries("book1", entries)

        results = await vector_store.search_summaries("book1", [0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert "love" in results[0].chunk.text

    @pytest.mark.asyncio
    async def test_search_summaries_no_summaries(self, vector_store):
        results = await vector_store.search_summaries("nobook", [1.0, 0.0], top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_remove_deletes_summary_files(self, vector_store):
        entries = [make_entry("book1", 0, "Summary.", [1.0])]
        await vector_store.save_summaries("book1", entries)
        summaries_path = vector_store._get_summaries_path("book1")
        assert summaries_path.exists()
        await vector_store.remove_book("book1")
        assert not summaries_path.exists()

    @pytest.mark.asyncio
    async def test_summaries_cached_in_memory(self, vector_store):
        entries = [make_entry("book1", 0, "Summary.", [1.0, 0.0])]
        await vector_store.save_summaries("book1", entries)
        assert "book1" in vector_store.summary_entries


class TestVectorStoreStats:
    @pytest.mark.asyncio
    async def test_empty_stats(self, vector_store):
        stats = vector_store.get_stats()
        assert stats == {"book_count": 0, "total_chunks": 0}

    @pytest.mark.asyncio
    async def test_stats_after_add(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        stats = vector_store.get_stats()
        assert stats["book_count"] == 1
        assert stats["total_chunks"] == 5


class TestVectorStoreSerialization:
    def test_serialize_deserialize_roundtrip(self, vector_store):
        entry = make_entry("b", 0, "Some text.", [1.0, 2.0, 3.0])
        serialized = vector_store._serialize_entry(entry)
        deserialized = vector_store._deserialize_entry(serialized)
        assert deserialized.chunk.id == entry.chunk.id
        assert deserialized.chunk.text == entry.chunk.text
        assert deserialized.chunk.metadata == entry.chunk.metadata
        assert deserialized.embedding == entry.embedding


class TestBM25LazyBuild:
    @pytest.mark.asyncio
    async def test_bm25_built_lazily_on_hybrid_search(self, tmp_path, sample_entries):
        """BM25 index should be built on demand when loading from disk."""
        store1 = VectorStore(storage_dir=str(tmp_path / "vectors"))
        await store1.add_book("book1", sample_entries)

        # Fresh store loads from disk — no BM25 index yet
        store2 = VectorStore(storage_dir=str(tmp_path / "vectors"))
        assert "book1" not in store2.bm25_indices

        # hybrid_search triggers lazy BM25 build
        results = await store2.hybrid_search("book1", [1.0, 0.0, 0.0], "beginning", top_k=3)
        assert "book1" in store2.bm25_indices
        assert len(results) >= 1
