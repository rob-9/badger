"""Tests for badger.core.vector_store — vector storage, search, BM25, RRF."""

import json
from pathlib import Path

import pytest

from badger.core.chunker import TextChunk
from badger.core.vector_store import (
    VectorEntry,
    VectorStore,
    SearchResult,
    BM25Index,
    reciprocal_rank_fusion,
    _tokenize,
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
        results = await vector_store.search("nobook", [1.0, 0.0, 0.0], top_k=5)
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
        assert idx is None

    @pytest.mark.asyncio
    async def test_find_chunk_containing_empty_text(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        idx = await vector_store.find_chunk_containing("book1", "")
        assert idx is None

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
        results = await vector_store.hybrid_search("nobook", [1.0, 0.0, 0.0], "test", top_k=5)
        assert results == []


class TestVectorStorePersistence:

    @pytest.mark.asyncio
    async def test_data_persists_across_clients(self, tmp_path, sample_entries):
        """Data written by one client is readable by another pointing to the same path."""
        storage = str(tmp_path / "qdrant")
        store1 = VectorStore(storage_dir=storage, embedding_dim=3)
        await store1.initialize()
        await store1.add_book("book1", sample_entries)
        # Release the file lock before opening a second client on the same path.
        # Embedded mode uses only _sync_client; _async_client is None.
        store1._sync_client.close()

        store2 = VectorStore(storage_dir=storage, embedding_dim=3)
        await store2.initialize()
        assert store2.has_book("book1")
        results = await store2.search("book1", [1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.metadata["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_has_book_from_persisted(self, tmp_path, sample_entries):
        storage = str(tmp_path / "qdrant")
        store1 = VectorStore(storage_dir=storage, embedding_dim=3)
        await store1.initialize()
        await store1.add_book("book1", sample_entries)
        # Release the file lock before opening a second client on the same path.
        # Embedded mode uses only _sync_client; _async_client is None.
        store1._sync_client.close()

        store2 = VectorStore(storage_dir=storage, embedding_dim=3)
        await store2.initialize()
        assert store2.has_book("book1")

    @pytest.mark.asyncio
    async def test_remove_deletes_from_collection(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        assert vector_store.has_book("book1")
        await vector_store.remove_book("book1")
        assert not vector_store.has_book("book1")
        results = await vector_store.search("book1", [1.0, 0.0, 0.0])
        assert results == []


class TestVectorStoreSummaries:

    @pytest.mark.asyncio
    async def test_save_and_search_summaries(self, vector_store):
        entries = [
            make_entry("book1", 0, "Chapter 1 summary about themes.", [1.0, 0.0, 0.0]),
            make_entry("book1", 1, "Chapter 2 summary about conflict.", [0.0, 1.0, 0.0]),
        ]
        await vector_store.save_summaries("book1", entries)
        results = await vector_store.search_summaries("book1", [0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert "themes" in results[0].chunk.text

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
        results = await vector_store.search_summaries("nobook", [1.0, 0.0, 0.0], top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_remove_deletes_summaries(self, vector_store):
        entries = [make_entry("book1", 0, "Summary.", [1.0, 0.0, 0.0])]
        await vector_store.save_summaries("book1", entries)
        results = await vector_store.search_summaries("book1", [1.0, 0.0, 0.0])
        assert len(results) == 1
        await vector_store.remove_book("book1")
        results = await vector_store.search_summaries("book1", [1.0, 0.0, 0.0])
        assert results == []


class TestVectorStoreStats:
    @pytest.mark.asyncio
    async def test_empty_stats(self, vector_store):
        stats = vector_store.get_stats()
        assert stats["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_add(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        stats = vector_store.get_stats()
        assert stats["total_chunks"] == 5


class TestAutoMigration:
    """Test that legacy .data/vectors/*.json files are auto-migrated into Qdrant."""

    def _write_legacy_json(self, dir_path: Path, book_id: str, entries: list, version: int = 2):
        """Write a legacy-format JSON vector file."""
        data = {
            "version": version,
            "book_id": book_id,
            "entry_count": len(entries),
            "entries": [
                {
                    "chunk": {
                        "id": e.chunk.id,
                        "text": e.chunk.text,
                        "metadata": e.chunk.metadata,
                    },
                    "embedding": e.embedding,
                }
                for e in entries
            ],
        }
        file_path = dir_path / f"{book_id}.json"
        file_path.write_text(json.dumps(data))
        return file_path

    @pytest.mark.asyncio
    async def test_migrates_legacy_chunks(self, tmp_path, sample_entries):
        """Legacy JSON chunks should be imported into Qdrant on first initialize()."""
        json_dir = tmp_path / ".data" / "vectors"
        json_dir.mkdir(parents=True)
        self._write_legacy_json(json_dir, "book1", sample_entries)

        store = VectorStore(storage_dir=str(tmp_path / "qdrant"), embedding_dim=3)
        store._old_json_dir = json_dir
        await store.initialize()

        assert store.has_book("book1")
        results = await store.search("book1", [1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.metadata["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_migrates_legacy_summaries(self, tmp_path, sample_entries):
        """Legacy summary files (*_summaries.json) should be migrated alongside chunks."""
        json_dir = tmp_path / ".data" / "vectors"
        json_dir.mkdir(parents=True)
        self._write_legacy_json(json_dir, "book1", sample_entries)

        summary_entries = [make_entry("book1", 0, "Chapter summary.", [0.5, 0.5, 0.0])]
        self._write_legacy_json(json_dir, "book1_summaries", summary_entries)

        store = VectorStore(storage_dir=str(tmp_path / "qdrant"), embedding_dim=3)
        store._old_json_dir = json_dir
        await store.initialize()

        results = await store.search_summaries("book1", [0.5, 0.5, 0.0], top_k=1)
        assert len(results) == 1
        assert "summary" in results[0].chunk.text.lower()

    @pytest.mark.asyncio
    async def test_skips_outdated_version(self, tmp_path, sample_entries):
        """Legacy files with version < CURRENT_INDEX_VERSION should be skipped."""
        json_dir = tmp_path / ".data" / "vectors"
        json_dir.mkdir(parents=True)
        self._write_legacy_json(json_dir, "book1", sample_entries, version=0)

        store = VectorStore(storage_dir=str(tmp_path / "qdrant"), embedding_dim=3)
        store._old_json_dir = json_dir
        await store.initialize()

        assert not store.has_book("book1")

    @pytest.mark.asyncio
    async def test_skips_if_qdrant_has_data(self, tmp_path, sample_entries):
        """Auto-migration should not run if Qdrant already has data."""
        json_dir = tmp_path / ".data" / "vectors"
        json_dir.mkdir(parents=True)

        # First: create a store and add data directly (book_id matches metadata)
        store1 = VectorStore(storage_dir=str(tmp_path / "qdrant"), embedding_dim=3)
        store1._old_json_dir = json_dir  # no files yet
        await store1.initialize()
        await store1.add_book("book1", sample_entries)
        store1._sync_client.close()

        # Now write a legacy JSON file for a different book
        other_entries = [make_entry("legacy", 0, "Legacy book.", [0.1, 0.2, 0.3])]
        self._write_legacy_json(json_dir, "legacy", other_entries)

        # Second store: should skip migration since Qdrant already has data
        store2 = VectorStore(storage_dir=str(tmp_path / "qdrant"), embedding_dim=3)
        store2._old_json_dir = json_dir
        await store2.initialize()

        assert store2.has_book("book1")
        assert not store2.has_book("legacy")

    @pytest.mark.asyncio
    async def test_no_legacy_dir_is_noop(self, tmp_path):
        """Missing legacy directory should not cause errors."""
        store = VectorStore(storage_dir=str(tmp_path / "qdrant"), embedding_dim=3)
        store._old_json_dir = tmp_path / "nonexistent"
        await store.initialize()
        # Should initialize cleanly with no data
        assert not store.has_book("anything")


class TestMultiBookIsolation:
    """Verify that book_id filtering properly isolates books."""

    @pytest.mark.asyncio
    async def test_search_returns_only_target_book(self, vector_store):
        book1 = [make_entry("book1", 0, "Alpha content.", [1.0, 0.0, 0.0])]
        book2 = [make_entry("book2", 0, "Beta content.", [1.0, 0.0, 0.0])]
        await vector_store.add_book("book1", book1)
        await vector_store.add_book("book2", book2)

        results = await vector_store.search("book1", [1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0].chunk.metadata["book_id"] == "book1"

    @pytest.mark.asyncio
    async def test_remove_only_affects_target_book(self, vector_store):
        book1 = [make_entry("book1", 0, "Alpha.", [1.0, 0.0, 0.0])]
        book2 = [make_entry("book2", 0, "Beta.", [0.0, 1.0, 0.0])]
        await vector_store.add_book("book1", book1)
        await vector_store.add_book("book2", book2)

        await vector_store.remove_book("book1")
        assert not vector_store.has_book("book1")
        assert vector_store.has_book("book2")


class TestBM25LazyBuild:
    @pytest.mark.asyncio
    async def test_bm25_built_lazily_on_hybrid_search(self, sample_entries):
        """BM25 index should be built on demand when not yet cached."""
        store = VectorStore(location=":memory:", embedding_dim=3)
        await store.initialize()
        await store.add_book("book1", sample_entries)
        # Clear the BM25 cache to simulate cold start
        store.bm25_indices.clear()
        assert "book1" not in store.bm25_indices

        # hybrid_search triggers lazy BM25 build
        results = await store.hybrid_search("book1", [1.0, 0.0, 0.0], "beginning", top_k=3)
        assert "book1" in store.bm25_indices
        assert len(results) >= 1


class TestChapterMethods:
    """Tests for chapter-aware vector store methods."""

    @pytest.mark.asyncio
    async def test_has_chapter_metadata_true(self, vector_store, sample_entries_with_chapters):
        await vector_store.add_book("book1", sample_entries_with_chapters)
        assert await vector_store.has_chapter_metadata("book1") is True

    @pytest.mark.asyncio
    async def test_has_chapter_metadata_false(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        assert await vector_store.has_chapter_metadata("book1") is False

    @pytest.mark.asyncio
    async def test_has_chapter_metadata_no_book(self, vector_store):
        assert await vector_store.has_chapter_metadata("nobook") is False

    @pytest.mark.asyncio
    async def test_get_chapter_list(self, vector_store, sample_entries_with_chapters):
        await vector_store.add_book("book1", sample_entries_with_chapters)
        chapters = await vector_store.get_chapter_list("book1")
        assert len(chapters) == 2
        assert chapters[0]["chapter_index"] == 0
        assert chapters[0]["chapter_title"] == "The Beginning"
        assert chapters[0]["first_chunk_index"] == 0
        assert chapters[0]["last_chunk_index"] == 2
        assert chapters[1]["chapter_index"] == 1
        assert chapters[1]["chapter_title"] == "The Journey"
        assert chapters[1]["first_chunk_index"] == 3
        assert chapters[1]["last_chunk_index"] == 6

    @pytest.mark.asyncio
    async def test_get_chapter_list_no_chapters(self, vector_store, sample_entries):
        await vector_store.add_book("book1", sample_entries)
        chapters = await vector_store.get_chapter_list("book1")
        assert chapters == []

    @pytest.mark.asyncio
    async def test_search_by_chapter(self, vector_store, sample_entries_with_chapters):
        await vector_store.add_book("book1", sample_entries_with_chapters)
        # Search chapter 1 (The Journey) for "villain"
        results = await vector_store.search_by_chapter(
            "book1", 1, [0.7, 0.7, 0.0], "villain", top_k=5,
        )
        assert len(results) >= 1
        # All results should be from chapter 1
        for r in results:
            assert r.chunk.metadata.get("chapter_index") == 1

    @pytest.mark.asyncio
    async def test_search_by_chapter_no_results(self, vector_store, sample_entries_with_chapters):
        await vector_store.add_book("book1", sample_entries_with_chapters)
        # Search nonexistent chapter
        results = await vector_store.search_by_chapter(
            "book1", 99, [1.0, 0.0, 0.0], "anything", top_k=5,
        )
        assert results == []
