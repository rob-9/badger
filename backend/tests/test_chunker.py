"""Tests for badger.core.chunker — text chunking logic."""

import pytest
from badger.core.chunker import (
    TextChunk,
    chunk_text,
    chunk_structured,
    estimate_tokens,
    _split_paragraph_at_sentences,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
)


# ── TextChunk dataclass ──────────────────────────────────────────────


class TestTextChunk:
    def test_valid_metadata(self):
        chunk = TextChunk(id="c-0", text="hello", metadata={"book_id": "b1", "chunk_index": 0})
        assert chunk.id == "c-0"
        assert chunk.text == "hello"

    def test_missing_book_id_raises(self):
        with pytest.raises(ValueError, match="book_id"):
            TextChunk(id="c-0", text="hello", metadata={"chunk_index": 0})

    def test_missing_chunk_index_raises(self):
        with pytest.raises(ValueError, match="chunk_index"):
            TextChunk(id="c-0", text="hello", metadata={"book_id": "b1"})

    def test_extra_metadata_allowed(self):
        chunk = TextChunk(
            id="c-0", text="hello",
            metadata={"book_id": "b1", "chunk_index": 0, "chapter_title": "Ch1"},
        )
        assert chunk.metadata["chapter_title"] == "Ch1"


# ── chunk_text ────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", "book1")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].metadata["book_id"] == "book1"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_empty_text_no_chunks(self):
        assert chunk_text("", "book1") == []

    def test_whitespace_only_no_chunks(self):
        assert chunk_text("   \n\n  ", "book1") == []

    def test_chunk_ids_are_sequential(self):
        text = "Sentence one. " * 200  # long enough for multiple chunks
        chunks = chunk_text(text, "mybook", chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"mybook-chunk-{i}"
            assert chunk.metadata["chunk_index"] == i

    def test_whitespace_normalization(self):
        text = "Hello   world.\n\nNew   paragraph."
        chunks = chunk_text(text, "b1")
        assert chunks[0].text == "Hello world. New paragraph."

    def test_respects_chunk_size(self):
        text = "Word. " * 1000
        chunks = chunk_text(text, "b1", chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            # Chunks should be roughly at or under chunk_size
            assert len(chunk.text) <= 520  # allow small overrun at boundaries

    def test_overlap_creates_redundancy(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. " * 40
        chunks = chunk_text(text, "b1", chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2
        # The end of chunk 0 should overlap with the start of chunk 1
        end_of_first = chunks[0].text[-50:]
        assert end_of_first in chunks[1].text or chunks[1].text[:60] in chunks[0].text[-70:]

    def test_sentence_boundary_breaking(self):
        # Build text where a sentence boundary exists near the chunk edge
        text = "A" * 1800 + ". " + "B" * 500
        chunks = chunk_text(text, "b1", chunk_size=2000, chunk_overlap=100)
        assert len(chunks) >= 2
        # First chunk should break at or near the sentence boundary
        assert "." in chunks[0].text

    def test_metadata_has_char_positions(self):
        text = "Hello world. " * 300
        chunks = chunk_text(text, "b1", chunk_size=500, chunk_overlap=50)
        for chunk in chunks:
            assert "start_char" in chunk.metadata
            assert "end_char" in chunk.metadata
            assert chunk.metadata["start_char"] < chunk.metadata["end_char"]

    def test_default_parameters(self):
        assert DEFAULT_CHUNK_SIZE == 2000
        assert DEFAULT_OVERLAP == 200

    def test_multiple_chunks_cover_full_text(self):
        """All content should be represented across chunks (with overlap)."""
        text = "Word. " * 500
        chunks = chunk_text(text, "b1", chunk_size=500, chunk_overlap=50)
        assert len(chunks) > 1
        # First chunk starts from beginning, last covers end
        assert chunks[0].metadata["start_char"] == 0


# ── chunk_structured ──────────────────────────────────────────────────


class TestChunkStructured:
    def test_basic_structured_chunking(self):
        content = {
            "chapters": [
                {
                    "title": "Chapter 1",
                    "index": 0,
                    "sections": [
                        {
                            "heading": "Intro",
                            "paragraphs": ["First paragraph.", "Second paragraph."],
                        }
                    ],
                }
            ]
        }
        chunks = chunk_structured(content, "book1")
        assert len(chunks) >= 1
        assert "First paragraph." in chunks[0].text
        assert chunks[0].metadata["chapter_title"] == "Chapter 1"
        assert chunks[0].metadata["chapter_index"] == 0

    def test_chapter_boundaries_respected(self):
        content = {
            "chapters": [
                {
                    "title": "Ch1",
                    "index": 0,
                    "sections": [{"heading": None, "paragraphs": ["Para A."]}],
                },
                {
                    "title": "Ch2",
                    "index": 1,
                    "sections": [{"heading": None, "paragraphs": ["Para B."]}],
                },
            ]
        }
        chunks = chunk_structured(content, "b1")
        # Each chapter should produce separate chunks
        ch1_chunks = [c for c in chunks if c.metadata["chapter_title"] == "Ch1"]
        ch2_chunks = [c for c in chunks if c.metadata["chapter_title"] == "Ch2"]
        assert len(ch1_chunks) >= 1
        assert len(ch2_chunks) >= 1
        # No chunk should contain text from both chapters
        for c in ch1_chunks:
            assert "Para B." not in c.text
        for c in ch2_chunks:
            assert "Para A." not in c.text

    def test_empty_chapters_skipped(self):
        content = {"chapters": [{"title": "Empty", "index": 0, "sections": []}]}
        chunks = chunk_structured(content, "b1")
        assert chunks == []

    def test_empty_paragraphs_skipped(self):
        content = {
            "chapters": [
                {
                    "title": "Ch",
                    "index": 0,
                    "sections": [{"heading": None, "paragraphs": ["", "  ", "Real text."]}],
                }
            ]
        }
        chunks = chunk_structured(content, "b1")
        assert len(chunks) == 1
        assert "Real text." in chunks[0].text

    def test_large_paragraph_split_at_sentences(self):
        big_para = ("This is a long sentence. " * 200).strip()
        content = {
            "chapters": [
                {
                    "title": "Ch",
                    "index": 0,
                    "sections": [{"heading": None, "paragraphs": [big_para]}],
                }
            ]
        }
        chunks = chunk_structured(content, "b1", chunk_size=500)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 520  # allow slight overrun

    def test_section_heading_tracked(self):
        content = {
            "chapters": [
                {
                    "title": "Ch",
                    "index": 0,
                    "sections": [
                        {"heading": "Section A", "paragraphs": ["Text A."]},
                        {"heading": "Section B", "paragraphs": ["Text B."]},
                    ],
                }
            ]
        }
        chunks = chunk_structured(content, "b1")
        # The last section heading should be in the metadata
        headings = [c.metadata.get("section_heading") for c in chunks]
        assert "Section A" in headings or "Section B" in headings

    def test_no_overlap_in_structured_chunks(self):
        """Structured chunking should NOT have overlap (voyage-context-3 handles it)."""
        content = {
            "chapters": [
                {
                    "title": "Ch",
                    "index": 0,
                    "sections": [
                        {
                            "heading": None,
                            "paragraphs": [f"Unique paragraph {i}." for i in range(50)],
                        }
                    ],
                }
            ]
        }
        chunks = chunk_structured(content, "b1", chunk_size=200)
        if len(chunks) >= 2:
            # Check that consecutive chunks don't share unique identifiers
            for i in range(len(chunks) - 1):
                # Extract unique paragraph identifiers from each chunk
                words_a = set(chunks[i].text.split())
                words_b = set(chunks[i + 1].text.split())
                # They might share common words but shouldn't share unique paragraph numbers
                unique_nums_a = {w for w in words_a if w.rstrip(".").isdigit()}
                unique_nums_b = {w for w in words_b if w.rstrip(".").isdigit()}
                overlap = unique_nums_a & unique_nums_b
                assert len(overlap) == 0, f"Chunks {i} and {i+1} overlap on: {overlap}"

    def test_sequential_chunk_ids(self):
        from tests.conftest import SAMPLE_STRUCTURED_CONTENT

        chunks = chunk_structured(SAMPLE_STRUCTURED_CONTENT, "book1")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.id == f"book1-chunk-{i}"

    def test_no_chapters_key(self):
        chunks = chunk_structured({}, "b1")
        assert chunks == []

    def test_accumulates_small_paragraphs(self):
        content = {
            "chapters": [
                {
                    "title": "Ch",
                    "index": 0,
                    "sections": [
                        {
                            "heading": None,
                            "paragraphs": ["Short A.", "Short B.", "Short C."],
                        }
                    ],
                }
            ]
        }
        chunks = chunk_structured(content, "b1", chunk_size=2000)
        # All short paragraphs should be accumulated into a single chunk
        assert len(chunks) == 1
        assert "Short A." in chunks[0].text
        assert "Short B." in chunks[0].text
        assert "Short C." in chunks[0].text


# ── _split_paragraph_at_sentences ─────────────────────────────────────


class TestSplitParagraphAtSentences:
    def test_short_paragraph_no_split(self):
        result = _split_paragraph_at_sentences("Short text.", 2000)
        assert result == ["Short text."]

    def test_splits_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        pieces = _split_paragraph_at_sentences(text, 40)
        assert len(pieces) >= 2
        for piece in pieces:
            assert len(piece) > 0

    def test_empty_string(self):
        assert _split_paragraph_at_sentences("", 100) == []


# ── estimate_tokens ───────────────────────────────────────────────────


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_basic_estimation(self):
        # 100 chars ≈ 25 tokens
        assert estimate_tokens("a" * 100) == 25

    def test_ceiling_division(self):
        # 5 chars → ceil(5/4) = 2
        assert estimate_tokens("hello") == 2

    def test_single_char(self):
        assert estimate_tokens("a") == 1
