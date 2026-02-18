"""Tests for boom.core.graph — LangGraph helpers, labeling, routing, logging."""

import json
import pytest

from boom.core.vector_store import SearchResult
from boom.core.graph import (
    strip_code_fences,
    label_chunks,
    build_query,
    build_context_string,
    prepare_generate,
    log_query,
    _build_log_entry,
    VALID_TYPES,
)
from boom.core.prompts import SYSTEM_PROMPTS
from tests.conftest import make_chunk


# ── strip_code_fences ─────────────────────────────────────────────────


class TestStripCodeFences:
    def test_no_fences(self):
        assert strip_code_fences('{"type": "context"}') == '{"type": "context"}'

    def test_json_fences(self):
        text = '```json\n{"type": "lookup"}\n```'
        assert strip_code_fences(text) == '{"type": "lookup"}'

    def test_plain_fences(self):
        text = '```\nhello world\n```'
        assert strip_code_fences(text) == "hello world"

    def test_fences_no_newline(self):
        text = '```some content```'
        assert strip_code_fences(text) == "some content"

    def test_empty_between_fences(self):
        text = '```\n```'
        assert strip_code_fences(text) == ""


# ── label_chunks ──────────────────────────────────────────────────────


class TestLabelChunks:
    def _make_result(self, index: int, score: float = 0.9) -> SearchResult:
        chunk = make_chunk("b", index, f"Text of chunk {index}")
        return SearchResult(chunk=chunk, score=score)

    def test_all_past(self):
        results = [self._make_result(0), self._make_result(1)]
        labeled = label_chunks(results, reader_position=0.5, total_chunks=10)
        assert all(c["label"] == "PAST" for c in labeled)

    def test_all_ahead(self):
        results = [self._make_result(8), self._make_result(9)]
        labeled = label_chunks(results, reader_position=0.1, total_chunks=10)
        assert all(c["label"] == "AHEAD" for c in labeled)

    def test_mixed_labels(self):
        results = [self._make_result(1), self._make_result(8)]
        labeled = label_chunks(results, reader_position=0.5, total_chunks=10)
        labels = {c["chunk_index"]: c["label"] for c in labeled}
        assert labels[1] == "PAST"
        assert labels[8] == "AHEAD"

    def test_reader_at_start(self):
        results = [self._make_result(0), self._make_result(5)]
        labeled = label_chunks(results, reader_position=0.0, total_chunks=10)
        labels = {c["chunk_index"]: c["label"] for c in labeled}
        assert labels[0] == "PAST"  # chunk 0 <= reader_idx 0
        assert labels[5] == "AHEAD"

    def test_reader_at_end(self):
        results = [self._make_result(0), self._make_result(9)]
        labeled = label_chunks(results, reader_position=1.0, total_chunks=10)
        assert all(c["label"] == "PAST" for c in labeled)

    def test_zero_total_chunks(self):
        results = [self._make_result(0)]
        labeled = label_chunks(results, reader_position=0.5, total_chunks=0)
        assert labeled[0]["label"] == "PAST"  # reader_idx = 0

    def test_preserves_score(self):
        results = [self._make_result(0, score=0.85)]
        labeled = label_chunks(results, reader_position=0.5, total_chunks=10)
        assert labeled[0]["score"] == 0.85

    def test_preserves_text(self):
        results = [self._make_result(3)]
        labeled = label_chunks(results, reader_position=0.5, total_chunks=10)
        assert labeled[0]["text"] == "Text of chunk 3"


# ── build_query ───────────────────────────────────────────────────────


class TestBuildQuery:
    def test_question_only(self):
        result = build_query("What happens?", None, [])
        assert result == "What happens?"

    def test_with_selected_text(self):
        result = build_query("What does this mean?", "some text", [])
        assert "What does this mean?" in result
        assert "Referring to: some text" in result

    def test_with_entities(self):
        result = build_query("Question?", None, ["Alice", "rabbit"])
        assert "Key terms: Alice, rabbit" in result

    def test_with_all(self):
        result = build_query("What?", "selected", ["entity"])
        lines = result.split("\n")
        assert len(lines) == 3


# ── build_context_string ──────────────────────────────────────────────


class TestBuildContextString:
    def test_past_only(self):
        chunks = [{"text": "Past text.", "label": "PAST", "chunk_index": 0, "score": 0.9}]
        ctx = build_context_string(chunks)
        assert "[ALREADY READ]" in ctx
        assert "[COMING UP]" not in ctx

    def test_ahead_only(self):
        chunks = [{"text": "Future text.", "label": "AHEAD", "chunk_index": 5, "score": 0.8}]
        ctx = build_context_string(chunks)
        assert "[COMING UP]" in ctx
        assert "[ALREADY READ]" not in ctx

    def test_mixed(self):
        chunks = [
            {"text": "Past.", "label": "PAST", "chunk_index": 0, "score": 0.9},
            {"text": "Future.", "label": "AHEAD", "chunk_index": 5, "score": 0.8},
        ]
        ctx = build_context_string(chunks)
        assert "[ALREADY READ]" in ctx
        assert "[COMING UP]" in ctx
        assert "===" in ctx

    def test_empty_chunks(self):
        assert build_context_string([]) == ""

    def test_passages_numbered(self):
        chunks = [
            {"text": "A", "label": "PAST", "chunk_index": 0, "score": 0.9},
            {"text": "B", "label": "PAST", "chunk_index": 1, "score": 0.8},
        ]
        ctx = build_context_string(chunks)
        assert "[Passage 1]" in ctx
        assert "[Passage 2]" in ctx


# ── _build_log_entry ──────────────────────────────────────────────────


class TestBuildLogEntry:
    def test_basic_log_entry(self):
        state = {
            "question": "What is this?",
            "book_id": "book1",
            "reader_position": 0.5,
            "total_chunks": 100,
            "selected_text": "some text",
            "question_type": "vocabulary",
            "entities": ["word"],
            "classify_raw_response": '{"type":"vocabulary"}',
            "classify_tokens_in": 50,
            "classify_tokens_out": 10,
            "chunks": [
                {"text": "chunk text", "chunk_index": 5, "score": 0.9, "label": "PAST"},
            ],
            "retrieval_strategy": "keyword",
            "retrieval_query": "some text",
            "retrieval_top_k": 5,
            "retrieval_embedding_dims": 1024,
            "answer": "It means...",
            "gen_model": "claude-sonnet-4-20250514",
            "gen_tokens_in": 200,
            "gen_tokens_out": 50,
            "gen_stop_reason": "end_turn",
            "gen_system_prompt": "You are...",
            "gen_user_prompt": "The user...",
        }
        entry = _build_log_entry(state)
        assert entry["question"] == "What is this?"
        assert entry["book_id"] == "book1"
        assert entry["reader_chunk_index"] == 50
        assert entry["question_type"] == "vocabulary"
        assert len(entry["retrieved_chunks"]) == 1
        assert entry["response"]["answer"] == "It means..."

    def test_minimal_state(self):
        state = {"question": "Test?"}
        entry = _build_log_entry(state)
        assert entry["question"] == "Test?"
        assert entry["reader_chunk_index"] == 0
        assert entry["retrieved_chunks"] == []


# ── VALID_TYPES ───────────────────────────────────────────────────────


class TestValidTypes:
    def test_all_types_present(self):
        assert VALID_TYPES == {"vocabulary", "context", "lookup", "analysis"}

    def test_types_match_prompts(self):
        from boom.core.prompts import SYSTEM_PROMPTS

        for t in VALID_TYPES:
            assert t in SYSTEM_PROMPTS, f"Missing system prompt for type: {t}"


# ── _label_and_log ───────────────────────────────────────────────────


class TestLabelAndLog:
    def test_returns_labeled_chunks(self):
        from boom.core.graph import _label_and_log

        results = [
            SearchResult(chunk=make_chunk("b", 2, "Text"), score=0.8),
            SearchResult(chunk=make_chunk("b", 8, "More"), score=0.7),
        ]
        state = {"reader_position": 0.5, "total_chunks": 10}
        labeled = _label_and_log(results, state)
        assert len(labeled) == 2
        labels = {c["chunk_index"]: c["label"] for c in labeled}
        assert labels[2] == "PAST"
        assert labels[8] == "AHEAD"


# ── _write_readable_log ─────────────────────────────────────────────


class TestPrepareGenerate:
    def test_with_chunks_and_selected_text(self):
        state = {
            "question": "What does this mean?",
            "question_type": "context",
            "selected_text": "some passage",
            "chunks": [
                {"text": "chunk text", "label": "PAST", "chunk_index": 0, "score": 0.9},
            ],
        }
        result = prepare_generate(state)
        assert result["system_prompt"] == SYSTEM_PROMPTS["context"]
        assert 'Selected text: "some passage"' in result["user_prompt"]
        assert "Context from the book:" in result["user_prompt"]
        assert "What does this mean?" in result["user_prompt"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["chunk_index"] == 0

    def test_with_chunks_no_selected_text(self):
        state = {
            "question": "What happens next?",
            "question_type": "lookup",
            "chunks": [
                {"text": "chunk", "label": "PAST", "chunk_index": 0, "score": 0.9},
            ],
        }
        result = prepare_generate(state)
        assert result["system_prompt"] == SYSTEM_PROMPTS["lookup"]
        assert "Context from the book:" in result["user_prompt"]
        assert "Selected text" not in result["user_prompt"]

    def test_no_chunks(self):
        state = {
            "question": "What?",
            "question_type": "vocabulary",
            "selected_text": "word",
            "chunks": [],
        }
        result = prepare_generate(state)
        assert 'Selected text: "word"' in result["user_prompt"]
        assert "Context from the book" not in result["user_prompt"]
        assert result["sources"] == []

    def test_defaults_to_context_type(self):
        state = {"question": "What?", "chunks": []}
        result = prepare_generate(state)
        assert result["system_prompt"] == SYSTEM_PROMPTS["context"]

    def test_source_truncation(self):
        long_text = "x" * 500
        state = {
            "question": "What?",
            "chunks": [
                {"text": long_text, "label": "PAST", "chunk_index": 0, "score": 0.9},
            ],
        }
        result = prepare_generate(state)
        assert result["sources"][0]["text"] == "x" * 200 + "..."
        assert result["sources"][0]["full_text"] == long_text


class TestLogQuery:
    def test_writes_jsonl_and_log(self, tmp_path):
        import boom.core.graph as graph_mod

        original_dir = graph_mod.LOG_DIR
        graph_mod.LOG_DIR = tmp_path
        try:
            state = {
                "question": "Test question?",
                "book_id": "b1",
                "chunks": [],
                "reader_position": 0.5,
                "total_chunks": 100,
            }
            log_query(state)
            assert (tmp_path / "queries.jsonl").exists()
            assert (tmp_path / "queries.log").exists()

            jsonl = (tmp_path / "queries.jsonl").read_text()
            entry = json.loads(jsonl.strip())
            assert entry["question"] == "Test question?"
            assert entry["book_id"] == "b1"

            log_text = (tmp_path / "queries.log").read_text()
            assert "QUERY @" in log_text
        finally:
            graph_mod.LOG_DIR = original_dir


class TestWriteReadableLog:
    def test_writes_log_file(self, tmp_path):
        """Readable log should be written without errors."""
        import boom.core.graph as graph_mod
        from boom.core.graph import _write_readable_log

        original_dir = graph_mod.LOG_DIR
        graph_mod.LOG_DIR = tmp_path
        try:
            state = {
                "book_id": "b1",
                "reader_position": 0.5,
                "total_chunks": 100,
                "selected_text": "test text",
                "question": "What?",
                "question_type": "vocabulary",
                "entities": ["word"],
                "classify_raw_response": '{"type":"vocabulary"}',
                "classify_tokens_in": 50,
                "classify_tokens_out": 10,
                "chunks": [
                    {"text": "chunk text", "chunk_index": 5, "score": 0.9, "label": "PAST"},
                ],
                "retrieval_strategy": "keyword",
                "retrieval_query": "test text",
                "gen_system_prompt": "You are a reading companion.",
                "gen_user_prompt": "Question: What?",
                "gen_model": "claude-sonnet-4-20250514",
                "gen_tokens_in": 200,
                "gen_tokens_out": 50,
                "gen_stop_reason": "end_turn",
                "answer": "The answer.",
            }
            log_entry = _build_log_entry(state)
            _write_readable_log(state, log_entry)

            log_content = (tmp_path / "queries.log").read_text()
            assert "QUERY @" in log_content
            assert "vocabulary" in log_content
        finally:
            graph_mod.LOG_DIR = original_dir
