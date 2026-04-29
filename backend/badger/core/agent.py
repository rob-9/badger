"""
Tool-calling agent for the RAG pipeline.

Replaces the fixed LangGraph DAG with a loop where Claude decides which
tools to call, how many times, and in what order. Seven tools:

- search_book: semantic / keyword / hybrid search with reranking
- get_surrounding_context: proximity retrieval around a chunk
- get_chapter_summary: broad thematic search over chapter summaries
- search_by_chapter: scoped search within a single chapter
- find_first_mention: earliest occurrence of a term by position
- get_reading_position_context: chunks before reader's current position
- get_book_structure: chapter list with PAST/CURRENT/AHEAD labels

The agent loop runs tool-call turns non-streaming (fast, 1-2 turns typical),
then streams only the final answer. This preserves the SSE event sequence
the frontend expects: status* → sources → token* → done.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic, AsyncAnthropic

from badger import config
from .prompts import AGENT_SYSTEM_PROMPT, EVALUATE_PROMPT
from .graph import label_chunks, strip_code_fences
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

LOG_DIR = Path(".data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_TURNS = config.AGENT_MAX_TURNS


def _format_conversation_history(
    history: list[dict],
    max_turns: int = 3,
    max_selected_len: int = 80,
    max_answer_len: int = 200,
) -> str:
    """Format conversation history as a condensed block for system prompt injection.

    Groups into Q&A pairs, strips [Source N] citations, truncates long content,
    and keeps only the last `max_turns` exchanges.
    """
    import re as _re

    # Normalize Pydantic models to dicts
    history = [t.model_dump() if hasattr(t, 'model_dump') else t for t in history]

    # Group into Q&A pairs
    pairs: list[dict] = []
    current_pair: dict = {}
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            if current_pair.get("user"):
                # Previous pair had no assistant response — save it anyway
                pairs.append(current_pair)
            current_pair = {
                "user": content,
                "selected_text": turn.get("selected_text"),
                "reader_position": turn.get("reader_position"),
            }
        elif role == "assistant" and current_pair.get("user"):
            # Strip [Source N] citations from prior answers
            cleaned = _re.sub(r'\[(?:Source\s+)?\d+(?:,\s*(?:Source\s+)?\d+)*\]', '', content)
            cleaned = _re.sub(r'\s{2,}', ' ', cleaned).strip()
            current_pair["assistant"] = cleaned
            pairs.append(current_pair)
            current_pair = {}

    # Don't lose a trailing user message without assistant response
    if current_pair.get("user"):
        pairs.append(current_pair)

    if not pairs:
        return ""

    # Keep only last max_turns exchanges
    pairs = pairs[-max_turns:]

    lines = ["\n\n[CONVERSATION HISTORY] (prior exchanges — source numbers below are from NEW searches, not these):\n"]
    for i, pair in enumerate(pairs, 1):
        # Position info
        pos = pair.get("reader_position")
        pos_str = f"Reader at {int(pos * 100)}%" if pos is not None else ""

        # Selected text (truncated)
        sel = pair.get("selected_text", "")
        if sel:
            if len(sel) > max_selected_len:
                sel = sel[:max_selected_len] + "..."
            sel_str = f' | Selected: "{sel}"'
        else:
            sel_str = ""

        header_parts = [p for p in [pos_str, sel_str.lstrip(" | ")] if p]
        if header_parts:
            if pos_str and sel_str:
                lines.append(f"[Turn {i}] {pos_str}{sel_str}")
            elif pos_str:
                lines.append(f"[Turn {i}] {pos_str}")
            else:
                lines.append(f"[Turn {i}] {sel_str.lstrip(' | ')}")
        else:
            lines.append(f"[Turn {i}]")

        # Question
        lines.append(f"Q: {pair['user']}")

        # Answer (truncated)
        assistant = pair.get("assistant", "")
        if assistant:
            if len(assistant) > max_answer_len:
                assistant = assistant[:max_answer_len] + "..."
            lines.append(f"A: {assistant}")

        lines.append("")  # blank line between turns

    return "\n".join(lines)


def _adaptive_cutoff(chunks: list[dict]) -> list[dict]:
    """Select chunks using score drop-off: keep chunks before the largest gap."""
    floor = config.CUTOFF_FLOOR
    ceiling = config.CUTOFF_CEILING

    if len(chunks) <= floor:
        return chunks

    scores = [c["score"] for c in chunks]
    limit = min(len(scores) - 1, ceiling)

    max_drop = 0
    cut_at = limit  # default: keep up to ceiling

    for i in range(floor - 1, limit):
        drop = scores[i] - scores[i + 1]
        if drop > max_drop:
            max_drop = drop
            cut_at = i + 1

    # Flat scores = no clear relevance signal; be conservative
    if max_drop < 0.02:
        cut_at = floor

    kept = chunks[:cut_at]

    # Drop chunks below the minimum relevance floor
    if config.RELEVANCE_FLOOR > 0:
        kept = [c for c in kept if c["score"] >= config.RELEVANCE_FLOOR]

    logger.info("  Adaptive cutoff: %d → %d chunks (max drop=%.4f at position %d)",
                len(chunks), len(kept), max_drop, cut_at)
    return kept


def _bookend_reorder(chunks: list[dict]) -> list[dict]:
    """Reorder chunks to place highest-relevance at start and end (lost-in-the-middle mitigation).

    Input: chunks sorted by descending score.
    Output: [best, worst, ..., second-worst, second-best]
    """
    if len(chunks) <= 2:
        return list(chunks)

    best = chunks[0]
    second_best = chunks[1]
    middle = chunks[2:]  # already weakest-last; reverse so weakest-first
    middle.reverse()

    return [best] + middle + [second_best]


def _extract_relevant_sentences(chunk_text: str, query: str, top_n: int = 5) -> str:
    """Compress a chunk by keeping only the most query-relevant sentences.

    Scores each sentence by normalized word overlap with the query.
    Returns top_n sentences in their original order to preserve narrative flow.
    Returns unchanged text if the chunk has fewer than top_n + 2 sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n+', chunk_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < top_n + 2:
        return chunk_text

    query_words = set(query.lower().split())
    if not query_words:
        return chunk_text

    scored = []
    for i, sentence in enumerate(sentences):
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words) / len(query_words)
        scored.append((i, overlap, sentence))

    # Sort by score descending, take top_n
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted(idx for idx, _, _ in scored[:top_n])

    # Reassemble in original order
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Tool schemas (for Claude's `tools` parameter)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search_book",
        "description": (
            "Search the book for passages relevant to a query. "
            "Returns passages the reader has already read, with source numbers for citation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — be specific with names, terms, or concepts.",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "hybrid"],
                    "description": (
                        "Search strategy. 'hybrid' (default) combines semantic + keyword. "
                        "'keyword' for exact term matching. 'semantic' for conceptual similarity."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to retrieve (default 20, reranked to ~5).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_surrounding_context",
        "description": (
            "Get passages surrounding a specific chunk index. "
            "Use when you found a relevant passage and need more narrative context around it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_index": {
                    "type": "integer",
                    "description": "The chunk index to center on (from a previous search result).",
                },
                "window": {
                    "type": "integer",
                    "description": "Number of chunks before and after to include (default 3).",
                },
            },
            "required": ["chunk_index"],
        },
    },
    {
        "name": "get_chapter_summary",
        "description": (
            "Search chapter-level summaries for broad thematic matches. "
            "Use for questions about overall themes, arcs, or 'what is the book's stance on X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The thematic query to search chapter summaries for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chapter summaries to return (default 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_by_chapter",
        "description": (
            "Search within a specific chapter for relevant passages. "
            "Use when the reader asks about events in a particular chapter. "
            "Requires a chapter_index (use get_book_structure to find it)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "chapter_index": {
                    "type": "integer",
                    "description": "The 0-based chapter index to search within.",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["semantic", "keyword", "hybrid"],
                    "description": "Search strategy (default 'hybrid').",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to retrieve (default 20, reranked to ~5).",
                },
            },
            "required": ["query", "chapter_index"],
        },
    },
    {
        "name": "find_first_mention",
        "description": (
            "Find the earliest occurrence(s) of a term in the book by position. "
            "Use for 'when was X first introduced/mentioned?' questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The exact term to search for (case-insensitive).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum earliest mentions to return (default 3, max 10).",
                },
            },
            "required": ["term"],
        },
    },
    {
        "name": "get_reading_position_context",
        "description": (
            "Get the chunks immediately before the reader's current position. "
            "Use for 'what just happened?' or recap questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "window": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 5, max 10).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_book_structure",
        "description": (
            "Get the book's chapter list with titles and reading progress. "
            "Use for 'what chapter am I in?' or navigation questions. No search needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

def _format_sources(
    chunks: list[dict],
    source_counter: int,
    seen_chunk_indices: set[int],
    include_chunk_id: bool = True,
) -> tuple[str, int]:
    parts = []
    for c in chunks:
        seen_chunk_indices.add(c["chunk_index"])
        c["source_number"] = source_counter
        chapter = c.get("chapter_title", "")
        header = f"[Source {source_counter}]"
        if chapter:
            header += f" (Chapter: {chapter})"
        if include_chunk_id:
            header += f" [chunk {c['chunk_index']}]"
        parts.append(f"{header}\n{c['text']}")
        source_counter += 1
    formatted = "\n\n---\n\n".join(parts)
    return formatted, source_counter


def build_tool_executors(vector_store: VectorStore, voyage_client):
    """Build tool executor functions closed over shared resources."""

    _embed_cache: dict[str, list[float]] = {}
    _EMBED_CACHE_MAX = 500

    async def _embed_query(text: str) -> list[float]:
        cache_key = text.lower().strip()
        if cache_key in _embed_cache:
            return _embed_cache[cache_key]

        result = await asyncio.to_thread(
            voyage_client.contextualized_embed,
            inputs=[[text]],
            model=config.VOYAGE_CONTEXT_MODEL,
            input_type="query",
        )
        embedding = result.results[0].embeddings[0]

        if len(_embed_cache) >= _EMBED_CACHE_MAX:
            _embed_cache.clear()
        _embed_cache[cache_key] = embedding

        return embedding

    async def _rerank_and_cutoff(
        chunks: list[dict], query: str, selected_text: str | None,
    ) -> list[dict]:
        """Rerank with Voyage rerank-2.5, apply adaptive cutoff."""
        if len(chunks) <= 3:
            return chunks

        rerank_query = query
        if selected_text:
            rerank_query += f"\n\nReferring to: {selected_text}"

        if config.RERANK_ENABLED:
            reranking = await asyncio.to_thread(
                voyage_client.rerank,
                query=rerank_query,
                documents=[c["text"] for c in chunks],
                model=config.VOYAGE_RERANK_MODEL,
                top_k=15,
            )
            reranked = [
                {**chunks[r.index], "score": r.relevance_score}
                for r in reranking.results
            ]
        else:
            reranked = list(chunks)

        reranked.sort(key=lambda c: c["score"], reverse=True)

        return _adaptive_cutoff(reranked)

    async def execute_search_book(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        selected_text: str | None,
        source_counter: int,
        seen_chunk_indices: set[int],
        is_first_search: list[bool] | None = None,
    ) -> tuple[str, list[dict], int]:
        """Execute search_book tool. Returns (formatted_text, raw_chunks, new_counter)."""
        query = tool_input["query"]
        strategy = tool_input.get("strategy", "hybrid")
        top_k = max(1, min(int(tool_input.get("top_k", 20)), 50))

        # On the first search, append selected_text to ground the query
        if is_first_search and is_first_search[0] and selected_text:
            query = f"{query} {selected_text}"
            is_first_search[0] = False

        total_chunks = await vector_store.get_total_chunks(book_id)
        # Pre-filter retrieval to PAST chunks only — avoids AHEAD results
        # crowding out PAST results in the top-k window
        reader_idx = int(reader_position * total_chunks) if total_chunks > 0 else None

        if strategy == "keyword":
            bm25 = await vector_store._get_bm25(book_id)
            results = bm25.search(query, top_k=top_k, max_chunk_index=reader_idx) if bm25 else []
        else:
            embedding = await _embed_query(query)
            if strategy == "semantic":
                results = await vector_store.search(
                    book_id, embedding, top_k=top_k, max_chunk_index=reader_idx,
                )
            else:  # hybrid
                results = await vector_store.hybrid_search(
                    book_id, embedding, query_text=query, top_k=top_k,
                    max_chunk_index=reader_idx,
                )

        labeled = label_chunks(results, reader_position, total_chunks)

        # Rerank (all results are PAST now, no AHEAD waste)
        reranked = await _rerank_and_cutoff(labeled, query, selected_text)

        # Dedup before numbering
        past_only = [
            c for c in reranked
            if c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            return "No relevant passages found in content you've already read.", [], source_counter

        # Bookend reorder: best at start, second-best at end (lost-in-the-middle)
        past_only = _bookend_reorder(past_only)

        # Format with global source numbering (post-dedup, so numbers are stable)
        display_chunks = [
            {**c, "text": _extract_relevant_sentences(c["text"], query) if config.COMPRESS_CONTEXT else c["text"]}
            for c in past_only
        ]
        formatted, source_counter = _format_sources(display_chunks, source_counter, seen_chunk_indices)
        for c, dc in zip(past_only, display_chunks):
            c["source_number"] = dc["source_number"]
        return formatted, past_only, source_counter

    async def execute_get_surrounding_context(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        source_counter: int,
        seen_chunk_indices: set[int],
    ) -> tuple[str, list[dict], int]:
        """Execute get_surrounding_context tool."""
        chunk_index = tool_input["chunk_index"]
        window = max(0, min(int(tool_input.get("window", 3)), 8))

        total_chunks = await vector_store.get_total_chunks(book_id)
        start = max(0, chunk_index - window)
        end = chunk_index + window

        results = await vector_store.get_chunks_by_range(book_id, start, end)
        labeled = label_chunks(results, reader_position, total_chunks)

        # Filter to PAST only, dedup before numbering
        past_only = [
            c for c in labeled
            if c["label"] == "PAST" and c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            return "No surrounding context available in content you've already read.", [], source_counter

        formatted, source_counter = _format_sources(past_only, source_counter, seen_chunk_indices)
        return formatted, past_only, source_counter

    async def execute_get_chapter_summary(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        source_counter: int,
        seen_chunk_indices: set[int],
    ) -> tuple[str, list[dict], int]:
        """Execute get_chapter_summary tool."""
        query = tool_input["query"]
        top_k = max(1, min(int(tool_input.get("top_k", 3)), 10))

        embedding = await _embed_query(query)
        total_chunks = await vector_store.get_total_chunks(book_id)
        results = await vector_store.search_summaries(book_id, embedding, top_k=top_k)

        labeled = label_chunks(results, reader_position, total_chunks)
        past_only = [
            c for c in labeled
            if c["label"] == "PAST" and c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            return "No chapter summaries available for content you've already read.", [], source_counter

        formatted, source_counter = _format_sources(past_only, source_counter, seen_chunk_indices, include_chunk_id=False)
        return formatted, past_only, source_counter

    async def execute_search_by_chapter(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        selected_text: str | None,
        source_counter: int,
        seen_chunk_indices: set[int],
        is_first_search: list[bool] | None = None,
    ) -> tuple[str, list[dict], int]:
        """Execute search_by_chapter tool. Falls back to search_book if no chapter metadata."""
        chapter_index = tool_input["chapter_index"]

        # Fall back to search_book if book lacks chapter metadata
        if not await vector_store.has_chapter_metadata(book_id):
            return await execute_search_book(
                tool_input, book_id, reader_position, selected_text,
                source_counter, seen_chunk_indices, is_first_search=is_first_search,
            )

        query = tool_input["query"]
        strategy = tool_input.get("strategy", "hybrid")
        top_k = max(1, min(int(tool_input.get("top_k", 20)), 50))

        # On the first search, append selected_text to ground the query
        if is_first_search and is_first_search[0] and selected_text:
            query = f"{query} {selected_text}"
            is_first_search[0] = False

        total_chunks = await vector_store.get_total_chunks(book_id)
        reader_idx = int(reader_position * total_chunks) if total_chunks > 0 else None

        if strategy == "keyword":
            bm25 = await vector_store._get_bm25(book_id)
            all_results = bm25.search(query, top_k=top_k * 3, max_chunk_index=reader_idx) if bm25 else []
            results = [
                r for r in all_results
                if r.chunk.metadata.get("chapter_index") == chapter_index
            ][:top_k]
        else:
            embedding = await _embed_query(query)
            if strategy == "semantic":
                # Use chapter + position filtered search via Qdrant
                from .vector_store import Filter, FieldCondition, MatchValue, Range
                conditions = [
                    FieldCondition(key="book_id", match=MatchValue(value=book_id)),
                    FieldCondition(key="chapter_index", match=MatchValue(value=chapter_index)),
                ]
                if reader_idx is not None:
                    conditions.append(
                        FieldCondition(key="chunk_index", range=Range(lte=reader_idx))
                    )
                chapter_filter = Filter(must=conditions)
                raw = await vector_store._ac.query_points(
                    collection_name="chunks",
                    query=embedding,
                    query_filter=chapter_filter,
                    limit=top_k,
                    with_payload=True,
                )
                results = [vector_store._point_to_result(p, p.score) for p in raw.points]
            else:  # hybrid
                results = await vector_store.search_by_chapter(
                    book_id, chapter_index, embedding, query_text=query, top_k=top_k,
                )

        labeled = label_chunks(results, reader_position, total_chunks)
        reranked = await _rerank_and_cutoff(labeled, query, selected_text)

        past_only = [
            c for c in reranked
            if c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            return "No relevant passages found in this chapter from content you've already read.", [], source_counter

        past_only = _bookend_reorder(past_only)

        display_chunks = [
            {**c, "text": _extract_relevant_sentences(c["text"], query) if config.COMPRESS_CONTEXT else c["text"]}
            for c in past_only
        ]
        formatted, source_counter = _format_sources(display_chunks, source_counter, seen_chunk_indices)
        for c, dc in zip(past_only, display_chunks):
            c["source_number"] = dc["source_number"]
        return formatted, past_only, source_counter

    async def execute_find_first_mention(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        source_counter: int,
        seen_chunk_indices: set[int],
    ) -> tuple[str, list[dict], int]:
        """Execute find_first_mention tool. Returns earliest occurrences by chunk position."""
        term = tool_input["term"]
        max_results = max(1, min(int(tool_input.get("max_results", 3)), 10))

        if len(term.strip()) < 2:
            return "Term too short — please provide at least 2 characters.", [], source_counter

        total_chunks = await vector_store.get_total_chunks(book_id)
        results = await vector_store.keyword_search(book_id, term)

        if not results:
            return f'No matches found for "{term}" in the book.', [], source_counter

        # Sort by chunk_index ascending (earliest first)
        results.sort(key=lambda r: r.chunk.metadata["chunk_index"])

        labeled = label_chunks(results, reader_position, total_chunks)

        past_only = [
            c for c in labeled
            if c["label"] == "PAST" and c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            # Don't confirm that the term appears later — that itself is a spoiler signal.
            return f'No matches found for "{term}" in what you\'ve read so far.', [], source_counter

        total_past = len(past_only)
        first_mentions = past_only[:max_results]

        formatted, source_counter = _format_sources(first_mentions, source_counter, seen_chunk_indices)
        summary = f'\nShowing {len(first_mentions)} earliest mention(s) of "{term}" (of {total_past} total in content you\'ve read).'
        return formatted + "\n\n" + summary, first_mentions, source_counter

    async def execute_get_reading_position_context(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        source_counter: int,
        seen_chunk_indices: set[int],
    ) -> tuple[str, list[dict], int]:
        """Execute get_reading_position_context tool. Returns chunks before reader's position."""
        window = max(1, min(int(tool_input.get("window", 5)), 10))

        total_chunks = await vector_store.get_total_chunks(book_id)
        if total_chunks == 0:
            return "No content indexed for this book.", [], source_counter

        reader_idx = min(int(reader_position * total_chunks), total_chunks - 1)
        start = max(0, reader_idx - window + 1)
        end = reader_idx

        results = await vector_store.get_chunks_by_range(book_id, start, end)
        labeled = label_chunks(results, reader_position, total_chunks)

        past_only = [
            c for c in labeled
            if c["label"] == "PAST" and c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            return "No context available at your current reading position.", [], source_counter

        formatted, source_counter = _format_sources(past_only, source_counter, seen_chunk_indices)
        return formatted, past_only, source_counter

    async def execute_get_book_structure(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        source_counter: int,
        seen_chunk_indices: set[int],
    ) -> tuple[str, list[dict], int]:
        """Execute get_book_structure tool. Returns chapter list with reading progress."""
        chapters = await vector_store.get_chapter_list(book_id)

        if not chapters:
            return "Chapter structure is not available for this book.", [], source_counter

        total_chunks = await vector_store.get_total_chunks(book_id)
        reader_idx = int(reader_position * total_chunks) if total_chunks > 0 else 0

        # Find current chapter
        current_chapter = None
        for ch in chapters:
            if ch["first_chunk_index"] <= reader_idx <= ch["last_chunk_index"]:
                current_chapter = ch["chapter_index"]
                break
        # If between chapters, pick the last one whose first_chunk_index <= reader_idx
        if current_chapter is None:
            for ch in reversed(chapters):
                if ch["first_chunk_index"] <= reader_idx:
                    current_chapter = ch["chapter_index"]
                    break

        lines = []
        current_name = None
        for ch in chapters:
            ci = ch["chapter_index"]
            real_title = ch["chapter_title"] or f"Chapter {ci + 1}"
            if ch["last_chunk_index"] <= reader_idx:
                label = "PAST"
            elif ch["first_chunk_index"] > reader_idx:
                label = "AHEAD"
            else:
                label = "CURRENT"

            # Redact AHEAD chapter titles — titles can be major spoilers
            display_title = f"Chapter {ci + 1}" if label == "AHEAD" else real_title
            marker = ">>>" if ci == current_chapter else "   "
            lines.append(f"{marker} {ci + 1}. {display_title} [{label}]")
            if ci == current_chapter:
                current_name = real_title

        pct = int(reader_position * 100)
        header = f"Currently reading: {current_name or 'Unknown'} ({pct}% through the book, {len(chapters)} chapters total)"
        return header + "\n\n" + "\n".join(lines), [], source_counter

    return {
        "search_book": execute_search_book,
        "get_surrounding_context": execute_get_surrounding_context,
        "get_chapter_summary": execute_get_chapter_summary,
        "search_by_chapter": execute_search_by_chapter,
        "find_first_mention": execute_find_first_mention,
        "get_reading_position_context": execute_get_reading_position_context,
        "get_book_structure": execute_get_book_structure,
        "embed_query": _embed_query,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def _anchor_lookup(
    vector_store: VectorStore,
    book_id: str,
    selected_text: str,
    reader_position: float,
    embed_fn=None,
) -> tuple[str, list[dict], int]:
    """Find the chunk containing selected_text and return +-1 context as ANCHOR sources.

    Returns (anchor_text, anchor_chunks, next_source_number) where anchor_text
    is formatted source blocks to prepend to the user message.
    """
    idx = await vector_store.find_chunk_containing(book_id, selected_text)

    # Fallback: if substring match fails, try embedding search
    if idx is None and embed_fn is not None:
        try:
            embedding = await embed_fn(selected_text)
            sem_results = await vector_store.search(book_id, embedding, top_k=3)
            if sem_results and sem_results[0].score > 0.5:
                idx = sem_results[0].chunk.metadata["chunk_index"]
                logger.info("  Anchor fallback: embedding search matched chunk %d (score=%.3f)",
                            idx, sem_results[0].score)
        except Exception as e:
            logger.warning("  Anchor embedding fallback failed: %s", e)

    if idx is None:
        return "", [], 1

    total_chunks = await vector_store.get_total_chunks(book_id)
    start = max(0, idx - 1)
    end = idx + 1
    results = await vector_store.get_chunks_by_range(book_id, start, end)
    labeled = label_chunks(results, reader_position, total_chunks)

    # The reader selected text from chunk idx, so they've definitively read
    # up to that point — but only if the chunk is close to reader_position.
    # A large gap (>5% of book) means the text matched a distant section
    # and forcing it to PAST could leak spoilers.
    reader_idx = int(reader_position * total_chunks) if total_chunks > 0 else 0
    tolerance = max(10, int(total_chunks * 0.05))
    if idx - reader_idx <= tolerance:
        for c in labeled:
            if c["chunk_index"] <= idx:
                c["label"] = "PAST"

    past_only = [c for c in labeled if c["label"] == "PAST"]
    if not past_only:
        return "", [], 1

    parts = []
    source_counter = 1
    anchor_chunks = []
    for c in past_only:
        c["source_number"] = source_counter
        chapter = c.get("chapter_title", "")
        header = f"[Source {source_counter}]"
        if chapter:
            header += f" (Chapter: {chapter})"
        header += f" [chunk {c['chunk_index']}]"
        parts.append(f"{header}\n{c['text']}")
        anchor_chunks.append(c)
        source_counter += 1

    anchor_text = "\n\n---\n\n".join(parts)
    return anchor_text, anchor_chunks, source_counter


def _build_user_message(
    question: str,
    selected_text: str | None,
    anchor_text: str = "",
    reader_position: float = 0.0,
) -> str:
    """Build the initial user message for the agent."""
    parts = []
    if reader_position > 0:
        pct = int(reader_position * 100)
        parts.append(f"[The reader is {pct}% through the book. Answer from this point in the story — not earlier, not later.]")
    if anchor_text:
        parts.append(f"[ANCHOR — passage around the reader's selection]\n\n{anchor_text}")
    if selected_text:
        parts.append(f'The reader selected this text: "{selected_text}"')
    prefix = "Their question" if selected_text else "The reader's question"
    parts.append(f"{prefix}: {question}")
    return "\n\n".join(parts)


def _audit_citations(answer: str, max_source: int) -> str:
    """Strip phantom [Source N] citations where N > max_source. Log any removed."""
    def _replace(m):
        n = int(m.group(1))
        if n < 1 or n > max_source:
            logger.warning("  Stripped phantom citation [Source %d] (max valid: %d)", n, max_source)
            return ""
        return m.group(0)
    return re.sub(r'\[Source (\d+)\]', _replace, answer)


async def _dispatch_tool(
    executors: dict,
    tool_name: str,
    tool_input: dict,
    book_id: str,
    reader_position: float,
    selected_text,
    source_counter: int,
    seen_chunk_indices: set[int],
    is_first_search: list[bool] | None = None,
) -> tuple[str, list[dict], int]:
    if tool_name == "search_book":
        return await executors["search_book"](
            tool_input, book_id, reader_position, selected_text, source_counter, seen_chunk_indices,
            is_first_search=is_first_search,
        )
    elif tool_name == "get_surrounding_context":
        return await executors["get_surrounding_context"](
            tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
        )
    elif tool_name == "get_chapter_summary":
        return await executors["get_chapter_summary"](
            tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
        )
    elif tool_name == "search_by_chapter":
        return await executors["search_by_chapter"](
            tool_input, book_id, reader_position, selected_text, source_counter, seen_chunk_indices,
            is_first_search=is_first_search,
        )
    elif tool_name == "find_first_mention":
        return await executors["find_first_mention"](
            tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
        )
    elif tool_name == "get_reading_position_context":
        return await executors["get_reading_position_context"](
            tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
        )
    elif tool_name == "get_book_structure":
        return await executors["get_book_structure"](
            tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
        )
    else:
        return f"Unknown tool: {tool_name}", [], source_counter


async def run_agent(
    anthropic: Anthropic,
    executors: dict,
    book_id: str,
    question: str,
    selected_text: str | None = None,
    reader_position: float = 0.0,
    vector_store: VectorStore | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """Run the agent loop (non-streaming). Returns {answer, sources, tool_calls, ...}."""
    source_counter = 1
    all_chunks: list[dict] = []
    seen_chunk_indices: set[int] = set()
    anchor_text = ""

    # Anchor lookup: find the chunk containing selected_text before the tool loop
    embed_fn = executors.get("embed_query")
    if selected_text and book_id and vector_store:
        anchor_text, anchor_chunks, source_counter = await _anchor_lookup(
            vector_store, book_id, selected_text, reader_position,
            embed_fn=embed_fn,
        )
        if anchor_chunks:
            all_chunks.extend(anchor_chunks)
            for c in anchor_chunks:
                seen_chunk_indices.add(c["chunk_index"])
            logger.info("  Anchor lookup: %d chunks (indices %s)",
                        len(anchor_chunks), [c["chunk_index"] for c in anchor_chunks])

    system = AGENT_SYSTEM_PROMPT
    if conversation_history:
        system += _format_conversation_history(conversation_history)

    messages = [{"role": "user", "content": _build_user_message(
        question, selected_text, anchor_text, reader_position,
    )}]
    tool_calls_log: list[dict] = []
    answer = ""
    empty_streak = 0
    is_first_search: list[bool] = [True]

    for turn in range(MAX_TURNS):
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Check if Claude wants to answer (no tool calls)
        if response.stop_reason == "end_turn":
            answer = ""
            for block in response.content:
                if block.type == "text":
                    answer += block.text
            break

        # Process tool calls
        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input

            logger.info("  Agent tool call [turn %d]: %s(%s)", turn + 1, tool_name, json.dumps(tool_input)[:200])

            formatted, chunks, source_counter = await _dispatch_tool(
                executors, tool_name, tool_input, book_id, reader_position,
                selected_text, source_counter, seen_chunk_indices,
                is_first_search=is_first_search,
            )

            # Track consecutive empty results (get_book_structure always returns [] by design)
            if not chunks and tool_name != "get_book_structure":
                empty_streak += 1
            else:
                empty_streak = 0

            # After 2 consecutive empty searches, nudge Claude to stop
            if empty_streak >= 2:
                formatted += "\n\nTwo searches returned no results. Answer with what you have or tell the reader to keep reading."

            # Chunks already deduped inside executors
            all_chunks.extend(chunks)

            tool_calls_log.append({
                "tool": tool_name,
                "input": tool_input,
                "chunks_returned": len(chunks),
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": formatted,
            })

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Cap total accumulated chunks — keep highest-scored
    MAX_TOTAL_CHUNKS = 8
    if len(all_chunks) > MAX_TOTAL_CHUNKS:
        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        all_chunks = all_chunks[:MAX_TOTAL_CHUNKS]

    if not answer:
        # Inject no-context constraint when retrieval returned nothing
        if not all_chunks:
            no_context_msg = (
                "You retrieved NO passages from the book. You MUST:\n"
                "- Say you don't have enough context from what the reader has read so far\n"
                "- Do NOT answer from your own knowledge of this book\n"
                "- Suggest the reader keep reading or ask about a specific passage"
            )
            # Append to last user message to avoid consecutive user messages
            last = messages[-1]
            if last["role"] == "user" and isinstance(last["content"], str):
                messages[-1] = {"role": "user", "content": last["content"] + "\n\n" + no_context_msg}
            else:
                messages.append({"role": "assistant", "content": "I was unable to find relevant passages."})
                messages.append({"role": "user", "content": no_context_msg})
        elif not selected_text:
            grounding_msg = (
                "IMPORTANT: No specific passage was highlighted for this question. "
                "Only state information that literally appears in the provided sources. "
                "If the sources don't contain the answer, tell the reader you couldn't find the relevant passage."
            )
            last = messages[-1]
            if last["role"] == "user" and isinstance(last["content"], str):
                messages[-1] = {"role": "user", "content": last["content"] + "\n\n" + grounding_msg}
            else:
                messages.append({"role": "assistant", "content": "I have retrieved some passages."})
                messages.append({"role": "user", "content": grounding_msg})

        # Max turns reached or no-context injected — get final answer
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1500,
            system=system,
            messages=messages,
        )
        answer = ""
        for block in response.content:
            if block.type == "text":
                answer += block.text

    # Post-generation: audit citations and detect truncation
    answer = _audit_citations(answer, max((c.get("source_number", 0) for c in all_chunks), default=0))
    if answer and not answer.rstrip().endswith(('.', '!', '?', '"', ')', ']', ':', ';', '\u201d', '\u2019', '\u2026')):
        answer = answer.rstrip() + "..."

    sources = [
        {
            "text": c["text"][:200] + "...",
            "full_text": c["text"],
            "score": c["score"],
            "chunk_index": c["chunk_index"],
            "source_number": c.get("source_number", i + 1),
            "label": c["label"],
            "chapter_title": c.get("chapter_title", ""),
        }
        for i, c in enumerate(all_chunks)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "tool_calls": tool_calls_log,
        "messages": messages,
    }


async def run_agent_streaming(
    anthropic: Anthropic,
    async_anthropic: AsyncAnthropic,
    executors: dict,
    book_id: str,
    question: str,
    selected_text: str | None = None,
    reader_position: float = 0.0,
    vector_store: VectorStore | None = None,
    conversation_history: list[dict] | None = None,
):
    """Async generator yielding SSE-compatible events.

    Event types:
      {"type": "status", "stage": str, "detail": str|None}
      {"type": "sources", "sources": list}
      {"type": "token", "text": str}
      {"type": "done"}
      {"type": "result", "state": dict}  # internal, for eval/logging
    """
    source_counter = 1
    all_chunks: list[dict] = []
    tool_calls_log: list[dict] = []
    seen_chunk_indices: set[int] = set()
    anchor_text = ""

    # Anchor lookup: find the chunk containing selected_text before the tool loop
    embed_fn = executors.get("embed_query")
    if selected_text and book_id and vector_store:
        anchor_text, anchor_chunks, source_counter = await _anchor_lookup(
            vector_store, book_id, selected_text, reader_position,
            embed_fn=embed_fn,
        )
        if anchor_chunks:
            all_chunks.extend(anchor_chunks)
            for c in anchor_chunks:
                seen_chunk_indices.add(c["chunk_index"])
            logger.info("  Anchor lookup: %d chunks (indices %s)",
                        len(anchor_chunks), [c["chunk_index"] for c in anchor_chunks])

    system = AGENT_SYSTEM_PROMPT
    if conversation_history:
        system += _format_conversation_history(conversation_history)

    messages = [{"role": "user", "content": _build_user_message(
        question, selected_text, anchor_text, reader_position,
    )}]

    # Phase 1: Tool-calling loop (non-streaming)
    yield {"type": "status", "stage": "thinking"}

    final_response = None
    empty_streak = 0
    is_first_search: list[bool] = [True]
    for turn in range(MAX_TURNS):
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_response = response
            break

        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input

            if tool_name == "get_reading_position_context":
                detail = f"{tool_name}: last {tool_input.get('window', 5)} chunks"
            elif tool_name == "get_book_structure":
                detail = f"{tool_name}"
            else:
                detail = f"{tool_name}: {tool_input.get('query', tool_input.get('chunk_index', tool_input.get('term', '')))}"
            yield {"type": "status", "stage": "searching", "detail": detail}

            logger.info("  Agent tool call [turn %d]: %s(%s)", turn + 1, tool_name, json.dumps(tool_input)[:200])

            formatted, chunks, source_counter = await _dispatch_tool(
                executors, tool_name, tool_input, book_id, reader_position,
                selected_text, source_counter, seen_chunk_indices,
                is_first_search=is_first_search,
            )

            # Track consecutive empty results (get_book_structure always returns [] by design)
            if not chunks and tool_name != "get_book_structure":
                empty_streak += 1
            else:
                empty_streak = 0

            # After 2 consecutive empty searches, nudge Claude to stop
            if empty_streak >= 2:
                formatted += "\n\nTwo searches returned no results. Answer with what you have or tell the reader to keep reading."

            # Chunks already deduped inside executors
            all_chunks.extend(chunks)

            tool_calls_log.append({
                "tool": tool_name,
                "input": tool_input,
                "chunks_returned": len(chunks),
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": formatted,
            })

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    # Cap total accumulated chunks — keep highest-scored
    MAX_TOTAL_CHUNKS = 8
    if len(all_chunks) > MAX_TOTAL_CHUNKS:
        all_chunks.sort(key=lambda c: c.get("score", 0), reverse=True)
        all_chunks = all_chunks[:MAX_TOTAL_CHUNKS]

    # Inject no-context constraint when retrieval returned nothing
    if not all_chunks and not final_response:
        no_context_msg = (
            "You retrieved NO passages from the book. You MUST:\n"
            "- Say you don't have enough context from what the reader has read so far\n"
            "- Do NOT answer from your own knowledge of this book\n"
            "- Suggest the reader keep reading or ask about a specific passage"
        )
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            messages[-1] = {"role": "user", "content": last["content"] + "\n\n" + no_context_msg}
        else:
            messages.append({"role": "assistant", "content": "I was unable to find relevant passages."})
            messages.append({"role": "user", "content": no_context_msg})
    elif all_chunks and not selected_text and not final_response:
        grounding_msg = (
            "IMPORTANT: No specific passage was highlighted for this question. "
            "Only state information that literally appears in the provided sources. "
            "If the sources don't contain the answer, tell the reader you couldn't find the relevant passage."
        )
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str):
            messages[-1] = {"role": "user", "content": last["content"] + "\n\n" + grounding_msg}
        else:
            messages.append({"role": "assistant", "content": "I have retrieved some passages."})
            messages.append({"role": "user", "content": grounding_msg})

    # Phase 2: Streaming answer
    sources = [
        {
            "text": c["text"][:200] + "...",
            "full_text": c["text"],
            "score": c["score"],
            "chunk_index": c["chunk_index"],
            "source_number": c.get("source_number", i + 1),
            "label": c["label"],
            "chapter_title": c.get("chapter_title", ""),
        }
        for i, c in enumerate(all_chunks)
    ]
    yield {"type": "sources", "sources": sources}
    yield {"type": "status", "stage": "generating"}

    if final_response and final_response.stop_reason == "end_turn":
        # Agent answered without needing streaming (simple answer from tool loop)
        # Still stream the text for consistent frontend behavior
        answer_text = ""
        for block in final_response.content:
            if block.type == "text":
                answer_text += block.text
        # Post-generation: audit citations and detect truncation
        answer_text = _audit_citations(answer_text, max((c.get("source_number", 0) for c in all_chunks), default=0))
        if answer_text and not answer_text.rstrip().endswith(('.', '!', '?', '"', ')', ']', ':', ';', '\u201d', '\u2019', '\u2026')):
            answer_text = answer_text.rstrip() + "..."
        # Stream in chunks for visual effect
        chunk_size = 12
        for i in range(0, len(answer_text), chunk_size):
            yield {"type": "token", "text": answer_text[i:i + chunk_size]}
        gen_model = final_response.model
        gen_tokens_in = final_response.usage.input_tokens
        gen_tokens_out = final_response.usage.output_tokens
        gen_stop_reason = final_response.stop_reason
    else:
        # Stream the final answer via AsyncAnthropic (no tools — force text answer)
        full_answer = []
        async with async_anthropic.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=1500,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full_answer.append(text)
                yield {"type": "token", "text": text}
            final_message = await stream.get_final_message()

        answer_text = "".join(full_answer)
        # Post-generation: detect truncation (send extra token to client)
        if answer_text and not answer_text.rstrip().endswith(('.', '!', '?', '"', ')', ']', ':', ';', '\u201d', '\u2019', '\u2026')):
            yield {"type": "token", "text": "..."}
            answer_text = answer_text.rstrip() + "..."
        # Audit citations for logging (tokens already sent to client)
        answer_text = _audit_citations(answer_text, max((c.get("source_number", 0) for c in all_chunks), default=0))
        gen_model = final_message.model
        gen_tokens_in = final_message.usage.input_tokens
        gen_tokens_out = final_message.usage.output_tokens
        gen_stop_reason = final_message.stop_reason

    yield {"type": "done"}

    # Build state for evaluation/logging
    state = {
        "question": question,
        "selected_text": selected_text,
        "reader_position": reader_position,
        "book_id": book_id,
        "answer": answer_text,
        "sources": sources,
        "tool_calls": tool_calls_log,
        "gen_model": gen_model,
        "gen_tokens_in": gen_tokens_in,
        "gen_tokens_out": gen_tokens_out,
        "gen_stop_reason": gen_stop_reason,
    }
    yield {"type": "result", "state": state}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

async def evaluate_answer(state: dict, anthropic: Anthropic) -> dict:
    """Score answer quality with Haiku. Same logic as graph.py evaluate_node."""
    answer = state.get("answer", "")
    question = state.get("question", "")
    sources = state.get("sources", [])

    if not answer:
        return {}

    context_preview = "\n".join(s.get("full_text", s.get("text", ""))[:200] for s in sources[:3])
    user_prompt = (
        f"Question: {question}\n\n"
        f"Sources (first 3):\n{context_preview}\n\n"
        f"Answer: {answer}"
    )

    try:
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_HAIKU_MODEL,
            max_tokens=100,
            system=EVALUATE_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        parsed = json.loads(strip_code_fences(raw))
        relevance = max(1, min(5, int(parsed.get("relevance", 3))))
        grounding = max(1, min(5, int(parsed.get("grounding", 3))))

        flags = []
        if relevance <= 2:
            flags.append("low_relevance")
        if grounding <= 2:
            flags.append("low_grounding")

        logger.info("  Eval: relevance=%d/5, grounding=%d/5", relevance, grounding)

        return {
            "eval_relevance": relevance,
            "eval_grounding": grounding,
            "eval_flags": flags,
            "eval_low_confidence": bool(flags),
            "eval_tokens_in": response.usage.input_tokens,
            "eval_tokens_out": response.usage.output_tokens,
        }
    except Exception as e:
        logger.warning("  Evaluation failed: %s", e)
        return {
            "eval_relevance": 0,
            "eval_grounding": 0,
            "eval_flags": ["eval_error"],
            "eval_low_confidence": True,
            "eval_tokens_in": 0,
            "eval_tokens_out": 0,
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_agent_query(state: dict):
    """Log a completed agent query to JSONL and human-readable log files."""
    W = 80

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "agent",
        "book_id": state.get("book_id"),
        "reader_position": state.get("reader_position"),
        "question": state.get("question"),
        "selected_text": state.get("selected_text", ""),
        "tool_calls": state.get("tool_calls", []),
        "sources": [
            {
                "chunk_index": s.get("chunk_index"),
                "score": round(s.get("score", 0), 4),
                "source_number": s.get("source_number"),
                "label": s.get("label"),
                "text": s.get("full_text", s.get("text", "")),
            }
            for s in state.get("sources", [])
        ],
        "response": {
            "answer": state.get("answer", ""),
            "model": state.get("gen_model"),
            "input_tokens": state.get("gen_tokens_in"),
            "output_tokens": state.get("gen_tokens_out"),
            "stop_reason": state.get("gen_stop_reason"),
        },
        "evaluation": {
            "relevance": state.get("eval_relevance"),
            "grounding": state.get("eval_grounding"),
            "flags": state.get("eval_flags", []),
            "low_confidence": state.get("eval_low_confidence", False),
            "input_tokens": state.get("eval_tokens_in", 0),
            "output_tokens": state.get("eval_tokens_out", 0),
        },
    }

    # JSONL
    with open(LOG_DIR / "queries.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Human-readable
    with open(LOG_DIR / "queries.log", "a") as f:
        def section(title: str):
            f.write(f"\n── {title} " + "─" * (W - 4 - len(title)) + "\n")

        f.write("\n\n" + "═" * W + "\n")
        f.write(f"AGENT QUERY @ {log_entry['timestamp']}\n")
        f.write("═" * W + "\n\n")

        f.write(f"Book:     {state.get('book_id')}\n")
        f.write(f"Position: {state.get('reader_position', 0):.1%}\n")
        selected = state.get("selected_text") or ""
        f.write(f'Selected: "{selected[:200]}{"…" if len(selected) > 200 else ""}"\n')
        f.write(f"Question: {state.get('question')}\n")

        # Tool calls
        tool_calls = state.get("tool_calls", [])
        section(f"Tool Calls ({len(tool_calls)})")
        for i, tc in enumerate(tool_calls):
            f.write(f"\n  [{i+1}] {tc['tool']}({json.dumps(tc['input'])[:120]})\n")
            f.write(f"      → {tc['chunks_returned']} chunks\n")

        # Sources
        sources = state.get("sources", [])
        section(f"Sources ({len(sources)})")
        for s in sources:
            text = s.get("full_text", s.get("text", ""))
            f.write(f"\n  [Source {s.get('source_number')}] chunk {s.get('chunk_index')} | score={s.get('score', 0):.4f}\n")
            f.write(f"      {text[:120]}…\n")

        # Answer
        section("Answer")
        f.write(f"\n  Model:  {state.get('gen_model')}\n")
        f.write(f"  Tokens: {state.get('gen_tokens_in')} in / {state.get('gen_tokens_out')} out\n\n")
        for line in state.get("answer", "").split("\n"):
            f.write(f"    {line}\n")

        # Evaluation
        if state.get("eval_relevance") is not None:
            section("Evaluation")
            f.write(f"\n  Relevance:  {state.get('eval_relevance')}/5\n")
            f.write(f"  Grounding:  {state.get('eval_grounding')}/5\n")
            flags = state.get("eval_flags", [])
            if flags:
                f.write(f"  Flags:      {', '.join(flags)}\n")

        f.write("\n" + "═" * W + "\n\n")

    logger.info(
        "  Logged agent query — %d tool calls, %d sources, gen: %d/%d tokens",
        len(tool_calls), len(sources),
        state.get("gen_tokens_in", 0), state.get("gen_tokens_out", 0),
    )


# ---------------------------------------------------------------------------
# Builder (entry point for server.py)
# ---------------------------------------------------------------------------

def build_agent(
    anthropic: Anthropic,
    async_anthropic: AsyncAnthropic,
    vector_store: VectorStore,
    voyage_client,
) -> dict:
    """Build the agent, returning callables for server.py."""
    executors = build_tool_executors(vector_store, voyage_client)

    async def _run_agent(
        book_id: str,
        question: str,
        selected_text: str | None = None,
        reader_position: float = 0.0,
        conversation_history: list[dict] | None = None,
    ) -> dict:
        return await run_agent(
            anthropic, executors, book_id, question, selected_text, reader_position,
            vector_store=vector_store,
            conversation_history=conversation_history,
        )

    async def _run_agent_streaming(
        book_id: str,
        question: str,
        selected_text: str | None = None,
        reader_position: float = 0.0,
        conversation_history: list[dict] | None = None,
    ):
        async for event in run_agent_streaming(
            anthropic, async_anthropic, executors,
            book_id, question, selected_text, reader_position,
            vector_store=vector_store,
            conversation_history=conversation_history,
        ):
            yield event

    async def _evaluate(state: dict) -> dict:
        if config.AGENT_SKIP_EVAL:
            return {}
        return await evaluate_answer(state, anthropic)

    return {
        "run_agent": _run_agent,
        "run_agent_streaming": _run_agent_streaming,
        "evaluate": _evaluate,
        "log": log_agent_query,
    }
