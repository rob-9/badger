"""
Tool-calling agent for the RAG pipeline.

Replaces the fixed LangGraph DAG with a loop where Claude decides which
tools to call, how many times, and in what order. Three tools:

- search_book: semantic / keyword / hybrid search with reranking
- get_surrounding_context: proximity retrieval around a chunk
- get_chapter_summary: broad thematic search over chapter summaries

The agent loop runs tool-call turns non-streaming (fast, 1-2 turns typical),
then streams only the final answer. This preserves the SSE event sequence
the frontend expects: status* → sources → token* → done.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic, AsyncAnthropic

from boom import config
from .prompts import AGENT_SYSTEM_PROMPT, EVALUATE_PROMPT
from .graph import label_chunks, strip_code_fences
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

LOG_DIR = Path(".data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_TURNS = 3


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
    import re

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
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

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
        """Rerank with Voyage rerank-2.5, apply AHEAD penalty + adaptive cutoff."""
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

        # Penalize AHEAD chunks
        for c in reranked:
            if c["label"] == "AHEAD":
                c["score"] *= 0.3
        reranked.sort(key=lambda c: c["score"], reverse=True)

        return _adaptive_cutoff(reranked)

    async def execute_search_book(
        tool_input: dict,
        book_id: str,
        reader_position: float,
        selected_text: str | None,
        source_counter: int,
        seen_chunk_indices: set[int],
    ) -> tuple[str, list[dict], int]:
        """Execute search_book tool. Returns (formatted_text, raw_chunks, new_counter)."""
        query = tool_input["query"]
        strategy = tool_input.get("strategy", "hybrid")
        top_k = max(1, min(int(tool_input.get("top_k", 20)), 50))

        total_chunks = await vector_store.get_total_chunks(book_id)

        if strategy == "keyword":
            bm25 = await vector_store._get_bm25(book_id)
            results = bm25.search(query, top_k=top_k) if bm25 else []
        else:
            embedding = await _embed_query(query)
            if strategy == "semantic":
                results = await vector_store.search(book_id, embedding, top_k=top_k)
            else:  # hybrid
                results = await vector_store.hybrid_search(
                    book_id, embedding, query_text=query, top_k=top_k,
                )

        labeled = label_chunks(results, reader_position, total_chunks)

        # Rerank
        reranked = await _rerank_and_cutoff(labeled, query, selected_text)

        # Filter to PAST only, dedup before numbering
        past_only = [
            c for c in reranked
            if c["label"] == "PAST" and c["chunk_index"] not in seen_chunk_indices
        ]

        if not past_only:
            return "No relevant passages found in content you've already read.", [], source_counter

        # Bookend reorder: best at start, second-best at end (lost-in-the-middle)
        past_only = _bookend_reorder(past_only)

        # Format with global source numbering (post-dedup, so numbers are stable)
        parts = []
        for c in past_only:
            seen_chunk_indices.add(c["chunk_index"])
            c["source_number"] = source_counter
            chapter = c.get("chapter_title", "")
            header = f"[Source {source_counter}]"
            if chapter:
                header += f" (Chapter: {chapter})"
            header += f" [chunk {c['chunk_index']}]"
            display_text = (
                _extract_relevant_sentences(c["text"], query)
                if config.COMPRESS_CONTEXT else c["text"]
            )
            parts.append(f"{header}\n{display_text}")
            source_counter += 1

        formatted = "\n\n---\n\n".join(parts)
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

        parts = []
        for c in past_only:
            seen_chunk_indices.add(c["chunk_index"])
            c["source_number"] = source_counter
            chapter = c.get("chapter_title", "")
            header = f"[Source {source_counter}]"
            if chapter:
                header += f" (Chapter: {chapter})"
            header += f" [chunk {c['chunk_index']}]"
            parts.append(f"{header}\n{c['text']}")
            source_counter += 1

        formatted = "\n\n---\n\n".join(parts)
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

        parts = []
        for c in past_only:
            seen_chunk_indices.add(c["chunk_index"])
            c["source_number"] = source_counter
            chapter = c.get("chapter_title", "")
            header = f"[Source {source_counter}]"
            if chapter:
                header += f" (Chapter: {chapter})"
            parts.append(f"{header}\n{c['text']}")
            source_counter += 1

        formatted = "\n\n---\n\n".join(parts)
        return formatted, past_only, source_counter

    return {
        "search_book": execute_search_book,
        "get_surrounding_context": execute_get_surrounding_context,
        "get_chapter_summary": execute_get_chapter_summary,
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def _anchor_lookup(
    vector_store: VectorStore,
    book_id: str,
    selected_text: str,
    reader_position: float,
) -> tuple[str, list[dict], int]:
    """Find the chunk containing selected_text and return +-1 context as ANCHOR sources.

    Returns (anchor_text, anchor_chunks, next_source_number) where anchor_text
    is formatted source blocks to prepend to the user message.
    """
    idx = await vector_store.find_chunk_containing(book_id, selected_text)
    if idx == 0 and selected_text:
        # find_chunk_containing returns 0 on miss AND when found at index 0.
        # Double-check by verifying the text is actually in chunk 0.
        entries = vector_store.entries.get(book_id)
        if entries and selected_text[:50].lower() not in entries[0].chunk.text.lower():
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
) -> str:
    """Build the initial user message for the agent."""
    parts = []
    if anchor_text:
        parts.append(f"[ANCHOR — passage around the reader's selection]\n\n{anchor_text}")
    if selected_text:
        parts.append(f'The reader selected this text: "{selected_text}"')
    prefix = "Their question" if selected_text else "The reader's question"
    parts.append(f"{prefix}: {question}")
    return "\n\n".join(parts)


async def run_agent(
    anthropic: Anthropic,
    executors: dict,
    book_id: str,
    question: str,
    selected_text: str | None = None,
    reader_position: float = 0.0,
    vector_store: VectorStore | None = None,
) -> dict:
    """Run the agent loop (non-streaming). Returns {answer, sources, tool_calls, ...}."""
    source_counter = 1
    all_chunks: list[dict] = []
    seen_chunk_indices: set[int] = set()
    anchor_text = ""

    # Anchor lookup: find the chunk containing selected_text before the tool loop
    if selected_text and book_id and vector_store:
        anchor_text, anchor_chunks, source_counter = await _anchor_lookup(
            vector_store, book_id, selected_text, reader_position,
        )
        if anchor_chunks:
            all_chunks.extend(anchor_chunks)
            for c in anchor_chunks:
                seen_chunk_indices.add(c["chunk_index"])
            logger.info("  Anchor lookup: %d chunks (indices %s)",
                        len(anchor_chunks), [c["chunk_index"] for c in anchor_chunks])

    messages = [{"role": "user", "content": _build_user_message(question, selected_text, anchor_text)}]
    tool_calls_log: list[dict] = []
    answer = ""
    empty_streak = 0

    for turn in range(MAX_TURNS):
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
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

            if tool_name == "search_book":
                formatted, chunks, source_counter = await executors["search_book"](
                    tool_input, book_id, reader_position, selected_text, source_counter, seen_chunk_indices,
                )
            elif tool_name == "get_surrounding_context":
                formatted, chunks, source_counter = await executors["get_surrounding_context"](
                    tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
                )
            elif tool_name == "get_chapter_summary":
                formatted, chunks, source_counter = await executors["get_chapter_summary"](
                    tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
                )
            else:
                formatted = f"Unknown tool: {tool_name}"
                chunks = []

            # Track consecutive empty results
            if len(chunks) == 0:
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
    else:
        # Max turns reached — get final answer
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            messages=messages,
        )
        answer = ""
        for block in response.content:
            if block.type == "text":
                answer += block.text

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
    if selected_text and book_id and vector_store:
        anchor_text, anchor_chunks, source_counter = await _anchor_lookup(
            vector_store, book_id, selected_text, reader_position,
        )
        if anchor_chunks:
            all_chunks.extend(anchor_chunks)
            for c in anchor_chunks:
                seen_chunk_indices.add(c["chunk_index"])
            logger.info("  Anchor lookup: %d chunks (indices %s)",
                        len(anchor_chunks), [c["chunk_index"] for c in anchor_chunks])

    messages = [{"role": "user", "content": _build_user_message(question, selected_text, anchor_text)}]

    # Phase 1: Tool-calling loop (non-streaming)
    yield {"type": "status", "stage": "thinking"}

    final_response = None
    empty_streak = 0
    for turn in range(MAX_TURNS):
        response = await asyncio.to_thread(
            anthropic.messages.create,
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
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

            detail = f"{tool_name}: {tool_input.get('query', tool_input.get('chunk_index', ''))}"
            yield {"type": "status", "stage": "searching", "detail": detail}

            logger.info("  Agent tool call [turn %d]: %s(%s)", turn + 1, tool_name, json.dumps(tool_input)[:200])

            if tool_name == "search_book":
                formatted, chunks, source_counter = await executors["search_book"](
                    tool_input, book_id, reader_position, selected_text, source_counter, seen_chunk_indices,
                )
            elif tool_name == "get_surrounding_context":
                formatted, chunks, source_counter = await executors["get_surrounding_context"](
                    tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
                )
            elif tool_name == "get_chapter_summary":
                formatted, chunks, source_counter = await executors["get_chapter_summary"](
                    tool_input, book_id, reader_position, source_counter, seen_chunk_indices,
                )
            else:
                formatted = f"Unknown tool: {tool_name}"
                chunks = []

            # Track consecutive empty results
            if len(chunks) == 0:
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
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full_answer.append(text)
                yield {"type": "token", "text": text}
            final_message = await stream.get_final_message()

        answer_text = "".join(full_answer)
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
    ) -> dict:
        return await run_agent(
            anthropic, executors, book_id, question, selected_text, reader_position,
            vector_store=vector_store,
        )

    async def _run_agent_streaming(
        book_id: str,
        question: str,
        selected_text: str | None = None,
        reader_position: float = 0.0,
    ):
        async for event in run_agent_streaming(
            anthropic, async_anthropic, executors,
            book_id, question, selected_text, reader_position,
            vector_store=vector_store,
        ):
            yield event

    async def _evaluate(state: dict) -> dict:
        return await evaluate_answer(state, anthropic)

    return {
        "run_agent": _run_agent,
        "run_agent_streaming": _run_agent_streaming,
        "evaluate": _evaluate,
        "log": log_agent_query,
    }
