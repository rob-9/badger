"""
LangGraph agent workflow for the RAG pipeline.

Routes questions through type-specific retrieval strategies:
- vocabulary: keyword/semantic top_k=5, concise definition
- context: proximity ±3 chunks, passage explanation
- lookup: semantic top_k=20 → rerank with adaptive cutoff (3-10)
- analysis: hybrid + summaries → rerank with adaptive cutoff (3-10)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing_extensions import TypedDict

from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from boom import config
from .prompts import SYSTEM_PROMPTS
from .vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)

LOG_DIR = Path(".data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES = {"vocabulary", "context", "lookup", "analysis"}
LOG_SEPARATOR = "──────────────────────────────────────────────"


# === State ===

class QAState(TypedDict, total=False):
    # Input
    question: str
    selected_text: Optional[str]
    reader_position: float
    book_id: str

    # Classification
    question_type: str
    entities: list[str]
    classify_raw_response: str
    classify_tokens_in: int
    classify_tokens_out: int

    # Retrieval
    chunks: list[dict]
    retrieval_strategy: str
    retrieval_query: str
    retrieval_center_chunk: int
    retrieval_range_start: int
    retrieval_range_end: int
    retrieval_top_k: int
    retrieval_embedding_dims: int
    retrieval_chunk_match: str  # "exact" | "fallback"

    # Generation
    answer: str
    sources: list[dict]
    gen_model: str
    gen_tokens_in: int
    gen_tokens_out: int
    gen_stop_reason: str
    gen_system_prompt: str
    gen_user_prompt: str

    # Internal
    total_chunks: int


# === Helpers ===

def strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def label_chunks(results: list[SearchResult], reader_position: float, total_chunks: int) -> list[dict]:
    """Tag chunks as PAST or AHEAD based on reader position."""
    reader_idx = int(reader_position * total_chunks) if total_chunks > 0 else 0
    return [
        {
            "text": r.chunk.text,
            "chunk_index": r.chunk.metadata["chunk_index"],
            "score": r.score,
            "label": "PAST" if r.chunk.metadata["chunk_index"] <= reader_idx else "AHEAD",
        }
        for r in results
    ]


def _label_and_log(results: list[SearchResult], state: QAState) -> list[dict]:
    """Label chunks with PAST/AHEAD and log retrieval results."""
    position = state.get("reader_position", 0)
    total = state.get("total_chunks", 0)
    labeled = label_chunks(results, position, total)
    reader_chunk = int(position * total) if total > 0 else 0
    logger.info("  Reader at chunk %d — labeling PAST/AHEAD", reader_chunk)
    logger.info("  Retrieved %d chunks:", len(labeled))
    for i, c in enumerate(labeled):
        logger.info(
            "    [%d] score=%.4f | chunk %d | %s | %s…",
            i + 1, c["score"], c["chunk_index"], c["label"], c["text"][:80],
        )
    logger.info(LOG_SEPARATOR)
    return labeled


def build_query(question: str, selected_text: Optional[str], entities: list[str]) -> str:
    """Build search query from question + context."""
    parts = [question]
    if selected_text:
        parts.append(f"Referring to: {selected_text}")
    if entities:
        parts.append(f"Key terms: {', '.join(entities)}")
    return "\n".join(parts)


def build_context_string(chunks: list[dict]) -> str:
    """Build context string with PAST/AHEAD labels from labeled chunks."""
    def format_group(group: list[dict]) -> str:
        return "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{c['text']}" for i, c in enumerate(group)
        )

    past = [c for c in chunks if c["label"] == "PAST"]
    ahead = [c for c in chunks if c["label"] == "AHEAD"]

    parts = []
    if past:
        parts.append(f"[ALREADY READ]\n{format_group(past)}")
    if ahead:
        parts.append(f"[COMING UP]\n{format_group(ahead)}")

    return "\n\n===\n\n".join(parts)


def prepare_generate(state: QAState) -> dict:
    """Build prompts and sources for generation without calling the LLM."""
    q_type = state.get("question_type", "context")
    chunks = state.get("chunks", [])
    selected = state.get("selected_text") or ""

    system_prompt = SYSTEM_PROMPTS.get(q_type, SYSTEM_PROMPTS["context"])

    if not chunks:
        user_prompt = f'Selected text: "{selected}"\n\nQuestion: {state["question"]}'
    elif selected:
        context = build_context_string(chunks)
        user_prompt = f'Selected text: "{selected}"\n\nContext from the book:\n{context}\n\nQuestion: {state["question"]}'
    else:
        context = build_context_string(chunks)
        user_prompt = f'Context from the book:\n{context}\n\nQuestion: {state["question"]}'

    sources = [
        {
            "text": c["text"][:200] + "...",
            "full_text": c["text"],
            "score": c["score"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "sources": sources,
    }


# === Logging ===

def _build_log_entry(state: QAState) -> dict:
    """Build structured log entry from final pipeline state."""
    chunks = state.get("chunks", [])
    total = state.get("total_chunks", 0)
    position = state.get("reader_position", 0)

    return {
        "timestamp": datetime.now().isoformat(),
        "book_id": state.get("book_id"),
        "reader_position": position,
        "reader_chunk_index": int(position * total) if total else 0,
        "total_chunks": total,
        "question": state["question"],
        "selected_text": state.get("selected_text") or "",
        # Classification
        "question_type": state.get("question_type"),
        "entities": state.get("entities", []),
        "classify_raw_response": state.get("classify_raw_response"),
        "classify_tokens": {
            "in": state.get("classify_tokens_in"),
            "out": state.get("classify_tokens_out"),
        },
        # Retrieval
        "retrieval_strategy": state.get("retrieval_strategy"),
        "retrieval_query": state.get("retrieval_query"),
        "retrieval_top_k": state.get("retrieval_top_k"),
        "retrieval_embedding_dims": state.get("retrieval_embedding_dims"),
        "retrieval_center_chunk": state.get("retrieval_center_chunk"),
        "retrieval_range": [state.get("retrieval_range_start"), state.get("retrieval_range_end")],
        "retrieval_chunk_match": state.get("retrieval_chunk_match"),
        "retrieved_chunks": [
            {
                "chunk_index": c["chunk_index"],
                "score": round(c["score"], 4),
                "label": c["label"],
                "text": c["text"],
            }
            for c in chunks
        ],
        # Generation
        "system_prompt": state.get("gen_system_prompt"),
        "user_prompt": state.get("gen_user_prompt"),
        "response": {
            "answer": state.get("answer", ""),
            "model": state.get("gen_model"),
            "input_tokens": state.get("gen_tokens_in"),
            "output_tokens": state.get("gen_tokens_out"),
            "stop_reason": state.get("gen_stop_reason"),
        },
    }


def _write_readable_log(state: QAState, log_entry: dict):
    """Write human-readable log entry to queries.log."""
    W = 80
    chunks = state.get("chunks", [])
    selected = state.get("selected_text") or ""
    total = state.get("total_chunks", 0)
    position = state.get("reader_position", 0)
    reader_chunk = int(position * total) if total else 0

    with open(LOG_DIR / "queries.log", "a") as f:
        def section(title: str):
            f.write(f"\n── {title} " + "─" * (W - 4 - len(title)) + "\n")

        # Header
        f.write("\n\n" + "═" * W + "\n")
        f.write(f"QUERY @ {log_entry['timestamp']}\n")
        f.write("═" * W + "\n\n")

        f.write(f"Book:     {state.get('book_id')}\n")
        f.write(f"Position: {position:.1%} (chunk {reader_chunk}/{total})\n")
        f.write(f'Selected: "{selected[:200]}{"…" if len(selected) > 200 else ""}"\n')
        f.write(f"Question: {state['question']}\n")

        # Classification
        section("Classification")
        f.write(f"\n  Model:    {config.CLAUDE_HAIKU_MODEL}\n")
        f.write(f"  Tokens:   {state.get('classify_tokens_in')} in / {state.get('classify_tokens_out')} out\n")
        f.write(f"  Raw:      {state.get('classify_raw_response')}\n")
        f.write(f"  Type:     {state.get('question_type')}\n")
        f.write(f"  Entities: {', '.join(state.get('entities', [])) or '(none)'}\n")

        # Retrieval
        strategy = state.get("retrieval_strategy")
        section("Retrieval")
        f.write(f"\n  Strategy: {strategy}\n")
        if strategy == "proximity":
            f.write(f"  Chunk match: {state.get('retrieval_chunk_match', 'unknown')}\n")
            f.write(f"  Center:   chunk {state.get('retrieval_center_chunk')}\n")
            f.write(f"  Range:    [{state.get('retrieval_range_start')}, {state.get('retrieval_range_end')}]\n")
        elif strategy and "keyword" in strategy:
            f.write(f'  Keyword:  "{state.get("retrieval_query", "")}"\n')
        else:
            f.write(f"  Query:    {(state.get('retrieval_query') or '')[:200]}\n")
            f.write(f"  Top K:    {state.get('retrieval_top_k')}\n")
            f.write(f"  Embedding: {state.get('retrieval_embedding_dims')} dimensions ({config.VOYAGE_CONTEXT_MODEL})\n")

        f.write(f"\n  Results: {len(chunks)} chunks\n")
        for i, c in enumerate(chunks):
            f.write(f"\n    [{i+1}] score={c['score']:.4f} | chunk {c['chunk_index']} | {c['label']}\n")
            chunk_text = c["text"]
            if selected and selected.lower() in chunk_text.lower():
                idx = chunk_text.lower().find(selected.lower())
                chunk_text = (
                    chunk_text[:idx]
                    + ">>>" + chunk_text[idx:idx + len(selected)] + "<<<"
                    + chunk_text[idx + len(selected):]
                )
            for line in chunk_text.split("\n"):
                f.write(f"        {line}\n")

        # LLM Input
        sys_prompt = state.get("gen_system_prompt") or ""
        usr_prompt = state.get("gen_user_prompt") or ""
        section("LLM Input")
        f.write(f"\n  System ({len(sys_prompt)} chars):\n")
        for line in sys_prompt.split("\n"):
            f.write(f"    {line}\n")
        f.write(f"\n  User ({len(usr_prompt)} chars):\n")
        usr_display = usr_prompt if len(usr_prompt) <= 500 else usr_prompt[:500] + f"\n    … [{len(usr_prompt) - 500} more chars]"
        for line in usr_display.split("\n"):
            f.write(f"    {line}\n")

        # LLM Output
        section("LLM Output")
        f.write(f"\n  Model:  {state.get('gen_model')}\n")
        f.write(f"  Tokens: {state.get('gen_tokens_in')} in / {state.get('gen_tokens_out')} out\n")
        f.write(f"  Stop:   {state.get('gen_stop_reason')}\n")
        f.write("\n  Answer:\n")
        for line in state.get("answer", "").split("\n"):
            f.write(f"    {line}\n")

        f.write("\n" + "═" * W + "\n\n")


def log_query(state: QAState):
    """Log a completed query to JSONL and human-readable log files."""
    logger.info("── LOG %s", LOG_SEPARATOR[6:])
    logger.info("  Writing to queries.jsonl + queries.log")

    log_entry = _build_log_entry(state)

    with open(LOG_DIR / "queries.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    _write_readable_log(state, log_entry)

    logger.info(
        "  Done — classify: %d/%d tokens, generate: %d/%d tokens",
        state.get("classify_tokens_in", 0), state.get("classify_tokens_out", 0),
        state.get("gen_tokens_in", 0), state.get("gen_tokens_out", 0),
    )
    logger.info(LOG_SEPARATOR)


# === Graph builder ===

def build_qa_graph(anthropic: Anthropic, vector_store: VectorStore, voyage_client) -> dict:
    """Build and compile the QA LangGraph."""

    def _embed_query(text: str) -> list[float]:
        """Embed a query using voyage-context-3."""
        return voyage_client.contextualized_embed(
            inputs=[[text]],
            model=config.VOYAGE_CONTEXT_MODEL,
            input_type="query",
        ).results[0].embeddings[0]

    # --- Node: classify ---

    async def classify_node(state: QAState) -> dict:
        """Classify question type with Haiku."""
        selected = state.get("selected_text") or ""
        question = state["question"]
        book_id = state["book_id"]

        logger.info("── CLASSIFY %s", LOG_SEPARATOR[11:])
        logger.info("  Book:     %s", book_id)
        logger.info("  Question: %s", question[:100])
        logger.info('  Selected: "%s%s"', selected[:80], "…" if len(selected) > 80 else "")
        logger.info("  Model:    %s", config.CLAUDE_HAIKU_MODEL)

        classify_prompt = (
            f'Question: "{question}"\n'
            f'Selected text: "{selected[:200]}"\n\n'
            f"Classify as exactly one of: vocabulary, context, lookup, analysis\n"
            f"Extract key entities (names, terms, concepts).\n\n"
            f'Return: {{"type": "...", "entities": [...]}}'
        )
        logger.info("  Prompt:   %s", classify_prompt.replace("\n", " ")[:200])

        response = anthropic.messages.create(
            model=config.CLAUDE_HAIKU_MODEL,
            max_tokens=150,
            system="Classify the reading question. Return JSON only, no markdown.",
            messages=[{"role": "user", "content": classify_prompt}],
        )
        raw_text = response.content[0].text.strip()
        logger.info("  Raw response: %s", raw_text)
        logger.info("  Tokens: %d in / %d out", response.usage.input_tokens, response.usage.output_tokens)

        try:
            parsed = json.loads(strip_code_fences(raw_text))
        except json.JSONDecodeError:
            logger.warning("  FAILED to parse classification JSON, falling back to 'context': %s", raw_text)
            parsed = {"type": "context", "entities": []}

        q_type = parsed.get("type", "context")
        if q_type not in VALID_TYPES:
            logger.warning("  Invalid type '%s', falling back to 'context'", q_type)
            q_type = "context"

        total = await vector_store.get_total_chunks(book_id)
        entities = parsed.get("entities", [])

        logger.info("  Result: type=%s, entities=%s, total_chunks=%d", q_type, entities, total)
        logger.info(LOG_SEPARATOR)

        return {
            "question_type": q_type,
            "entities": entities,
            "total_chunks": total,
            "classify_raw_response": raw_text,
            "classify_tokens_in": response.usage.input_tokens,
            "classify_tokens_out": response.usage.output_tokens,
        }

    # --- Node: vocabulary retrieval ---

    async def vocabulary_node(state: QAState) -> dict:
        """Keyword search for vocabulary — find chunks containing the word."""
        selected = state.get("selected_text") or ""
        book_id = state["book_id"]
        logger.info("── RETRIEVE (vocabulary) ─────────────────────")
        logger.info('  Strategy: keyword search for "%s"', selected[:100])

        results = await vector_store.keyword_search(book_id, selected)
        strategy = "keyword"
        logger.info('  Keyword matches: %d chunks contain "%s"', len(results), selected[:50])

        if not results:
            logger.info("  No keyword matches — falling back to semantic search")
            strategy = "keyword→semantic_fallback"
            query = selected if selected else state["question"]
            embedding = _embed_query(query)
            logger.info('  Semantic query: "%s", %d dimensions', query[:100], len(embedding))
            results = await vector_store.search(book_id, embedding, top_k=5)

        results = results[:5]
        labeled = _label_and_log(results, state)

        return {
            "chunks": labeled,
            "retrieval_strategy": strategy,
            "retrieval_query": selected,
        }

    # --- Node: context (proximity) retrieval ---

    async def context_node(state: QAState) -> dict:
        """Proximity retrieval ±3 chunks for passage explanation."""
        selected = state.get("selected_text") or ""
        book_id = state["book_id"]
        logger.info("── RETRIEVE (context) ────────────────────────")
        logger.info("  Strategy: proximity ±3 chunks")
        logger.info('  Looking for selected text in chunks: "%s…"', selected[:60])

        center = await vector_store.find_chunk_containing(book_id, selected)
        chunk_match = "exact" if (center > 0 or not selected) else "fallback"
        start, end = max(0, center - 3), center + 3

        logger.info("  Chunk match: %s", chunk_match)
        logger.info("  Center chunk: %d, range: [%d, %d]", center, start, end)
        if chunk_match == "fallback" and selected:
            logger.warning("  find_chunk_containing fell back to 0 — selected text not found in any chunk")

        results = await vector_store.get_chunks_by_range(book_id, start, end)
        labeled = _label_and_log(results, state)

        return {
            "chunks": labeled,
            "retrieval_strategy": "proximity",
            "retrieval_center_chunk": center,
            "retrieval_range_start": start,
            "retrieval_range_end": end,
            "retrieval_chunk_match": chunk_match,
        }

    # --- Node factory: semantic retrieval (lookup & analysis) ---

    def _make_semantic_node(top_k: int, strategy: str):
        """Create a pure semantic retrieval node (cosine similarity only)."""
        async def semantic_node(state: QAState) -> dict:
            logger.info("── RETRIEVE (%s) %s", strategy, "─" * (33 - len(strategy)))
            logger.info("  Strategy: semantic search, top_k=%d", top_k)

            query = build_query(state["question"], state.get("selected_text"), state.get("entities", []))
            logger.info("  Query: %s", query[:200])
            logger.info("  Voyage model: %s, input_type: query", config.VOYAGE_CONTEXT_MODEL)

            embedding = _embed_query(query)
            logger.info("  Embedding: %d dimensions", len(embedding))
            logger.info("  Searching %d total chunks in book %s", state.get("total_chunks", 0), state["book_id"])

            results = await vector_store.search(state["book_id"], embedding, top_k=top_k)
            labeled = _label_and_log(results, state)

            return {
                "chunks": labeled,
                "retrieval_strategy": strategy,
                "retrieval_query": query,
                "retrieval_top_k": top_k,
                "retrieval_embedding_dims": len(embedding),
            }
        return semantic_node

    lookup_node = _make_semantic_node(top_k=20, strategy="semantic")

    # Analysis uses hybrid search + chapter summaries for broad thematic questions.
    # Summaries act as chapter-level index entries that help match questions like
    # "what does the book say about X" to the right chapter, even when no single
    # passage chunk would score well on its own.
    async def analysis_node(state: QAState) -> dict:
        logger.info("── RETRIEVE (analysis) ───────────────────────")
        logger.info("  Strategy: hybrid + chapter summaries")

        query = build_query(state["question"], state.get("selected_text"), state.get("entities", []))
        logger.info("  Query: %s", query[:200])

        embedding = _embed_query(query)
        logger.info("  Embedding: %d dimensions", len(embedding))

        # Passage-level: hybrid (semantic + BM25), top 20
        passage_results = await vector_store.hybrid_search(
            state["book_id"], embedding, query_text=query, top_k=20,
        )
        logger.info("  Passage results: %d", len(passage_results))

        # Chapter-level: summary vectors, top 3
        summary_results = await vector_store.search_summaries(
            state["book_id"], embedding, top_k=3,
        )
        logger.info("  Summary results: %d", len(summary_results))

        # Merge both tiers — reranker + adaptive cutoff will select 3-10
        all_results = passage_results + summary_results
        labeled = _label_and_log(all_results, state)

        return {
            "chunks": labeled,
            "retrieval_strategy": "hybrid_broad+summaries",
            "retrieval_query": query,
            "retrieval_top_k": 20,
            "retrieval_embedding_dims": len(embedding),
        }

    # --- Node: rerank ---

    def _adaptive_cutoff(chunks: list[dict], floor: int = 3, ceiling: int = 10) -> list[dict]:
        """Select chunks using score drop-off: keep chunks before the largest gap."""
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

        kept = chunks[:cut_at]
        logger.info("  Adaptive cutoff: %d → %d chunks (max drop=%.4f at position %d)",
                     len(chunks), len(kept), max_drop, cut_at)
        return kept

    async def rerank_node(state: QAState) -> dict:
        """
        Rerank retrieved chunks using Voyage's rerank-2.5 model.

        Takes up to 20 candidates from retrieval, reranks to top 15,
        then applies adaptive cutoff (floor=3, ceiling=10) based on
        the largest score drop-off between consecutive chunks.
        """
        chunks = state.get("chunks", [])
        if len(chunks) <= 3:
            logger.info("── RERANK (skipped, %d chunks ≤ 3) %s", len(chunks), LOG_SEPARATOR[35:])
            return {}

        logger.info("── RERANK %s", LOG_SEPARATOR[10:])
        logger.info("  Input: %d chunks → reranking to top 15", len(chunks))

        query = state["question"]
        if state.get("selected_text"):
            query += f"\n\nReferring to: {state['selected_text']}"

        reranking = voyage_client.rerank(
            query=query,
            documents=[c["text"] for c in chunks],
            model=config.VOYAGE_RERANK_MODEL,
            top_k=15,
        )

        reranked = [
            {**chunks[r.index], "score": r.relevance_score}
            for r in reranking.results
        ]

        reranked = _adaptive_cutoff(reranked)

        logger.info("  Output: %d chunks", len(reranked))
        for i, c in enumerate(reranked):
            logger.info(
                "    [%d] score=%.4f | chunk %d | %s…",
                i + 1, c["score"], c["chunk_index"], c["text"][:80],
            )
        logger.info(LOG_SEPARATOR)

        return {"chunks": reranked}

    # --- Node: generate ---

    async def generate_node(state: QAState) -> dict:
        """Generate answer with type-specific system prompt."""
        q_type = state.get("question_type", "context")
        chunks = state.get("chunks", [])

        logger.info("── GENERATE %s", LOG_SEPARATOR[11:])
        logger.info("  Type:   %s", q_type)
        logger.info("  Model:  %s", config.CLAUDE_MODEL)
        logger.info("  Chunks: %d", len(chunks))

        prepared = prepare_generate(state)
        system_prompt = prepared["system_prompt"]
        user_prompt = prepared["user_prompt"]

        if not chunks:
            logger.info("  No chunks — generating from selected text only")

        logger.info("  System prompt (%d chars): %s…", len(system_prompt), system_prompt[:150])
        logger.info("  User prompt (%d chars):   %s…", len(user_prompt), user_prompt[:150])

        response = anthropic.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        answer = response.content[0].text if response.content else "Unable to generate response"

        logger.info("  Tokens: %d in / %d out", response.usage.input_tokens, response.usage.output_tokens)
        logger.info("  Stop:   %s", response.stop_reason)
        logger.info("  Answer (%d chars): %s…", len(answer), answer[:200])
        logger.info(LOG_SEPARATOR)

        return {
            "answer": answer,
            "sources": prepared["sources"],
            "gen_model": response.model,
            "gen_tokens_in": response.usage.input_tokens,
            "gen_tokens_out": response.usage.output_tokens,
            "gen_stop_reason": response.stop_reason,
            "gen_system_prompt": system_prompt,
            "gen_user_prompt": user_prompt,
        }

    # --- Node: log ---

    async def log_node(state: QAState) -> dict:
        """Log query to JSONL and human-readable log files."""
        log_query(state)
        return {}

    # --- Routing ---

    def route_by_type(state: QAState) -> str:
        return state.get("question_type", "context")

    # --- Streaming helpers ---

    async def run_pre_generate(params):
        """Run pre-generation pipeline, yielding stage names then final state."""
        state = dict(params)

        yield "classifying"
        result = await classify_node(state)
        state.update(result)

        yield "retrieving"
        q_type = state.get("question_type", "context")
        retrieval_fns = {
            "vocabulary": vocabulary_node,
            "context": context_node,
            "lookup": lookup_node,
            "analysis": analysis_node,
        }
        result = await retrieval_fns.get(q_type, context_node)(state)
        state.update(result)

        yield "reranking"
        result = await rerank_node(state)
        state.update(result)

        yield state

    # --- Build graph ---

    graph = StateGraph(QAState)

    graph.add_node("classify", classify_node)
    graph.add_node("vocabulary", vocabulary_node)
    graph.add_node("context", context_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges("classify", route_by_type, {
        "vocabulary": "vocabulary",
        "context": "context",
        "lookup": "lookup",
        "analysis": "analysis",
    })

    for node in ["vocabulary", "context", "lookup", "analysis"]:
        graph.add_edge(node, "rerank")

    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", "log")
    graph.add_edge("log", END)

    return {
        "graph": graph.compile(),
        "run_pre_generate": run_pre_generate,
    }
