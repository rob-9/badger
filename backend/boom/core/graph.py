"""
LangGraph agent workflow for the RAG pipeline.

Routes questions through type-specific retrieval strategies:
- vocabulary: semantic top_k=5, concise definition
- context: proximity ±3 chunks, passage explanation
- lookup: semantic top_k=5, factual answer
- analysis: broad semantic top_k=8, deeper analysis
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

def label_chunks(results: list[SearchResult], reader_position: float, total_chunks: int) -> list[dict]:
    """Tag chunks as PAST or AHEAD based on reader position."""
    reader_chunk_index = int(reader_position * total_chunks) if total_chunks > 0 else 0
    labeled = []
    for r in results:
        idx = r.chunk.metadata['chunk_index']
        labeled.append({
            "text": r.chunk.text,
            "chunk_index": idx,
            "score": r.score,
            "label": "PAST" if idx <= reader_chunk_index else "AHEAD",
        })
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
    past = [c for c in chunks if c["label"] == "PAST"]
    ahead = [c for c in chunks if c["label"] == "AHEAD"]

    parts = []
    if past:
        past_text = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{c['text']}" for i, c in enumerate(past)
        )
        parts.append(f"[ALREADY READ]\n{past_text}")
    if ahead:
        ahead_text = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{c['text']}" for i, c in enumerate(ahead)
        )
        parts.append(f"[COMING UP - guide only, do not spoil]\n{ahead_text}")

    return "\n\n===\n\n".join(parts)


# === Graph builder ===

def build_qa_graph(anthropic: Anthropic, vector_store: VectorStore, voyage_client) -> StateGraph:
    """Build and compile the QA LangGraph."""

    # --- Nodes ---

    async def classify_node(state: QAState) -> dict:
        """Fast classification with Haiku."""
        selected = state.get("selected_text") or ""
        question = state["question"]
        book_id = state["book_id"]

        logger.info("── CLASSIFY ──────────────────────────────────")
        logger.info("  Book:     %s", book_id)
        logger.info("  Question: %s", question[:100])
        logger.info("  Selected: \"%s%s\"", selected[:80], "…" if len(selected) > 80 else "")
        logger.info("  Model:    claude-haiku-4-5-20251001")

        classify_prompt = f"""Question: "{question}"
Selected text: "{selected[:200]}"

Classify as exactly one of: vocabulary, context, lookup, analysis
Extract key entities (names, terms, concepts).

Return: {{"type": "...", "entities": [...]}}"""

        logger.info("  Prompt:   %s", classify_prompt.replace("\n", " ")[:200])

        response = anthropic.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system="Classify the reading question. Return JSON only, no markdown.",
            messages=[{"role": "user", "content": classify_prompt}],
        )
        raw_text = response.content[0].text.strip()
        logger.info("  Raw response: %s", raw_text)
        logger.info("  Tokens: %d in / %d out", response.usage.input_tokens, response.usage.output_tokens)

        # Strip markdown code fences if present
        text = raw_text
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("  FAILED to parse classification JSON, falling back to 'context': %s", text)
            parsed = {"type": "context", "entities": []}

        q_type = parsed.get("type", "context")
        if q_type not in VALID_TYPES:
            logger.warning("  Invalid type '%s', falling back to 'context'", q_type)
            q_type = "context"

        total = await vector_store.get_total_chunks(book_id)

        logger.info("  Result: type=%s, entities=%s, total_chunks=%d", q_type, parsed.get("entities", []), total)
        logger.info("──────────────────────────────────────────────")

        return {
            "question_type": q_type,
            "entities": parsed.get("entities", []),
            "total_chunks": total,
            "classify_raw_response": raw_text,
            "classify_tokens_in": response.usage.input_tokens,
            "classify_tokens_out": response.usage.output_tokens,
        }

    async def vocabulary_node(state: QAState) -> dict:
        """Keyword search for vocabulary — find chunks that contain the word."""
        selected = state.get("selected_text") or ""
        book_id = state["book_id"]
        logger.info("── RETRIEVE (vocabulary) ─────────────────────")
        logger.info("  Strategy: keyword search for \"%s\"", selected[:100])

        # Keyword search first — find chunks that literally contain the term
        results = await vector_store.keyword_search(book_id, selected)
        strategy = "keyword"
        logger.info("  Keyword matches: %d chunks contain \"%s\"", len(results), selected[:50])

        if not results:
            # Fall back to semantic search if the word isn't found literally
            logger.info("  No keyword matches — falling back to semantic search")
            strategy = "keyword→semantic_fallback"
            query = selected if selected else state["question"]
            embedding = voyage_client.embed(
                texts=[query], model=config.VOYAGE_MODEL, input_type="query"
            ).embeddings[0]
            logger.info("  Semantic query: \"%s\", %d dimensions", query[:100], len(embedding))
            results = await vector_store.search(book_id, embedding, top_k=5)

        # Cap at 5 for keyword results
        results = results[:5]

        labeled = label_chunks(results, state.get("reader_position", 0), state.get("total_chunks", 0))
        reader_chunk = int(state.get("reader_position", 0) * state.get("total_chunks", 0))
        logger.info("  Reader at chunk %d — labeling PAST/AHEAD", reader_chunk)
        logger.info("  Retrieved %d chunks:", len(labeled))
        for i, c in enumerate(labeled):
            logger.info("    [%d] score=%.4f | chunk %d | %s | %s…", i + 1, c["score"], c["chunk_index"], c["label"], c["text"][:80])
        logger.info("──────────────────────────────────────────────")
        return {
            "chunks": labeled,
            "retrieval_strategy": strategy,
            "retrieval_query": selected,
        }

    async def context_node(state: QAState) -> dict:
        """Proximity retrieval ±3 chunks for passage explanation."""
        selected = state.get("selected_text") or ""
        book_id = state["book_id"]
        logger.info("── RETRIEVE (context) ────────────────────────")
        logger.info("  Strategy: proximity ±3 chunks")
        logger.info("  Looking for selected text in chunks: \"%s…\"", selected[:60])
        center = await vector_store.find_chunk_containing(book_id, selected)
        chunk_match = "exact" if (center > 0 or not selected) else "fallback"
        start, end = max(0, center - 3), center + 3
        logger.info("  Chunk match: %s", chunk_match)
        logger.info("  Center chunk: %d, range: [%d, %d]", center, start, end)
        if chunk_match == "fallback" and selected:
            logger.warning("  find_chunk_containing fell back to 0 — selected text not found in any chunk")
        results = await vector_store.get_chunks_by_range(book_id, start, end)
        labeled = label_chunks(results, state.get("reader_position", 0), state.get("total_chunks", 0))
        reader_chunk = int(state.get("reader_position", 0) * state.get("total_chunks", 0))
        logger.info("  Reader at chunk %d — labeling PAST/AHEAD", reader_chunk)
        logger.info("  Retrieved %d chunks:", len(labeled))
        for i, c in enumerate(labeled):
            logger.info("    [%d] chunk %d | %s | score=%.4f | %s…", i + 1, c["chunk_index"], c["label"], c["score"], c["text"][:80])
        logger.info("──────────────────────────────────────────────")
        return {
            "chunks": labeled,
            "retrieval_strategy": "proximity",
            "retrieval_center_chunk": center,
            "retrieval_range_start": start,
            "retrieval_range_end": end,
            "retrieval_chunk_match": chunk_match,
        }

    async def lookup_node(state: QAState) -> dict:
        """Semantic search top_k=5 for factual lookups."""
        logger.info("── RETRIEVE (lookup) ─────────────────────────")
        logger.info("  Strategy: semantic search, top_k=5")
        query = build_query(state["question"], state.get("selected_text"), state.get("entities", []))
        logger.info("  Query: %s", query[:200])
        logger.info("  Voyage model: %s, input_type: query", config.VOYAGE_MODEL)
        embedding = voyage_client.embed(
            texts=[query], model=config.VOYAGE_MODEL, input_type="query"
        ).embeddings[0]
        logger.info("  Embedding: %d dimensions", len(embedding))
        logger.info("  Searching %d total chunks in book %s", state.get("total_chunks", 0), state["book_id"])
        results = await vector_store.search(state["book_id"], embedding, top_k=5)
        labeled = label_chunks(results, state.get("reader_position", 0), state.get("total_chunks", 0))
        reader_chunk = int(state.get("reader_position", 0) * state.get("total_chunks", 0))
        logger.info("  Reader at chunk %d — labeling PAST/AHEAD", reader_chunk)
        logger.info("  Retrieved %d chunks:", len(labeled))
        for i, c in enumerate(labeled):
            logger.info("    [%d] score=%.4f | chunk %d | %s | %s…", i + 1, c["score"], c["chunk_index"], c["label"], c["text"][:80])
        logger.info("──────────────────────────────────────────────")
        return {
            "chunks": labeled,
            "retrieval_strategy": "semantic",
            "retrieval_query": query,
            "retrieval_top_k": 5,
            "retrieval_embedding_dims": len(embedding),
        }

    async def analysis_node(state: QAState) -> dict:
        """Broad semantic search top_k=8 for deeper analysis."""
        logger.info("── RETRIEVE (analysis) ───────────────────────")
        logger.info("  Strategy: broad semantic search, top_k=8")
        query = build_query(state["question"], state.get("selected_text"), state.get("entities", []))
        logger.info("  Query: %s", query[:200])
        logger.info("  Voyage model: %s, input_type: query", config.VOYAGE_MODEL)
        embedding = voyage_client.embed(
            texts=[query], model=config.VOYAGE_MODEL, input_type="query"
        ).embeddings[0]
        logger.info("  Embedding: %d dimensions", len(embedding))
        logger.info("  Searching %d total chunks in book %s", state.get("total_chunks", 0), state["book_id"])
        results = await vector_store.search(state["book_id"], embedding, top_k=8)
        labeled = label_chunks(results, state.get("reader_position", 0), state.get("total_chunks", 0))
        reader_chunk = int(state.get("reader_position", 0) * state.get("total_chunks", 0))
        logger.info("  Reader at chunk %d — labeling PAST/AHEAD", reader_chunk)
        logger.info("  Retrieved %d chunks:", len(labeled))
        for i, c in enumerate(labeled):
            logger.info("    [%d] score=%.4f | chunk %d | %s | %s…", i + 1, c["score"], c["chunk_index"], c["label"], c["text"][:80])
        logger.info("──────────────────────────────────────────────")
        return {
            "chunks": labeled,
            "retrieval_strategy": "broad_semantic",
            "retrieval_query": query,
            "retrieval_top_k": 8,
            "retrieval_embedding_dims": len(embedding),
        }

    async def generate_node(state: QAState) -> dict:
        """Generate answer with type-specific system prompt."""
        q_type = state.get("question_type", "context")
        chunks = state.get("chunks", [])
        selected = state.get("selected_text") or ""

        logger.info("── GENERATE ──────────────────────────────────")
        logger.info("  Type:   %s", q_type)
        logger.info("  Model:  %s", config.CLAUDE_MODEL)
        logger.info("  Chunks: %d", len(chunks))

        system_prompt = SYSTEM_PROMPTS.get(q_type, SYSTEM_PROMPTS["context"])

        if not chunks:
            user_prompt = f'The user selected: "{selected}"\n\nQuestion: {state["question"]}'
            logger.info("  No chunks — generating from selected text only")
        else:
            context = build_context_string(chunks)
            if selected:
                user_prompt = f'The user selected this text: "{selected}"\n\nContext from the book:\n{context}\n\nQuestion: {state["question"]}'
            else:
                user_prompt = f'Context from the book:\n{context}\n\nQuestion: {state["question"]}'

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
        logger.info("──────────────────────────────────────────────")

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
            "answer": answer,
            "sources": sources,
            "gen_model": response.model,
            "gen_tokens_in": response.usage.input_tokens,
            "gen_tokens_out": response.usage.output_tokens,
            "gen_stop_reason": response.stop_reason,
            "gen_system_prompt": system_prompt,
            "gen_user_prompt": user_prompt,
        }

    async def log_node(state: QAState) -> dict:
        """Log query to JSONL and human-readable log files."""
        reader_position = state.get("reader_position", 0)
        total_chunks = state.get("total_chunks", 0)
        reader_chunk_index = int(reader_position * total_chunks) if total_chunks else 0
        chunks = state.get("chunks", [])
        selected = state.get("selected_text") or ""

        logger.info("── LOG ───────────────────────────────────────")
        logger.info("  Writing to queries.jsonl + queries.log")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "book_id": state.get("book_id"),
            "reader_position": reader_position,
            "reader_chunk_index": reader_chunk_index,
            "total_chunks": total_chunks,
            "question": state["question"],
            "selected_text": selected,
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

        # JSONL log
        log_file = LOG_DIR / "queries.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Human-readable log
        readable_log = LOG_DIR / "queries.log"
        with open(readable_log, "a") as f:
            W = 80

            f.write("\n\n" + "═" * W + "\n")
            f.write(f"QUERY @ {log_entry['timestamp']}\n")
            f.write("═" * W + "\n\n")

            f.write(f"Book:     {state.get('book_id')}\n")
            f.write(f"Position: {reader_position:.1%} (chunk {reader_chunk_index}/{total_chunks})\n")
            f.write(f"Selected: \"{selected[:200]}{'…' if len(selected) > 200 else ''}\"\n")
            f.write(f"Question: {state['question']}\n")

            # Classification section
            f.write("\n" + "── Classification " + "─" * (W - 18) + "\n\n")
            f.write(f"  Model:    claude-haiku-4-5-20251001\n")
            f.write(f"  Tokens:   {state.get('classify_tokens_in')} in / {state.get('classify_tokens_out')} out\n")
            f.write(f"  Raw:      {state.get('classify_raw_response')}\n")
            f.write(f"  Type:     {state.get('question_type')}\n")
            f.write(f"  Entities: {', '.join(state.get('entities', [])) or '(none)'}\n")

            # Retrieval section
            strategy = state.get("retrieval_strategy")
            f.write("\n" + "── Retrieval " + "─" * (W - 13) + "\n\n")
            f.write(f"  Strategy: {strategy}\n")
            if strategy == "proximity":
                f.write(f"  Chunk match: {state.get('retrieval_chunk_match', 'unknown')}\n")
                f.write(f"  Center:   chunk {state.get('retrieval_center_chunk')}\n")
                f.write(f"  Range:    [{state.get('retrieval_range_start')}, {state.get('retrieval_range_end')}]\n")
            elif strategy and "keyword" in strategy:
                f.write(f"  Keyword:  \"{state.get('retrieval_query', '')}\"\n")
            else:
                f.write(f"  Query:    {(state.get('retrieval_query') or '')[:200]}\n")
                f.write(f"  Top K:    {state.get('retrieval_top_k')}\n")
                f.write(f"  Embedding: {state.get('retrieval_embedding_dims')} dimensions ({config.VOYAGE_MODEL})\n")

            f.write(f"\n  Results: {len(chunks)} chunks\n")
            for i, c in enumerate(chunks):
                f.write(f"\n    [{i+1}] score={c['score']:.4f} | chunk {c['chunk_index']} | {c['label']}\n")
                # Show full chunk text, with selected text highlighted via >>>markers<<<
                chunk_text = c["text"]
                if selected and selected.lower() in chunk_text.lower():
                    # Find and highlight the match
                    lower_text = chunk_text.lower()
                    lower_sel = selected.lower()
                    idx = lower_text.find(lower_sel)
                    highlighted = (
                        chunk_text[:idx]
                        + ">>>" + chunk_text[idx:idx + len(selected)] + "<<<"
                        + chunk_text[idx + len(selected):]
                    )
                    for line in highlighted.split("\n"):
                        f.write(f"        {line}\n")
                else:
                    for line in chunk_text.split("\n"):
                        f.write(f"        {line}\n")

            # LLM Input section
            sys_prompt = state.get("gen_system_prompt") or ""
            usr_prompt = state.get("gen_user_prompt") or ""
            f.write("\n" + "── LLM Input " + "─" * (W - 13) + "\n\n")
            f.write(f"  System ({len(sys_prompt)} chars):\n")
            for line in sys_prompt.split("\n"):
                f.write(f"    {line}\n")
            f.write(f"\n  User ({len(usr_prompt)} chars):\n")
            usr_display = usr_prompt if len(usr_prompt) <= 500 else usr_prompt[:500] + f"\n    … [{len(usr_prompt) - 500} more chars]"
            for line in usr_display.split("\n"):
                f.write(f"    {line}\n")

            # LLM Output section
            f.write("\n" + "── LLM Output " + "─" * (W - 14) + "\n\n")
            f.write(f"  Model:  {state.get('gen_model')}\n")
            f.write(f"  Tokens: {state.get('gen_tokens_in')} in / {state.get('gen_tokens_out')} out\n")
            f.write(f"  Stop:   {state.get('gen_stop_reason')}\n")
            f.write(f"\n  Answer:\n")
            for line in state.get("answer", "").split("\n"):
                f.write(f"    {line}\n")

            f.write("\n" + "═" * W + "\n\n")

        logger.info("  Done — classify: %d/%d tokens, generate: %d/%d tokens",
                     state.get("classify_tokens_in", 0), state.get("classify_tokens_out", 0),
                     state.get("gen_tokens_in", 0), state.get("gen_tokens_out", 0))
        logger.info("──────────────────────────────────────────────")

        return {}

    # --- Routing ---

    def route_by_type(state: QAState) -> str:
        return state.get("question_type", "context")

    # --- Build graph ---

    graph = StateGraph(QAState)

    graph.add_node("classify", classify_node)
    graph.add_node("vocabulary", vocabulary_node)
    graph.add_node("context", context_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("analysis", analysis_node)
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
        graph.add_edge(node, "generate")

    graph.add_edge("generate", "log")
    graph.add_edge("log", END)

    return graph.compile()
