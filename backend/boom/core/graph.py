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
from .prompts import SYSTEM_PROMPTS, EVALUATE_PROMPT
from .vector_store import VectorStore, SearchResult, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

LOG_DIR = Path(".data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES = {"vocabulary", "context", "lookup", "analysis"}
LOG_SEPARATOR = "──────────────────────────────────────────────"

TOKEN_LIMITS = {
    "vocabulary": 256,
    "context": 512,
    "lookup": 512,
    "analysis": 1024,
}


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
    gen_max_tokens: int
    gen_model: str
    gen_tokens_in: int
    gen_tokens_out: int
    gen_stop_reason: str
    gen_system_prompt: str
    gen_user_prompt: str

    # Relevance filtering
    relevance_filtered_count: int
    relevance_threshold_used: float

    # Sanitization
    sanitize_ahead_count: int
    sanitize_tokens_in: int
    sanitize_tokens_out: int

    # Query decomposition / HyDE
    sub_queries: list[str]
    hyde_passage: str

    # Evaluation
    eval_relevance: int
    eval_grounding: int
    eval_flags: list[str]
    eval_low_confidence: bool
    eval_tokens_in: int
    eval_tokens_out: int

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


def parse_decompose_response(raw: str) -> dict:
    """Parse Haiku's decomposition JSON → {"queries": [...], "use_hyde": bool}.

    Returns a safe default if parsing fails or fields are missing.
    """
    if not isinstance(raw, str):
        return {"queries": [], "use_hyde": False}
    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return {"queries": [], "use_hyde": False}

    queries = parsed.get("queries")
    if not isinstance(queries, list):
        queries = []
    # Filter to non-empty strings only
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    use_hyde = parsed.get("use_hyde")
    if not isinstance(use_hyde, bool):
        use_hyde = False

    return {"queries": queries, "use_hyde": use_hyde}


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


def filter_by_relevance(chunks: list[dict], threshold: float) -> list[dict]:
    """Drop chunks below threshold, always keeping the best one."""
    if not chunks:
        return chunks
    best = max(chunks, key=lambda c: c["score"])
    kept = [c for c in chunks if c["score"] >= threshold]
    if not kept:
        kept = [best]
    return kept


def build_context_string(chunks: list[dict]) -> str:
    """Build context string with PAST/AHEAD labels and global [Source N] numbering."""
    past = [c for c in chunks if c["label"] == "PAST"]
    ahead = [c for c in chunks if c["label"] == "AHEAD"]

    # Global sequential numbering: PAST first, then AHEAD
    counter = 1

    def format_group(group: list[dict]) -> str:
        nonlocal counter
        parts = []
        for c in group:
            parts.append(f"[Source {counter}]\n{c['text']}")
            counter += 1
        return "\n\n---\n\n".join(parts)

    sections = []
    if past:
        sections.append(f"[ALREADY READ]\n{format_group(past)}")
    if ahead:
        sections.append(f"[COMING UP]\n{format_group(ahead)}")

    return "\n\n===\n\n".join(sections)


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

    # Order: PAST first, then AHEAD — matches build_context_string numbering
    past = [c for c in chunks if c["label"] == "PAST"]
    ahead = [c for c in chunks if c["label"] == "AHEAD"]
    ordered = past + ahead

    sources = [
        {
            "text": c["text"][:200] + "...",
            "full_text": c["text"],
            "score": c["score"],
            "chunk_index": c["chunk_index"],
            "source_number": i + 1,
            "label": c["label"],
        }
        for i, c in enumerate(ordered)
    ]

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "sources": sources,
        "max_tokens": TOKEN_LIMITS.get(q_type, 1024),
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
        # Relevance filtering
        "relevance_filtered_count": state.get("relevance_filtered_count", 0),
        "relevance_threshold_used": state.get("relevance_threshold_used"),
        # Sanitization
        "sanitize_ahead_count": state.get("sanitize_ahead_count", 0),
        "sanitize_tokens": {
            "in": state.get("sanitize_tokens_in", 0),
            "out": state.get("sanitize_tokens_out", 0),
        },
        # Query decomposition / HyDE
        "sub_queries": state.get("sub_queries", []),
        "hyde_passage": state.get("hyde_passage", ""),
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
            "max_tokens": state.get("gen_max_tokens"),
            "input_tokens": state.get("gen_tokens_in"),
            "output_tokens": state.get("gen_tokens_out"),
            "stop_reason": state.get("gen_stop_reason"),
        },
        # Evaluation
        "evaluation": {
            "relevance": state.get("eval_relevance"),
            "grounding": state.get("eval_grounding"),
            "flags": state.get("eval_flags", []),
            "low_confidence": state.get("eval_low_confidence", False),
            "input_tokens": state.get("eval_tokens_in", 0),
            "output_tokens": state.get("eval_tokens_out", 0),
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

        # Decomposition / HyDE
        sub_queries = state.get("sub_queries", [])
        hyde_passage = state.get("hyde_passage", "")
        if sub_queries or hyde_passage:
            section("Decomposition")
            if len(sub_queries) > 1:
                f.write(f"\n  Sub-queries ({len(sub_queries)}):\n")
                for i, sq in enumerate(sub_queries):
                    f.write(f"    [{i+1}] {sq}\n")
            elif sub_queries:
                f.write(f"\n  Query (not decomposed): {sub_queries[0]}\n")
            if hyde_passage:
                f.write(f"\n  HyDE passage ({len(hyde_passage)} chars):\n")
                for line in hyde_passage[:500].split("\n"):
                    f.write(f"    {line}\n")

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

        # Relevance filtering
        filtered_count = state.get("relevance_filtered_count", 0)
        if filtered_count:
            section("Relevance Filter")
            f.write(f"\n  Threshold: {state.get('relevance_threshold_used', 0):.2f}\n")
            f.write(f"  Dropped:   {filtered_count} chunks\n")

        # Sanitization
        sanitize_count = state.get("sanitize_ahead_count", 0)
        if sanitize_count:
            section("Sanitization")
            f.write(f"\n  AHEAD chunks dropped: {sanitize_count}\n")

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

        # Evaluation
        eval_rel = state.get("eval_relevance")
        if eval_rel is not None:
            section("Evaluation")
            f.write(f"\n  Model:      {config.CLAUDE_HAIKU_MODEL}\n")
            f.write(f"  Tokens:     {state.get('eval_tokens_in', 0)} in / {state.get('eval_tokens_out', 0)} out\n")
            f.write(f"  Relevance:  {eval_rel}/5\n")
            f.write(f"  Grounding:  {state.get('eval_grounding')}/5\n")
            flags = state.get("eval_flags", [])
            if flags:
                f.write(f"  Flags:      {', '.join(flags)}\n")

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
        "  Done — classify: %d/%d, sanitize: %d/%d, generate: %d/%d, evaluate: %d/%d tokens",
        state.get("classify_tokens_in", 0), state.get("classify_tokens_out", 0),
        state.get("sanitize_tokens_in", 0), state.get("sanitize_tokens_out", 0),
        state.get("gen_tokens_in", 0), state.get("gen_tokens_out", 0),
        state.get("eval_tokens_in", 0), state.get("eval_tokens_out", 0),
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

    # --- Helpers: query decomposition + HyDE ---

    def _decompose_query(question: str, selected_text: str | None) -> dict:
        """Ask Haiku whether to decompose the question into sub-queries and/or use HyDE."""
        selected_ctx = f'\nSelected text: "{selected_text[:200]}"' if selected_text else ""
        prompt = (
            f'Question: "{question}"{selected_ctx}\n\n'
            "Should this question be broken into 2-4 simpler sub-queries for better retrieval? "
            "Is it abstract enough that a hypothetical answer passage (HyDE) would help?\n\n"
            'Return JSON: {"queries": ["sub1", ...], "use_hyde": true/false}\n'
            "If the question is already simple/specific, return the original question as the only item in queries."
        )
        try:
            response = anthropic.messages.create(
                model=config.CLAUDE_HAIKU_MODEL,
                max_tokens=300,
                system="You decompose reading comprehension questions for retrieval. Return JSON only, no markdown.",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            logger.info("  Decompose raw: %s", raw)
            logger.info("  Decompose tokens: %d in / %d out", response.usage.input_tokens, response.usage.output_tokens)
            result = parse_decompose_response(raw)
        except Exception as e:
            logger.warning("  Decompose failed: %s — using original question", e)
            result = {"queries": [], "use_hyde": False}

        # Fallback: if no valid queries returned, use the original question
        if not result["queries"]:
            result["queries"] = [question]

        return result

    def _generate_hyde(question: str, selected_text: str | None) -> str:
        """Generate a hypothetical answer passage (~100 tokens) via Haiku."""
        selected_ctx = f'\nThe reader highlighted: "{selected_text[:200]}"' if selected_text else ""
        prompt = (
            f"Question: {question}{selected_ctx}\n\n"
            "Write a ~100-word passage that would answer this question, "
            "as if it were an excerpt from a book. "
            "Do not say 'the book says' — write as if you ARE the book text."
        )
        try:
            response = anthropic.messages.create(
                model=config.CLAUDE_HAIKU_MODEL,
                max_tokens=200,
                system="You write hypothetical book passages for retrieval augmentation. Write naturally, as if from the source text.",
                messages=[{"role": "user", "content": prompt}],
            )
            passage = response.content[0].text.strip()
            logger.info("  HyDE passage (%d chars): %s…", len(passage), passage[:120])
            logger.info("  HyDE tokens: %d in / %d out", response.usage.input_tokens, response.usage.output_tokens)
            return passage
        except Exception as e:
            logger.warning("  HyDE generation failed: %s", e)
            return ""

    def _embed_queries(texts: list[str]) -> list[list[float]]:
        """Batch-embed multiple query texts in a single Voyage API call."""
        result = voyage_client.contextualized_embed(
            inputs=[[t] for t in texts],
            model=config.VOYAGE_CONTEXT_MODEL,
            input_type="query",
        )
        return [r.embeddings[0] for r in result.results]

    def _embed_document(text: str) -> list[float]:
        """Embed a HyDE passage as a document (matching the indexed chunk space)."""
        return voyage_client.contextualized_embed(
            inputs=[[text]],
            model=config.VOYAGE_CONTEXT_MODEL,
            input_type="document",
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
            f"Classify as exactly one of:\n"
            f"- vocabulary: The selected text IS the word/phrase to define\n"
            f"- context: Reader wants to understand a specific passage they selected\n"
            f"- lookup: Reader wants factual info from elsewhere in the book\n"
            f"- analysis: Reader wants literary interpretation or thematic discussion\n\n"
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
        """Hybrid search for vocabulary — BM25 catches exact terms, semantic finds related context."""
        selected = state.get("selected_text") or ""
        book_id = state["book_id"]
        query = selected if selected else state["question"]

        logger.info("── RETRIEVE (vocabulary) ─────────────────────")
        logger.info('  Strategy: hybrid search for "%s", top_k=20', query[:100])

        embedding = _embed_query(query)
        logger.info("  Embedding: %d dimensions", len(embedding))

        results = await vector_store.hybrid_search(
            book_id, embedding, query_text=query, top_k=20,
        )
        labeled = _label_and_log(results, state)

        return {
            "chunks": labeled,
            "retrieval_strategy": "hybrid_vocabulary",
            "retrieval_query": query,
            "retrieval_top_k": 20,
            "retrieval_embedding_dims": len(embedding),
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

    # --- Node: lookup retrieval (with decomposition + optional HyDE) ---

    async def lookup_node(state: QAState) -> dict:
        """Hybrid retrieval with query decomposition and optional HyDE."""
        logger.info("── RETRIEVE (lookup) ─────────────────────────")

        question = state["question"]
        selected = state.get("selected_text")
        book_id = state["book_id"]

        # Decompose the query
        decomp = _decompose_query(question, selected)
        sub_queries = decomp["queries"]
        use_hyde = decomp["use_hyde"]
        logger.info("  Sub-queries: %s", sub_queries)
        logger.info("  Use HyDE: %s", use_hyde)

        # Build full query strings (with entities/selected context)
        entities = state.get("entities", [])
        full_queries = [build_query(sq, selected, entities) for sq in sub_queries]

        # Generate HyDE passage if recommended
        hyde_passage = ""
        if use_hyde:
            hyde_passage = _generate_hyde(question, selected)

        if len(full_queries) == 1 and not hyde_passage:
            # Simple case: single query, no HyDE — hybrid search
            logger.info("  Strategy: hybrid search (single query), top_k=20")
            embedding = _embed_query(full_queries[0])
            logger.info("  Embedding: %d dimensions", len(embedding))
            results = await vector_store.hybrid_search(
                book_id, embedding, query_text=full_queries[0], top_k=20,
            )
            labeled = _label_and_log(results, state)
            return {
                "chunks": labeled,
                "retrieval_strategy": "hybrid",
                "retrieval_query": full_queries[0],
                "retrieval_top_k": 20,
                "retrieval_embedding_dims": len(embedding),
                "sub_queries": sub_queries,
                "hyde_passage": hyde_passage,
            }

        # Multi-query and/or HyDE: embed all, search independently, fuse
        query_embeddings = _embed_queries(full_queries)
        logger.info("  Embedded %d queries (%d dimensions each)", len(query_embeddings), len(query_embeddings[0]))

        result_lists = []
        for i, (q, emb) in enumerate(zip(full_queries, query_embeddings)):
            results = await vector_store.hybrid_search(
                book_id, emb, query_text=q, top_k=20,
            )
            logger.info("  Sub-query %d: %d results — %s", i + 1, len(results), q[:80])
            result_lists.append(results)

        if hyde_passage:
            hyde_emb = _embed_document(hyde_passage)
            logger.info("  HyDE embedded as document (%d dimensions)", len(hyde_emb))
            hyde_results = await vector_store.search(book_id, hyde_emb, top_k=20)
            logger.info("  HyDE: %d results", len(hyde_results))
            result_lists.append(hyde_results)

        fused = reciprocal_rank_fusion(*result_lists)
        strategy = "decomposed" if len(sub_queries) > 1 else "semantic"
        if hyde_passage:
            strategy += "+hyde"
        logger.info("  Strategy: %s → %d fused results", strategy, len(fused))

        labeled = _label_and_log(fused, state)
        return {
            "chunks": labeled,
            "retrieval_strategy": strategy,
            "retrieval_query": "\n---\n".join(full_queries),
            "retrieval_top_k": 20,
            "retrieval_embedding_dims": len(query_embeddings[0]),
            "sub_queries": sub_queries,
            "hyde_passage": hyde_passage,
        }

    # Analysis uses decomposition + hybrid search + chapter summaries.
    # Always decomposes; Haiku decides whether HyDE is useful.
    async def analysis_node(state: QAState) -> dict:
        logger.info("── RETRIEVE (analysis) ───────────────────────")
        logger.info("  Strategy: decompose + hyde + hybrid + summaries")

        question = state["question"]
        selected = state.get("selected_text")
        book_id = state["book_id"]
        entities = state.get("entities", [])

        # Always decompose for analysis; Haiku decides whether HyDE helps
        decomp = _decompose_query(question, selected)
        sub_queries = decomp["queries"]
        use_hyde = decomp["use_hyde"]
        logger.info("  Sub-queries: %s", sub_queries)
        logger.info("  Use HyDE: %s", use_hyde)

        hyde_passage = ""
        if use_hyde:
            hyde_passage = _generate_hyde(question, selected)

        # Build full query strings
        full_queries = [build_query(sq, selected, entities) for sq in sub_queries]

        # Embed sub-queries as queries
        query_embeddings = _embed_queries(full_queries)
        logger.info("  Embedded %d queries (%d dimensions each)", len(query_embeddings), len(query_embeddings[0]))

        result_lists = []

        # For each sub-query: hybrid search (semantic + BM25), top 20
        for i, (q, emb) in enumerate(zip(full_queries, query_embeddings)):
            hybrid_results = await vector_store.hybrid_search(
                book_id, emb, query_text=q, top_k=20,
            )
            logger.info("  Sub-query %d hybrid: %d results — %s", i + 1, len(hybrid_results), q[:80])
            result_lists.append(hybrid_results)

        # HyDE passage: embed as document, semantic search, top 20
        if hyde_passage:
            hyde_emb = _embed_document(hyde_passage)
            logger.info("  HyDE embedded as document (%d dimensions)", len(hyde_emb))
            hyde_results = await vector_store.search(book_id, hyde_emb, top_k=20)
            logger.info("  HyDE semantic: %d results", len(hyde_results))
            result_lists.append(hyde_results)
        else:
            logger.info("  HyDE skipped (generation returned empty)")

        # Chapter summaries: semantic search using first query embedding, top 3
        summary_results = await vector_store.search_summaries(
            book_id, query_embeddings[0], top_k=3,
        )
        logger.info("  Summary results: %d", len(summary_results))
        if summary_results:
            result_lists.append(summary_results)

        # Fuse all result lists
        fused = reciprocal_rank_fusion(*result_lists)
        logger.info("  Fused: %d results from %d lists", len(fused), len(result_lists))

        labeled = _label_and_log(fused, state)

        return {
            "chunks": labeled,
            "retrieval_strategy": "decomposed+hyde+hybrid+summaries",
            "retrieval_query": "\n---\n".join(full_queries),
            "retrieval_top_k": 20,
            "retrieval_embedding_dims": len(query_embeddings[0]),
            "sub_queries": sub_queries,
            "hyde_passage": hyde_passage,
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

        When RERANK_ENABLED is false, skips the Voyage API call but
        still applies adaptive cutoff on existing scores.
        """
        chunks = state.get("chunks", [])
        if len(chunks) <= 3:
            logger.info("── RERANK (skipped, %d chunks ≤ 3) %s", len(chunks), LOG_SEPARATOR[35:])
            return {}

        if not config.RERANK_ENABLED:
            logger.info("── RERANK (disabled, applying adaptive cutoff only) %s", LOG_SEPARATOR[50:])
            for c in chunks:
                if c["label"] == "AHEAD":
                    c["score"] *= 0.3
            chunks.sort(key=lambda c: c["score"], reverse=True)
            cutoff = _adaptive_cutoff(chunks)
            logger.info("  Output: %d chunks", len(cutoff))
            return {"chunks": cutoff}

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

        # Penalize AHEAD chunks to reduce spoiler exposure
        for c in reranked:
            if c["label"] == "AHEAD":
                c["score"] *= 0.3
        reranked.sort(key=lambda c: c["score"], reverse=True)

        reranked = _adaptive_cutoff(reranked)

        logger.info("  Output: %d chunks", len(reranked))
        for i, c in enumerate(reranked):
            logger.info(
                "    [%d] score=%.4f | chunk %d | %s…",
                i + 1, c["score"], c["chunk_index"], c["text"][:80],
            )
        logger.info(LOG_SEPARATOR)

        return {"chunks": reranked}

    # --- Node: relevance filter ---

    async def relevance_filter_node(state: QAState) -> dict:
        """Drop chunks below relevance threshold, always keeping the best one."""
        chunks = state.get("chunks", [])
        threshold = config.RELEVANCE_THRESHOLD
        before = len(chunks)
        filtered = filter_by_relevance(chunks, threshold)
        dropped = before - len(filtered)

        if dropped:
            logger.info("── FILTER %s", LOG_SEPARATOR[10:])
            logger.info("  Threshold: %.2f — dropped %d/%d chunks", threshold, dropped, before)
            for c in filtered:
                logger.info("    kept score=%.4f | chunk %d", c["score"], c["chunk_index"])
            logger.info(LOG_SEPARATOR)
        else:
            logger.info("── FILTER (all %d chunks above %.2f) %s", before, threshold, LOG_SEPARATOR[40:])

        return {
            "chunks": filtered,
            "relevance_filtered_count": dropped,
            "relevance_threshold_used": threshold,
        }

    # --- Node: sanitize ---

    async def sanitize_node(state: QAState) -> dict:
        """Drop AHEAD chunks to prevent spoiler leakage."""
        chunks = state.get("chunks", [])
        ahead = [c for c in chunks if c["label"] == "AHEAD"]

        if not ahead:
            logger.info("── SANITIZE (skipped, no AHEAD chunks) %s", LOG_SEPARATOR[40:])
            return {}

        logger.info("── SANITIZE %s", LOG_SEPARATOR[12:])
        logger.info("  Dropping %d AHEAD chunks (spoiler prevention)", len(ahead))

        past_only = [c for c in chunks if c["label"] == "PAST"]

        logger.info("  Result: %d PAST chunks kept, %d AHEAD dropped", len(past_only), len(ahead))
        logger.info(LOG_SEPARATOR)

        return {
            "chunks": past_only,
            "sanitize_ahead_count": len(ahead),
            "sanitize_tokens_in": 0,
            "sanitize_tokens_out": 0,
        }

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
        max_tokens = prepared["max_tokens"]

        if not chunks:
            logger.info("  No chunks — generating from selected text only")

        logger.info("  Max tokens: %d", max_tokens)
        logger.info("  System prompt (%d chars): %s…", len(system_prompt), system_prompt[:150])
        logger.info("  User prompt (%d chars):   %s…", len(user_prompt), user_prompt[:150])

        response = anthropic.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
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
            "gen_max_tokens": max_tokens,
            "gen_model": response.model,
            "gen_tokens_in": response.usage.input_tokens,
            "gen_tokens_out": response.usage.output_tokens,
            "gen_stop_reason": response.stop_reason,
            "gen_system_prompt": system_prompt,
            "gen_user_prompt": user_prompt,
        }

    # --- Node: evaluate ---

    async def evaluate_node(state: QAState) -> dict:
        """Score answer quality with Haiku."""
        answer = state.get("answer", "")
        question = state.get("question", "")
        chunks = state.get("chunks", [])

        if not answer:
            return {}

        logger.info("── EVALUATE %s", LOG_SEPARATOR[12:])

        context_preview = "\n".join(c["text"][:200] for c in chunks[:3])
        user_prompt = (
            f"Question: {question}\n\n"
            f"Sources (first 3):\n{context_preview}\n\n"
            f"Answer: {answer}"
        )

        try:
            response = anthropic.messages.create(
                model=config.CLAUDE_HAIKU_MODEL,
                max_tokens=100,
                system=EVALUATE_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text.strip()
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens

            parsed = json.loads(strip_code_fences(raw))
            relevance = max(1, min(5, int(parsed.get("relevance", 3))))
            grounding = max(1, min(5, int(parsed.get("grounding", 3))))

            flags = []
            if relevance <= 2:
                flags.append("low_relevance")
            if grounding <= 2:
                flags.append("low_grounding")

            low_confidence = bool(flags)

            logger.info("  Model:      %s", config.CLAUDE_HAIKU_MODEL)
            logger.info("  Tokens:     %d in / %d out", tokens_in, tokens_out)
            logger.info("  Relevance:  %d/5", relevance)
            logger.info("  Grounding:  %d/5", grounding)
            if flags:
                logger.warning("  Flags: %s", ", ".join(flags))
            logger.info(LOG_SEPARATOR)

            return {
                "eval_relevance": relevance,
                "eval_grounding": grounding,
                "eval_flags": flags,
                "eval_low_confidence": low_confidence,
                "eval_tokens_in": tokens_in,
                "eval_tokens_out": tokens_out,
            }

        except Exception as e:
            logger.warning("  Evaluation failed: %s", e)
            logger.info(LOG_SEPARATOR)
            return {
                "eval_relevance": 0,
                "eval_grounding": 0,
                "eval_flags": ["eval_error"],
                "eval_low_confidence": True,
                "eval_tokens_in": 0,
                "eval_tokens_out": 0,
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

        yield "filtering"
        result = await relevance_filter_node(state)
        state.update(result)

        yield "sanitizing"
        result = await sanitize_node(state)
        state.update(result)

        yield state

    # --- Build graph ---

    graph = StateGraph(QAState)

    graph.add_node("classify", classify_node)
    graph.add_node("vocabulary", vocabulary_node)
    graph.add_node("context", context_node)
    graph.add_node("lookup", lookup_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("relevance_filter", relevance_filter_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
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

    graph.add_edge("rerank", "relevance_filter")
    graph.add_edge("relevance_filter", "sanitize")
    graph.add_edge("sanitize", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", "log")
    graph.add_edge("log", END)

    return {
        "graph": graph.compile(),
        "run_pre_generate": run_pre_generate,
        "evaluate": evaluate_node,
    }
