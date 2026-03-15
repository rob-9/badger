"""
LLM-as-judge scoring for RAG benchmark responses.

Calls Haiku to score each response on four dimensions (0-3 scale):
  Relevance, Conciseness, Accuracy, Spoiler Safety.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from anthropic import Anthropic

from badger import config

logger = logging.getLogger(__name__)

# --- Judge cache ---
JUDGE_CACHE_PATH = Path(".data/benchmarks/judge_cache.json")
_judge_cache: dict | None = None
_cache_enabled: bool = True
_judge_cache_dirty: bool = False


def set_cache_enabled(enabled: bool):
    """Enable or disable the judge response cache."""
    global _cache_enabled
    _cache_enabled = enabled


def _load_judge_cache() -> dict:
    """Load the judge cache from disk (lazily, on first access)."""
    global _judge_cache
    if _judge_cache is not None:
        return _judge_cache
    if JUDGE_CACHE_PATH.exists():
        try:
            _judge_cache = json.loads(JUDGE_CACHE_PATH.read_text())
            logger.info("Loaded judge cache: %d entries", len(_judge_cache))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load judge cache, starting fresh")
            _judge_cache = {}
    else:
        _judge_cache = {}
    return _judge_cache


def _save_judge_cache():
    """Persist the judge cache to disk if it has been modified."""
    global _judge_cache_dirty
    if _judge_cache is None or not _judge_cache_dirty:
        return
    os.makedirs(JUDGE_CACHE_PATH.parent, exist_ok=True)
    JUDGE_CACHE_PATH.write_text(json.dumps(_judge_cache, indent=2))
    _judge_cache_dirty = False


def flush_judge_cache():
    """Flush the judge cache to disk. Call at end of a benchmark run."""
    _save_judge_cache()


_RUBRIC_VERSION = "v5"  # Bump when RUBRIC text changes


def _judge_cache_key(case_id: str, response_text: str) -> str:
    """Compute a stable cache key from rubric version + case ID + full response."""
    raw = _RUBRIC_VERSION + "\x00" + case_id + "\x00" + response_text
    return hashlib.sha256(raw.encode()).hexdigest()

RUBRIC = """\
You are evaluating a reading-assistant AI that answers questions about a novel.

## Inputs

**Test case:**
- Question: {question}
- Selected text: "{selected_text}"
- Reader position: {reader_position:.0%} through the book
- Expected gist: {expected_gist}

**Retrieved chunks** (what the AI had access to):
{chunks_text}

**AI response:**
{response}

## Scoring rubric (0-3 for each dimension)

**Relevance** — Does the response answer the question?
- 3: Directly and fully answers the question
- 2: Answers the question but misses a minor aspect
- 1: Partially relevant but doesn't clearly answer
- 0: Off-topic or non-responsive

**Conciseness** — Is it appropriately brief?
- 3: No fluff, no unnecessary plot recaps, well-scoped
- 2: Mostly concise with minor padding
- 1: Noticeably verbose or includes unnecessary context
- 0: Rambling, excessive plot summary, or wall of text

**Accuracy** — Is it consistent with the retrieved chunks and expected gist?
- 3: Factually correct and aligned with expected gist
- 2: Mostly accurate with a minor inaccuracy or omission
- 1: Contains a significant factual error or contradiction
- 0: Largely inaccurate or fabricated
When the response quotes text and cites a source, verify the quote actually appears in that chunk before flagging it as fabricated. Paraphrases and minor wording variations are acceptable.

**Spoiler safety** — Does the AI's RESPONSE text reveal things the reader shouldn't know yet?
- 3: Safe — answers from the reader's current perspective without revealing future information
- 2: Minor allusion to future content but nothing plot-critical
- 1: Introduces doubt or suspicion not warranted at the reader's position; or alludes to a future event
- 0: Major spoiler — explicitly reveals deaths, twists, or key plot points from later in the book
IMPORTANT: Score ONLY the AI's response text. Do NOT penalize for what the retrieved chunks contain — chunks are provided for your reference only. Chunks labeled PAST are content the reader has ALREADY READ, so referencing PAST content is NEVER a spoiler. Only penalize if the response reveals information the reader has not yet encountered or draws conclusions beyond what the text establishes at the reader's position.

First, write your reasoning for each dimension. Then assign scores based on that reasoning.
Return JSON only, no markdown fences. Notes MUST come before scores:
{{"notes": "one sentence overall assessment", "relevance_note": "why this score", "accuracy_note": "cite specific fabrications or errors if any — quote the RESPONSE text, not the chunks", "spoiler_note": "what future content was revealed IN THE RESPONSE, if any — quote the specific response text that spoils", "relevance": 0-3, "conciseness": 0-3, "accuracy": 0-3, "spoiler_safety": 0-3}}
"""


def score_response(
    anthropic: Anthropic,
    case: dict,
    chunks: list[dict],
    response: str,
) -> dict:
    """
    Score a RAG response using Haiku as judge.

    Returns:
        {
            "relevance": int,
            "conciseness": int,
            "accuracy": int,
            "spoiler_safety": int,
            "notes": str,
            "judge_tokens_in": int,
            "judge_tokens_out": int,
        }
    """
    # Check cache first
    case_id = case.get("id", "")
    cache_key = _judge_cache_key(case_id, response)
    if _cache_enabled:
        cache = _load_judge_cache()
        if cache_key in cache:
            logger.info("  Judge cache hit: %s", case_id)
            return cache[cache_key]

    chunks_text = "\n\n".join(
        f"[Chunk {i+1}, label={c.get('label', '?')}] {c['text']}"
        for i, c in enumerate(chunks)
    ) or "(no chunks retrieved)"

    prompt = RUBRIC.format(
        question=case["question"],
        selected_text=case.get("selected_text", ""),
        reader_position=case.get("reader_position", 0),
        expected_gist=case["expected_gist"],
        chunks_text=chunks_text,
        response=response,
    )

    result = anthropic.messages.create(
        model=config.CLAUDE_HAIKU_MODEL,
        max_tokens=800,
        system="You are a strict evaluator. Read the AI response carefully and evaluate ONLY what the response says — do not confuse chunk content with response content. Return JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = result.content[0].text.strip()
    tokens_in = result.usage.input_tokens
    tokens_out = result.usage.output_tokens

    logger.info("  Judge raw: %s", raw[:200])
    logger.info("  Judge tokens: %d in / %d out", tokens_in, tokens_out)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try stripping code fences
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            parsed = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            # Final fallback: extract first {...} block via regex
            match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None

            if parsed is None:
                logger.warning("  Failed to parse judge response: %s", raw)
                parsed = {
                    "relevance": -1,
                    "conciseness": -1,
                    "accuracy": -1,
                    "spoiler_safety": -1,
                    "notes": f"PARSE ERROR: {raw[:100]}",
                }

    result_dict = {
        "relevance": parsed.get("relevance", -1),
        "conciseness": parsed.get("conciseness", -1),
        "accuracy": parsed.get("accuracy", -1),
        "spoiler_safety": parsed.get("spoiler_safety", -1),
        "notes": parsed.get("notes", ""),
        "relevance_note": parsed.get("relevance_note", ""),
        "accuracy_note": parsed.get("accuracy_note", ""),
        "spoiler_note": parsed.get("spoiler_note", ""),
        "judge_tokens_in": tokens_in,
        "judge_tokens_out": tokens_out,
    }

    # Write to cache (skip parse errors — they are transient)
    has_parse_error = any(
        result_dict.get(d) == -1
        for d in ("relevance", "conciseness", "accuracy", "spoiler_safety")
    )
    if _cache_enabled and not has_parse_error:
        global _judge_cache_dirty
        cache = _load_judge_cache()
        cache[cache_key] = result_dict
        _judge_cache_dirty = True
        logger.info("  Judge cache write: %s", case_id)

    return result_dict
