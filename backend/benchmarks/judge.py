"""
LLM-as-judge scoring for RAG benchmark responses.

Calls Haiku to score each response on four dimensions (0-3 scale):
  Relevance, Conciseness, Accuracy, Spoiler Safety.
"""

import json
import logging
import re

from anthropic import Anthropic

from boom import config

logger = logging.getLogger(__name__)

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

**Spoiler safety** — Does it respect the reader's position?
- 3: Reveals nothing from AHEAD/COMING UP content; safe for reader at this position
- 2: Minor allusion to future content but nothing plot-critical
- 1: Reveals a future event or outcome the reader hasn't reached
- 0: Major spoiler — reveals deaths, twists, or key plot points from later in the book

Return JSON only, no markdown fences:
{{"relevance": 0-3, "conciseness": 0-3, "accuracy": 0-3, "spoiler_safety": 0-3, "notes": "one sentence overall assessment", "relevance_note": "why this score", "accuracy_note": "cite specific fabrications or errors if any", "spoiler_note": "what future content was revealed, if any"}}
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
    # Truncate chunks for the judge prompt
    chunks_text = "\n\n".join(
        f"[Chunk {i+1}, label={c.get('label', '?')}] {c['text'][:500]}..."
        if len(c.get("text", "")) > 500
        else f"[Chunk {i+1}, label={c.get('label', '?')}] {c['text']}"
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
        max_tokens=500,
        system="You are a strict evaluator. Return JSON only.",
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

    return {
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
