"""
Question Generation: grounded in the reader's mental model.

Generates questions that a simulated reader would naturally ask at
each stop point, with type distribution evolving based on position.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from badger.core.graph import strip_code_fences
from reader.mind import TokenUsage, _usage
from reader.prompts import QUESTION_GEN_PROMPT

logger = logging.getLogger(__name__)


def _normalize_ws(text: str) -> str:
    """Collapse whitespace for substring comparison."""
    return re.sub(r'\s+', ' ', text).strip()


def _repair_selected_text(question: str, recent_text: str) -> str:
    """Attempt to extract a ~120-char anchor window from recent_text for a question.

    Tries candidates in priority order:
      1. Quoted phrases (any quote marks including smart quotes)
      2. Words with non-ASCII characters (diacritics, CJK) — foreign terms
      3. Multi-word capitalized sequences (proper names)
      4. Single capitalized words that aren't sentence-initial

    For the first candidate found in recent_text, returns a ~120-char window
    centered on the match snapped to word boundaries. Returns empty string if
    no candidate matches.
    """
    candidates: list[str] = []

    # 1. Quoted phrases
    for m in re.finditer(r'[\u201c\u2018\u0022\u0027]([^\u201d\u2019\u0022\u0027]{2,60})[\u201d\u2019\u0022\u0027]', question):
        candidates.append(m.group(1).strip())

    # 2. Words with non-ASCII characters
    for m in re.finditer(r'\b\w*[^\x00-\x7F]\w*\b', question):
        candidates.append(m.group(0))

    # 3. Multi-word capitalized sequences (2+ consecutive title-case words)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question):
        candidates.append(m.group(1))

    # 4. Single capitalized words that aren't sentence-initial (preceded by non-period)
    for m in re.finditer(r'(?<=[a-z,;:]\s)([A-Z][a-z]{2,})\b', question):
        candidates.append(m.group(1))

    for candidate in candidates:
        # Case-sensitive search first, then case-insensitive fallback
        idx = recent_text.find(candidate)
        if idx == -1:
            lower_text = recent_text.lower()
            lower_candidate = candidate.lower()
            idx = lower_text.find(lower_candidate)
        if idx == -1:
            continue

        # Build ~120-char window centered on match, snapped to word boundaries
        half = 60
        start = max(0, idx - half)
        end = min(len(recent_text), idx + len(candidate) + half)

        # Snap start forward to next word boundary
        if start > 0:
            space = recent_text.find(' ', start)
            if space != -1 and space < idx:
                start = space + 1

        # Snap end back to previous word boundary
        if end < len(recent_text):
            space = recent_text.rfind(' ', idx + len(candidate), end)
            if space != -1:
                end = space

        return recent_text[start:end].strip()

    return ""


@dataclass
class GeneratedQuestion:
    question: str
    selected_text: str             # exact substring from the section
    question_type: str             # vocabulary | context | lookup | analysis
    motivation: str                # why the reader would ask this
    expected_answer: str           # sketch of good answer (for judge)
    triggered_by: str | None       # "theory:3" or "unresolved:1" or None (organic)
    answerable_by_retrieval: bool = True  # false → direct answer (interpretation/analysis)


def _type_guidance(position: float) -> str:
    """Return question type guidance string based on reader position."""
    if position < 0.2:
        return (
            "- HEAVY on vocabulary and lookup questions (new world, new terms, unfamiliar names)\n"
            "- LIGHT on context questions\n"
            "- MINIMAL analysis (too early for deep thematic questions)"
        )
    elif position < 0.7:
        return (
            "- MIXED types: vocabulary still relevant but decreasing\n"
            "- INCREASING analysis and theory-testing questions\n"
            "- Context questions about relationships and motivations\n"
            "- Try to include at least one question that tests a specific theory from your mental model"
        )
    else:
        return (
            "- HEAVY on analysis, connection-making, and thematic synthesis\n"
            "- Questions about how themes have evolved\n"
            "- Theory-testing and theory-confirming questions\n"
            "- LIGHT on vocabulary (you should know the world by now)"
        )


async def generate_questions(
    client,
    recent_text: str,
    mind,
    journal_context: str,
    position: float,
    label: str,
    max_questions: int,
    model: str,
) -> tuple[list[GeneratedQuestion], TokenUsage]:
    """Generate reader questions using QUESTION_GEN_PROMPT.

    Validates that selected_text is a verbatim substring of recent_text.
    Returns (questions, usage).
    """
    prompt = QUESTION_GEN_PROMPT.format(
        position=position,
        label=label,
        recent_text=recent_text[:12000],
        mind_context=mind.to_prompt_context(),
        journal_context=journal_context[:4000],
        max_questions=max_questions,
        type_guidance=_type_guidance(position),
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=4096,
        system="You are a reader generating questions. Return only a valid JSON array.",
        messages=[{"role": "user", "content": prompt}],
    )

    usage = _usage(response, "question_gen")

    if not response.content or not hasattr(response.content[0], "text"):
        logger.warning("  Unexpected question gen API response format")
        return [], usage

    raw = response.content[0].text.strip()
    logger.info("  Question gen: %d chars, %d/%d tokens",
                len(raw), usage.input_tokens, usage.output_tokens)

    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("  Failed to parse question gen JSON: %s", raw[:200])
        return [], usage

    if not isinstance(parsed, list):
        logger.warning("  Question gen returned non-list: %s", type(parsed).__name__)
        return [], usage

    # Pre-compute normalized text for whitespace-tolerant validation
    normalized_text = _normalize_ws(recent_text)

    questions: list[GeneratedQuestion] = []
    total_answerable = 0
    has_anchor = 0
    repaired = 0

    for q in parsed[:max_questions]:
        if not isinstance(q, dict):
            continue

        question_text = str(q.get("question", "")).strip()
        selected = str(q.get("selected_text", "")).strip()
        qtype = str(q.get("question_type", "context")).strip()

        if not question_text:
            continue

        # Validate question_type
        if qtype not in ("vocabulary", "context", "lookup", "analysis"):
            qtype = "context"

        # Validate selected_text is a substring (whitespace-tolerant)
        if selected:
            if selected not in recent_text and _normalize_ws(selected) not in normalized_text:
                logger.warning("  Hallucinated selected_text for question: %s", question_text[:60])
                selected = ""

        # Repair missing selected_text for retrieval-anchor types
        if not selected and qtype in ("vocabulary", "lookup", "context"):
            repaired_text = _repair_selected_text(question_text, recent_text)
            if repaired_text:
                selected = repaired_text
                repaired += 1

        # Determine if this question can be answered by passage retrieval
        raw_answerable = q.get("answerable_by_retrieval")
        if isinstance(raw_answerable, bool):
            answerable = raw_answerable
        else:
            answerable = qtype in ("vocabulary", "lookup")

        if answerable:
            total_answerable += 1
            if selected:
                has_anchor += 1

        questions.append(GeneratedQuestion(
            question=question_text,
            selected_text=selected,
            question_type=qtype,
            motivation=str(q.get("motivation", "")),
            expected_answer=str(q.get("expected_answer", "")),
            triggered_by=q.get("triggered_by"),
            answerable_by_retrieval=answerable,
        ))

    logger.info("  Generated %d valid questions (of %d returned)", len(questions), len(parsed))
    logger.info("  selected_text: %d/%d answerable questions have anchors (%d repaired)",
                has_anchor, total_answerable, repaired)
    return questions, usage
