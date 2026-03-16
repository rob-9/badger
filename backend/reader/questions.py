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


@dataclass
class GeneratedQuestion:
    question: str
    selected_text: str             # exact substring from the section
    question_type: str             # vocabulary | context | lookup | analysis
    motivation: str                # why the reader would ask this
    expected_answer: str           # sketch of good answer (for judge)
    triggered_by: str | None       # "theory:3" or "unresolved:1" or None (organic)


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

        questions.append(GeneratedQuestion(
            question=question_text,
            selected_text=selected,
            question_type=qtype,
            motivation=str(q.get("motivation", "")),
            expected_answer=str(q.get("expected_answer", "")),
            triggered_by=q.get("triggered_by"),
        ))

    logger.info("  Generated %d valid questions (of %d returned)", len(questions), len(parsed))
    return questions, usage
