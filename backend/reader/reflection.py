"""
Self-Correction: evaluate RAG responses and generate follow-ups.

After each question is answered by the RAG pipeline, the reader
reflects on whether the response was satisfying, contradicts their
mental model, or might contain spoilers.
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from badger.core.graph import strip_code_fences
from reader.prompts import REFLECT_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    satisfactory: bool
    contradicts_model: bool
    reveals_new_info: bool
    possible_spoiler: bool
    follow_up: str | None
    follow_up_reason: str | None
    mind_update: str | None


async def reflect_on_response(
    client,
    question: str,
    response: str,
    mind,
    position: float,
    model: str,
) -> Reflection:
    """Evaluate a RAG response from the reader's perspective."""
    prompt = REFLECT_PROMPT.format(
        position=position,
        question=question,
        response=response[:4000],
        mind_context=mind.to_prompt_context(),
    )

    result = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=1024,
        system="You are a reader reflecting on a response. Return only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = result.content[0].text.strip()
    logger.info("  Reflect: %d chars, %d/%d tokens",
                len(raw), result.usage.input_tokens, result.usage.output_tokens)

    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("  Failed to parse reflection JSON")
        return Reflection(
            satisfactory=True,
            contradicts_model=False,
            reveals_new_info=False,
            possible_spoiler=False,
            follow_up=None,
            follow_up_reason=None,
            mind_update=None,
        )

    return Reflection(
        satisfactory=parsed.get("satisfactory", True),
        contradicts_model=parsed.get("contradicts_model", False),
        reveals_new_info=parsed.get("reveals_new_info", False),
        possible_spoiler=parsed.get("possible_spoiler", False),
        follow_up=parsed.get("follow_up"),
        follow_up_reason=parsed.get("follow_up_reason"),
        mind_update=parsed.get("mind_update"),
    )
