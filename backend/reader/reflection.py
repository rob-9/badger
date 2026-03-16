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
from reader.mind import TokenUsage, _usage
from reader.prompts import REFLECT_PROMPT

logger = logging.getLogger(__name__)


def _to_bool(value, default: bool = False) -> bool:
    """Coerce an LLM-produced value to bool safely."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1")
    return bool(value) if value is not None else default


def _to_optional_str(value) -> str | None:
    """Coerce to str or None, stripping empty strings."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


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
) -> tuple[Reflection, TokenUsage]:
    """Evaluate a RAG response from the reader's perspective. Returns (reflection, usage)."""
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

    usage = _usage(result, "reflect")

    if not result.content or not hasattr(result.content[0], "text"):
        logger.warning("  Unexpected reflection API response format")
        return _default_reflection(), usage

    raw = result.content[0].text.strip()
    logger.info("  Reflect: %d chars, %d/%d tokens",
                len(raw), usage.input_tokens, usage.output_tokens)

    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("  Failed to parse reflection JSON: %s", raw[:200])
        return _default_reflection(), usage

    if not isinstance(parsed, dict):
        logger.warning("  Reflection is not a dict: %s", type(parsed).__name__)
        return _default_reflection(), usage

    return Reflection(
        satisfactory=_to_bool(parsed.get("satisfactory"), default=True),
        contradicts_model=_to_bool(parsed.get("contradicts_model")),
        reveals_new_info=_to_bool(parsed.get("reveals_new_info")),
        possible_spoiler=_to_bool(parsed.get("possible_spoiler")),
        follow_up=_to_optional_str(parsed.get("follow_up")),
        follow_up_reason=_to_optional_str(parsed.get("follow_up_reason")),
        mind_update=_to_optional_str(parsed.get("mind_update")),
    ), usage


def _default_reflection() -> Reflection:
    return Reflection(
        satisfactory=True,
        contradicts_model=False,
        reveals_new_info=False,
        possible_spoiler=False,
        follow_up=None,
        follow_up_reason=None,
        mind_update=None,
    )
