"""
Reader's Mind: theories, character map, and evolving mental model.

Tracks everything the simulated reader believes, suspects, and wonders
about as they progress through the book section by section.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

from badger.core.graph import strip_code_fences
from reader.prompts import REACT_PROMPT, MIND_UPDATE_PROMPT

logger = logging.getLogger(__name__)


def _extract_text(response) -> str:
    """Safely extract text from an Anthropic API response."""
    if not response.content or not hasattr(response.content[0], "text"):
        logger.warning("  Unexpected API response format: no text content")
        return ""
    return response.content[0].text.strip()


@dataclass
class Theory:
    text: str                    # "I think Lovell is hiding something about Robin's mother"
    confidence: str              # "speculation" | "likely" | "confirmed" | "contradicted"
    formed_at: float             # reader_position when theory was formed
    updated_at: float            # last update position
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "formed_at": self.formed_at,
            "updated_at": self.updated_at,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Theory":
        return cls(
            text=d["text"],
            confidence=d.get("confidence", "speculation"),
            formed_at=d.get("formed_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            evidence=d.get("evidence", []),
        )


@dataclass
class Character:
    name: str
    description: str             # brief, evolving description
    relationships: list[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "relationships": self.relationships,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            relationships=d.get("relationships", []),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
        )


@dataclass
class MindUpdate:
    """Structured update produced by the THINK step."""
    new_characters: list[dict] = field(default_factory=list)
    updated_characters: list[dict] = field(default_factory=list)
    new_theories: list[dict] = field(default_factory=list)
    theory_updates: list[dict] = field(default_factory=list)
    new_unresolved: list[str] = field(default_factory=list)
    resolved: list[int] = field(default_factory=list)
    new_themes: list[str] = field(default_factory=list)
    new_surprises: list[str] = field(default_factory=list)
    emotional_state: str = ""
    events_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "new_characters": self.new_characters,
            "updated_characters": self.updated_characters,
            "new_theories": self.new_theories,
            "theory_updates": self.theory_updates,
            "new_unresolved": self.new_unresolved,
            "resolved": self.resolved,
            "new_themes": self.new_themes,
            "new_surprises": self.new_surprises,
            "emotional_state": self.emotional_state,
            "events_summary": self.events_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MindUpdate":
        if not isinstance(d, dict):
            logger.warning("  MindUpdate.from_dict got %s, expected dict", type(d).__name__)
            return cls(emotional_state="confused", events_summary="(invalid data)")

        def _ensure_list(val):
            return val if isinstance(val, list) else []

        def _ensure_int_list(val):
            if not isinstance(val, list):
                return []
            return [x for x in val if isinstance(x, int) and x >= 0]

        return cls(
            new_characters=_ensure_list(d.get("new_characters")),
            updated_characters=_ensure_list(d.get("updated_characters")),
            new_theories=_ensure_list(d.get("new_theories")),
            theory_updates=_ensure_list(d.get("theory_updates")),
            new_unresolved=_ensure_list(d.get("new_unresolved")),
            resolved=_ensure_int_list(d.get("resolved")),
            new_themes=_ensure_list(d.get("new_themes")),
            new_surprises=_ensure_list(d.get("new_surprises")),
            emotional_state=str(d.get("emotional_state", "")),
            events_summary=str(d.get("events_summary", "")),
        )


@dataclass
class ReaderMind:
    characters: dict[str, Character] = field(default_factory=dict)
    theories: list[Theory] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    emotional_state: str = "curious"
    surprises: list[str] = field(default_factory=list)

    def to_prompt_context(self, max_tokens: int = 3000) -> str:
        """Format mental model for inclusion in prompts.

        Prioritizes: active theories, recent characters, unresolved questions.
        Truncates if the rendered text exceeds max_tokens (~4 chars/token).
        """
        max_chars = max_tokens * 4
        parts: list[str] = []

        # Emotional state
        parts.append(f"Current mood: {self.emotional_state}")

        # Active theories (non-contradicted first)
        if self.theories:
            parts.append("\nActive theories:")
            for idx, t in enumerate(self.theories):
                if t.confidence == "contradicted":
                    continue
                evidence = "; ".join(t.evidence[-2:]) if t.evidence else "no evidence yet"
                parts.append(f"  [{idx}] ({t.confidence}) {t.text} — {evidence}")

            contradicted = [(i, t) for i, t in enumerate(self.theories) if t.confidence == "contradicted"]
            if contradicted:
                parts.append("\nContradicted theories:")
                for idx, t in contradicted[-3:]:
                    parts.append(f"  [{idx}] {t.text}")

        # Unresolved questions
        if self.unresolved:
            parts.append("\nOpen questions:")
            for i, q in enumerate(self.unresolved):
                parts.append(f"  [{i}] {q}")

        # Characters (sorted by last_seen descending — most recent first)
        if self.characters:
            sorted_chars = sorted(
                self.characters.values(),
                key=lambda c: c.last_seen,
                reverse=True,
            )
            parts.append("\nCharacters:")
            for c in sorted_chars:
                rels = ", ".join(c.relationships) if c.relationships else "no known relationships"
                parts.append(f"  - {c.name}: {c.description} ({rels})")

        # Themes
        if self.themes:
            parts.append(f"\nThemes: {', '.join(self.themes)}")

        # Surprises (last 3)
        if self.surprises:
            parts.append("\nRecent surprises:")
            for s in self.surprises[-3:]:
                parts.append(f"  - {s}")

        text = "\n".join(parts)

        # Truncate if too long (cut from the end, preserving structure)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n  ... (truncated)"

        return text

    def apply_update(self, update: MindUpdate, position: float) -> None:
        """Apply a structured update from the THINK step."""
        # New characters
        for c in update.new_characters:
            name = c.get("name", "").strip()
            if not name:
                continue
            self.characters[name] = Character(
                name=name,
                description=c.get("description", ""),
                relationships=c.get("relationships", []),
                first_seen=c.get("first_seen", position),
                last_seen=position,
            )

        # Updated characters
        for c in update.updated_characters:
            name = c.get("name", "").strip()
            if not name or name not in self.characters:
                continue
            char = self.characters[name]
            if c.get("description_update"):
                char.description += f" {c['description_update']}"
            if c.get("new_relationships"):
                char.relationships.extend(c["new_relationships"])
            char.last_seen = c.get("last_seen", position)

        # New theories
        for t in update.new_theories:
            self.theories.append(Theory(
                text=t.get("text", ""),
                confidence=t.get("confidence", "speculation"),
                formed_at=position,
                updated_at=position,
                evidence=t.get("evidence", []),
            ))

        # Theory updates
        for t in update.theory_updates:
            idx = t.get("index", -1)
            if 0 <= idx < len(self.theories):
                theory = self.theories[idx]
                if t.get("confidence"):
                    theory.confidence = t["confidence"]
                if t.get("evidence_note"):
                    theory.evidence.append(t["evidence_note"])
                theory.updated_at = position

        # Resolved unresolved questions (process in reverse order to keep indices valid)
        for idx in sorted(update.resolved, reverse=True):
            if 0 <= idx < len(self.unresolved):
                self.unresolved.pop(idx)

        # New unresolved
        self.unresolved.extend(update.new_unresolved)

        # Themes
        for theme in update.new_themes:
            if theme not in self.themes:
                self.themes.append(theme)

        # Surprises
        self.surprises.extend(update.new_surprises)

        # Emotional state
        if update.emotional_state:
            self.emotional_state = update.emotional_state

    def to_dict(self) -> dict:
        return {
            "characters": {n: c.to_dict() for n, c in self.characters.items()},
            "theories": [t.to_dict() for t in self.theories],
            "unresolved": self.unresolved,
            "themes": self.themes,
            "emotional_state": self.emotional_state,
            "surprises": self.surprises,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReaderMind":
        return cls(
            characters={n: Character.from_dict(c) for n, c in d.get("characters", {}).items()},
            theories=[Theory.from_dict(t) for t in d.get("theories", [])],
            unresolved=d.get("unresolved", []),
            themes=d.get("themes", []),
            emotional_state=d.get("emotional_state", "curious"),
            surprises=d.get("surprises", []),
        )


async def react_to_section(
    client,
    recent_text: str,
    mind: ReaderMind,
    position: float,
    label: str,
    model: str,
) -> str:
    """Chain-of-thought reaction to a section. Returns raw reasoning text."""
    prompt = REACT_PROMPT.format(
        label=label,
        position=position,
        recent_text=recent_text[:12000],  # cap to avoid token overflow
        mind_context=mind.to_prompt_context(),
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=2048,
        system="You are a thoughtful reader experiencing a book for the first time.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_text(response)
    logger.info("  React: %d chars, %d/%d tokens",
                len(raw), response.usage.input_tokens, response.usage.output_tokens)
    return raw


async def update_mind(
    client,
    recent_text: str,
    reaction: str,
    mind: ReaderMind,
    position: float,
    label: str,
    model: str,
) -> MindUpdate:
    """Structured mental model update. Returns MindUpdate dataclass."""
    prompt = MIND_UPDATE_PROMPT.format(
        label=label,
        position=position,
        recent_text=recent_text[:12000],
        reaction=reaction[:4000],
        mind_context=mind.to_prompt_context(),
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=2048,
        system="You are updating a reader's mental model. Return only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _extract_text(response)
    logger.info("  Mind update: %d chars, %d/%d tokens",
                len(raw), response.usage.input_tokens, response.usage.output_tokens)

    try:
        parsed = json.loads(strip_code_fences(raw))
    except (json.JSONDecodeError, TypeError):
        logger.warning("  Failed to parse mind update JSON, returning empty update")
        parsed = {}

    if not isinstance(parsed, dict):
        logger.warning("  Mind update is not a dict: %s", type(parsed).__name__)
        parsed = {}

    if not parsed.get("events_summary"):
        parsed["events_summary"] = "(no summary)"
    if not parsed.get("emotional_state"):
        parsed["emotional_state"] = "uncertain"

    return MindUpdate.from_dict(parsed)
