"""
Reading Journal: section-by-section observations and reactions.

Maintains a log of what the reader experienced at each stop point,
providing context for question generation and the final report.
"""

from dataclasses import dataclass


@dataclass
class JournalEntry:
    position: float
    label: str                     # "Chapter Three" or "15%"
    events: str                    # 2-3 sentence summary
    reaction: str                  # the raw chain-of-thought from REACT step
    mood: str                      # emotional tone


def format_journal_context(entries: list[JournalEntry], recent_n: int = 3) -> str:
    """Compressed journal for prompt context.

    Full detail for the last ``recent_n`` entries, abbreviated
    (label + events only) for older ones.
    """
    if not entries:
        return "(No prior reading.)"

    parts: list[str] = []

    # Older entries — abbreviated
    older = entries[:-recent_n] if len(entries) > recent_n else []
    for e in older:
        parts.append(f"- {e.label} ({e.position:.0%}): {e.events}")

    # Recent entries — full detail
    recent = entries[-recent_n:]
    if older:
        parts.append("")
        parts.append("Recent sections (detailed):")
    for e in recent:
        parts.append(f"\n### {e.label} ({e.position:.0%})")
        parts.append(f"Events: {e.events}")
        parts.append(f"Mood: {e.mood}")
        # Include first ~500 chars of reaction for context
        if e.reaction:
            snippet = e.reaction[:500]
            if len(e.reaction) > 500:
                snippet += "..."
            parts.append(f"Reaction: {snippet}")

    return "\n".join(parts)


def render_journal_markdown(entries: list[JournalEntry], mind) -> str:
    """Render the complete journal + mental model as readable markdown."""
    lines: list[str] = []
    lines.append("# Reading Journal")
    lines.append("")

    # Table of contents
    lines.append("## Sections")
    lines.append("")
    for i, e in enumerate(entries):
        lines.append(f"{i + 1}. [{e.label}](#{_slug(e.label)}) ({e.position:.0%})")
    lines.append("")

    # Each section
    for i, e in enumerate(entries):
        lines.append(f"## {e.label}")
        lines.append(f"*Position: {e.position:.0%} | Mood: {e.mood}*")
        lines.append("")
        lines.append(f"**What happened:** {e.events}")
        lines.append("")
        lines.append("**Reaction:**")
        lines.append("")
        lines.append(e.reaction)
        lines.append("")
        lines.append("---")
        lines.append("")

    # Mental model snapshot
    lines.append("## Final Mental Model")
    lines.append("")

    # Characters
    if mind.characters:
        lines.append("### Characters")
        lines.append("")
        for name, char in sorted(mind.characters.items()):
            rels = ", ".join(char.relationships) if char.relationships else ""
            lines.append(f"- **{name}**: {char.description}")
            if rels:
                lines.append(f"  - Relationships: {rels}")
        lines.append("")

    # Theories
    if mind.theories:
        lines.append("### Theories")
        lines.append("")
        for i, t in enumerate(mind.theories):
            emoji = {"speculation": "?", "likely": "~", "confirmed": "+", "contradicted": "x"}.get(t.confidence, "?")
            lines.append(f"- [{emoji}] {t.text} *(confidence: {t.confidence})*")
            if t.evidence:
                for ev in t.evidence:
                    lines.append(f"  - {ev}")
        lines.append("")

    # Themes
    if mind.themes:
        lines.append("### Themes")
        lines.append("")
        for theme in mind.themes:
            lines.append(f"- {theme}")
        lines.append("")

    # Unresolved
    if mind.unresolved:
        lines.append("### Open Questions")
        lines.append("")
        for q in mind.unresolved:
            lines.append(f"- {q}")
        lines.append("")

    return "\n".join(lines)


def _slug(text: str) -> str:
    """Create a markdown-compatible anchor slug."""
    import re
    if not text:
        return "section"
    slug = text.strip().lower()
    slug = slug.replace(" ", "-")
    slug = re.sub(r'[^\w\-]', '', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug or "section"
