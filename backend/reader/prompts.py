"""All prompt templates for the readthrough agent."""

REACT_PROMPT = """\
You are reading a book for the first time. You have just finished reading \
the section labeled "{label}" (approximately {position:.0%} through the book).

Here is what you just read:

---
{recent_text}
---

Here is your mental model so far (your theories, characters you've met, \
open questions):

{mind_context}

Respond as if you are experiencing this section for the first time. \
Write a few paragraphs covering:

1. **What just happened** — summarize the key events in 2-3 sentences.
2. **What surprised you** — anything that defied your expectations.
3. **What confused you** — anything unclear, ambiguous, or unexplained.
4. **Theories forming** — what you think might happen or what you suspect \
is going on beneath the surface.
5. **Emotional reaction** — how this section made you feel as a reader.

CRITICAL RULE: You only know what has been read so far. Do not use any \
external knowledge of this book. Do not reference events, characters, or \
themes that have not yet appeared in the text you've read. If you recognize \
this book, ignore everything you know about it beyond what the text has \
shown you up to this point.\
"""

MIND_UPDATE_PROMPT = """\
You are updating your mental model of a book after reading a new section.

Here is the section you just read ("{label}", ~{position:.0%} through):

---
{recent_text}
---

Here is your chain-of-thought reaction to this section:

---
{reaction}
---

Here is your current mental model (the state BEFORE this section):

{mind_context}

Based on the section text and your reaction, produce a structured JSON \
update to your mental model. Include ONLY what changed or is new. \
Return a JSON object with these fields:

{{
  "new_characters": [
    {{"name": "...", "description": "brief description", "relationships": ["relationship to other character"], "first_seen": {position}}}
  ],
  "updated_characters": [
    {{"name": "...", "description_update": "what changed about them", "new_relationships": ["new relationship"], "last_seen": {position}}}
  ],
  "new_theories": [
    {{"text": "your theory", "confidence": "speculation|likely|confirmed|contradicted", "evidence": ["brief evidence note"]}}
  ],
  "theory_updates": [
    {{"index": 0, "confidence": "speculation|likely|confirmed|contradicted", "evidence_note": "why the update"}}
  ],
  "new_unresolved": ["open question the reader is wondering about"],
  "resolved": [0, 1],
  "new_themes": ["theme or motif recognized"],
  "new_surprises": ["moment that defied expectations"],
  "emotional_state": "one or two words describing your mood",
  "events_summary": "2-3 sentence summary of what happened in this section"
}}

Return ONLY valid JSON. Omit fields with empty lists. Always include \
events_summary and emotional_state.\
"""

QUESTION_GEN_PROMPT = """\
You are a reader at {position:.0%} through a book. You just finished the \
section labeled "{label}".

Here is the text of the section you just read:

---
{recent_text}
---

Here is your current mental model (theories, characters, open questions):

{mind_context}

Here is a summary of your reading so far:

{journal_context}

Generate up to {max_questions} questions that you, as this reader, would \
naturally want to ask a reading assistant right now. Each question should \
be grounded in what you just read.

Question type guidelines based on your position ({position:.0%}):
{type_guidance}

For each question, provide:
- "question": the question text (natural, conversational)
- "selected_text": an EXACT substring from the section text above that \
triggered this question. Must be a verbatim copy of text from the section. \
REQUIRED (must not be empty) for question_type "vocabulary", "lookup", and \
"context" — these types need an anchor passage to retrieve the right content. \
Copy the specific word, phrase, name, or sentence that prompted the question.
- "question_type": one of "vocabulary", "context", "lookup", "analysis"
- "motivation": why the reader would ask this (1 sentence)
- "expected_answer": a sketch of what a good answer would include (2-3 sentences)
- "triggered_by": if this question was triggered by a specific theory or \
unresolved question, write "theory:N" or "unresolved:N" (using the index). \
Otherwise null.
- "answerable_by_retrieval": true if this question can be answered by finding \
and quoting specific passages from the book (vocabulary definitions, factual \
lookups, locating a specific scene). false if the question requires \
interpretation, thematic synthesis, character motivation analysis, or \
connecting ideas across multiple sections. When in doubt, prefer false.

Examples of correctly filled selected_text:

vocabulary example — the foreign word itself must be copied:
{{"question": "What does 'mujahid' mean?", "selected_text": "mujahid", \
"question_type": "vocabulary", "motivation": "Unfamiliar Arabic term.", \
"expected_answer": "A Muslim fighter in a holy war.", "triggered_by": null, \
"answerable_by_retrieval": true}}

lookup example — the name or concept phrase must be copied:
{{"question": "Who is Professor Lovell and why does Letty trust him?", \
"selected_text": "Professor Lovell", "question_type": "lookup", \
"motivation": "Lovell appears suddenly and Letty defers to him completely.", \
"expected_answer": "Lovell is a senior Oxford professor who recruited Robin.", \
"triggered_by": null, "answerable_by_retrieval": true}}

Return a JSON array of question objects. Return ONLY valid JSON.\
"""

REFLECT_PROMPT = """\
You are a reader at {position:.0%} through a book. You asked a question \
and received a response from a reading assistant backed by RAG retrieval.

Your question: {question}

The response you received:
---
{response}
---

Your current mental model (what you know/believe so far):

{mind_context}

Evaluate this response from your perspective as a reader at this point \
in the book. Return a JSON object:

{{
  "satisfactory": true/false,
  "contradicts_model": true/false,
  "reveals_new_info": true/false,
  "possible_spoiler": true/false,
  "follow_up": "a follow-up question if the answer was unsatisfying or \
opened a new thread, otherwise null",
  "follow_up_reason": "why you want to ask this follow-up, otherwise null",
  "mind_update": "brief note on how this response changes your mental \
model, or null if it doesn't"
}}

Be honest: if the answer was satisfying and complete, say so. Only ask \
a follow-up if there's a genuine gap or interesting thread to pull.

Return ONLY valid JSON.\
"""

DIRECT_ANSWER_PROMPT = """\
You are a reading companion. The reader is at {position:.0%} through the book \
and just finished "{label}". They have a question that calls for interpretation \
or analysis — not just finding a passage.

The reader's question: {question}

{selected_text_block}

Here is the section the reader just finished:

---
{recent_text}
---

Here is the reader's evolving understanding of the book:

{mind_context}

Answer thoughtfully but concisely (2-4 sentences). You may:
- Interpret events, dialogue, and character behavior
- Connect themes and patterns across the reading so far
- Reference the reader's own theories and open questions
- Offer your own reading of ambiguous moments

You must NOT:
- Reveal ANY events, character fates, or plot points beyond {position:.0%}
- Use knowledge of this book from outside what the reader has read
- Hint at twists, betrayals, deaths, or reversals not yet encountered
- Add dramatic irony or foreshadowing based on what you know happens later

If you recognize this book, suppress everything beyond what has been read.

STYLE:
- Use character names, not generic labels
- Talk about the story world directly — no "the text shows" or "the narrative suggests"
- Be a thoughtful reading companion, not an essayist
- Address the reader as "you"\
"""
