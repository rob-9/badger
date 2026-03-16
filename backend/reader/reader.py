"""
Core Orchestrator: the reading loop.

Drives the simulated readthrough from start to finish, coordinating
the react-think-ask cycle at each stop point and recording all outputs.
"""

import asyncio
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from badger.core.vector_store import VectorStore
from benchmarks.judge import score_response
from benchmarks.run import compute_retrieval_metrics

from reader.mind import ReaderMind, MindUpdate, react_to_section, update_mind
from reader.journal import JournalEntry, format_journal_context, render_journal_markdown
from reader.questions import generate_questions
from reader.reflection import reflect_on_response

logger = logging.getLogger(__name__)

# Chapter titles to skip: front-matter, back-matter, footnotes (case-insensitive)
SKIP_CHAPTER_RE = re.compile(
    r"^("
    r"dedication|contents|copyright|title\s*page|cover"
    r"|acknowledge?ments?"
    r"|about\s+the\s+(author|publisher)"
    r"|also\s+by\b.*"
    r"|.*\bfootnote\s*\d*"
    r")$",
    re.IGNORECASE,
)


@dataclass
class ReadthroughConfig:
    book_id: str
    stop_strategy: str = "chapter"            # "chapter" | "pct:N" | "chunks:N"
    max_questions_per_stop: int = 5
    max_follow_ups: int = 1                   # per question
    skip_judge: bool = False
    delay: float = 1.0                        # seconds between API calls
    dry_run: bool = False
    start_at: float = 0.0                     # resume from position (0-1)
    think_model: str = ""                     # model for REACT/THINK
    question_model: str = ""                  # model for question gen
    reflect_model: str = ""                   # model for reflection
    no_cache: bool = False


@dataclass
class StopPoint:
    position: float                 # 0.0 - 1.0
    chunk_range: tuple[int, int]    # (start_idx, end_idx) inclusive
    label: str                      # "Chapter Three" or "15%"
    chapter_index: int | None = None


SKIP_JUDGE_SCORES = {
    "relevance": -1,
    "conciseness": -1,
    "accuracy": -1,
    "spoiler_safety": -1,
    "notes": "judge skipped",
    "relevance_note": "",
    "accuracy_note": "",
    "spoiler_note": "",
    "judge_tokens_in": 0,
    "judge_tokens_out": 0,
}


# ---------------------------------------------------------------------------
# Stop Point Resolution
# ---------------------------------------------------------------------------

async def resolve_stops(
    vector_store: VectorStore,
    book_id: str,
    strategy: str,
) -> list[StopPoint]:
    """Resolve stop points based on strategy.

    Strategies:
      - "chapter": stop at each chapter boundary (auto-fallback to pct:10)
      - "pct:N": stop every N% of total chunks
      - "chunks:N": stop every N chunks
    """
    total_chunks = await vector_store.get_total_chunks(book_id)
    if total_chunks == 0:
        return []

    if strategy == "chapter":
        return await _resolve_chapter_stops(vector_store, book_id, total_chunks)
    elif strategy.startswith("pct:"):
        pct = int(strategy.split(":")[1])
        return _resolve_pct_stops(total_chunks, pct)
    elif strategy.startswith("chunks:"):
        n = int(strategy.split(":")[1])
        return _resolve_chunk_stops(total_chunks, n)
    else:
        logger.warning("Unknown stop strategy %r, falling back to pct:10", strategy)
        return _resolve_pct_stops(total_chunks, 10)


async def _resolve_chapter_stops(
    vector_store: VectorStore,
    book_id: str,
    total_chunks: int,
) -> list[StopPoint]:
    """Resolve chapter-based stop points by scanning chunk metadata."""
    # Load all chunks to inspect chapter metadata
    all_results = await vector_store.get_chunks_by_range(book_id, 0, total_chunks - 1)
    if not all_results:
        return _resolve_pct_stops(total_chunks, 10)

    # Group chunks by chapter_title
    chapters: list[dict] = []  # [{title, start_idx, end_idx, chapter_index}]
    current_title = None
    current_start = 0

    for r in all_results:
        chunk_idx = r.chunk.metadata.get("chunk_index", 0)
        chapter_title = r.chunk.metadata.get("chapter_title", "")
        chapter_index = r.chunk.metadata.get("chapter_index")

        if chapter_title != current_title:
            if current_title is not None:
                chapters.append({
                    "title": current_title,
                    "start_idx": current_start,
                    "end_idx": chunk_idx - 1,
                    "chapter_index": chapters[-1]["chapter_index"] + 1 if chapters else 0,
                })
            current_title = chapter_title
            current_start = chunk_idx

    # Don't forget the last chapter
    if current_title is not None:
        chapters.append({
            "title": current_title,
            "start_idx": current_start,
            "end_idx": total_chunks - 1,
            "chapter_index": chapters[-1]["chapter_index"] + 1 if chapters else 0,
        })

    # Filter out front-matter
    chapters = [
        c for c in chapters
        if c["title"] and not SKIP_CHAPTER_RE.match(c["title"].strip())
    ]

    if len(chapters) < 2:
        logger.warning("No chapter structure found, falling back to pct:10")
        return _resolve_pct_stops(total_chunks, 10)

    # Cap very long books at ~30 stops (evenly spaced, always include last)
    max_stops = 40
    if len(chapters) > max_stops:
        indices = [round(i * (len(chapters) - 1) / (max_stops - 1)) for i in range(max_stops)]
        chapters = [chapters[i] for i in dict.fromkeys(indices)]

    stops: list[StopPoint] = []
    for c in chapters:
        position = (c["end_idx"] + 1) / total_chunks
        stops.append(StopPoint(
            position=min(position, 1.0),
            chunk_range=(c["start_idx"], c["end_idx"]),
            label=c["title"] or f"Section {c['chapter_index']}",
            chapter_index=c.get("chapter_index"),
        ))

    return stops


def _resolve_pct_stops(total_chunks: int, pct: int) -> list[StopPoint]:
    """Stop every pct% of the book."""
    if total_chunks < 20:
        # Very short book — every 5 chunks
        return _resolve_chunk_stops(total_chunks, 5)

    step = max(1, int(total_chunks * pct / 100))
    stops: list[StopPoint] = []
    start = 0

    while start < total_chunks:
        end = min(start + step - 1, total_chunks - 1)
        position = (end + 1) / total_chunks
        stops.append(StopPoint(
            position=min(position, 1.0),
            chunk_range=(start, end),
            label=f"{int(position * 100)}%",
        ))
        start = end + 1

    return stops


def _resolve_chunk_stops(total_chunks: int, n: int) -> list[StopPoint]:
    """Stop every n chunks."""
    if total_chunks < 20:
        n = min(n, 5)

    stops: list[StopPoint] = []
    start = 0

    while start < total_chunks:
        end = min(start + n - 1, total_chunks - 1)
        position = (end + 1) / total_chunks
        stops.append(StopPoint(
            position=min(position, 1.0),
            chunk_range=(start, end),
            label=f"Chunks {start}-{end}",
        ))
        start = end + 1

    return stops


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

async def run_readthrough(
    cfg: ReadthroughConfig,
    anthropic,
    vector_store: VectorStore,
    voyage_client,
    agent: dict,
) -> None:
    """The main readthrough loop."""
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(f".data/readthrough/{cfg.book_id}/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve stop points
    stops = await resolve_stops(vector_store, cfg.book_id, cfg.stop_strategy)
    if not stops:
        print("ERROR: No stop points resolved. Is the book indexed?")
        return

    print(f"Resolved {len(stops)} stop points ({cfg.stop_strategy})")
    if cfg.dry_run:
        print("\nDry run — stop points:")
        for i, s in enumerate(stops):
            print(f"  [{i+1}] {s.label:30s} pos={s.position:.2%}  chunks={s.chunk_range}")
        return

    # Initialize state
    mind = ReaderMind()
    journal: list[JournalEntry] = []
    completed_stops: list[str] = []

    # Resume support: load state if start_at > 0
    state_path = output_dir / "state.json"
    if cfg.start_at > 0:
        mind, journal, completed_stops = _try_resume(output_dir, cfg.start_at)

    # Output file handles / paths
    mind_path = output_dir / "mind.jsonl"
    questions_path = output_dir / "questions.jsonl"
    responses_path = output_dir / "responses.jsonl"
    traces_path = output_dir / "traces.jsonl"
    journal_path = output_dir / "journal.md"

    # Tracking for report
    all_results: list[dict] = []
    total_questions = 0
    total_follow_ups = 0
    stop_summaries: list[dict] = []

    print(f"Output: {output_dir}")
    print(f"Book: {cfg.book_id}")
    print()

    dims = ["relevance", "conciseness", "accuracy", "spoiler_safety"]
    errors = 0

    for stop_idx, stop in enumerate(stops):
        if stop.position <= cfg.start_at:
            continue

        t0 = time.time()
        print(f"[{stop_idx + 1}/{len(stops)}] {stop.label} ({stop.position:.0%})...", end=" ", flush=True)

        try:
            # 1. READ — load chunks for this section
            chunks = await vector_store.get_chunks_by_range(
                cfg.book_id, stop.chunk_range[0], stop.chunk_range[1],
            )
            recent_text = "\n\n".join(r.chunk.text for r in chunks)
            if not recent_text.strip():
                print("(empty section, skipping)")
                continue

            # 2. REACT — chain-of-thought reaction
            reaction = await react_to_section(
                anthropic, recent_text, mind, stop.position, stop.label, cfg.think_model,
            )
            await asyncio.sleep(cfg.delay)

            # 3. THINK — structured mind update
            mind_update = await update_mind(
                anthropic, recent_text, reaction, mind, stop.position, stop.label, cfg.think_model,
            )
            mind.apply_update(mind_update, stop.position)
            await asyncio.sleep(cfg.delay)

            # 4. CREATE JOURNAL ENTRY
            entry = JournalEntry(
                position=stop.position,
                label=stop.label,
                events=mind_update.events_summary,
                reaction=reaction,
                mood=mind_update.emotional_state or mind.emotional_state,
            )
            journal.append(entry)

            # 5. ASK — generate questions
            journal_ctx = format_journal_context(journal)
            questions = await generate_questions(
                anthropic, recent_text, mind, journal_ctx,
                stop.position, stop.label, cfg.max_questions_per_stop, cfg.question_model,
            )
            await asyncio.sleep(cfg.delay)

        except Exception as e:
            elapsed = time.time() - t0
            errors += 1
            print(f"ERROR in react/think/ask: {e} ({elapsed:.1f}s)")
            logger.exception("Stop %d (%s) failed during react/think/ask", stop_idx, stop.label)
            # Save what we have so far before continuing
            _append_jsonl(mind_path, {
                "position": stop.position, "label": stop.label, "mind": mind.to_dict(),
            })
            completed_stops.append(stop.label)
            _write_state(state_path, stop.position, completed_stops, run_id)
            continue

        stop_scores: list[float] = []
        stop_questions = 0
        stop_follow_ups = 0

        # 6. For each question
        for q_idx, q in enumerate(questions):
            try:
                stop_questions += 1
                total_questions += 1

                # 6a. QUERY — run through the agent
                agent_result = await agent["run_agent"](
                    book_id=cfg.book_id,
                    question=q.question,
                    selected_text=q.selected_text or None,
                    reader_position=stop.position,
                )
                answer = agent_result.get("answer", "")
                sources = agent_result.get("sources", [])
                tool_calls = agent_result.get("tool_calls", [])

                # Build state dict for judge/metrics
                q_state = {
                    "answer": answer,
                    "sources": sources,
                    "tool_calls": tool_calls,
                    "selected_text": q.selected_text,
                    "book_id": cfg.book_id,
                    "reader_position": stop.position,
                    "question": q.question,
                }

                # 6b. JUDGE
                if cfg.skip_judge:
                    judge_scores = dict(SKIP_JUDGE_SCORES)
                else:
                    judge_chunks = [
                        {
                            "text": s.get("full_text", s.get("text", "")),
                            "label": s.get("label", "?"),
                            "chunk_index": s.get("chunk_index"),
                            "score": s.get("score", 0),
                        }
                        for s in sources
                    ]
                    judge_case = {
                        "id": f"reader-{stop_idx}-{q_idx}",
                        "question": q.question,
                        "selected_text": q.selected_text,
                        "reader_position": stop.position,
                        "expected_gist": q.expected_answer,
                        "tags": [q.question_type],
                    }
                    judge_scores = score_response(
                        anthropic=anthropic,
                        case=judge_case,
                        chunks=judge_chunks,
                        response=answer,
                    )

                retrieval_metrics = compute_retrieval_metrics(q_state)

                valid_scores = [judge_scores[d] for d in dims if judge_scores.get(d, -1) >= 0]
                avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
                stop_scores.append(avg_score)

                # 6c. REFLECT
                reflection = await reflect_on_response(
                    anthropic, q.question, answer, mind, stop.position, cfg.reflect_model,
                )

                # Build trace entry
                trace = {
                    "stop_index": stop_idx,
                    "stop_label": stop.label,
                    "position": stop.position,
                    "question_index": q_idx,
                    "question": q.question,
                    "question_type": q.question_type,
                    "selected_text": q.selected_text,
                    "motivation": q.motivation,
                    "expected_answer": q.expected_answer,
                    "triggered_by": q.triggered_by,
                    "answer": answer,
                    "tool_calls": tool_calls,
                    "sources_count": len(sources),
                    "retrieval_metrics": retrieval_metrics,
                    "judge": judge_scores,
                    "reflection": {
                        "satisfactory": reflection.satisfactory,
                        "contradicts_model": reflection.contradicts_model,
                        "reveals_new_info": reflection.reveals_new_info,
                        "possible_spoiler": reflection.possible_spoiler,
                        "follow_up": reflection.follow_up,
                        "follow_up_reason": reflection.follow_up_reason,
                        "mind_update": reflection.mind_update,
                    },
                    "is_follow_up": False,
                }

                follow_up_traces: list[dict] = []

                # 6d. FOLLOW-UP if needed
                if (
                    reflection.follow_up
                    and stop_follow_ups < cfg.max_follow_ups
                    and not reflection.satisfactory
                ):
                    stop_follow_ups += 1
                    total_follow_ups += 1

                    fu_result = await agent["run_agent"](
                        book_id=cfg.book_id,
                        question=reflection.follow_up,
                        selected_text=q.selected_text or None,
                        reader_position=stop.position,
                    )
                    fu_answer = fu_result.get("answer", "")
                    fu_sources = fu_result.get("sources", [])
                    fu_tool_calls = fu_result.get("tool_calls", [])

                    fu_q_state = {
                        "answer": fu_answer,
                        "sources": fu_sources,
                        "tool_calls": fu_tool_calls,
                        "selected_text": q.selected_text,
                        "book_id": cfg.book_id,
                        "reader_position": stop.position,
                        "question": reflection.follow_up,
                    }

                    if cfg.skip_judge:
                        fu_judge = dict(SKIP_JUDGE_SCORES)
                    else:
                        fu_chunks = [
                            {
                                "text": s.get("full_text", s.get("text", "")),
                                "label": s.get("label", "?"),
                                "chunk_index": s.get("chunk_index"),
                                "score": s.get("score", 0),
                            }
                            for s in fu_sources
                        ]
                        fu_case = {
                            "id": f"reader-{stop_idx}-{q_idx}-fu",
                            "question": reflection.follow_up,
                            "selected_text": q.selected_text,
                            "reader_position": stop.position,
                            "expected_gist": reflection.follow_up_reason or "",
                            "tags": [q.question_type],
                        }
                        fu_judge = score_response(
                            anthropic=anthropic,
                            case=fu_case,
                            chunks=fu_chunks,
                            response=fu_answer,
                        )

                    fu_metrics = compute_retrieval_metrics(fu_q_state)
                    fu_valid = [fu_judge[d] for d in dims if fu_judge.get(d, -1) >= 0]
                    fu_avg = sum(fu_valid) / len(fu_valid) if fu_valid else 0
                    stop_scores.append(fu_avg)

                    # Reflect on follow-up (no further follow-ups)
                    fu_reflection = await reflect_on_response(
                        anthropic, reflection.follow_up, fu_answer, mind,
                        stop.position, cfg.reflect_model,
                    )

                    fu_trace = {
                        "stop_index": stop_idx,
                        "stop_label": stop.label,
                        "position": stop.position,
                        "question_index": q_idx,
                        "question": reflection.follow_up,
                        "question_type": q.question_type,
                        "selected_text": q.selected_text,
                        "motivation": reflection.follow_up_reason or "",
                        "expected_answer": "",
                        "triggered_by": f"follow_up:{q_idx}",
                        "answer": fu_answer,
                        "tool_calls": fu_tool_calls,
                        "sources_count": len(fu_sources),
                        "retrieval_metrics": fu_metrics,
                        "judge": fu_judge,
                        "reflection": {
                            "satisfactory": fu_reflection.satisfactory,
                            "contradicts_model": fu_reflection.contradicts_model,
                            "reveals_new_info": fu_reflection.reveals_new_info,
                            "possible_spoiler": fu_reflection.possible_spoiler,
                            "follow_up": None,
                            "follow_up_reason": None,
                            "mind_update": fu_reflection.mind_update,
                        },
                        "is_follow_up": True,
                    }
                    follow_up_traces.append(fu_trace)

                # 6e. RECORD — append to output files
                _append_jsonl(questions_path, {
                    "stop_index": stop_idx,
                    "position": stop.position,
                    "label": stop.label,
                    "question": q.question,
                    "selected_text": q.selected_text,
                    "question_type": q.question_type,
                    "motivation": q.motivation,
                    "expected_answer": q.expected_answer,
                    "triggered_by": q.triggered_by,
                })

                _append_jsonl(responses_path, {
                    "stop_index": stop_idx,
                    "position": stop.position,
                    "question": q.question,
                    "answer": answer,
                    "judge": judge_scores,
                    "reflection": trace["reflection"],
                })

                _append_jsonl(traces_path, trace)
                for ft in follow_up_traces:
                    _append_jsonl(traces_path, ft)

                all_results.append(trace)
                for ft in follow_up_traces:
                    all_results.append(ft)

            except Exception as e:
                errors += 1
                logger.exception("Question %d at stop %d failed: %s", q_idx, stop_idx, e)
                print(f"\n    ⚠  Question {q_idx + 1} failed: {e}")
                continue

            # 6f. Delay between questions
            await asyncio.sleep(cfg.delay)

        # Console output
        elapsed = time.time() - t0
        avg = sum(stop_scores) / len(stop_scores) if stop_scores else 0
        print(f"{stop_questions} questions, avg={avg:.1f}/3 ({elapsed:.1f}s)")

        # Check for spoiler leaks
        for r in all_results[-(stop_questions + stop_follow_ups):]:
            spoiler_score = r.get("judge", {}).get("spoiler_safety", 3)
            if 0 <= spoiler_score <= 1:
                print(f"    ⚠  SPOILER LEAK: score={spoiler_score}/3 at position={r['position']:.2f}")

        stop_summaries.append({
            "stop_index": stop_idx,
            "label": stop.label,
            "position": stop.position,
            "questions": stop_questions,
            "follow_ups": stop_follow_ups,
            "avg_score": round(avg, 2),
            "elapsed": round(elapsed, 1),
        })

        # 7. SNAPSHOT — append mind state (include events_summary for resume)
        _append_jsonl(mind_path, {
            "position": stop.position,
            "label": stop.label,
            "events_summary": mind_update.events_summary,
            "mind": mind.to_dict(),
        })

        # 8. RENDER journal FIRST (so it's consistent if crash after state write)
        journal_md = render_journal_markdown(journal, mind)
        journal_path.write_text(journal_md)

        # 9. UPDATE STATE (last — signals this stop is complete)
        completed_stops.append(stop.label)
        _write_state(state_path, stop.position, completed_stops, run_id)

    # Generate report
    print()
    report = generate_readthrough_report(
        cfg, run_id, stops, all_results, stop_summaries, mind, journal, output_dir,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report)

    print(f"Readthrough complete: {total_questions} questions, {total_follow_ups} follow-ups", end="")
    if errors:
        print(f", {errors} errors")
    else:
        print()
    print(f"Output: {output_dir}")
    print(f"  report.md")
    print(f"  journal.md")
    print(f"  traces.jsonl")
    print(f"  questions.jsonl")
    print(f"  responses.jsonl")
    print(f"  mind.jsonl")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_readthrough_report(
    cfg: ReadthroughConfig,
    run_id: str,
    stops: list[StopPoint],
    all_results: list[dict],
    stop_summaries: list[dict],
    mind: ReaderMind,
    journal: list[JournalEntry],
    output_dir: Path,
) -> str:
    """Generate report.md with aggregate analysis."""
    lines: list[str] = []
    dims = ["relevance", "conciseness", "accuracy", "spoiler_safety"]

    lines.append(f"# Readthrough Report: {cfg.book_id}")
    lines.append(f"**Run ID:** {run_id}  |  **Stops:** {len(stops)}  |  **Questions:** {len(all_results)}")
    lines.append(f"**Strategy:** {cfg.stop_strategy}  |  **Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # 1. Overall scores
    agg: dict[str, dict] = {}
    for d in dims:
        vals = [r["judge"][d] for r in all_results if r.get("judge", {}).get(d, -1) >= 0]
        agg[d] = {
            "mean": round(sum(vals) / len(vals), 2) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
        }

    lines.append("## Overall Scores")
    lines.append("| Dimension | Mean | Min | Max |")
    lines.append("|-----------|------|-----|-----|")
    overall_scores = []
    for d in dims:
        a = agg[d]
        label = d.replace("_", " ").title()
        lines.append(f"| {label} | {a['mean']} | {a['min']} | {a['max']} |")
        overall_scores.append(a["mean"])
    overall = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0
    lines.append("")
    lines.append(f"**Overall: {overall} / 3.00**")
    lines.append("")

    # 2. Quality vs position curve
    lines.append("## Quality vs Position")
    lines.append("| Stop | Label | Position | Questions | Avg Score | Time |")
    lines.append("|------|-------|----------|-----------|-----------|------|")
    for ss in stop_summaries:
        lines.append(
            f"| {ss['stop_index'] + 1} | {ss['label']} | {ss['position']:.0%} "
            f"| {ss['questions']} | {ss['avg_score']}/3 | {ss['elapsed']}s |"
        )
    lines.append("")

    # 3. Spoiler safety map
    spoiler_incidents = [
        r for r in all_results
        if 0 <= r.get("judge", {}).get("spoiler_safety", 3) <= 1
    ]
    if spoiler_incidents:
        lines.append("## Spoiler Safety Issues")
        for r in spoiler_incidents:
            lines.append(
                f"- **{r.get('stop_label', '?')}** ({r['position']:.0%}): "
                f"score={r['judge']['spoiler_safety']}/3 — "
                f"Q: \"{r['question'][:80]}\""
            )
        lines.append("")
    else:
        lines.append("## Spoiler Safety")
        lines.append("No spoiler leaks detected.")
        lines.append("")

    # 4. Theory tracking summary
    lines.append("## Theory Tracking")
    if mind.theories:
        confirmed = [t for t in mind.theories if t.confidence == "confirmed"]
        contradicted = [t for t in mind.theories if t.confidence == "contradicted"]
        active = [t for t in mind.theories if t.confidence in ("speculation", "likely")]
        lines.append(f"- Formed: {len(mind.theories)} theories total")
        lines.append(f"- Confirmed: {len(confirmed)}")
        lines.append(f"- Contradicted: {len(contradicted)}")
        lines.append(f"- Still active: {len(active)}")
        lines.append("")
        for t in mind.theories:
            marker = {"speculation": "?", "likely": "~", "confirmed": "+", "contradicted": "x"}.get(t.confidence, "?")
            lines.append(f"  [{marker}] {t.text}")
    else:
        lines.append("No theories tracked.")
    lines.append("")

    # 5. Follow-up analysis
    follow_ups = [r for r in all_results if r.get("is_follow_up")]
    non_follow_ups = [r for r in all_results if not r.get("is_follow_up")]
    lines.append("## Follow-up Analysis")
    lines.append(f"- Total follow-ups: {len(follow_ups)}")
    if follow_ups:
        fu_avgs = []
        for r in follow_ups:
            vals = [r["judge"][d] for d in dims if r.get("judge", {}).get(d, -1) >= 0]
            if vals:
                fu_avgs.append(sum(vals) / len(vals))
        orig_avgs = []
        for r in non_follow_ups:
            vals = [r["judge"][d] for d in dims if r.get("judge", {}).get(d, -1) >= 0]
            if vals:
                orig_avgs.append(sum(vals) / len(vals))
        fu_mean = round(sum(fu_avgs) / len(fu_avgs), 2) if fu_avgs else 0
        orig_mean = round(sum(orig_avgs) / len(orig_avgs), 2) if orig_avgs else 0
        lines.append(f"- Follow-up avg score: {fu_mean}/3")
        lines.append(f"- Original avg score: {orig_mean}/3")
        improvement = round(fu_mean - orig_mean, 2)
        lines.append(f"- Delta: {'+' if improvement >= 0 else ''}{improvement}")
    lines.append("")

    # 6. Question type distribution
    lines.append("## Question Type Distribution")
    type_counts: dict[str, int] = {}
    for r in all_results:
        qt = r.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1
    for qt, count in sorted(type_counts.items()):
        lines.append(f"- {qt}: {count}")
    lines.append("")

    # 7. Token usage & cost estimates
    lines.append("## Token Usage")
    judge_in = sum(r.get("judge", {}).get("judge_tokens_in", 0) for r in all_results)
    judge_out = sum(r.get("judge", {}).get("judge_tokens_out", 0) for r in all_results)
    lines.append("| Stage | Input | Output |")
    lines.append("|-------|-------|--------|")
    lines.append(f"| Judge | {judge_in:,} | {judge_out:,} |")
    lines.append("")

    # 8. Timing
    total_time = sum(ss.get("elapsed", 0) for ss in stop_summaries)
    times = [ss.get("elapsed", 0) for ss in stop_summaries if ss.get("elapsed", 0) > 0]
    lines.append("## Timing")
    lines.append(f"- Total: {total_time:.0f}s ({total_time / 60:.1f}m)")
    if times:
        lines.append(f"- Median per stop: {statistics.median(times):.1f}s")
        lines.append(f"- Max per stop: {max(times):.1f}s")
    lines.append("")

    # 9. Stop-by-stop details
    lines.append("## Stop-by-Stop Details")
    lines.append("")
    for ss in stop_summaries:
        stop_results = [
            r for r in all_results
            if r.get("stop_index") == ss["stop_index"]
        ]
        lines.append(f"### {ss['label']} ({ss['position']:.0%})")
        lines.append(f"*{ss['questions']} questions, {ss['follow_ups']} follow-ups, avg={ss['avg_score']}/3, {ss['elapsed']}s*")
        lines.append("")
        for r in stop_results:
            j = r.get("judge", {})
            valid = [j[d] for d in dims if j.get(d, -1) >= 0]
            avg = round(sum(valid) / len(valid), 1) if valid else 0
            fu_marker = " (follow-up)" if r.get("is_follow_up") else ""
            lines.append(f"- [{r.get('question_type', '?')}] \"{r['question'][:80]}\" — {avg}/3{fu_marker}")
            notes = j.get("notes", "")
            if notes and notes != "judge skipped":
                lines.append(f"  - {notes}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, data: dict) -> None:
    """Append a JSON line to a file."""
    with open(path, "a") as f:
        f.write(json.dumps(data, default=str) + "\n")


def _write_state(path: Path, last_position: float, completed: list[str], run_id: str) -> None:
    """Write run state for resume capability."""
    state = {
        "run_id": run_id,
        "last_position": last_position,
        "completed_stops": completed,
        "updated_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(state, indent=2))


def _try_resume(
    output_dir: Path,
    start_at: float,
) -> tuple[ReaderMind, list[JournalEntry], list[str]]:
    """Try to resume from existing state files.

    Restores mind state, journal entries, and completed stops from
    the output directory. Falls back to empty state on any error.
    """
    mind = ReaderMind()
    journal: list[JournalEntry] = []
    completed: list[str] = []

    # Restore mind + journal entries from mind.jsonl
    mind_path = output_dir / "mind.jsonl"
    if mind_path.exists():
        try:
            latest = None
            for line in mind_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                entry = json.loads(line)
                pos = entry.get("position", 0)
                if pos <= start_at:
                    latest = entry
                    # Build journal entry from mind snapshot
                    journal.append(JournalEntry(
                        position=pos,
                        label=entry.get("label", f"{pos:.0%}"),
                        events=entry.get("events_summary", ""),
                        reaction=entry.get("reaction", "(resumed)"),
                        mood=entry.get("mind", {}).get("emotional_state", ""),
                    ))
            if latest and "mind" in latest:
                mind = ReaderMind.from_dict(latest["mind"])
                logger.info("Resumed mind state from position %.2f (%d journal entries)",
                            latest["position"], len(journal))
        except Exception as e:
            logger.warning("Failed to resume mind state: %s", e)
            journal = []

    # Restore state.json for completed stops
    state_path = output_dir / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            completed = state.get("completed_stops", [])
        except Exception as e:
            logger.warning("Failed to load state.json: %s", e)

    return mind, journal, completed
