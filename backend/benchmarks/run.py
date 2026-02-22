"""
RAG benchmark runner.

Runs test cases through the full LangGraph pipeline, scores with an LLM judge,
and produces a detailed diagnostic report.

Usage:
    python -m benchmarks.run                          # all cases
    python -m benchmarks.run --tags vocabulary         # filter by tag
    python -m benchmarks.run --ids vocab-canton        # specific cases
    python -m benchmarks.run --dry-run                 # validate only
    python -m benchmarks.run --delay 3                 # seconds between cases
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

from boom import config
from boom.core.rag import RAGService
from boom.core.graph import build_qa_graph, _build_log_entry
from benchmarks.judge import score_response

logger = logging.getLogger(__name__)

CASES_FILE = Path(__file__).parent / "test_cases.json"
OUTPUT_DIR = Path(".data/benchmarks")
LOGS_DIR = Path(__file__).parent / "logs"

BOOK_ALIASES = {
    "babel": "1771123370474-gtejcsha5",
    "cloud-atlas": "1771736822957-fjia2yspy",
    "gone-girl": "1771737236835-15c8dfryt",
}


# === Loading & Filtering ===


def load_cases(path: Path = CASES_FILE) -> dict:
    """Load test suite from JSON file."""
    with open(path) as f:
        return json.load(f)


def filter_cases(
    cases: list[dict],
    ids: list[str] | None = None,
    tags: list[str] | None = None,
    books: list[str] | None = None,
) -> list[dict]:
    """Filter cases by --ids, --tags, or --book."""
    if ids:
        id_set = set(ids)
        cases = [c for c in cases if c["id"] in id_set]
    if tags:
        tag_set = set(tags)
        cases = [c for c in cases if tag_set & set(c.get("tags", []))]
    if books:
        resolved = [BOOK_ALIASES.get(b, b) for b in books]
        cases = [c for c in cases if any(b in c["book_id"] for b in resolved)]
    return cases


# === Retrieval Metrics ===


def compute_retrieval_metrics(state: dict) -> dict:
    """Compute retrieval metrics from pipeline state (no API call)."""
    chunks = state.get("chunks", [])
    scores = [c.get("score", 0) for c in chunks]
    selected = state.get("selected_text") or ""
    total = state.get("total_chunks", 0)
    position = state.get("reader_position", 0)
    reader_chunk = int(position * total) if total else 0

    selected_in_chunks = False
    if selected:
        needle = selected.lower()
        selected_in_chunks = any(needle in c.get("text", "").lower() for c in chunks)

    return {
        "num_chunks_retrieved": len(chunks),
        "avg_score": round(sum(scores) / len(scores), 4) if scores else 0,
        "max_score": round(max(scores), 4) if scores else 0,
        "min_score": round(min(scores), 4) if scores else 0,
        "past_chunk_count": sum(1 for c in chunks if c.get("label") == "PAST"),
        "ahead_chunk_count": sum(1 for c in chunks if c.get("label") == "AHEAD"),
        "selected_text_in_chunks": selected_in_chunks,
        "reader_chunk_index": reader_chunk,
        "total_chunks": total,
        "chunk_indices": [c.get("chunk_index", -1) for c in chunks],
    }


# === Diagnostic Flags ===


def compute_diagnostics(case: dict, state: dict, judge: dict, metrics: dict) -> list[str]:
    """Flag specific problems for easy scanning."""
    flags = []

    # Classification mismatch
    expected_type = case.get("question_type")
    actual_type = state.get("question_type")
    if expected_type and expected_type != actual_type:
        flags.append(f"MISCLASSIFIED: expected={expected_type}, got={actual_type}")

    # Spoiler leak
    if judge.get("spoiler_safety", 3) <= 1:
        ahead = metrics.get("ahead_chunk_count", 0)
        total = metrics.get("num_chunks_retrieved", 0)
        flags.append(f"SPOILER LEAK: score={judge['spoiler_safety']}/3, ahead_chunks={ahead}/{total}")

    # Fabrication (accuracy=0 or 1 with judge noting fabrication)
    if judge.get("accuracy", 3) <= 1:
        notes = judge.get("notes", "").lower()
        if "fabricat" in notes or "unsupported" in notes or "not in" in notes:
            flags.append(f"FABRICATION: accuracy={judge['accuracy']}/3 — {judge.get('notes', '')[:80]}")
        else:
            flags.append(f"LOW ACCURACY: score={judge['accuracy']}/3")

    # Retrieval miss — selected text not found in chunks
    if case.get("selected_text") and not metrics.get("selected_text_in_chunks"):
        flags.append(f"RETRIEVAL MISS: selected text not found in retrieved chunks")

    # All chunks AHEAD — nothing from what the reader has actually read
    if metrics.get("past_chunk_count", 0) == 0 and metrics.get("num_chunks_retrieved", 0) > 0:
        flags.append("ALL CHUNKS AHEAD: no PAST content retrieved")

    # Verbose response
    if judge.get("conciseness", 3) <= 1:
        answer_len = len(state.get("answer", ""))
        flags.append(f"VERBOSE: conciseness={judge['conciseness']}/3, answer_length={answer_len} chars")

    return flags


# === Detailed Trace File ===


W = 80


def write_detailed_trace(f, case: dict, state: dict, judge: dict, metrics: dict, diagnostics: list[str], elapsed: float):
    """Write a human-readable diagnostic trace for one case."""
    def header(title):
        f.write(f"\n{'=' * W}\n")
        f.write(f"  {title}\n")
        f.write(f"{'=' * W}\n\n")

    def section(title):
        f.write(f"\n── {title} {'─' * (W - 4 - len(title))}\n\n")

    case_id = case["id"]
    dims = ["relevance", "conciseness", "accuracy", "spoiler_safety"]
    scores = [judge[d] for d in dims if judge.get(d, -1) >= 0]
    avg = round(sum(scores) / len(scores), 1) if scores else 0

    header(f"{case_id}  —  avg={avg}/3  ({elapsed:.1f}s)")

    # Diagnostics (the most important part — what went wrong)
    if diagnostics:
        section("PROBLEMS")
        for flag in diagnostics:
            f.write(f"  ⚠  {flag}\n")
    else:
        section("STATUS")
        f.write("  ✓  No issues detected\n")

    # Test case inputs
    section("Input")
    f.write(f"  Question:        {case['question']}\n")
    f.write(f"  Selected text:   \"{case.get('selected_text', '')}\"\n")
    f.write(f"  Reader position: {case.get('reader_position', 0):.0%} (chunk {metrics['reader_chunk_index']}/{metrics['total_chunks']})\n")
    f.write(f"  Expected type:   {case.get('question_type', '(not specified)')}\n")
    f.write(f"  Expected gist:   {case['expected_gist']}\n")
    f.write(f"  Tags:            {', '.join(case.get('tags', []))}\n")

    # Classification
    section("Classification")
    f.write(f"  Model:     {config.CLAUDE_HAIKU_MODEL}\n")
    f.write(f"  Result:    {state.get('question_type')}\n")
    expected = case.get("question_type")
    if expected:
        match = "✓ MATCH" if expected == state.get("question_type") else f"✗ MISMATCH (expected {expected})"
        f.write(f"  Match:     {match}\n")
    f.write(f"  Entities:  {state.get('entities', [])}\n")
    f.write(f"  Raw:       {state.get('classify_raw_response', '')}\n")
    f.write(f"  Tokens:    {state.get('classify_tokens_in', 0)} in / {state.get('classify_tokens_out', 0)} out\n")

    # Retrieval
    section("Retrieval")
    f.write(f"  Strategy:        {state.get('retrieval_strategy')}\n")
    if state.get("retrieval_query"):
        f.write(f"  Query:           {state['retrieval_query'][:200]}\n")
    if state.get("retrieval_center_chunk") is not None:
        f.write(f"  Center chunk:    {state['retrieval_center_chunk']}\n")
        f.write(f"  Range:           [{state.get('retrieval_range_start')}, {state.get('retrieval_range_end')}]\n")
        f.write(f"  Chunk match:     {state.get('retrieval_chunk_match')}\n")
    f.write(f"  Chunks:          {metrics['num_chunks_retrieved']} ({metrics['past_chunk_count']} PAST, {metrics['ahead_chunk_count']} AHEAD)\n")
    f.write(f"  Scores:          avg={metrics['avg_score']}, max={metrics['max_score']}, min={metrics['min_score']}\n")
    f.write(f"  Selected found:  {'Yes' if metrics['selected_text_in_chunks'] else 'No'}\n")

    # Full chunk dump
    chunks = state.get("chunks", [])
    f.write(f"\n  --- Retrieved Chunks ({len(chunks)}) ---\n")
    for i, c in enumerate(chunks):
        f.write(f"\n  [{i+1}] chunk_index={c.get('chunk_index')} | score={c.get('score', 0):.4f} | {c.get('label', '?')}\n")
        text = c.get("text", "")
        # Show full text, indented
        for line in text.split("\n"):
            f.write(f"      {line}\n")

    # Generation
    section("Generation")
    f.write(f"  Model:   {state.get('gen_model')}\n")
    f.write(f"  Tokens:  {state.get('gen_tokens_in', 0)} in / {state.get('gen_tokens_out', 0)} out\n")
    f.write(f"  Stop:    {state.get('gen_stop_reason')}\n")

    f.write(f"\n  --- System Prompt ({len(state.get('gen_system_prompt', ''))} chars) ---\n")
    for line in (state.get("gen_system_prompt") or "").split("\n"):
        f.write(f"      {line}\n")

    f.write(f"\n  --- User Prompt ({len(state.get('gen_user_prompt', ''))} chars) ---\n")
    for line in (state.get("gen_user_prompt") or "").split("\n"):
        f.write(f"      {line}\n")

    f.write(f"\n  --- AI Response ({len(state.get('answer', ''))} chars) ---\n")
    for line in (state.get("answer") or "").split("\n"):
        f.write(f"      {line}\n")

    # Judge
    section("Judge Scores")
    for d in dims:
        f.write(f"  {d:20s} {judge.get(d, -1)}/3\n")
    f.write(f"  {'average':20s} {avg}/3\n")
    f.write(f"\n  Notes:         {judge.get('notes', '')}\n")
    if judge.get("relevance_note"):
        f.write(f"  Relevance:     {judge['relevance_note']}\n")
    if judge.get("accuracy_note"):
        f.write(f"  Accuracy:      {judge['accuracy_note']}\n")
    if judge.get("spoiler_note"):
        f.write(f"  Spoiler:       {judge['spoiler_note']}\n")
    f.write(f"  Tokens:        {judge.get('judge_tokens_in', 0)} in / {judge.get('judge_tokens_out', 0)} out\n")

    # Expected vs actual comparison
    section("Expected vs Actual")
    f.write(f"  Expected gist:\n")
    f.write(f"      {case['expected_gist']}\n")
    f.write(f"\n  Actual answer (first 500 chars):\n")
    answer = state.get("answer", "")[:500]
    for line in answer.split("\n"):
        f.write(f"      {line}\n")

    f.write(f"\n{'=' * W}\n\n\n")


# === Report Generation ===


def generate_report(
    suite_name: str,
    run_id: str,
    results: list[dict],
    output_dir: Path,
) -> str:
    """Generate markdown report from benchmark results."""
    lines = []
    lines.append(f"# Benchmark Report: {suite_name}")
    lines.append(f"**Run ID:** {run_id}  |  **Cases:** {len(results)}  |  **Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # Aggregate scores
    dims = ["relevance", "conciseness", "accuracy", "spoiler_safety"]
    dim_labels = ["Relevance", "Conciseness", "Accuracy", "Spoiler Safety"]
    agg = {}
    for d in dims:
        vals = [r["judge"][d] for r in results if r["judge"].get(d, -1) >= 0]
        agg[d] = {
            "mean": round(sum(vals) / len(vals), 2) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
        }

    lines.append("## Aggregate Scores")
    lines.append("| Dimension | Mean | Min | Max |")
    lines.append("|-----------|------|-----|-----|")
    overall_scores = []
    for d, label in zip(dims, dim_labels):
        a = agg[d]
        lines.append(f"| {label} | {a['mean']} | {a['min']} | {a['max']} |")
        overall_scores.append(a["mean"])
    lines.append("")
    overall = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0
    lines.append(f"**Overall: {overall} / 3.00**")
    lines.append("")

    # Problems summary — the most actionable section
    all_diagnostics = []
    for r in results:
        for flag in r["diagnostics"]:
            all_diagnostics.append((r["case"]["id"], flag))
    if all_diagnostics:
        lines.append("## Problems Found")
        for case_id, flag in all_diagnostics:
            lines.append(f"- **{case_id}**: {flag}")
        lines.append("")

    # Classification accuracy
    correct = 0
    total_classified = 0
    mismatches = []
    for r in results:
        expected = r["case"].get("question_type")
        actual = r["state"].get("question_type")
        if expected:
            total_classified += 1
            if expected == actual:
                correct += 1
            else:
                mismatches.append(f"{r['case']['id']}: expected={expected}, got={actual}")
    if total_classified:
        lines.append("## Classification Accuracy")
        lines.append(f"{correct}/{total_classified} ({100 * correct // total_classified}%)")
        if mismatches:
            for m in mismatches:
                lines.append(f"- {m}")
        lines.append("")

    # Per-case results
    lines.append("## Per-Case Results")
    lines.append("| ID | Type | Rel | Con | Acc | Spoil | Avg | Flags | Notes |")
    lines.append("|----|------|-----|-----|-----|-------|-----|-------|-------|")
    for r in results:
        j = r["judge"]
        scores = [j[d] for d in dims if j.get(d, -1) >= 0]
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        q_type = r["state"].get("question_type", "?")
        expected_type = r["case"].get("question_type")
        type_display = q_type
        if expected_type and expected_type != q_type:
            type_display = f"~~{expected_type}~~→{q_type}"
        flag_count = len(r["diagnostics"])
        flag_str = f"{flag_count} issue{'s' if flag_count != 1 else ''}" if flag_count else "clean"
        notes = j.get("notes", "")[:50]
        lines.append(
            f"| {r['case']['id']} | {type_display} "
            f"| {j['relevance']} | {j['conciseness']} | {j['accuracy']} | {j['spoiler_safety']} "
            f"| {avg} | {flag_str} | {notes} |"
        )
    lines.append("")

    # Token usage
    lines.append("## Token Usage")
    lines.append("| Stage | Input | Output |")
    lines.append("|-------|-------|--------|")
    classify_in = sum(r["state"].get("classify_tokens_in", 0) for r in results)
    classify_out = sum(r["state"].get("classify_tokens_out", 0) for r in results)
    gen_in = sum(r["state"].get("gen_tokens_in", 0) for r in results)
    gen_out = sum(r["state"].get("gen_tokens_out", 0) for r in results)
    judge_in = sum(r["judge"].get("judge_tokens_in", 0) for r in results)
    judge_out = sum(r["judge"].get("judge_tokens_out", 0) for r in results)
    lines.append(f"| Classify (Haiku) | {classify_in:,} | {classify_out:,} |")
    lines.append(f"| Generate (Sonnet) | {gen_in:,} | {gen_out:,} |")
    lines.append(f"| Judge (Haiku) | {judge_in:,} | {judge_out:,} |")
    total_in = classify_in + gen_in + judge_in
    total_out = classify_out + gen_out + judge_out
    lines.append(f"| **Total** | **{total_in:,}** | **{total_out:,}** |")
    lines.append("")

    # Retrieval statistics
    lines.append("## Retrieval Statistics")
    lines.append("| ID | Strategy | Chunks | PAST/AHEAD | Avg Score | Selected Found |")
    lines.append("|----|----------|--------|------------|-----------|----------------|")
    for r in results:
        m = r["retrieval_metrics"]
        strategy = r["state"].get("retrieval_strategy", "?")
        found = "Yes" if m["selected_text_in_chunks"] else "No"
        if not r["case"].get("selected_text"):
            found = "N/A"
        past_ahead = f"{m['past_chunk_count']}/{m['ahead_chunk_count']}"
        lines.append(
            f"| {r['case']['id']} | {strategy} | {m['num_chunks_retrieved']} "
            f"| {past_ahead} | {m['avg_score']} | {found} |"
        )
    lines.append("")

    report = "\n".join(lines)
    report_path = output_dir / "report.md"
    report_path.write_text(report)
    return report


# === Main Runner ===


async def run_case(
    case: dict,
    graph,
    anthropic: Anthropic,
) -> dict:
    """Run a single test case through the pipeline and score it."""
    case_id = case["id"]
    logger.info("Running case: %s", case_id)

    params = {
        "question": case["question"],
        "book_id": case["book_id"],
        "reader_position": case.get("reader_position", 0),
    }
    if case.get("selected_text"):
        params["selected_text"] = case["selected_text"]

    state = await graph.ainvoke(params)

    trace = _build_log_entry(state)
    retrieval_metrics = compute_retrieval_metrics(state)

    judge_scores = score_response(
        anthropic=anthropic,
        case=case,
        chunks=state.get("chunks", []),
        response=state.get("answer", ""),
    )

    diagnostics = compute_diagnostics(case, state, judge_scores, retrieval_metrics)

    return {
        "case": case,
        "state": dict(state),
        "trace": trace,
        "retrieval_metrics": retrieval_metrics,
        "judge": judge_scores,
        "diagnostics": diagnostics,
    }


class TeeOutput:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


async def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmarks")
    parser.add_argument("--ids", nargs="+", help="Run specific case IDs")
    parser.add_argument("--tags", nargs="+", help="Filter by tags")
    parser.add_argument("--book", nargs="+", dest="books", help="Filter by book ID or substring (e.g. cloud-atlas, gone-girl)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't run")
    parser.add_argument("--delay", type=float, default=1, help="Seconds between cases")
    parser.add_argument("--cases-file", type=str, help="Path to test cases JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cases_path = Path(args.cases_file) if args.cases_file else CASES_FILE
    suite = load_cases(cases_path)
    suite_name = suite.get("suite_name", "unknown")
    cases = filter_cases(suite["cases"], ids=args.ids, tags=args.tags, books=args.books)

    if not cases:
        print("No cases matched filters.")
        sys.exit(1)

    print(f"Suite: {suite_name} — {len(cases)} case(s)")

    config.validate_keys()
    rag_service = RAGService(storage_dir=config.VECTOR_STORAGE_DIR)
    anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    vector_store = rag_service.vector_store

    pipeline = build_qa_graph(
        anthropic=anthropic,
        vector_store=vector_store,
        voyage_client=rag_service.voyage,
    )
    graph = pipeline["graph"]

    book_ids = {c["book_id"] for c in cases}
    missing = [bid for bid in book_ids if not vector_store.has_book(bid)]
    if missing:
        print(f"ERROR: Books not indexed: {missing}")
        print("Index them first via the API before running benchmarks.")
        sys.exit(1)
    print(f"Books verified: {len(book_ids)} indexed")

    if args.dry_run:
        print("\nDry run — all cases valid:")
        for c in cases:
            tags = ", ".join(c.get("tags", []))
            print(f"  {c['id']:30s} book={c['book_id'][:20]}… tags=[{tags}]")
        return

    # Create output directories
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "traces.jsonl"
    detail_path = LOGS_DIR / f"details-{run_id}.log"
    console_path = LOGS_DIR / f"console-{run_id}.log"

    # Tee console output to file
    tee = TeeOutput(console_path)
    sys.stdout = tee

    print(f"Output: {output_dir}")
    print()

    # Run cases
    results = []
    detail_file = open(detail_path, "w")
    detail_file.write(f"Benchmark Run: {suite_name}\n")
    detail_file.write(f"Run ID: {run_id}\n")
    detail_file.write(f"Cases: {len(cases)}\n")
    detail_file.write(f"Started: {datetime.now().isoformat()}\n")
    detail_file.write(f"\n{'=' * W}\n\n")

    dims = ["relevance", "conciseness", "accuracy", "spoiler_safety"]

    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {case['id']}...", end=" ", flush=True)
        t0 = time.time()

        result = await run_case(case, graph, anthropic)
        results.append(result)
        elapsed = time.time() - t0

        # Append JSONL trace
        trace_entry = {
            "case_id": case["id"],
            **result["trace"],
            "retrieval_metrics": result["retrieval_metrics"],
            "judge": result["judge"],
            "diagnostics": result["diagnostics"],
        }
        with open(traces_path, "a") as f:
            f.write(json.dumps(trace_entry, default=str) + "\n")

        # Write detailed trace
        write_detailed_trace(detail_file, case, result["state"], result["judge"], result["retrieval_metrics"], result["diagnostics"], elapsed)
        detail_file.flush()

        # Console summary
        j = result["judge"]
        score_vals = [j[d] for d in dims if j.get(d, -1) >= 0]
        avg_score = round(sum(score_vals) / len(score_vals), 1) if score_vals else 0
        flag_str = ""
        if result["diagnostics"]:
            flag_str = f" [{len(result['diagnostics'])} issue{'s' if len(result['diagnostics']) != 1 else ''}]"
        print(f"avg={avg_score}/3 ({elapsed:.1f}s){flag_str}")

        if result["diagnostics"]:
            for flag in result["diagnostics"]:
                print(f"    ⚠  {flag}")

        if i < len(cases) - 1:
            time.sleep(args.delay)

    detail_file.close()

    # Generate report
    report = generate_report(suite_name, run_id, results, output_dir)

    print()
    print(f"Output:")
    print(f"  {output_dir / 'report.md'}")
    print(f"  {output_dir / 'traces.jsonl'}")
    print(f"  {detail_path}")
    print(f"  {console_path}")
    print()

    # Print summary
    print("── Summary ─────────────────────────────")
    for d in dims:
        vals = [r["judge"][d] for r in results if r["judge"].get(d, -1) >= 0]
        mean = round(sum(vals) / len(vals), 2) if vals else 0
        bar = "█" * int(mean) + "░" * (3 - int(mean))
        print(f"  {d:18s} {bar} {mean}/3")

    all_vals = []
    for r in results:
        for d in dims:
            v = r["judge"].get(d, -1)
            if v >= 0:
                all_vals.append(v)
    overall = round(sum(all_vals) / len(all_vals), 2) if all_vals else 0
    print(f"  {'overall':18s} {'█' * int(overall)}{'░' * (3 - int(overall))} {overall}/3")
    print()

    # Problem summary
    total_issues = sum(len(r["diagnostics"]) for r in results)
    clean = sum(1 for r in results if not r["diagnostics"])
    print(f"  {clean}/{len(results)} cases clean, {total_issues} issues total")
    print()

    # Restore stdout
    sys.stdout = tee.stdout
    tee.close()


if __name__ == "__main__":
    asyncio.run(main())
