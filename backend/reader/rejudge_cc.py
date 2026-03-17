#!/usr/bin/env python3
"""
Rejudge readthrough traces using Claude Code CLI (`claude -p`).

Usage:
    python -m reader.rejudge_cc .data/readthrough/BOOK_ID/RUN_ID
    python -m reader.rejudge_cc .data/readthrough/BOOK_ID/RUN_ID --batch-size 5
    python -m reader.rejudge_cc .data/readthrough/BOOK_ID/RUN_ID --dry-run

Reads traces.jsonl, scores each trace via `claude -p`, writes updated
traces.jsonl, responses.jsonl, and report.md.

Progress is saved to scores_progress.jsonl so the run can resume if interrupted.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
When the response quotes text and cites a source, verify the quote actually appears in that chunk before flagging it as fabricated. Paraphrases and minor wording variations are acceptable.

**Spoiler safety** — Does the AI's RESPONSE text reveal things the reader shouldn't know yet?
- 3: Safe — answers from the reader's current perspective without revealing future information
- 2: Minor allusion to future content but nothing plot-critical
- 1: Introduces doubt or suspicion not warranted at the reader's position; or alludes to a future event
- 0: Major spoiler — explicitly reveals deaths, twists, or key plot points from later in the book
IMPORTANT: Score ONLY the AI's response text. Do NOT penalize for what the retrieved chunks contain — chunks are provided for your reference only. Chunks labeled PAST are content the reader has ALREADY READ, so referencing PAST content is NEVER a spoiler. Only penalize if the response reveals information the reader has not yet encountered or draws conclusions beyond what the text establishes at the reader's position.

Score each dimension, then return your results as a JSON array with one object per test case.
Each object MUST have these fields: index, notes, relevance_note, accuracy_note, spoiler_note, relevance, conciseness, accuracy, spoiler_safety.
Return ONLY the JSON array, no markdown fences or other text.
"""

BATCH_RUBRIC = """\
You are evaluating a reading-assistant AI that answers questions about a novel.
Score each test case below on four dimensions (0-3 each).

## Scoring rubric

**Relevance** — Does the response answer the question?
3=Directly and fully answers | 2=Answers but misses minor aspect | 1=Partially relevant | 0=Off-topic

**Conciseness** — Is it appropriately brief?
3=No fluff, well-scoped | 2=Mostly concise | 1=Noticeably verbose | 0=Rambling

**Accuracy** — Is it consistent with the retrieved chunks and expected gist?
3=Factually correct | 2=Mostly accurate | 1=Significant error | 0=Largely inaccurate
Paraphrases and minor wording variations are acceptable when quoting.

**Spoiler safety** — Does the AI's RESPONSE reveal things the reader shouldn't know yet?
3=Safe | 2=Minor allusion | 1=Introduces unwarranted doubt | 0=Major spoiler
Score ONLY the response text. PAST chunks are content the reader has already read — referencing them is NEVER a spoiler.

---

{cases}

---

Return a JSON array with one object per test case, in order.
Each object: {{"index": N, "notes": "...", "relevance_note": "...", "accuracy_note": "...", "spoiler_note": "...", "relevance": 0-3, "conciseness": 0-3, "accuracy": 0-3, "spoiler_safety": 0-3}}
Return ONLY valid JSON. No markdown fences, no extra text.
"""


def format_case(idx: int, trace: dict) -> str:
    """Format a single trace as a test case block."""
    sources = trace.get("sources", [])
    chunks_text = "\n".join(
        f"  [Chunk {i+1}] {c.get('text', '')[:300]}"
        for i, c in enumerate(sources)
    ) or "  (no chunks retrieved)"

    return (
        f"### Test case {idx}\n"
        f"- Question: {trace['question']}\n"
        f"- Selected text: \"{trace.get('selected_text', '')[:200]}\"\n"
        f"- Reader position: {trace.get('position', 0):.0%} through the book\n"
        f"- Expected gist: {trace.get('expected_answer', '')[:200]}\n"
        f"- Chunks:\n{chunks_text}\n"
        f"- AI response: {trace.get('answer', '')[:600]}\n"
    )


def call_claude(prompt: str, model: str = "haiku") -> str:
    """Call claude -p from /tmp to avoid CLAUDE.md interference."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    result = subprocess.run(
        ["claude", "-p", "--output-format", "json", "--model", model],
        input=prompt,
        capture_output=True,
        text=True,
        cwd="/tmp",
        env=env,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr[:300]}")

    output = json.loads(result.stdout)
    return output.get("result", "")


def parse_scores_array(text: str) -> list[dict]:
    """Extract a JSON array of score objects from model output."""
    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try extracting [...] block
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try extracting individual {...} blocks
    results = []
    for m in re.finditer(r'\{[^{}]+\}', text):
        try:
            results.append(json.loads(m.group()))
        except json.JSONDecodeError:
            continue
    return results


def score_to_dict(score: dict) -> dict:
    """Normalize a parsed score into the standard judge format."""
    return {
        "relevance": int(score.get("relevance", -1)),
        "conciseness": int(score.get("conciseness", -1)),
        "accuracy": int(score.get("accuracy", -1)),
        "spoiler_safety": int(score.get("spoiler_safety", -1)),
        "notes": score.get("notes", ""),
        "relevance_note": score.get("relevance_note", ""),
        "accuracy_note": score.get("accuracy_note", ""),
        "spoiler_note": score.get("spoiler_note", ""),
        "judge_tokens_in": 0,
        "judge_tokens_out": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Rejudge readthrough traces via Claude Code CLI")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--batch-size", type=int, default=10, help="Traces per batch (default 10)")
    parser.add_argument("--model", default="haiku", help="Model for judging (default haiku)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    run_dir = args.run_dir
    traces_path = run_dir / "traces.jsonl"
    progress_path = run_dir / "scores_progress.jsonl"

    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found")
        sys.exit(1)

    # Load traces
    traces = []
    for line in traces_path.read_text().strip().split("\n"):
        if line.strip():
            traces.append(json.loads(line))

    # Load progress (already-scored indices)
    scored: dict[int, dict] = {}
    if progress_path.exists():
        for line in progress_path.read_text().strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                scored[entry["trace_index"]] = entry["judge"]

    remaining = [i for i in range(len(traces)) if i not in scored]
    print(f"Traces: {len(traces)} total, {len(scored)} already scored, {len(remaining)} remaining")

    if args.dry_run:
        print(f"Would score {len(remaining)} traces in batches of {args.batch_size}")
        return

    # Process in batches
    dims = ["relevance", "conciseness", "accuracy", "spoiler_safety"]
    batch_num = 0
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_indices = remaining[batch_start:batch_start + args.batch_size]
        batch_num += 1
        batch_total = (len(remaining) + args.batch_size - 1) // args.batch_size

        cases_text = "\n".join(
            format_case(i, traces[idx]) for i, idx in enumerate(batch_indices)
        )
        prompt = BATCH_RUBRIC.format(cases=cases_text)

        print(f"Batch {batch_num}/{batch_total} ({len(batch_indices)} traces)...", end=" ", flush=True)
        t0 = time.time()

        try:
            raw = call_claude(prompt, model=args.model)
            scores = parse_scores_array(raw)
        except Exception as e:
            print(f"FAILED: {e}")
            # Try individual fallback for this batch
            scores = []
            for i, idx in enumerate(batch_indices):
                case_text = format_case(0, traces[idx])
                single_prompt = BATCH_RUBRIC.format(cases=case_text)
                try:
                    raw = call_claude(single_prompt, model=args.model)
                    s = parse_scores_array(raw)
                    scores.extend(s[:1])
                except Exception as e2:
                    print(f"  trace {idx} failed: {e2}")
                    scores.append({})

        elapsed = time.time() - t0

        # Map scores back to trace indices
        for i, idx in enumerate(batch_indices):
            if i < len(scores):
                judge = score_to_dict(scores[i])
            else:
                judge = score_to_dict({})  # fallback: -1 scores

            scored[idx] = judge
            traces[idx]["judge"] = judge

            # Save progress incrementally
            with open(progress_path, "a") as f:
                f.write(json.dumps({"trace_index": idx, "judge": judge}) + "\n")

        # Print batch summary
        batch_scores = []
        for idx in batch_indices:
            valid = [scored[idx][d] for d in dims if scored[idx].get(d, -1) >= 0]
            if valid:
                batch_scores.append(sum(valid) / len(valid))
        avg = sum(batch_scores) / len(batch_scores) if batch_scores else 0
        print(f"{avg:.1f}/3 avg ({elapsed:.1f}s)")

    # Rewrite traces.jsonl
    with open(traces_path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace, default=str) + "\n")

    # Rewrite responses.jsonl
    responses_path = run_dir / "responses.jsonl"
    with open(responses_path, "w") as f:
        for trace in traces:
            f.write(json.dumps({
                "stop_index": trace["stop_index"],
                "position": trace["position"],
                "question": trace["question"],
                "answer": trace.get("answer", ""),
                "answer_mode": trace.get("answer_mode", ""),
                "sources": trace.get("sources", []),
                "judge": trace["judge"],
                "reflection": trace.get("reflection", {}),
            }, default=str) + "\n")

    # Regenerate report
    from reader.reader import generate_readthrough_report, ReadthroughConfig
    from reader.mind import ReaderMind

    # Reconstruct stop_summaries
    stop_data: dict[int, dict] = {}
    for trace in traces:
        si = trace["stop_index"]
        if si not in stop_data:
            stop_data[si] = {
                "stop_index": si,
                "label": trace.get("stop_label", ""),
                "position": trace.get("position", 0),
                "questions": 0,
                "follow_ups": 0,
                "scores": [],
                "elapsed": 0,
                "tokens_in": 0,
                "tokens_out": 0,
            }
        sd = stop_data[si]
        if trace.get("is_follow_up"):
            sd["follow_ups"] += 1
        else:
            sd["questions"] += 1
        valid = [trace["judge"][d] for d in dims if trace["judge"].get(d, -1) >= 0]
        if valid:
            sd["scores"].append(sum(valid) / len(valid))

    stop_summaries = []
    for si in sorted(stop_data.keys()):
        sd = stop_data[si]
        scores = sd.pop("scores")
        sd["avg_score"] = round(sum(scores) / len(scores), 2) if scores else 0
        stop_summaries.append(sd)

    # Restore mind
    mind = ReaderMind()
    mind_path = run_dir / "mind.jsonl"
    if mind_path.exists():
        for line in mind_path.read_text().strip().split("\n"):
            if line.strip():
                entry = json.loads(line)
                if "mind" in entry:
                    mind = ReaderMind.from_dict(entry["mind"])

    # Detect book_id from path (run_dir = .data/readthrough/BOOK_ID/RUN_ID)
    book_id = run_dir.parent.name
    run_id = run_dir.name

    cfg = ReadthroughConfig(book_id=book_id)
    report = generate_readthrough_report(
        cfg=cfg, run_id=run_id, stops=[], all_results=traces,
        stop_summaries=stop_summaries, mind=mind, journal=[], output_dir=run_dir,
    )
    report_path = run_dir / "report.md"
    report_path.write_text(report)

    # Clean up progress file
    progress_path.unlink(missing_ok=True)

    # Summary
    all_valid = []
    for trace in traces:
        valid = [trace["judge"][d] for d in dims if trace["judge"].get(d, -1) >= 0]
        if valid:
            all_valid.append(sum(valid) / len(valid))
    overall = round(sum(all_valid) / len(all_valid), 2) if all_valid else 0
    print(f"\nDone. Overall: {overall}/3.00")
    print(f"Updated: {traces_path}")
    print(f"Updated: {responses_path}")
    print(f"Updated: {report_path}")


if __name__ == "__main__":
    main()
