"""
CLI entry point for the readthrough agent.

Usage:
    python -m reader.run --book-id babel
    python -m reader.run --book-id babel --stops pct:10 --max-questions 3
    python -m reader.run --book-id babel --dry-run
    python -m reader.run --book-id babel --start-at 0.5 --skip-judge
    python -m reader.run --book-id babel --rejudge 20260316-140707
    python -m reader.run --book-id babel --rejudge 20260316-140707 --rejudge-api
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic, AsyncAnthropic

from badger import config
from badger.core.rag import RAGService
from badger.core.agent import build_agent
from benchmarks.run import BOOK_ALIASES, TeeOutput
from benchmarks.judge import set_cache_enabled, flush_judge_cache

from reader.reader import ReadthroughConfig, run_readthrough, rejudge_run


def main():
    parser = argparse.ArgumentParser(
        description="Book Readthrough Agent — autonomous RAG evaluation",
    )
    parser.add_argument(
        "--book-id", required=True,
        help="Book to read (supports aliases: babel, cloud-atlas, gone-girl)",
    )
    parser.add_argument(
        "--stops", default="chapter",
        help='Stop strategy: "chapter" (default), "pct:N", "chunks:N"',
    )
    parser.add_argument(
        "--max-questions", type=int, default=5,
        help="Max questions per stop point (default 5)",
    )
    parser.add_argument(
        "--max-follow-ups", type=int, default=1,
        help="Max follow-up questions per original question (default 1)",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judge scoring",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass caches",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between API calls (default 1.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show stop points without running",
    )
    parser.add_argument(
        "--start-at", type=float, default=0.0,
        help="Resume from position 0-1 (default 0)",
    )
    parser.add_argument(
        "--think-model", default="",
        help=f"Model for REACT/THINK steps (default: {config.CLAUDE_MODEL})",
    )
    parser.add_argument(
        "--question-model", default="",
        help=f"Model for question generation (default: {config.CLAUDE_MODEL})",
    )
    parser.add_argument(
        "--reflect-model", default="",
        help=f"Model for reflection (default: {config.CLAUDE_HAIKU_MODEL})",
    )
    parser.add_argument(
        "--rejudge", metavar="RUN_ID",
        help="Re-score an existing run (e.g. 20260316-140707). Uses Claude Code CLI by default.",
    )
    parser.add_argument(
        "--rejudge-api", action="store_true",
        help="Use direct API calls for rejudge instead of Claude Code CLI",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve book alias
    book_id = BOOK_ALIASES.get(args.book_id, args.book_id)

    # Validate API keys
    config.validate_keys()

    # Configure caches
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # --- Rejudge mode ---
    if args.rejudge:
        if args.rejudge_api:
            # Direct API calls (needs ANTHROPIC_API_KEY with credits)
            anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            try:
                rejudge_run(book_id, args.rejudge, anthropic)
            finally:
                flush_judge_cache()
        else:
            # Claude Code CLI (default — uses Claude Code subscription)
            import subprocess
            run_dir = Path(f".data/readthrough/{book_id}/{args.rejudge}")
            if not (run_dir / "traces.jsonl").exists():
                print(f"ERROR: No traces found at {run_dir}/traces.jsonl")
                sys.exit(1)
            subprocess.run(
                [sys.executable, "-m", "reader.rejudge_cc", str(run_dir)],
                cwd=Path(__file__).resolve().parent.parent,
            )
        return

    # Initialize services
    rag_service = RAGService()
    anthropic = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    async_anthropic = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    vector_store = rag_service.vector_store
    voyage_client = rag_service.voyage

    # Verify book is indexed
    if not vector_store.has_book(book_id):
        print(f"ERROR: Book not indexed: {book_id}")
        print("Index it first via the API before running the readthrough.")
        sys.exit(1)

    # Build agent
    agent = build_agent(
        anthropic=anthropic,
        async_anthropic=async_anthropic,
        vector_store=vector_store,
        voyage_client=voyage_client,
    )

    # Build config
    cfg = ReadthroughConfig(
        book_id=book_id,
        stop_strategy=args.stops,
        max_questions_per_stop=args.max_questions,
        max_follow_ups=args.max_follow_ups,
        skip_judge=args.skip_judge,
        delay=args.delay,
        dry_run=args.dry_run,
        start_at=args.start_at,
        think_model=args.think_model or config.CLAUDE_MODEL,
        question_model=args.question_model or config.CLAUDE_MODEL,
        reflect_model=args.reflect_model or config.CLAUDE_HAIKU_MODEL,
        no_cache=args.no_cache,
    )

    # Set up console tee logging (unless dry-run)
    tee = None
    if not args.dry_run:
        log_dir = Path(".data/readthrough") / book_id
        log_dir.mkdir(parents=True, exist_ok=True)
        console_log = log_dir / "console.log"
        tee = TeeOutput(console_log)
        sys.stdout = tee

    try:
        print(f"Book Readthrough Agent")
        print(f"Book: {book_id} (alias: {args.book_id})")
        print(f"Strategy: {cfg.stop_strategy}")
        print(f"Models: think={cfg.think_model}, question={cfg.question_model}, reflect={cfg.reflect_model}")
        print()

        asyncio.run(_run(cfg, anthropic, vector_store, voyage_client, agent))
    finally:
        flush_judge_cache()
        if tee:
            sys.stdout = tee.stdout
            tee.close()


async def _run(cfg, anthropic, vector_store, voyage_client, agent):
    """Async wrapper for the readthrough."""
    await run_readthrough(
        cfg=cfg,
        anthropic=anthropic,
        vector_store=vector_store,
        voyage_client=voyage_client,
        agent=agent,
    )


if __name__ == "__main__":
    main()
