#!/usr/bin/env python3
"""
Migrate legacy JSON vector store files to Qdrant.

Reads .data/vectors/*.json and upserts into Qdrant collections.
Old files are preserved (not deleted) for easy rollback.

Usage:
    python -m scripts.migrate_to_qdrant
    # or
    python scripts/migrate_to_qdrant.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add backend to path if running as standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from badger.core.vector_store import VectorStore, VectorEntry
from badger.core.chunker import TextChunk
from badger import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def deserialize_entry(data: dict) -> VectorEntry:
    """Deserialize a legacy JSON entry to VectorEntry."""
    chunk = TextChunk(
        id=data["chunk"]["id"],
        text=data["chunk"]["text"],
        metadata=data["chunk"]["metadata"],
    )
    return VectorEntry(chunk=chunk, embedding=data["embedding"])


def load_legacy_file(path: Path) -> tuple[str, list[VectorEntry]]:
    """Load a legacy JSON vector file. Returns (book_id, entries)."""
    with open(path) as f:
        data = json.load(f)
    book_id = data["book_id"]
    entries = [deserialize_entry(e) for e in data["entries"]]
    return book_id, entries


async def migrate():
    json_dir = Path(config.VECTOR_STORAGE_DIR)
    if not json_dir.exists():
        logger.info("No legacy vector directory found at %s — nothing to migrate", json_dir)
        return

    # Find all chunk files (exclude summaries)
    chunk_files = sorted(
        p for p in json_dir.glob("*.json")
        if not p.name.endswith("_summaries.json")
    )
    if not chunk_files:
        logger.info("No legacy JSON files found in %s", json_dir)
        return

    logger.info("Found %d legacy vector files in %s", len(chunk_files), json_dir)

    store = VectorStore(
        storage_dir=config.QDRANT_STORAGE_DIR,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    await store.initialize()

    migrated = 0
    for path in chunk_files:
        try:
            book_id, entries = load_legacy_file(path)
            if store.has_book(book_id):
                logger.info("Skipping %s — already in Qdrant", book_id)
                continue
            await store.add_book(book_id, entries)
            logger.info("Migrated %s: %d chunks from %s", book_id, len(entries), path.name)

            # Check for corresponding summaries file
            summaries_path = json_dir / f"{path.stem}_summaries.json"
            if summaries_path.exists():
                _, summary_entries = load_legacy_file(summaries_path)
                await store.save_summaries(book_id, summary_entries)
                logger.info("Migrated %d summaries for %s", len(summary_entries), book_id)

            migrated += 1
        except Exception as e:
            logger.error("Failed to migrate %s: %s", path.name, e)

    logger.info("Migration complete: %d/%d books migrated", migrated, len(chunk_files))
    logger.info("Legacy files preserved in %s (safe to delete manually after verification)", json_dir)


if __name__ == "__main__":
    asyncio.run(migrate())
