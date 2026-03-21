"""
Deduplicate documents in the documents2 table.

Two strategies run sequentially:
1. Metadata-based: Remove rows with duplicate law_abbreviation + section_identifier,
   keeping the row with the highest quality_score (or newest if no score).
2. Content-based: Remove rows with identical md5(content), keeping the newest.

Usage:
    cd backend
    python -m scripts.dedup_documents --dry-run   # report only
    python -m scripts.dedup_documents              # actually delete
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.client import get_pool, close_pool  # noqa: E402

logger = structlog.get_logger(__name__)

BATCH_DELETE_SIZE = 500


async def _batched_delete(pool, ids: list) -> int:
    """Delete IDs in batches to avoid statement timeout."""
    deleted = 0
    for i in range(0, len(ids), BATCH_DELETE_SIZE):
        batch = ids[i : i + BATCH_DELETE_SIZE]
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM documents2 WHERE id = ANY($1::uuid[])",
                batch,
                timeout=300,
            )
        deleted += len(batch)
        print(f"    Deleted {deleted}/{len(ids)}...")
    return deleted


async def dedup_metadata(pool, dry_run: bool) -> int:
    """
    Remove duplicate rows sharing the same law_abbreviation + section_identifier.
    Keeps the row with the highest quality_score, or the newest (by id) if tied.
    """
    query = """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        metadata->>'law_abbreviation',
                        metadata->>'section_identifier'
                    ORDER BY
                        COALESCE((metadata->>'quality_score')::float, 0) DESC,
                        id DESC
                ) AS rn
            FROM documents2
            WHERE metadata->>'law_abbreviation' IS NOT NULL
              AND metadata->>'section_identifier' IS NOT NULL
        )
        SELECT id FROM ranked WHERE rn > 1;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, timeout=300)
    count = len(rows)

    if count == 0:
        print("  Metadata dedup: no duplicates found.")
        return 0

    if dry_run:
        print(f"  Metadata dedup: would remove {count} duplicate rows.")
        return count

    ids = [r["id"] for r in rows]
    await _batched_delete(pool, ids)
    print(f"  Metadata dedup: removed {count} duplicate rows.")
    return count


async def dedup_content(pool, dry_run: bool) -> int:
    """
    Remove rows with identical content (by md5 hash), keeping the newest.
    """
    query = """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY md5(content)
                    ORDER BY id DESC
                ) AS rn
            FROM documents2
        )
        SELECT id FROM ranked WHERE rn > 1;
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, timeout=300)
    count = len(rows)

    if count == 0:
        print("  Content dedup: no duplicates found.")
        return 0

    if dry_run:
        print(f"  Content dedup: would remove {count} duplicate rows.")
        return count

    ids = [r["id"] for r in rows]
    await _batched_delete(pool, ids)
    print(f"  Content dedup: removed {count} duplicate rows.")
    return count


async def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate documents2 table")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't delete")
    args = parser.parse_args()

    pool = await get_pool()

    # Count before
    total_before = await pool.fetchval("SELECT COUNT(*) FROM documents2")
    print(f"Total documents before: {total_before}")

    if args.dry_run:
        print("\n[DRY RUN MODE]")

    print("\nStep 1: Metadata-based dedup (law_abbreviation + section_identifier)")
    meta_removed = await dedup_metadata(pool, args.dry_run)

    print("\nStep 2: Content-based dedup (md5 hash)")
    content_removed = await dedup_content(pool, args.dry_run)

    total_removed = meta_removed + content_removed

    if args.dry_run:
        print(f"\n[DRY RUN] Would remove up to {total_removed} rows total.")
        print("(Actual count may be lower due to overlap between strategies.)")
    else:
        total_after = await pool.fetchval("SELECT COUNT(*) FROM documents2")
        print(f"\nTotal documents after: {total_after}")
        print(f"Removed: {total_before - total_after}")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
