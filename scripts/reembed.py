"""
Re-embed all documents in documents2 using Gemini embedding model.

Fetches in pages (cursor-based) to avoid hanging on large tables.
Updates embeddings in-place — no re-download/re-parse needed.

Usage:
    python -m scripts.reembed [--batch-size 16] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

import asyncpg
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIM = 1536
BATCH_SIZE = 100       # texts per Gemini API call (Gemini supports up to 100)
PAGE_SIZE = 500        # rows fetched per DB page
SLEEP_BETWEEN = 0.2    # seconds between API calls
MAX_CHARS = 14000      # truncate content beyond this for safety


def load_env():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                if "#" in val:
                    val = val[:val.index("#")].strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)


def embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=OUTPUT_DIM,
        ),
    )
    return [e.values for e in result.embeddings]


async def main(batch_size: int, dry_run: bool, skip: int = 0):
    load_env()
    db_url = os.environ["DATABASE_URL"]
    api_key = os.environ["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)

    # Quick API key test
    print("Testing Gemini API key...")
    try:
        embed_batch(client, ["test"])
        print("  API key OK")
    except Exception as e:
        print(f"  API key FAILED: {e}")
        return

    print("Connecting to database...")
    conn = await asyncpg.connect(db_url, timeout=30)

    total = await conn.fetchval("SELECT count(*) FROM documents2")
    print(f"Total documents: {total}")

    if dry_run:
        print("[DRY RUN] Would re-embed all documents. Exiting.")
        await conn.close()
        return

    # If resuming, skip ahead
    last_id = "00000000-0000-0000-0000-000000000000"
    if skip > 0:
        print(f"Skipping first {skip} rows...")
        row = await conn.fetchrow(
            "SELECT id FROM documents2 ORDER BY id OFFSET $1 LIMIT 1",
            skip - 1,
            timeout=120,
        )
        if row:
            last_id = row["id"]
            print(f"  Resuming from ID {last_id}")

    # Process with cursor-based pagination (no huge SELECT)
    updated = skip
    errors = 0
    t0 = time.time()

    while True:
        # Fetch next page
        rows = await conn.fetch(
            "SELECT id, content FROM documents2 WHERE id > $1 ORDER BY id LIMIT $2",
            last_id,
            PAGE_SIZE,
            timeout=120,
        )
        if not rows:
            break

        last_id = rows[-1]["id"]

        # Process this page in embedding batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            ids = [r["id"] for r in batch]
            texts = [r["content"][:MAX_CHARS] for r in batch]

            try:
                embeddings = embed_batch(client, texts)
            except Exception as e:
                print(f"\n  ERROR at doc {updated}: {e}")
                errors += len(batch)
                time.sleep(2)
                continue

            # Bulk UPDATE — single round-trip for entire batch
            # Build VALUES list for a single UPDATE FROM statement
            values_parts = []
            params = []
            for j, (row_id, emb) in enumerate(zip(ids, embeddings)):
                emb_str = "[" + ",".join(str(v) for v in emb) + "]"
                idx = j * 2
                values_parts.append(f"(${idx+1}::uuid, ${idx+2}::vector)")
                params.extend([row_id, emb_str])

            sql = (
                "UPDATE documents2 AS d SET embedding = v.emb "
                "FROM (VALUES " + ",".join(values_parts) + ") AS v(id, emb) "
                "WHERE d.id = v.id"
            )
            await conn.execute(sql, *params, timeout=120)
            updated += len(batch)

            elapsed = time.time() - t0
            rate = updated / elapsed if elapsed > 0 else 0
            eta = (total - updated) / rate if rate > 0 else 0
            print(
                f"  [{updated}/{total}] "
                f"({updated/total*100:.1f}%) "
                f"{rate:.1f} docs/s "
                f"ETA={eta/60:.0f}min "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

            time.sleep(SLEEP_BETWEEN)

    elapsed = time.time() - t0
    print(f"\nDone! Updated: {updated}, Errors: {errors}, Time: {elapsed:.0f}s")
    await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N rows (for resuming)")
    args = parser.parse_args()
    asyncio.run(main(args.batch_size, args.dry_run, args.skip))
