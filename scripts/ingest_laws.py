"""
Ingest core German law codes from gesetze-im-internet.de into documents2.

Usage:
    cd backend
    python -m scripts.ingest_laws                  # ingest all 7 codes
    python -m scripts.ingest_laws --laws bgb,stgb  # specific codes only
    python -m scripts.ingest_laws --dry-run         # count chunks, don't insert
    python -m scripts.ingest_laws --force            # re-ingest even if exists
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

# Ensure backend/ is on sys.path so `app.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.db.client import get_pool, close_pool  # noqa: E402
from app.llm.embeddings import embed_batch  # noqa: E402

logger = structlog.get_logger(__name__)

# ── Law definitions ──────────────────────────────────────────────────────────

LAWS: dict[str, str] = {
    "bgb": "Bürgerliches Gesetzbuch",
    "stgb": "Strafgesetzbuch",
    "hgb": "Handelsgesetzbuch",
    "gg": "Grundgesetz",
    "zpo": "Zivilprozessordnung",
    "stpo": "Strafprozeßordnung",
    "vwgo": "Verwaltungsgerichtsordnung",
}

BASE_URL = "https://www.gesetze-im-internet.de"
BATCH_SIZE = 16  # texts per embedding API call
MAX_CONCURRENT = 5  # concurrent embedding requests
SLEEP_BETWEEN = 0.7  # seconds between batches (~85 RPM)
MAX_CHUNK_CHARS = 2000
MAX_EMBED_SAFE_CHARS = 14000  # ~7000 tokens at 2 chars/token, well under 8192 limit


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


# ── Download & parse ─────────────────────────────────────────────────────────

async def download_law_xml(slug: str) -> bytes:
    """Download ZIP from gesetze-im-internet.de and return the XML bytes."""
    url = f"{BASE_URL}/{slug}/xml.zip"
    logger.info("downloading", url=url)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_names:
            raise ValueError(f"No XML found in ZIP for {slug}")
        return zf.read(xml_names[0])


def _text_of(element: ET.Element | None) -> str:
    """Recursively extract all text from an element."""
    if element is None:
        return ""
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_text_of(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts).strip()


def _find_deep(element: ET.Element, tag: str) -> ET.Element | None:
    """Find a tag anywhere in the subtree."""
    return element.find(f".//{tag}")


def _split_text(text: str, max_chars: int = MAX_EMBED_SAFE_CHARS) -> list[str]:
    """Split oversized text by sentence boundaries ('. ')."""
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    while len(text) > max_chars:
        # Find last sentence boundary before the limit
        cut = text.rfind(". ", 0, max_chars)
        if cut == -1:
            # No sentence boundary — hard cut at max
            cut = max_chars
        else:
            cut += 2  # include ". "
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def parse_norms(xml_bytes: bytes, slug: str) -> list[Chunk]:
    """Parse <norm> elements from the law XML into Chunk objects."""
    law_abbr = LAWS.get(slug, slug.upper())
    law_title = LAWS.get(slug, slug.upper())

    root = ET.fromstring(xml_bytes)
    chunks: list[Chunk] = []

    for norm in root.iter("norm"):
        meta = norm.find("metadaten")
        if meta is None:
            continue

        jurabk_el = meta.find("jurabk")
        enbez_el = meta.find("enbez")
        titel_el = meta.find("titel")

        jurabk = jurabk_el.text.strip() if jurabk_el is not None and jurabk_el.text else ""
        enbez = enbez_el.text.strip() if enbez_el is not None and enbez_el.text else ""
        titel = titel_el.text.strip() if titel_el is not None and titel_el.text else ""

        # Skip norms without a section identifier (table of contents, preamble, etc.)
        if not enbez:
            continue

        # Extract text content from <textdaten>/<text>/<Content>
        content_el = _find_deep(norm, "Content")
        if content_el is None:
            continue

        # Collect <P> elements for potential splitting
        p_elements = content_el.findall(".//P")
        if not p_elements:
            full_text = _text_of(content_el)
            if not full_text.strip():
                continue
            p_texts = [full_text]
        else:
            p_texts = [_text_of(p) for p in p_elements if _text_of(p).strip()]

        if not p_texts:
            continue

        # Build prefix for embedding quality
        prefix = f"{enbez} {jurabk}"
        if titel:
            prefix += f" - {titel}"

        base_metadata = {
            "law_abbreviation": jurabk or law_abbr,
            "law_title": law_title,
            "section_identifier": enbez,
            "section_title": titel,
            "title": jurabk or law_abbr,
            "section": f"{enbez} {titel}".strip(),
            "source": "gesetze-im-internet.de",
            "language": "de",
            "law_level": "federal",
            "legal_type": "federal_law",
            "is_legally_binding": True,
        }

        # Join all paragraphs
        full_content = "\n\n".join(p_texts)
        prefixed = f"{prefix}\n\n{full_content}"

        if len(prefixed) <= MAX_CHUNK_CHARS:
            chunks.append(Chunk(content=prefixed, metadata=dict(base_metadata)))
        else:
            # Split by paragraph boundaries, further split oversized <P>s
            current_parts: list[str] = []
            current_len = len(prefix) + 2  # prefix + \n\n

            # Flatten: split any oversized <P> by sentence boundaries
            flat_texts: list[str] = []
            for p_text in p_texts:
                if len(p_text) > MAX_EMBED_SAFE_CHARS - len(prefix) - 2:
                    flat_texts.extend(_split_text(p_text, MAX_EMBED_SAFE_CHARS - len(prefix) - 2))
                else:
                    flat_texts.append(p_text)

            for p_text in flat_texts:
                addition = len(p_text) + 2  # \n\n separator
                if current_len + addition > MAX_CHUNK_CHARS and current_parts:
                    chunk_content = f"{prefix}\n\n" + "\n\n".join(current_parts)
                    chunks.append(Chunk(content=chunk_content, metadata=dict(base_metadata)))
                    current_parts = []
                    current_len = len(prefix) + 2

                current_parts.append(p_text)
                current_len += addition

            if current_parts:
                chunk_content = f"{prefix}\n\n" + "\n\n".join(current_parts)
                chunks.append(Chunk(content=chunk_content, metadata=dict(base_metadata)))

    logger.info("parsed_norms", slug=slug, chunks=len(chunks))
    return chunks


# ── Embedding ────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~2 chars per token for German text."""
    return len(text) // 2


def _split_into_token_safe_batches(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Split chunks into batches that fit within the 8192 token limit."""
    max_batch_tokens = 7000  # leave headroom
    batches: list[list[Chunk]] = []
    current_batch: list[Chunk] = []
    current_tokens = 0

    for chunk in chunks:
        tok = _estimate_tokens(chunk.content)

        if current_tokens + tok > max_batch_tokens or len(current_batch) >= BATCH_SIZE:
            if current_batch:
                batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(chunk)
        current_tokens += tok

    if current_batch:
        batches.append(current_batch)

    return batches


async def embed_chunks(chunks: list[Chunk]) -> None:
    """Embed all chunks sequentially with rate limiting (~80 RPM)."""
    batches = _split_into_token_safe_batches(chunks)
    total = len(chunks)
    done = 0

    for batch in batches:
        texts = [c.content for c in batch]
        embeddings = await embed_batch(texts)
        for chunk, emb in zip(batch, embeddings):
            chunk.embedding = emb
        done += len(batch)
        if done % 160 == 0 or done == total:
            print(f"  Embedded {done}/{total} chunks")
        await asyncio.sleep(SLEEP_BETWEEN)


# ── Database insertion ───────────────────────────────────────────────────────

async def get_existing_sections(pool, law_abbr: str) -> set[str]:
    """Return set of section_identifiers already in DB for this law."""
    rows = await pool.fetch(
        """
        SELECT metadata->>'section_identifier' AS sec_id
        FROM documents2
        WHERE metadata->>'law_abbreviation' = $1
          AND metadata->>'section_identifier' IS NOT NULL
        """,
        law_abbr,
    )
    return {r["sec_id"] for r in rows}


async def insert_chunks(chunks: list[Chunk], force: bool = False) -> int:
    """Insert chunks into documents2. Returns number inserted."""
    pool = await get_pool()
    inserted = 0

    # Group by law abbreviation for efficient dedup checks
    by_law: dict[str, list[Chunk]] = {}
    for c in chunks:
        abbr = c.metadata.get("law_abbreviation", "")
        by_law.setdefault(abbr, []).append(c)

    for law_abbr, law_chunks in by_law.items():
        existing = set() if force else await get_existing_sections(pool, law_abbr)

        to_insert = []
        for chunk in law_chunks:
            sec_id = chunk.metadata.get("section_identifier", "")
            if sec_id in existing and not force:
                continue
            if chunk.embedding is None:
                continue
            to_insert.append(chunk)

        if not to_insert:
            continue

        # Batch insert using a single connection with longer timeout
        async with pool.acquire() as conn:
            for i, chunk in enumerate(to_insert):
                embedding_str = "[" + ",".join(str(x) for x in chunk.embedding) + "]"
                await conn.execute(
                    """
                    INSERT INTO documents2 (content, metadata, embedding)
                    VALUES ($1, $2::jsonb, $3::vector)
                    """,
                    chunk.content,
                    json.dumps(chunk.metadata),
                    embedding_str,
                    timeout=120,
                )
                inserted += 1
                if inserted % 500 == 0:
                    print(f"  Inserted {inserted} rows...")

    return inserted


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest German law codes")
    parser.add_argument(
        "--laws",
        type=str,
        default=None,
        help="Comma-separated law slugs (e.g. bgb,stgb). Default: all 7.",
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest even if exists")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count only")
    args = parser.parse_args()

    slugs = [s.strip().lower() for s in args.laws.split(",")] if args.laws else list(LAWS.keys())

    # Validate slugs
    for s in slugs:
        if s not in LAWS:
            print(f"Unknown law slug: {s}. Available: {', '.join(LAWS.keys())}")
            sys.exit(1)

    all_chunks: list[Chunk] = []

    for slug in slugs:
        print(f"\n{'='*60}")
        print(f"Processing: {LAWS[slug]} ({slug.upper()})")
        print(f"{'='*60}")

        xml_bytes = await download_law_xml(slug)
        chunks = parse_norms(xml_bytes, slug)
        print(f"  Parsed {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all laws: {len(all_chunks)}")

    if args.dry_run:
        print("\n[DRY RUN] No embeddings generated. No database changes.")
        # Show a sample
        if all_chunks:
            sample = all_chunks[0]
            print(f"\nSample chunk ({sample.metadata.get('section_identifier', '?')}):")
            print(f"  Content (first 200 chars): {sample.content[:200]}...")
            print(f"  Metadata: {json.dumps(sample.metadata, ensure_ascii=False, indent=2)}")
        return

    print("\nGenerating embeddings...")
    await embed_chunks(all_chunks)
    print(f"Embeddings generated for {len(all_chunks)} chunks")

    print("\nInserting into database...")
    inserted = await insert_chunks(all_chunks, force=args.force)
    print(f"Inserted {inserted} new chunks (skipped {len(all_chunks) - inserted} existing)")

    await close_pool()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
