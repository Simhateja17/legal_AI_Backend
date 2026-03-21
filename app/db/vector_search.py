from __future__ import annotations

import json

import structlog

from app.core.config import get_settings
from app.core.exceptions import VectorSearchError
from app.db.client import get_pool
from app.db.models import DocumentChunk

logger = structlog.get_logger(__name__)

SEARCH_QUERY = """
SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> $1::vector) AS similarity
FROM documents2
WHERE 1 - (embedding <=> $1::vector) >= $2
ORDER BY embedding <=> $1::vector
LIMIT $3;
"""


async def search_documents(
    query_embedding: list[float],
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    metadata_filter: dict | None = None,
) -> list[DocumentChunk]:
    """
    Perform cosine similarity search against documents2.

    Args:
        query_embedding: 1536-dim vector from text-embedding-3-small.
        top_k: Number of results to return.
        similarity_threshold: Minimum cosine similarity (0-1).
        metadata_filter: Optional JSONB containment filter.

    Returns:
        List of DocumentChunk ordered by descending similarity.
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    similarity_threshold = similarity_threshold or settings.similarity_threshold

    embedding_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SET hnsw.ef_search = 40;")
            if metadata_filter:
                query = """
                    SELECT
                        id, content, metadata,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM documents2
                    WHERE 1 - (embedding <=> $1::vector) >= $2
                      AND metadata @> $4::jsonb
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3;
                """
                rows = await conn.fetch(
                    query,
                    embedding_str,
                    similarity_threshold,
                    top_k,
                    json.dumps(metadata_filter),
                )
            else:
                rows = await conn.fetch(
                    SEARCH_QUERY,
                    embedding_str,
                    similarity_threshold,
                    top_k,
                )

        chunks = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            chunks.append(
                DocumentChunk(
                    id=row["id"],
                    content=row["content"],
                    metadata=meta or {},
                    similarity=float(row["similarity"]),
                )
            )

        logger.info(
            "vector_search_completed",
            results=len(chunks),
            top_k=top_k,
            threshold=similarity_threshold,
        )
        return chunks

    except VectorSearchError:
        raise
    except Exception as exc:
        logger.error("vector_search_failed", error=str(exc))
        raise VectorSearchError(str(exc)) from exc
