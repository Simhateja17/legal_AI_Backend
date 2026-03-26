from __future__ import annotations

from cachetools import TTLCache
from google import genai
from google.genai import types
import structlog

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)

_embedding_cache: TTLCache[str, list[float]] = TTLCache(maxsize=256, ttl=3600)

_EMBEDDING_MODEL = "gemini-embedding-001"
_OUTPUT_DIM = 1536  # matches existing DB vector columns


def _get_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)


async def embed_query(text: str) -> list[float]:
    cached = _embedding_cache.get(text)
    if cached is not None:
        logger.info("embedding_cache_hit", dimensions=len(cached))
        return cached

    client = _get_client()
    try:
        result = client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=_OUTPUT_DIM,
            ),
        )
        embedding: list[float] = result.embeddings[0].values
        _embedding_cache[text] = embedding
        logger.info("embedding_generated", dimensions=len(embedding), provider="gemini")
        return embedding
    except Exception as exc:
        logger.error("embedding_failed", error=str(exc), provider="gemini")
        raise EmbeddingError(str(exc)) from exc


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    client = _get_client()
    try:
        result = client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=_OUTPUT_DIM,
            ),
        )
        embeddings: list[list[float]] = [e.values for e in result.embeddings]
        logger.info("batch_embedding_generated", count=len(embeddings), provider="gemini")
        return embeddings
    except Exception as exc:
        logger.error("batch_embedding_failed", error=str(exc), provider="gemini")
        raise EmbeddingError(str(exc)) from exc
