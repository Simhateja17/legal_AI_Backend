from __future__ import annotations

from cachetools import TTLCache
from openai import AsyncAzureOpenAI
import structlog

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)

_client: AsyncAzureOpenAI | None = None
_embedding_cache: TTLCache[str, list[float]] = TTLCache(maxsize=256, ttl=3600)


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
    return _client


async def embed_query(text: str) -> list[float]:
    """
    Generate a 1536-dim embedding for a query string using
    Azure OpenAI text-embedding-3-small. Results cached for 1 hour.
    """
    cached = _embedding_cache.get(text)
    if cached is not None:
        logger.info("embedding_cache_hit", dimensions=len(cached))
        return cached

    settings = get_settings()
    client = _get_client()

    try:
        response = await client.embeddings.create(
            input=text,
            model=settings.azure_openai_embedding_deployment,
        )
        embedding = response.data[0].embedding
        _embedding_cache[text] = embedding
        logger.info("embedding_generated", dimensions=len(embedding))
        return embedding

    except Exception as exc:
        logger.error("embedding_failed", error=str(exc))
        raise EmbeddingError(str(exc)) from exc


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts (up to 16) in a single API call.
    Returns list of 1536-dim embeddings in the same order as input.
    """
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()

    try:
        response = await client.embeddings.create(
            input=texts,
            model=settings.azure_openai_embedding_deployment,
        )
        # API may return data out of order; sort by index
        sorted_data = sorted(response.data, key=lambda d: d.index)
        embeddings = [d.embedding for d in sorted_data]
        logger.info("batch_embedding_generated", count=len(embeddings))
        return embeddings

    except Exception as exc:
        logger.error("batch_embedding_failed", error=str(exc))
        raise EmbeddingError(str(exc)) from exc
