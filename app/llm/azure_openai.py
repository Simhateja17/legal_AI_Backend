from __future__ import annotations

from collections.abc import AsyncGenerator

from openai import AsyncAzureOpenAI
import structlog

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class AzureOpenAIProvider(BaseLLMProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        self._model = settings.azure_openai_chat_deployment

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.info(
                "llm_generation_completed",
                provider="azure",
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return content

        except Exception as exc:
            logger.error("llm_generation_failed", provider="azure", error=str(exc))
            raise LLMGenerationError(str(exc)) from exc

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as exc:
            logger.error("llm_stream_failed", provider="azure", error=str(exc))
            raise LLMGenerationError(str(exc)) from exc
