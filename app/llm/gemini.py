from __future__ import annotations

from collections.abc import AsyncGenerator

import google.generativeai as genai
import structlog

from app.core.config import get_settings
from app.core.exceptions import LLMGenerationError
from app.llm.base import BaseLLMProvider

logger = structlog.get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        settings = get_settings()
        genai.configure(api_key=settings.google_api_key)
        self._model_name = model_name

    def _to_gemini_history(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini format.
        Returns (system_instruction_text, chat_history_for_send_message).
        """
        system_text = None
        history: list[dict] = []
        last_user_msg = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_text = content
            elif role == "user":
                last_user_msg = content
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        # The last user message should not be in history — it will be sent via send_message
        if history and history[-1]["role"] == "user":
            history.pop()

        return system_text, history, last_user_msg

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        try:
            system_text, history, user_msg = self._to_gemini_history(messages)

            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=system_text,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            chat = model.start_chat(history=history)
            response = await chat.send_message_async(user_msg or "")
            content = response.text
            logger.info("llm_generation_completed", provider="gemini")
            return content

        except Exception as exc:
            logger.error("llm_generation_failed", provider="gemini", error=str(exc))
            raise LLMGenerationError(str(exc)) from exc

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        try:
            system_text, history, user_msg = self._to_gemini_history(messages)

            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=system_text,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            chat = model.start_chat(history=history)
            response = await chat.send_message_async(user_msg or "", stream=True)

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            logger.error("llm_stream_failed", provider="gemini", error=str(exc))
            raise LLMGenerationError(str(exc)) from exc
