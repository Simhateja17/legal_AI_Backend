from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import structlog

from app.core.config import get_settings
from app.db.models import ConversationMessage, DocumentChunk
from app.db.vector_search import search_documents
from app.llm.base import BaseLLMProvider
from app.llm.embeddings import embed_query
from app.rag.context import assemble_context
from app.llm.gemini import GeminiProvider
from app.rag.prompts import (
    GUARD_SYSTEM_PROMPT, GUARD_USER_TEMPLATE,
    MODE_PROMPTS, _GUARD_MODE_NOTES, SYSTEM_PROMPT,
)

logger = structlog.get_logger(__name__)


def _fix_markdown(text: str) -> str:
    """Fix malformed markdown patterns in LLM output."""
    # Fix bold markers with inner spaces (handles multi-line): "** text **" → "**text**"
    text = re.sub(r"\*\*\s+([\s\S]+?)\s+\*\*", r"**\1**", text)
    # Fix broken list: digit-dot on its own line, bold label on next line
    # "1.\n** Label **: content" → "1. **Label**: content"
    text = re.sub(r"^(\d+)\.\s*\n\*\*(.+?)\*\*(\s*:?)", r"\1. **\2**\3", text, flags=re.MULTILINE)
    # Fix numbered list items with extra spaces: "1 ." → "1."
    text = re.sub(r"(\d+)\s+\.", r"\1.", text)
    return text


@dataclass
class RAGResult:
    answer: str
    sources: list[DocumentChunk]
    query_used: str


class RAGPipeline:
    """Core orchestrator: embed → retrieve → generate."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider
        settings = get_settings()
        self._guard_enabled = settings.enable_guard
        if self._guard_enabled:
            self._guard = GeminiProvider(model_name="gemini-3-flash-preview")

    async def _retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[list[DocumentChunk], list[float]]:
        """Embed the query and retrieve relevant chunks."""
        embedding = await embed_query(query)
        chunks = await search_documents(
            query_embedding=embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter,
        )
        return chunks, embedding

    def _build_messages(
        self,
        context: str,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        mode: str = "normal",
    ) -> list[dict[str, str]]:
        """Assemble the message array for the LLM."""
        settings = get_settings()
        system_prompt = MODE_PROMPTS.get(mode, SYSTEM_PROMPT)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        if context:
            messages.append({"role": "user", "content": context})
            messages.append(
                {
                    "role": "assistant",
                    "content": "Ich habe die Quellen gelesen und werde meine Antwort darauf stützen.",
                }
            )

        if conversation_history:
            turns = conversation_history[-settings.max_conversation_turns :]
            for msg in turns:
                messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": query})
        return messages

    async def _run_guard(self, query: str, context: str, draft: str, mode: str = "normal") -> str:
        """Run the Gemini guard to fact-check and refine the draft answer."""
        logger.info("guard_started", mode=mode, draft_chars=len(draft))
        user_content = GUARD_USER_TEMPLATE.format(
            query=query,
            context=context if context else "Keine Quelldokumente verfügbar.",
            draft=draft if draft else "Keine Entwurfsantwort vorhanden.",
            mode_instruction=_GUARD_MODE_NOTES.get(mode, ""),
        )
        messages = [
            {"role": "system", "content": GUARD_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        result = await self._guard.generate(messages)
        logger.info("guard_completed", result_chars=len(result))
        return result

    async def run(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
        mode: str = "normal",
    ) -> RAGResult:
        """Full RAG pipeline: embed → retrieve → generate (non-streaming)."""
        settings = get_settings()
        logger.info(
            "rag_pipeline_started",
            query=query[:100],
            mode=mode,
            top_k=top_k or settings.retrieval_top_k,
            similarity_threshold=similarity_threshold or settings.similarity_threshold,
            history_turns=len(conversation_history) if conversation_history else 0,
            guard_enabled=self._guard_enabled,
        )

        chunks, _ = await self._retrieve(
            query, top_k, similarity_threshold, metadata_filter
        )
        context = assemble_context(chunks)
        logger.info(
            "retrieval_done",
            chunks_found=len(chunks),
            top_similarity=round(chunks[0].similarity, 3) if chunks else None,
            min_similarity=round(chunks[-1].similarity, 3) if chunks else None,
        )

        if chunks:
            messages = self._build_messages(context, query, conversation_history, mode)
            logger.info("llm_generate_started", message_count=len(messages))
            raw_answer = await self._llm.generate(messages)
            raw_answer = _fix_markdown(raw_answer)
            logger.info("llm_generate_done", answer_chars=len(raw_answer))
        else:
            logger.warning("no_chunks_found", query=query[:100])
            raw_answer = ""  # No sources — let guard answer from knowledge

        if self._guard_enabled:
            answer = await self._run_guard(query, context, raw_answer, mode)
        else:
            answer = raw_answer

        logger.info("rag_pipeline_completed", sources=len(chunks), answer_chars=len(answer))
        return RAGResult(answer=answer, sources=chunks, query_used=query)

    async def run_stream(
        self,
        query: str,
        conversation_history: list[ConversationMessage] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
        mode: str = "normal",
    ) -> tuple[AsyncGenerator[str, None], list[DocumentChunk]]:
        """
        Streaming RAG pipeline. Returns:
          - An async generator of answer tokens
          - The list of source chunks (available immediately after retrieval)
        """
        settings = get_settings()
        logger.info(
            "rag_stream_started",
            query=query[:100],
            mode=mode,
            top_k=top_k or settings.retrieval_top_k,
            similarity_threshold=similarity_threshold or settings.similarity_threshold,
            history_turns=len(conversation_history) if conversation_history else 0,
            guard_enabled=self._guard_enabled,
        )

        chunks, _ = await self._retrieve(
            query, top_k, similarity_threshold, metadata_filter
        )
        context = assemble_context(chunks)
        logger.info(
            "retrieval_done",
            chunks_found=len(chunks),
            top_similarity=round(chunks[0].similarity, 3) if chunks else None,
            min_similarity=round(chunks[-1].similarity, 3) if chunks else None,
        )

        if chunks:
            messages = self._build_messages(context, query, conversation_history, mode)
            logger.info("llm_stream_started", message_count=len(messages))
            # Collect full primary stream internally before guard
            tokens: list[str] = []
            async for token in self._llm.generate_stream(messages):
                tokens.append(token)
            raw_answer = _fix_markdown("".join(tokens))
            logger.info("llm_stream_done", answer_chars=len(raw_answer))
        else:
            logger.warning("no_chunks_found", query=query[:100])
            raw_answer = ""

        if self._guard_enabled:
            logger.info("guard_stream_started", mode=mode, draft_chars=len(raw_answer))
            user_content = GUARD_USER_TEMPLATE.format(
                query=query,
                context=context if context else "Keine Quelldokumente verfügbar.",
                draft=raw_answer if raw_answer else "Keine Entwurfsantwort vorhanden.",
                mode_instruction=_GUARD_MODE_NOTES.get(mode, ""),
            )
            guard_messages = [
                {"role": "system", "content": GUARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            token_stream = self._guard.generate_stream(guard_messages)
        else:
            async def _single(text: str) -> AsyncGenerator[str, None]:
                yield text
            token_stream = _single(raw_answer)

        return token_stream, chunks

    async def search_only(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> list[DocumentChunk]:
        """Vector search without LLM generation (for debugging retrieval)."""
        chunks, _ = await self._retrieve(
            query, top_k, similarity_threshold, metadata_filter
        )
        return chunks
