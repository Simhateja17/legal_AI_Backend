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


# Common German words (2-8 chars) that should NOT be merged by OCR fix.
# Includes articles, pronouns, prepositions, conjunctions, common verbs/adverbs.
_DE_COMMON_WORDS = frozenset(
    "ab alle als also am an auch auf aus bei beim bis bitte da dabei damit dann"
    " das dass dem den denn der des die dies doch dort du durch eben ein eine"
    " einem einen einer er erst es etwa etwas falls fest für ganz gar gegen"
    " gemacht gibt groß gut hab hat hatte hin ich ihm ihn ihr immer in ins"
    " ist ja je jede jedem jeden jeder jedes jedoch jetzt kann kein keine"
    " keinem keinen keiner kommt kurz lang laut machen macht man mehr mein"
    " mir mit muss nach neu nicht noch nun nur ob oder oft ohne pro rund"
    " sehr sei seit sich sie sind so sogar statt tun tut um und uns unter"
    " viel vom von vor wann war warum was weil weit wem wen wenn wer werde"
    " wie wir wird wo wohl zu zum zur zwei".split()
)


def _fix_ocr_splits(text: str) -> str:
    """Remove erroneous spaces within words caused by OCR artifacts in LLM output.

    Catches patterns like 'verfol gen', 'finanz ielle', 'Ohrfe ige' where a
    lowercase word fragment follows a space inside what should be one word.
    """
    from app.rag.context import _LEGAL_ABBREV_FIXES

    # Fix known legal abbreviation splits (B GB → BGB, etc.)
    for pattern, replacement in _LEGAL_ABBREV_FIXES:
        text = pattern.sub(replacement, text)

    # Fix space before punctuation: "Personen ," → "Personen,"
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)

    # Fix split words by iterating through word pairs.
    # Merge adjacent words when: both fragments end/start lowercase,
    # neither is a common German word, and the right fragment is short (≤7).
    # Run 2 passes for chains like "Rechts anw alt".
    for _ in range(2):
        words = text.split(" ")
        merged: list[str] = []
        i = 0
        while i < len(words):
            if i + 1 < len(words):
                w1 = words[i]
                w2 = words[i + 1]
                # Check if this looks like an OCR split:
                # w1 ends with a lowercase letter, w2 starts lowercase,
                # w2 is short, and neither is a common standalone word
                if (
                    len(w1) >= 2
                    and len(w2) <= 7
                    and w1[-1:].isalpha() and w1[-1:].lower() == w1[-1:]
                    and w2[:1].isalpha() and w2[:1].lower() == w2[:1]
                    and w1.lower() not in _DE_COMMON_WORDS
                    and w2.lower() not in _DE_COMMON_WORDS
                ):
                    merged.append(w1 + w2)
                    i += 2
                    continue
            merged.append(words[i])
            i += 1
        text = " ".join(merged)

    return text


@dataclass
class RAGResult:
    answer: str
    sources: list[DocumentChunk]
    query_used: str


# Common German words unlikely to appear in English queries.
# If enough are present, the query is already German — skip translation.
_GERMAN_MARKERS = re.compile(
    r"\b("
    r"der|die|das|den|dem|des"
    r"|ein|eine|einen|einem|einer|eines"
    r"|und|oder|aber|nicht|ist|sind|wird|wurde|werden|hat|haben"
    r"|auf|für|mit|von|zu|bei|nach|über|unter|vor|zwischen"
    r"|wie|was|wer|wann|warum|welche|welcher|welches|welchem|welchen"
    r"|ich|er|sie|es|wir|ihr"
    r"|kann|können|muss|müssen|soll|sollen|darf|dürfen"
    r"|kein|keine|keinen|keinem|keiner"
    r"|dieser|diese|dieses|diesem|diesen"
    r"|auch|noch|schon|nur|sehr|mehr"
    r"|§|Abs|Art|Satz|Nr|BGB|StGB|HGB|GG|ZPO|StPO|VwGO"
    r"|Recht|Gesetz|Anspruch|Vertrag|Klage|Haftung|Schuld"
    r"|Gericht|Urteil|Beschluss|Antrag|Frist|Kündigung"
    r")\b",
    re.IGNORECASE,
)

# Threshold: fraction of words that must be German markers
_GERMAN_THRESHOLD = 0.25


def _is_german(text: str) -> bool:
    """Fast local check: is the text likely German?"""
    words = text.split()
    if not words:
        return False
    matches = len(_GERMAN_MARKERS.findall(text))
    return matches / len(words) >= _GERMAN_THRESHOLD


_SMALLTALK_PATTERNS = re.compile(
    r"^("
    r"h(i|ey|ello|allo|ola)"
    r"|guten\s*(tag|morgen|abend)"
    r"|moin"
    r"|servus"
    r"|grüß\s*gott"
    r"|danke(schön|sehr)?"
    r"|thanks?(\s*you)?"
    r"|thank\s*you"
    r"|how\s*are\s*you"
    r"|wie\s*geht'?s?"
    r"|what'?s\s*up"
    r"|yo"
    r"|good\s*(morning|evening|afternoon|night)"
    r"|bye|goodbye|tschüss|ciao"
    r"|ok(ay)?"
    r"|yes|no|ja|nein"
    r")[\s?!.,]*$",
    re.IGNORECASE,
)


class RAGPipeline:
    """Core orchestrator: embed → retrieve → generate."""

    def __init__(self, llm_provider: BaseLLMProvider) -> None:
        self._llm = llm_provider
        settings = get_settings()
        self._guard_enabled = settings.enable_guard
        if self._guard_enabled:
            self._guard = GeminiProvider(model_name="gemini-3.1-flash-lite-preview")

    @staticmethod
    def _is_smalltalk(query: str) -> bool:
        """Return True if the query is a simple greeting or small-talk."""
        return len(query.split()) < 10 and _SMALLTALK_PATTERNS.match(query.strip()) is not None

    async def _translate_to_german(self, query: str) -> str:
        """Translate a query to German for better vector search against German legal docs."""
        if _is_german(query):
            logger.info("query_already_german", query=query[:100])
            return query

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a translator. Translate the user's query into German. "
                    "If the query is already in German, return it unchanged. "
                    "Output ONLY the translated text, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            translated = await self._guard.generate(messages, temperature=0.0, max_tokens=256)
            translated = translated.strip()
            logger.info("query_translated", original=query[:100], translated=translated[:100])
            return translated
        except Exception as exc:
            logger.warning("translation_failed_fallback", error=str(exc))
            return query

    async def _retrieve(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[list[DocumentChunk], list[float]]:
        """Embed the query and retrieve relevant chunks."""
        # Translate to German for better similarity with German legal documents
        if self._guard_enabled:
            search_query = await self._translate_to_german(query)
        else:
            search_query = query
        embedding = await embed_query(search_query)
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

        # Fast path: skip embedding + vector search for trivial greetings
        if self._is_smalltalk(query):
            logger.info("smalltalk_fast_path", query=query[:100])
            if self._guard_enabled:
                answer = await self._run_guard(query, "", "", mode)
            else:
                answer = ""
            return RAGResult(answer=answer, sources=[], query_used=query)

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
            raw_answer = _fix_ocr_splits(_fix_markdown(raw_answer))
            logger.info("llm_generate_done", answer_chars=len(raw_answer))
        else:
            logger.warning("no_chunks_found", query=query[:100])
            raw_answer = ""  # No sources — let guard answer from knowledge

        if self._guard_enabled:
            try:
                answer = await self._run_guard(query, context, raw_answer, mode)
            except Exception as exc:
                logger.warning("guard_failed_fallback", error=str(exc))
                answer = raw_answer
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

        # Fast path: skip embedding + vector search for trivial greetings
        if self._is_smalltalk(query):
            logger.info("smalltalk_fast_path_stream", query=query[:100])
            if self._guard_enabled:
                user_content = GUARD_USER_TEMPLATE.format(
                    query=query,
                    context="Keine Quelldokumente verfügbar.",
                    draft="Keine Entwurfsantwort vorhanden.",
                    mode_instruction=_GUARD_MODE_NOTES.get(mode, ""),
                )
                guard_messages = [
                    {"role": "system", "content": GUARD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                async def _smalltalk_guard_stream() -> AsyncGenerator[str, None]:
                    async for token in self._guard.generate_stream(guard_messages):
                        yield token

                return _smalltalk_guard_stream(), []

            async def _empty_stream() -> AsyncGenerator[str, None]:
                yield ""

            return _empty_stream(), []

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

            # Stream primary LLM tokens directly to the user for fast
            # time-to-first-token. The guard (fact-check) still runs in
            # the non-streaming `run()` path; for streaming we prioritise
            # perceived latency over the extra guard pass.
            async def _primary_stream() -> AsyncGenerator[str, None]:
                async for token in self._llm.generate_stream(messages):
                    yield token

            token_stream = _primary_stream()
        else:
            logger.warning("no_chunks_found", query=query[:100])

            if self._guard_enabled:
                # No sources — let guard answer from its own knowledge
                user_content = GUARD_USER_TEMPLATE.format(
                    query=query,
                    context="Keine Quelldokumente verfügbar.",
                    draft="Keine Entwurfsantwort vorhanden.",
                    mode_instruction=_GUARD_MODE_NOTES.get(mode, ""),
                )
                guard_messages = [
                    {"role": "system", "content": GUARD_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                async def _guard_fallback_stream() -> AsyncGenerator[str, None]:
                    async for token in self._guard.generate_stream(guard_messages):
                        yield token

                token_stream = _guard_fallback_stream()
            else:
                async def _empty_stream() -> AsyncGenerator[str, None]:
                    yield ""
                token_stream = _empty_stream()

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
