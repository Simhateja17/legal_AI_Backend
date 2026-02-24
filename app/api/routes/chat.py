import json

import structlog
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import get_pipeline
from app.api.schemas import ChatRequest, ChatResponse, SourceDocument
from app.db.models import ConversationMessage
from app.rag.pipeline import RAGPipeline

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


def _to_source_doc(chunk) -> SourceDocument:
    return SourceDocument(
        id=chunk.id,
        content=chunk.content,
        metadata=chunk.metadata,
        similarity=chunk.similarity,
        source_display=chunk.source_display,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    logger.info(
        "chat_request_received",
        query=request.query[:100],
        mode=request.mode,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        history_turns=len(request.conversation_history),
    )
    history = [
        ConversationMessage(role=m.role, content=m.content)
        for m in request.conversation_history
    ]

    result = await pipeline.run(
        query=request.query,
        conversation_history=history,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        metadata_filter=request.metadata_filter,
        mode=request.mode,
    )

    logger.info("chat_response_sent", sources=len(result.sources), answer_chars=len(result.answer))
    return ChatResponse(
        answer=result.answer,
        sources=[_to_source_doc(c) for c in result.sources],
        query=result.query_used,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    logger.info(
        "chat_stream_request_received",
        query=request.query[:100],
        mode=request.mode,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        history_turns=len(request.conversation_history),
    )
    history = [
        ConversationMessage(role=m.role, content=m.content)
        for m in request.conversation_history
    ]

    token_stream, chunks = await pipeline.run_stream(
        query=request.query,
        conversation_history=history,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        metadata_filter=request.metadata_filter,
        mode=request.mode,
    )

    sources_payload = json.dumps(
        [_to_source_doc(c).model_dump(mode="json") for c in chunks],
        ensure_ascii=False,
    )
    logger.info("chat_stream_sources_ready", sources=len(chunks))

    async def event_generator():
        token_count = 0
        async for token in token_stream:
            token_count += 1
            yield {"event": "token", "data": token}

        logger.info("chat_stream_done", tokens_sent=token_count, sources=len(chunks))
        yield {"event": "sources", "data": sources_payload}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
