from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ConversationMessageSchema(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    conversation_history: list[ConversationMessageSchema] = Field(default_factory=list)
    top_k: int | None = Field(None, ge=1, le=50)
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    metadata_filter: dict | None = None
    mode: Literal["normal", "student", "lawyer"] = "normal"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int | None = Field(None, ge=1, le=50)
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    metadata_filter: dict | None = None


class SourceDocument(BaseModel):
    id: UUID
    content: str
    metadata: dict
    similarity: float
    source_display: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str


class SearchResponse(BaseModel):
    results: list[SourceDocument]
    query: str
    count: int


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: str
    llm_provider: str


class ErrorResponse(BaseModel):
    detail: str
