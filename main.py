from contextlib import asynccontextmanager

import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.exceptions import RAGBaseError
from app.db.client import close_pool, get_pool
from app.middleware.error_handler import rag_exception_handler
from app.api.routes import chat, health, search

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )
    logger.info(
        "starting_app",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
    )
    # Don't block on database connection at startup
    try:
        import asyncio
        asyncio.create_task(get_pool())
    except Exception as exc:
        logger.warning("database_connection_async_task_failed", error=str(exc))
    yield
    await close_pool()
    logger.info("app_shutdown")


app = FastAPI(
    title="German Legal AI RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — restricted to production domains (override via CORS_ORIGINS env var)
_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
app.add_exception_handler(RAGBaseError, rag_exception_handler)

# Routes
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(search.router)
