"""FastAPI application entry point for the Production RAG API."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.guardrails import check_blocklist
from app.logging_conf import RequestLoggingMiddleware
from app.rag import ask as rag_ask
from app.settings import settings

# Logging
log = structlog.get_logger()


# Rate-limiter key function

def _rate_limit_key(request: Request) -> str:
    """Return rate-limit bucket key (API key if auth enabled, else IP)."""
    if settings.REQUIRE_API_KEY:
        api_key = request.headers.get("x-api-key", "")
        if api_key:
            # Bucket by API key (internal use only)
            return f"key:{api_key}"
    return get_remote_address(request)


# Rate limiter
limiter = Limiter(key_func=_rate_limit_key, default_limits=[])

# FastAPI app

app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    description="Retrieval-Augmented Generation API powered by Pinecone & OpenAI.",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Auth

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _mask_key(key: str) -> str:
    """Return a masked version of an API key for safe logging."""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _client_identity(request: Request, api_key: str | None) -> str:
    """Return a loggable client identity string (never the full key)."""
    if api_key:
        return f"key:{_mask_key(api_key)}"
    return f"ip:{get_remote_address(request)}"


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str | None:
    """Verify the API key if REQUIRE_API_KEY is enabled."""
    if not settings.REQUIRE_API_KEY:
        return None
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return api_key


# Request / Response schemas


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    meta: dict[str, Any]


# Endpoints


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
@limiter.limit(settings.RATE_LIMIT)
async def chat(
    body: ChatRequest,
    request: Request,
    _api_key: str | None = Security(verify_api_key),
) -> ChatResponse:
    """Ask a question and receive a RAG-powered answer."""
    question = body.question.strip()

    log.info("chat_request", client=_client_identity(request, _api_key), q=question[:60])

    # Guardrail check
    blocked = check_blocklist(question)
    if blocked:
        log.warning("question_blocked", question=question[:80])
        return ChatResponse(answer=blocked, sources=[], meta={"blocked": True})

    # RAG pipeline
    try:
        result = rag_ask(question, history=body.history)
    except Exception as exc:
        # Do not leak internal details
        log.error("rag_pipeline_error", error=type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="An error occurred while processing your question. Please try again later.",
        ) from None

    return ChatResponse(**result)
