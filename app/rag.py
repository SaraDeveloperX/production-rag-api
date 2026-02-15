"""Core Retrieval-Augmented Generation logic.

1. Embed the user question.
2. Query Pinecone for top-K similar chunks.
3. Build context, call chat model, return answer + sources.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from openai import OpenAI
from pinecone import Pinecone

from app.cache import _make_key, response_cache
from app.settings import settings

log = structlog.get_logger()

# System prompt

SYSTEM_PROMPT = (
    "Answer ONLY from the provided context. "
    "If you don't know, say: I don't know."
)

# Lazy-initialised clients

_openai_client: OpenAI | None = None
_pinecone_index: Any = None
_preflight_checked: bool = False


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
    return _pinecone_index


def _validate_dimensions(client: OpenAI, index: Any) -> None:
    """One-time check: embedding dim must match Pinecone index dim."""
    global _preflight_checked
    if _preflight_checked:
        return
    try:
        stats = index.describe_index_stats()
        index_dim = stats.get("dimension")
        if index_dim is None:
            log.warning("preflight_skip", reason="could not read index dimension")
            _preflight_checked = True
            return

        probe = client.embeddings.create(
            input=["dimension check"], model=settings.EMBEDDING_MODEL
        )
        embed_dim = len(probe.data[0].embedding)

        if embed_dim != index_dim:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) does not match "
                f"Pinecone index dimension ({index_dim}). "
                f"Re-ingest with the correct EMBEDDING_MODEL or recreate the index."
            )
        log.info("preflight_ok", embed_dim=embed_dim, index_dim=index_dim)
    except ValueError:
        raise  # re-raise dimension mismatch
    except Exception as exc:
        log.warning("preflight_error", error=type(exc).__name__)
    _preflight_checked = True


# Public API


def ask(
    question: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the full RAG pipeline and return answer + sources + meta."""
    # Cache lookup (stateless requests only)
    cache_key = _make_key(question) if not history else None
    if cache_key:
        cached = response_cache.get(cache_key)
        if cached is not None:
            cached["meta"]["cache_hit"] = True
            log.info("cache_hit", question=question[:60])
            return cached

    t0 = time.perf_counter()
    client = _get_openai()
    index = _get_index()

    # Preflight: validate embedding dim == index dim (once)
    _validate_dimensions(client, index)

    # 1. Embed the question
    embed_resp = client.embeddings.create(
        input=[question],
        model=settings.EMBEDDING_MODEL,
    )
    query_embedding = embed_resp.data[0].embedding

    # 2. Query Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=settings.TOP_K,
        include_metadata=True,
        namespace=settings.NAMESPACE,
        filter={"type": {"$ne": "manifest"}},  # exclude manifests
    )

    matches = search_results.get("matches", [])
    log.info("pinecone_results", matches=len(matches))

    # 3. Build context
    MAX_CHUNK_CHARS = 800  # limit prompt tokens
    context_parts: list[str] = []
    sources: list[dict[str, Any]] = []

    for m in matches:
        meta = m.get("metadata", {})
        text = meta.get("text", "")[:MAX_CHUNK_CHARS]
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        doc_hash = meta.get("doc_hash", "")
        chunk_id = meta.get("chunk_id", 0)
        score = round(m.get("score", 0.0), 4)

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{text}"
        )
        sources.append(
            {
                "source": source,
                "page": page,
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
                "score": score,
            }
        )

    context_block = "\n\n---\n\n".join(context_parts) if context_parts else ""

    # 4. Build messages
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Append conversation history
    if history:
        for entry in history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # User message with context
    user_content = (
        f"Context:\n{context_block}\n\nQuestion: {question}"
        if context_block
        else f"Question: {question}"
    )
    messages.append({"role": "user", "content": user_content})

    # 5. Call chat model
    chat_resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=0,
    )
    answer = chat_resp.choices[0].message.content or ""

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    usage = chat_resp.usage
    meta: dict[str, Any] = {
        "model": settings.OPENAI_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "top_k": settings.TOP_K,
        "namespace": settings.NAMESPACE,
        "chunks_retrieved": len(matches),
        "latency_ms": latency_ms,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }

    # Include raw chunks for RAGAS evaluation
    if settings.EVAL_MODE:
        meta["contexts"] = context_parts

    log.info("rag_complete", latency_ms=latency_ms, chunks=len(matches))

    result = {"answer": answer, "sources": sources, "meta": meta}

    # Store in cache
    if cache_key:
        response_cache.put(cache_key, result)

    return result
