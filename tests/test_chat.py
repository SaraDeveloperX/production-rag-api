"""Tests for the POST /chat endpoint.

A) Offline tests (default) — mocks app.rag.ask, no network calls.
B) Optional live test — requires RUN_LIVE_TESTS=true and API keys.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

# Predictable mock payload

MOCK_RAG_RESULT = {
    "answer": "The answer is 42.",
    "sources": [
        {
            "source": "doc.pdf",
            "page": 1,
            "doc_hash": "abc123",
            "chunk_id": 0,
            "score": 0.95,
        }
    ],
    "meta": {
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "top_k": 4,
        "namespace": "dev",
        "chunks_retrieved": 1,
        "latency_ms": 123.45,
        "prompt_tokens": 100,
        "completion_tokens": 20,
    },
}


# A) Offline tests (always run, no keys needed)


class TestChatOffline:
    """Tests with mocked RAG layer — run in CI without API keys."""

    def test_chat_schema(self, client):
        """Response must contain answer (str), sources (list), meta (dict)."""
        with patch("app.main.rag_ask", return_value=MOCK_RAG_RESULT):
            resp = client.post(
                "/chat",
                json={"question": "What is the meaning of life?", "history": []},
            )
        assert resp.status_code == 200
        body = resp.json()

        # Keys exist
        assert "answer" in body
        assert "sources" in body
        assert "meta" in body

        # Types
        assert isinstance(body["answer"], str)
        assert isinstance(body["sources"], list)
        assert isinstance(body["meta"], dict)

        # Values match mock
        assert body["answer"] == "The answer is 42."
        assert len(body["sources"]) == 1
        assert body["sources"][0]["source"] == "doc.pdf"

    def test_chat_empty_question_rejected(self, client):
        """An empty question must be rejected with 422."""
        resp = client.post(
            "/chat",
            json={"question": "", "history": []},
        )
        assert resp.status_code == 422

    def test_chat_missing_question_rejected(self, client):
        """A request without the question field must be rejected."""
        resp = client.post("/chat", json={"history": []})
        assert resp.status_code == 422

    def test_chat_rag_error_returns_502(self, client):
        """If the RAG pipeline raises, the endpoint returns 502."""
        with patch("app.main.rag_ask", side_effect=RuntimeError("boom")):
            resp = client.post(
                "/chat",
                json={"question": "test question", "history": []},
            )
        assert resp.status_code == 502
        body = resp.json()
        assert "detail" in body
        # Must NOT leak error details
        assert "boom" not in body["detail"]


# B) Optional live test (needs RUN_LIVE_TESTS=true + keys)

_live = (
    os.getenv("RUN_LIVE_TESTS", "").lower() == "true"
    and bool(os.getenv("OPENAI_API_KEY"))
    and bool(os.getenv("PINECONE_API_KEY"))
)


@pytest.mark.skipif(not _live, reason="Live tests disabled (set RUN_LIVE_TESTS=true with API keys)")
class TestChatLive:
    """End-to-end test against the real RAG pipeline.
    Only runs when RUN_LIVE_TESTS=true AND required API keys are set."""

    def test_chat_live_returns_valid_response(self, client):
        resp = client.post(
            "/chat",
            json={"question": "What is Saudi Vision 2030?", "history": []},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["answer"], str)
        assert isinstance(body["sources"], list)
        assert isinstance(body["meta"], dict)
        assert body["meta"]["chunks_retrieved"] >= 0
