"""Tests for the content-safety guardrails.

All tests run offline — no OpenAI or Pinecone calls.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


# Unit tests for is_blocked


class TestIsBlocked:
    """Test is_blocked() detects blocked terms case-insensitively."""

    @pytest.fixture(autouse=True)
    def _set_blocklist(self):
        os.environ["BLOCKLIST"] = "politics,violence"
        # Re-import to pick up fresh settings
        import importlib
        import app.settings
        importlib.reload(app.settings)
        import app.guardrails
        importlib.reload(app.guardrails)

    def test_detects_blocked_term(self):
        from app.guardrails import is_blocked
        assert is_blocked("tell me about politics") is True

    def test_case_insensitive(self):
        from app.guardrails import is_blocked
        assert is_blocked("POLITICS is a hot topic") is True
        assert is_blocked("Violence in Movies") is True

    def test_safe_question_not_blocked(self):
        from app.guardrails import is_blocked
        assert is_blocked("what is 2+2") is False

    def test_empty_string_not_blocked(self):
        from app.guardrails import is_blocked
        assert is_blocked("") is False

    def test_blocked_response_constant(self):
        from app.guardrails import BLOCKED_RESPONSE
        assert BLOCKED_RESPONSE == "I cannot answer this question due to safety policies."


# Integration: blocked question via /chat


class TestChatBlocked:
    """POST /chat with a blocked question must return immediately
    with the fixed BLOCKED_RESPONSE and must NOT call the RAG layer."""

    def test_chat_returns_blocked_response_no_rag(self, client):
        """rag_ask should never be called for blocked questions."""
        with patch("app.main.rag_ask", side_effect=RuntimeError("RAG should not be called")):
            resp = client.post(
                "/chat",
                json={"question": "tell me about politics", "history": []},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "I cannot answer this question due to safety policies."
        assert body["sources"] == []
        assert body["meta"]["blocked"] is True
