"""Tests for authentication, rate limiting, and response caching.

All tests run offline — no OpenAI or Pinecone calls.
"""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# Predictable mock payload

MOCK_RAG_RESULT = {
    "answer": "The answer is 42.",
    "sources": [],
    "meta": {
        "model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "top_k": 4,
        "namespace": "dev",
        "chunks_retrieved": 1,
        "latency_ms": 50.0,
        "prompt_tokens": 100,
        "completion_tokens": 20,
    },
}


# Helpers


@pytest.fixture()
def auth_client():
    """TestClient with API-key auth ENABLED."""
    os.environ["REQUIRE_API_KEY"] = "true"
    os.environ["API_KEY"] = "test-secret-key"
    os.environ["BLOCKLIST"] = "politics"

    # Reload settings to pick up the new env vars
    import importlib
    import app.settings
    importlib.reload(app.settings)
    import app.main
    importlib.reload(app.main)

    from app.main import app
    return TestClient(app)


@pytest.fixture()
def noauth_client():
    """TestClient with API-key auth DISABLED."""
    os.environ["REQUIRE_API_KEY"] = "false"
    os.environ["BLOCKLIST"] = "politics"

    import importlib
    import app.settings
    importlib.reload(app.settings)
    import app.main
    importlib.reload(app.main)

    from app.main import app
    return TestClient(app)


# A) Auth Tests


class TestAuth:
    """When REQUIRE_API_KEY=true, requests without valid X-API-Key get 401."""

    def test_missing_key_returns_401(self, auth_client):
        resp = auth_client.post(
            "/chat",
            json={"question": "hello", "history": []},
        )
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, auth_client):
        resp = auth_client.post(
            "/chat",
            json={"question": "hello", "history": []},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_valid_key_accepted(self, auth_client):
        with patch("app.main.rag_ask", return_value=MOCK_RAG_RESULT):
            resp = auth_client.post(
                "/chat",
                json={"question": "hello", "history": []},
                headers={"X-API-Key": "test-secret-key"},
            )
        assert resp.status_code == 200

    def test_no_auth_when_disabled(self, noauth_client):
        """When REQUIRE_API_KEY=false, no X-API-Key needed."""
        with patch("app.main.rag_ask", return_value=MOCK_RAG_RESULT):
            resp = noauth_client.post(
                "/chat",
                json={"question": "hello", "history": []},
            )
        assert resp.status_code == 200


# B) Rate Limit Key Function Tests


class TestRateLimitKey:
    """Verify rate-limiter key function uses API key or falls back to IP."""

    def test_key_func_uses_api_key_when_auth_enabled(self):
        """When REQUIRE_API_KEY=true, the limiter key should contain the API key."""
        os.environ["REQUIRE_API_KEY"] = "true"
        import importlib, app.settings
        importlib.reload(app.settings)
        import app.main
        importlib.reload(app.main)

        mock_request = MagicMock()
        mock_request.headers = {"x-api-key": "my-secret"}
        mock_request.client.host = "10.0.0.1"

        result = app.main._rate_limit_key(mock_request)
        assert result == "key:my-secret"

    def test_key_func_falls_back_to_ip_when_no_key(self):
        """When REQUIRE_API_KEY=false, fall back to IP."""
        os.environ["REQUIRE_API_KEY"] = "false"
        import importlib, app.settings
        importlib.reload(app.settings)
        import app.main
        importlib.reload(app.main)

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.1"

        result = app.main._rate_limit_key(mock_request)
        # Should be the IP address (from get_remote_address fallback)
        assert "key:" not in result


# C) Mask Key Tests


class TestMaskKey:
    """Verify API keys are properly masked for logging."""

    def test_mask_long_key(self):
        from app.main import _mask_key
        assert _mask_key("abcdefghijklmnop") == "abcd...mnop"

    def test_mask_short_key(self):
        from app.main import _mask_key
        assert _mask_key("short") == "***"


# D) Cache Tests


class TestCache:
    """Test the TTL cache in app/cache.py."""

    def test_cache_disabled_by_default(self):
        """With default CACHE_TTL_SECONDS=0, cache should not store items."""
        os.environ["CACHE_TTL_SECONDS"] = "0"
        import importlib, app.settings
        importlib.reload(app.settings)
        import app.cache
        importlib.reload(app.cache)

        assert app.cache.response_cache.enabled is False

    def test_cache_stores_and_retrieves(self):
        """When enabled, cache should store and retrieve results."""
        os.environ["CACHE_TTL_SECONDS"] = "60"
        os.environ["CACHE_MAX_ITEMS"] = "10"
        import importlib, app.settings
        importlib.reload(app.settings)
        import app.cache
        importlib.reload(app.cache)

        cache = app.cache.response_cache
        assert cache.enabled is True

        key = app.cache._make_key("test question")
        cache.put(key, {"answer": "cached", "sources": [], "meta": {}})
        result = cache.get(key)
        assert result is not None
        assert result["answer"] == "cached"

        # Cleanup
        cache.clear()
        os.environ["CACHE_TTL_SECONDS"] = "0"

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        os.environ["CACHE_TTL_SECONDS"] = "60"
        import importlib, app.settings
        importlib.reload(app.settings)
        import app.cache
        importlib.reload(app.cache)

        cache = app.cache.response_cache
        result = cache.get("nonexistent-key")
        assert result is None

        # Cleanup
        cache.clear()
        os.environ["CACHE_TTL_SECONDS"] = "0"
