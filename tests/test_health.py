"""Smoke test for /health. No API keys required."""

from __future__ import annotations


def test_health_returns_ok(client):
    """GET /health must return 200 with {"status": "ok"}."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok"}
