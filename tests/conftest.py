"""Shared pytest fixtures for the Production RAG API test suite."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Return a FastAPI TestClient for the production RAG API."""
    # Ensure BLOCKLIST is set so guardrail tests are deterministic
    os.environ.setdefault("BLOCKLIST", "politics,violence")
    # Disable API key requirement for tests
    os.environ.setdefault("REQUIRE_API_KEY", "false")

    from app.main import app

    return TestClient(app)
