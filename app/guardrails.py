"""Content-safety guardrails.

Checks incoming questions against a configurable blocklist.
"""

from __future__ import annotations

from app.settings import settings

BLOCKED_RESPONSE = "I cannot answer this question due to safety policies."


def is_blocked(text: str) -> bool:
    """Return True if the text contains any blocked term (case-insensitive)."""
    t_lower = text.lower()
    return any(term and term in t_lower for term in settings.BLOCKLIST)


def check_blocklist(question: str) -> str | None:
    """Return BLOCKED_RESPONSE if the question contains a blocked term,
    otherwise return None (meaning the question is safe)."""
    if is_blocked(question):
        return BLOCKED_RESPONSE
    return None

