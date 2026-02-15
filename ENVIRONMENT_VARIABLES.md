# Environment Variables Reference

Complete inventory of all environment variables used across the Production RAG API.

---

## Secrets

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | `settings.py`, `ingest.py`, `eval.py` | OpenAI API key for embeddings and chat completions | Use a dedicated production key with spend limits |
| `PINECONE_API_KEY` | **Yes** | — | `settings.py`, `ingest.py`, `eval.py`, `cleanup.py` | Pinecone vector database API key | Use a dedicated production key |
| `API_KEY` | When auth enabled | `change_me` | `settings.py`, `main.py` | API key that clients must send via `X-API-Key` header | Generate a strong random key (32+ chars) |

> ⚠️ **Never commit secrets to version control.** All three variables are loaded from `.env` (gitignored).

---

## Models

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `OPENAI_MODEL` | No | `gpt-4o-mini` | `settings.py`, `rag.py` | Chat completion model for answer generation | `gpt-4o-mini` (cost-effective) |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | `settings.py`, `rag.py`, `ingest.py` | Embedding model for query and chunk encoding | `text-embedding-3-small` |

---

## Retrieval / Cost

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `TOP_K` | No | `4` | `settings.py`, `rag.py` | Number of Pinecone chunks retrieved per query | `4` (do not increase without evaluating cost and latency) |
| `PINECONE_INDEX_NAME` | **Yes** | — | `settings.py`, `ingest.py`, `cleanup.py` | Name of the Pinecone index to query/write | Use a dedicated production index |

---

## Safety / Guardrails

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `BLOCKLIST` | No | _(empty)_ | `settings.py`, `guardrails.py` | Comma-separated list of blocked terms (case-insensitive) | `politics,violence` or domain-specific blocked terms |

---

## Auth

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `REQUIRE_API_KEY` | No | `false` | `settings.py`, `main.py`, `conftest.py`, `test_auth.py` | Enable/disable API key authentication | `true` — always require auth in production |
| `API_KEY` | When auth enabled | `change_me` | `settings.py`, `main.py` | The expected value of the `X-API-Key` header | Generate a strong random key |

---

## Rate Limiting

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `RATE_LIMIT` | No | `5/minute` | `settings.py`, `main.py` | Rate limit per API key (or per IP when auth disabled) | `60/minute` per API key for ~100 users |

---

## Data / Namespaces

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `NAMESPACE` | No | `dev` | `settings.py`, `rag.py`, `ingest.py`, `cleanup.py` | Pinecone namespace for vector isolation | `prod` |
| `DATA_DIR` | No | `./data` | `settings.py`, `ingest.py` | Directory containing PDF files for ingestion | `./data` |

---

## Evaluation

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `EVAL_MODE` | No | `false` | `settings.py`, `rag.py`, `eval.py` | Include raw chunk texts in response `meta.contexts` for RAGAS | `false` — never enable in production (increases payload) |

---

## Caching

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `CACHE_TTL_SECONDS` | No | `0` | `settings.py`, `cache.py`, `rag.py` | TTL for in-memory response cache (`0` = disabled) | `60` — reduces OpenAI cost for repeated questions |
| `CACHE_MAX_ITEMS` | No | `256` | `settings.py`, `cache.py` | Maximum number of cached responses (LRU eviction) | `256` |

---

## Deployment / Runtime

| Variable | Required | Default | Used In | Purpose | Production Recommendation |
|---|---|---|---|---|---|
| `PORT` | No | `8000` | `Dockerfile` (shell CMD) | Server port (used by Render/Railway via `$PORT`) | Platform-assigned or `8000` |

> **Proxy headers**: The Dockerfile includes `--proxy-headers --forwarded-allow-ips='*'` in the CMD.
> This trusts `X-Forwarded-For` headers from reverse proxies (Render, Railway, etc.).
> In production, restrict `--forwarded-allow-ips` to your proxy's IP range if possible.

---

## Test-Only Variables

| Variable | Default | Used In | Purpose |
|---|---|---|---|
| `RUN_LIVE_TESTS` | _(unset)_ | `test_chat.py` | Set to `true` to enable live API tests (skipped otherwise) |
| `BLOCKLIST` | `politics,violence` | `conftest.py` | Overridden in tests for deterministic guardrail behavior |
| `REQUIRE_API_KEY` | `false` | `conftest.py`, `test_auth.py` | Overridden in tests to test auth on/off |
| `CACHE_TTL_SECONDS` | `0` | `test_auth.py` | Overridden in cache tests |

---

## Summary

| Category | Count |
|---|---|
| **Secrets** | 3 |
| **Models** | 2 |
| **Retrieval / Cost** | 2 |
| **Safety** | 1 |
| **Auth** | 2 |
| **Rate Limiting** | 1 |
| **Data / Namespaces** | 2 |
| **Evaluation** | 1 |
| **Caching** | 2 |
| **Deployment** | 1 |
| **Total unique variables** | **16** (+ 1 test-only: `RUN_LIVE_TESTS`) |
