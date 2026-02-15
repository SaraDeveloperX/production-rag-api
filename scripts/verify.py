"""One-shot verification for the production RAG API.

Checks env vars, Pinecone, OpenAI, and /chat endpoint. Never prints secrets.
"""
import os
import sys
import json
import subprocess

from dotenv import load_dotenv

load_dotenv()

results = {}

# 1. Environment variables
print("=" * 60)
print("1. ENVIRONMENT VARIABLES")
print("=" * 60)
env_ok = True
for key in [
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "NAMESPACE",
    "OPENAI_MODEL",
    "EMBEDDING_MODEL",
    "TOP_K",
]:
    val = os.getenv(key, "")
    is_set = bool(val)
    print(f"  {key}: set={is_set}, len={len(val)}")
    if not is_set and key in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
        env_ok = False
results["1_env"] = env_ok
print(f"  => {'PASS' if env_ok else 'FAIL'}")

# 2. Git safety
print()
print("=" * 60)
print("2. GIT SAFETY")
print("=" * 60)
gitignore_path = os.path.join(os.path.dirname(__file__), "..", ".gitignore")
git_ok = False
if os.path.isfile(gitignore_path):
    with open(gitignore_path) as f:
        lines = [l.strip() for l in f.readlines()]
    if ".env" in lines:
        print("  .gitignore contains '.env': True")
        git_ok = True
    else:
        print("  WARNING: .env NOT found in .gitignore!")
else:
    print("  WARNING: .gitignore file not found!")

# Try git ls-files
try:
    out = subprocess.run(
        ["git", "ls-files", ".env"],
        capture_output=True, text=True, cwd=os.path.dirname(gitignore_path),
        timeout=5,
    )
    tracked = bool(out.stdout.strip())
    print(f"  .env tracked by git: {tracked}")
    if tracked:
        print("  CRITICAL: .env IS tracked! Run: git rm --cached .env && git commit")
        git_ok = False
except Exception as e:
    print(f"  git check skipped: {e}")
    # Still pass if .gitignore has .env
results["2_git"] = git_ok
print(f"  => {'PASS' if git_ok else 'WARN (git unavailable but .gitignore is correct)'}")

# 3. Pinecone connectivity
print()
print("=" * 60)
print("3. PINECONE CONNECTIVITY")
print("=" * 60)
pinecone_ok = False
pinecone_stats = None
try:
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    idx = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    stats = idx.describe_index_stats()
    ns = os.getenv("NAMESPACE", "dev")
    ns_stats = stats.get("namespaces", {}).get(ns, {})
    vec_count = ns_stats.get("vector_count", 0)
    total_count = stats.get("total_vector_count", 0)
    dimension = stats.get("dimension", "unknown")
    print(f"  Index dimension: {dimension}")
    print(f"  Total vector count: {total_count}")
    print(f"  Namespace '{ns}' vector count: {vec_count}")
    pinecone_stats = {"dimension": dimension, "total": total_count, "namespace_count": vec_count}
    if vec_count == 0:
        print(f"  WARNING: 0 vectors in namespace '{ns}'!")
        print(f"    - Check that ingestion used the same namespace (NAMESPACE={ns})")
        print(f"    - Check PINECONE_INDEX_NAME matches the ingested index")
        print(f"    - Available namespaces: {list(stats.get('namespaces', {}).keys())}")
    else:
        pinecone_ok = True
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
results["3_pinecone"] = pinecone_ok
print(f"  => {'PASS' if pinecone_ok else 'FAIL'}")

# 4. OpenAI connectivity
print()
print("=" * 60)
print("4. OPENAI CONNECTIVITY")
print("=" * 60)
openai_embed_ok = False
openai_chat_ok = False
embedding_dim = None
try:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 4a. Embedding test
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(input=["healthcheck"], model=emb_model)
    embedding_dim = len(resp.data[0].embedding)
    print(f"  Embedding test: SUCCESS (dim={embedding_dim})")
    openai_embed_ok = True

    # 4b. Chat test
    chat_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    chat_resp = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    reply = chat_resp.choices[0].message.content
    print(f"  Chat test: SUCCESS (model={chat_model}, reply_len={len(reply or '')})")
    openai_chat_ok = True
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
results["4a_openai_embed"] = openai_embed_ok
results["4b_openai_chat"] = openai_chat_ok

# 4c. Dimension match
if embedding_dim and pinecone_stats:
    pc_dim = pinecone_stats.get("dimension")
    if pc_dim and pc_dim != "unknown":
        match = embedding_dim == pc_dim
        print(f"  Dimension match: embedding={embedding_dim}, pinecone={pc_dim}, match={match}")
        if not match:
            print("  CRITICAL: Dimension mismatch! Re-ingest with correct embedding model.")
        results["4c_dim_match"] = match
    else:
        print("  Dimension match: could not determine index dimension")
        results["4c_dim_match"] = None

print(f"  => Embed: {'PASS' if openai_embed_ok else 'FAIL'}")
print(f"  => Chat:  {'PASS' if openai_chat_ok else 'FAIL'}")

# 5. /chat endpoint test
print()
print("=" * 60)
print("5. /chat ENDPOINT TEST")
print("=" * 60)
chat_endpoint_ok = False
try:
    import urllib.request
    import urllib.error

    payload = json.dumps({"question": "What is Saudi Vision 2030?", "history": []}).encode()
    req = urllib.request.Request(
        "http://localhost:8000/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    status = resp.status
    body = json.loads(resp.read().decode())
    has_keys = all(k in body for k in ("answer", "sources", "meta"))
    print(f"  HTTP status: {status}")
    print(f"  Response has answer/sources/meta: {has_keys}")
    print(f"  Answer preview: {body.get('answer', '')[:100]}...")
    print(f"  Sources count: {len(body.get('sources', []))}")
    chat_endpoint_ok = has_keys and status == 200
except urllib.error.HTTPError as e:
    print(f"  HTTP ERROR: {e.code}")
    try:
        err_body = e.read().decode()
        print(f"  Error body: {err_body[:200]}")
    except Exception:
        pass
    if e.code == 502:
        print("  ROOT CAUSE: RAG pipeline exception (OpenAI/Pinecone call failed)")
        print("  Check: API keys valid, Pinecone index exists, namespace has vectors")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")
results["5_chat_endpoint"] = chat_endpoint_ok
print(f"  => {'PASS' if chat_endpoint_ok else 'FAIL'}")

# Final summary
print()
print("=" * 60)
print("FINAL VERIFICATION REPORT")
print("=" * 60)
for key, val in results.items():
    icon = "[PASS]" if val else ("[WARN]" if val is None else "[FAIL]")
    print(f"  {icon} {key}: {val}")
print("=" * 60)
