"""Production ingestion pipeline for the RAG API.

Reads PDFs, chunks, embeds via OpenAI, and upserts into Pinecone.
Uses SHA-256 manifests to skip unchanged files.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --strategy replace
    python scripts/ingest.py --strategy skip --dir ./data
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone

# Logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# Constants
BATCH_SIZE = 100
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MANIFEST_FIXED_TEXT = "manifest record"


# Helpers


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts using the OpenAI embeddings API."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def manifest_id(doc_hash: str) -> str:
    """Deterministic Pinecone vector ID for a manifest record."""
    return f"manifest::{doc_hash}"


def check_manifest_exists(
    index, doc_hash: str, namespace: str
) -> bool:
    """Return True if a manifest vector already exists for this hash."""
    mid = manifest_id(doc_hash)
    try:
        result = index.fetch(ids=[mid], namespace=namespace)
        return mid in (result.get("vectors", None) or {})
    except Exception:
        return False


def delete_vectors_by_source(
    index, source: str, namespace: str
) -> None:
    """Delete all vectors (chunks + manifest) whose source matches."""
    log.info("deleting_existing_vectors", source=source, namespace=namespace)
    # Delete by metadata filter
    try:
        index.delete(
            filter={"source": {"$eq": source}},
            namespace=namespace,
        )
    except Exception as exc:
        log.warning("delete_by_filter_failed", error=str(exc))


def upsert_batched(
    index, vectors: list[dict], namespace: str
) -> int:
    """Upsert vectors into Pinecone in batches. Returns count upserted."""
    total = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        total += len(batch)
        log.debug("upserted_batch", count=len(batch), total=total)
    return total


# Core pipeline


def ingest_pdf(
    pdf_path: Path,
    *,
    index,
    openai_client: OpenAI,
    embedding_model: str,
    namespace: str,
    strategy: str,
) -> dict:
    """Ingest a single PDF into Pinecone. Returns a summary dict."""
    filename = pdf_path.name
    doc_hash = sha256_file(pdf_path)
    log.info("processing_file", file=filename, hash=doc_hash[:12])

    # Manifest check
    if strategy == "skip" and check_manifest_exists(index, doc_hash, namespace):
        log.info("skipping_unchanged_file", file=filename)
        return {"file": filename, "status": "skipped", "chunks": 0}

    if strategy == "replace":
        delete_vectors_by_source(index, filename, namespace)

    # Load & chunk
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    log.info("loaded_pages", file=filename, pages=len(pages))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    log.info("created_chunks", file=filename, chunks=len(chunks))

    if not chunks:
        log.warning("no_chunks_produced", file=filename)
        return {"file": filename, "status": "empty", "chunks": 0}

    # Embed chunks
    texts = [c.page_content for c in chunks]
    embeddings = embed_texts(openai_client, texts, embedding_model)

    # Build Pinecone vectors
    vectors = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        page_num = chunk.metadata.get("page", 0)
        vec_id = f"{doc_hash}::{idx}"
        vectors.append(
            {
                "id": vec_id,
                "values": embedding,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "doc_hash": doc_hash,
                    "chunk_id": idx,
                    "text": chunk.page_content[:1000],  # Pinecone metadata limit
                },
            }
        )

    # Create manifest vector (same dimension as chunks)
    manifest_embedding = embed_texts(
        openai_client, [MANIFEST_FIXED_TEXT], embedding_model
    )[0]
    vectors.append(
        {
            "id": manifest_id(doc_hash),
            "values": manifest_embedding,
            "metadata": {
                "type": "manifest",
                "doc_hash": doc_hash,
                "source": filename,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    )

    # Upsert
    upserted = upsert_batched(index, vectors, namespace)
    log.info("ingestion_complete", file=filename, vectors=upserted)
    return {"file": filename, "status": "ingested", "chunks": len(chunks)}


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into Pinecone for RAG."
    )
    parser.add_argument(
        "--strategy",
        choices=["skip", "replace"],
        default="skip",
        help='Duplicate handling: "skip" (default) or "replace".',
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Override DATA_DIR from .env.",
    )
    args = parser.parse_args()

    # Load env
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    namespace = os.getenv("NAMESPACE", "dev")
    data_dir = Path(args.dir or os.getenv("DATA_DIR", "./data"))

    # Validate
    missing = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not pinecone_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("PINECONE_INDEX_NAME")
    if missing:
        log.error("missing_env_vars", vars=missing)
        sys.exit(1)

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        log.info("created_data_dir", path=str(data_dir))

    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        log.warning(
            "no_pdfs_found",
            path=str(data_dir),
            hint="Place .pdf files in the data/ directory and re-run.",
        )
        sys.exit(0)

    # Clients
    openai_client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    log.info(
        "ingestion_started",
        files=len(pdfs),
        strategy=args.strategy,
        namespace=namespace,
        embedding_model=embedding_model,
    )

    # Process each PDF
    results = []
    for pdf in pdfs:
        try:
            result = ingest_pdf(
                pdf,
                index=index,
                openai_client=openai_client,
                embedding_model=embedding_model,
                namespace=namespace,
                strategy=args.strategy,
            )
            results.append(result)
        except Exception as exc:
            log.error("file_ingestion_failed", file=pdf.name, error=str(exc))
            results.append({"file": pdf.name, "status": "error", "chunks": 0})

    # Summary
    ingested = sum(1 for r in results if r["status"] == "ingested")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    total_chunks = sum(r["chunks"] for r in results)

    log.info(
        "pipeline_finished",
        ingested=ingested,
        skipped=skipped,
        errors=errors,
        total_chunks=total_chunks,
    )


if __name__ == "__main__":
    main()
