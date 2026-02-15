"""Pinecone vector cleanup utility.

Delete vectors by --source, --doc-hash, or --namespace.
Always use --dry-run first to preview.

Usage:
    python scripts/cleanup.py --source policy.pdf --dry-run
    python scripts/cleanup.py --source policy.pdf
    python scripts/cleanup.py --doc-hash abc123...
    python scripts/cleanup.py --namespace old-ns --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

import structlog
from dotenv import load_dotenv
from pinecone import Pinecone

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()


# Helpers


def count_by_filter(index, namespace: str, filter_dict: dict) -> int:
    """Estimate the number of vectors matching a metadata filter.

    Uses a dummy query vector to search with a filter. This is an
    approximation — Pinecone does not expose a direct count-by-filter API.
    """
    try:
        # Get index dimension for a zero-vector query
        stats = index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        total = ns_stats.get("vector_count", 0)
        dim = stats.get("dimension", 1536)

        if total == 0:
            return 0

        # Query with filter to estimate matches
        result = index.query(
            vector=[0.0] * dim,
            filter=filter_dict,
            namespace=namespace,
            top_k=10000,
            include_metadata=False,
        )
        return len(result.get("matches", []))
    except Exception as exc:
        log.warning("count_estimate_failed", error=str(exc))
        return -1


def delete_by_filter(index, namespace: str, filter_dict: dict, dry_run: bool) -> None:
    """Delete vectors matching a metadata filter."""
    count = count_by_filter(index, namespace, filter_dict)
    log.info("vectors_matched", count=count, filter=filter_dict, namespace=namespace)

    if dry_run:
        print(f"\n[DRY RUN] Would delete ~{count} vector(s) from namespace '{namespace}'")
        print(f"          Filter: {filter_dict}")
        print("          Re-run without --dry-run to execute.\n")
        return

    if count == 0:
        log.info("nothing_to_delete")
        return

    try:
        index.delete(filter=filter_dict, namespace=namespace)
        log.info("deleted", filter=filter_dict, namespace=namespace)
        print(f"\nDeleted vectors matching {filter_dict} from namespace '{namespace}'.\n")
    except Exception as exc:
        log.error("delete_failed", error=str(exc))
        sys.exit(1)


def delete_namespace(index, namespace: str, dry_run: bool) -> None:
    """Delete ALL vectors in a namespace."""
    try:
        stats = index.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        count = ns_stats.get("vector_count", 0)
    except Exception as exc:
        log.error("stats_failed", error=str(exc))
        sys.exit(1)

    log.info("namespace_stats", namespace=namespace, vector_count=count)

    if dry_run:
        print(f"\n[DRY RUN] Would delete ALL {count} vector(s) in namespace '{namespace}'")
        print("          Re-run without --dry-run to execute.\n")
        return

    if count == 0:
        log.info("namespace_empty", namespace=namespace)
        return

    try:
        index.delete(delete_all=True, namespace=namespace)
        log.info("namespace_deleted", namespace=namespace, vectors=count)
        print(f"\nDeleted all {count} vector(s) from namespace '{namespace}'.\n")
    except Exception as exc:
        log.error("namespace_delete_failed", error=str(exc))
        sys.exit(1)


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up Pinecone vectors by source, hash, or namespace."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Delete vectors by source filename (e.g. 'policy.pdf')")
    group.add_argument("--doc-hash", type=str, help="Delete vectors by document hash")
    group.add_argument("--namespace", type=str, dest="delete_namespace", help="Delete ALL vectors in this namespace")

    parser.add_argument(
        "--ns",
        type=str,
        default=None,
        help="Namespace to operate on (default: NAMESPACE from .env or 'dev')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the action without making changes",
    )
    args = parser.parse_args()

    # Load env
    load_dotenv()

    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not pinecone_key or not index_name:
        log.error("missing_env_vars", hint="Set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env")
        sys.exit(1)

    namespace = args.ns or os.getenv("NAMESPACE", "dev")

    # Connect
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    log.info("connected", index=index_name, namespace=namespace, dry_run=args.dry_run)

    # Dispatch
    if args.delete_namespace:
        delete_namespace(index, args.delete_namespace, args.dry_run)
    elif args.source:
        delete_by_filter(index, namespace, {"source": {"$eq": args.source}}, args.dry_run)
    elif args.doc_hash:
        delete_by_filter(index, namespace, {"doc_hash": {"$eq": args.doc_hash}}, args.dry_run)


if __name__ == "__main__":
    main()
