"""Comprehensive accuracy evaluation for the RAG pipeline.

Outputs RAGAS metrics, retrieval quality metrics, eval/results.json,
and eval/report.md. Requires EVAL_MODE=true and valid API keys.

Note: Running this script consumes OpenAI tokens. Keep the dataset small.

Usage:
    set EVAL_MODE=true
    python scripts/eval_accuracy.py
"""

from __future__ import annotations

import json
import math
import re
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so `from app.* import ...` works
# when this script is run directly (e.g. python scripts/eval_accuracy.py).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import structlog
import yaml
from dotenv import load_dotenv

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# Paths

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QA_FILE = EVAL_DIR / "qa.yaml"
RESULTS_FILE = EVAL_DIR / "results.json"
REPORT_FILE = EVAL_DIR / "report.md"


# NaN-safe helpers


def _is_valid_score(v: Any) -> bool:
    """Return True if v is a finite number (not NaN / None / str)."""
    if v is None or isinstance(v, str):
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _nan_safe_avg(values: list[Any]) -> tuple[float | None, int, int]:
    """Return (average, valid_count, total_count) ignoring NaN / None."""
    total = len(values)
    valid = [float(v) for v in values if _is_valid_score(v)]
    if not valid:
        return None, 0, total
    return sum(valid) / len(valid), len(valid), total


# Retrieval Metrics


def keyword_hit_rate(contexts: list[str], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the concatenated contexts."""
    if not expected_keywords:
        return 1.0
    joined = " ".join(contexts).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in joined)
    return round(hits / len(expected_keywords), 4)


def source_match(sources: list[dict[str, Any]], expected_source: str | None) -> bool | None:
    """True if any returned source matches expected_source. None if not specified."""
    if not expected_source:
        return None
    return any(s.get("source") == expected_source for s in sources)


def page_match(sources: list[dict[str, Any]], expected_pages: list[int] | None) -> bool | None:
    """True if any returned page is in expected_pages. None if not specified."""
    if not expected_pages:
        return None
    returned_pages = {s.get("page") for s in sources}
    return bool(returned_pages & set(expected_pages))


# Overlap Metrics (offline, no API calls)

# Common Arabic stopwords to exclude from overlap computation.
_ARABIC_STOPWORDS: set[str] = {
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "التى", "اللذان", "اللتان", "الذين", "اللاتي",
    "هو", "هي", "هم", "هن", "نحن", "أنا", "أنت", "أنتم",
    "كان", "كانت", "يكون", "تكون", "ليس", "ليست",
    "أن", "إن", "لا", "لم", "لن", "قد", "ما", "و", "أو", "ثم",
    "بل", "لكن", "حتى", "إذا", "إذ", "كل", "بعض", "غير", "بين",
    "عند", "منذ", "خلال", "حول", "فوق", "تحت", "أمام", "وراء",
    "ال", "بـ", "لـ", "كـ",
    # Common English stopwords (in case of mixed content)
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "not", "no", "it", "its", "this", "that",
}


def _extract_tokens(text: str) -> set[str]:
    """Extract meaningful tokens from text (Arabic-friendly).

    - Splits on whitespace and punctuation
    - Strips diacritics (tashkeel) and tatweel
    - Removes stopwords
    - Keeps tokens with >=2 chars or that contain digits
    """
    if not text:
        return set()
    # Remove Arabic diacritics (\u0610-\u065F) and tatweel (\u0640)
    cleaned = re.sub(r"[\u0610-\u065F\u0640]", "", text)
    # Split on non-word chars (keeps Arabic + Latin + digits)
    tokens = re.findall(r"[\w\u0600-\u06FF]+", cleaned.lower())
    return {
        t for t in tokens
        if t not in _ARABIC_STOPWORDS and (len(t) >= 2 or any(c.isdigit() for c in t))
    }


def _overlap_rate(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Fraction of tokens_a that appear in tokens_b. Returns 0.0 if tokens_a is empty."""
    if not tokens_a:
        return 0.0
    return round(len(tokens_a & tokens_b) / len(tokens_a), 4)


def answer_context_overlap(answer: str, contexts: list[str]) -> float:
    """Overlap between answer tokens and context tokens."""
    answer_tokens = _extract_tokens(answer)
    ctx_tokens = _extract_tokens(" ".join(contexts))
    return _overlap_rate(answer_tokens, ctx_tokens)


def question_context_overlap(question: str, contexts: list[str]) -> float:
    """Overlap between question tokens and context tokens."""
    q_tokens = _extract_tokens(question)
    ctx_tokens = _extract_tokens(" ".join(contexts))
    return _overlap_rate(q_tokens, ctx_tokens)


# Report Generation


def _fmt_score(v: Any) -> str:
    """Format a score for display: finite → 4 decimals, else '—'."""
    if _is_valid_score(v):
        return f"{float(v):.4f}"
    return "—"


def generate_report(
    results: list[dict[str, Any]],
    ragas_scores: dict[str, Any] | None,
    elapsed_s: float,
    invalid_ragas_samples: list[dict[str, str]] | None = None,
) -> str:
    """Generate a Markdown report from evaluation results."""
    n = len(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # NaN-safe averages with coverage
    avg_khr = sum(r["keyword_hit_rate"] for r in results) / n if n else 0
    source_matches = [r for r in results if r.get("source_match") is not None]
    page_matches = [r for r in results if r.get("page_match") is not None]
    smr = sum(1 for r in source_matches if r["source_match"]) / len(source_matches) if source_matches else None
    pmr = sum(1 for r in page_matches if r["page_match"]) / len(page_matches) if page_matches else None

    faith_avg, faith_valid, faith_total = _nan_safe_avg(
        [r.get("faithfulness") for r in results]
    )
    rel_avg, rel_valid, rel_total = _nan_safe_avg(
        [r.get("answer_relevancy") for r in results]
    )
    ao_avg, ao_valid, ao_total = _nan_safe_avg(
        [r.get("answer_overlap_rate") for r in results]
    )
    qo_avg, qo_valid, qo_total = _nan_safe_avg(
        [r.get("question_overlap_rate") for r in results]
    )

    lines = [
        "# RAG Accuracy Evaluation Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Questions evaluated:** {n}",
        f"**Total time:** {elapsed_s:.1f}s",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | Score | Coverage |",
        "|---|---|---|",
    ]

    # Faithfulness
    if faith_avg is not None:
        lines.append(f"| **avg_faithfulness** | {faith_avg:.4f} | {faith_valid}/{faith_total} questions |")
    else:
        lines.append(f"| **avg_faithfulness** | — | 0/{faith_total} questions |")

    # Answer relevancy
    if rel_avg is not None:
        lines.append(f"| **avg_answer_relevancy** | {rel_avg:.4f} | {rel_valid}/{rel_total} questions |")
    else:
        lines.append(f"| **avg_answer_relevancy** | — (see note below) | 0/{rel_total} questions |")

    lines.append(f"| **avg_keyword_hit_rate** | {avg_khr:.4f} | {n}/{n} questions |")

    # Overlap metrics (always available — offline)
    if ao_avg is not None:
        lines.append(f"| **avg_answer_overlap_rate** | {ao_avg:.4f} | {ao_valid}/{ao_total} questions |")
    if qo_avg is not None:
        lines.append(f"| **avg_question_overlap_rate** | {qo_avg:.4f} | {qo_valid}/{qo_total} questions |")

    if smr is not None:
        lines.append(f"| **source_match_rate** | {smr:.4f} | {len(source_matches)}/{n} questions |")
    if pmr is not None:
        lines.append(f"| **page_match_rate** | {pmr:.4f} | {len(page_matches)}/{n} questions |")

    # NaN answer_relevancy note
    if rel_avg is None and rel_total > 0:
        lines += [
            "",
            "> **⚠️ answer_relevancy returned NaN for all questions.**",
            "> This is a known RAGAS compatibility issue: the `answer_relevancy` metric",
            "> uses `OpenAIEmbeddings.embed_query()` which may not exist in your",
            "> version of `langchain-openai`. Fix: pin `langchain-openai<0.2` or",
            "> upgrade to `ragas>=0.2`.",
        ]

    # Invalid RAGAS samples
    if invalid_ragas_samples:
        lines += [
            "",
            "---",
            "",
            "## Samples Excluded from RAGAS",
            "",
            "| ID | Reason |",
            "|---|---|",
        ]
        for s in invalid_ragas_samples:
            lines.append(f"| {s['id']} | {s['reason']} |")

    # Per-question details
    lines += [
        "",
        "---",
        "",
        "## Per-Question Results",
        "",
        "| # | Question (truncated) | KW Hit | Ans Overlap | Q Overlap | Src | Faith. | Relev. |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
        sm = "✅" if r.get("source_match") is True else ("❌" if r.get("source_match") is False else "—")
        faith = _fmt_score(r.get("faithfulness"))
        rel = _fmt_score(r.get("answer_relevancy"))
        ao = f"{r.get('answer_overlap_rate', 0):.2f}"
        qo = f"{r.get('question_overlap_rate', 0):.2f}"
        lines.append(f"| {r['id']} | {q_short} | {r['keyword_hit_rate']:.2f} | {ao} | {qo} | {sm} | {faith} | {rel} |")

    # Worst 3 by composite score: min(keyword_hit_rate, question_overlap_rate)
    def _worst_score(r: dict[str, Any]) -> float:
        khr_val = r.get("keyword_hit_rate", 0.0)
        qo_val = r.get("question_overlap_rate", 0.0)
        return min(khr_val, qo_val)

    worst = sorted(results, key=_worst_score)[:3]
    lines += [
        "",
        "---",
        "",
        "## Worst 3 Questions (by min of keyword_hit_rate, question_overlap_rate)",
        "",
    ]
    for r in worst:
        lines.append(f"### {r['id']}: {r['question'][:80]}")
        lines.append(f"- **keyword_hit_rate:** {r['keyword_hit_rate']:.2f}")
        lines.append(f"- **answer_overlap_rate:** {r.get('answer_overlap_rate', 0):.2f}")
        lines.append(f"- **question_overlap_rate:** {r.get('question_overlap_rate', 0):.2f}")
        lines.append(f"- **answer:** {r['answer'][:200]}...")
        missing = r.get("missing_keywords", [])
        if missing:
            lines.append(f"- **missing keywords:** {', '.join(missing)}")
        lines.append("")

    # Guidance
    lines += [
        "---",
        "",
        "## What to Fix",
        "",
    ]

    if faith_avg is not None and faith_avg < 0.7:
        lines += [
            "### ⚠️ Low Faithfulness",
            "The model is generating answers not grounded in the retrieved context.",
            "- Tighten the system prompt (stricter \"answer only from context\")",
            "- Reduce context size (lower MAX_CHUNK_CHARS)",
            "- Review guardrails for off-topic answers",
            "",
        ]

    if rel_avg is not None and rel_avg < 0.7:
        lines += [
            "### ⚠️ Low Answer Relevancy",
            "Answers are not directly addressing the question.",
            "- Improve chunking strategy (smaller chunks, more overlap)",
            "- Check if query embedding captures Arabic well",
            "- Consider query translation or expansion",
            "",
        ]

    if avg_khr < 0.5:
        lines += [
            "### ⚠️ Low Keyword Hit Rate",
            "Retrieved contexts are missing expected keywords.",
            "- Review ingestion: are PDFs being chunked correctly?",
            "- Increase TOP_K to retrieve more context (test cost impact)",
            "- Check embedding model performance on Arabic text",
            "- Verify the data was ingested into the correct namespace",
            "",
        ]

    has_issues = any([
        faith_avg is not None and faith_avg < 0.7,
        rel_avg is not None and rel_avg < 0.7,
        avg_khr < 0.5,
    ])
    if not has_issues:
        lines.append("All metrics look healthy. No immediate action needed. ✅")
        lines.append("")

    lines += [
        "---",
        "",
        f"*Report generated by `scripts/eval_accuracy.py` on {timestamp}*",
    ]

    return "\n".join(lines)


# Main


def main() -> None:
    load_dotenv()

    # Preflight checks
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    eval_mode = os.getenv("EVAL_MODE", "false").lower() == "true"

    if not openai_key or not pinecone_key:
        log.warning(
            "keys_missing",
            hint="Set OPENAI_API_KEY and PINECONE_API_KEY in .env to run evaluation.",
        )
        sys.exit(0)

    if not eval_mode:
        log.warning(
            "eval_mode_disabled",
            hint="Set EVAL_MODE=true so the RAG pipeline returns contexts for evaluation.",
        )
        sys.exit(0)

    # Load dataset
    if not QA_FILE.exists():
        log.error("dataset_not_found", path=str(QA_FILE))
        sys.exit(1)

    with open(QA_FILE, encoding="utf-8") as f:
        dataset = yaml.safe_load(f)

    if not dataset or not isinstance(dataset, list):
        log.error("dataset_empty_or_invalid", path=str(QA_FILE))
        sys.exit(1)

    log.info("dataset_loaded", questions=len(dataset))

    # Import RAG + RAGAS
    try:
        from app.rag import ask  # noqa: WPS433
    except ImportError as exc:
        log.error("import_error", error=str(exc))
        sys.exit(1)

    ragas_available = False
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from datasets import Dataset as HFDataset
        ragas_available = True
    except ImportError:
        log.warning("ragas_not_available", hint="pip install ragas datasets — RAGAS metrics will be skipped")

    # Run evaluation
    t0 = time.time()
    results: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, str]] = []

    ragas_questions: list[str] = []
    ragas_answers: list[str] = []
    ragas_contexts: list[list[str]] = []
    ragas_ground_truths: list[str] = []
    ragas_id_map: list[str] = []  # Maps RAGAS row index → question id

    for i, item in enumerate(dataset):
        qid = item.get("id", f"q{i+1:02d}")
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        expected_kw = item.get("expected_keywords", [])
        exp_source = item.get("expected_source")
        exp_pages = item.get("expected_pages")

        log.info("evaluating", id=qid, question=question[:60])

        try:
            result = ask(question, history=[])
        except Exception as exc:
            log.error("rag_failed", id=qid, error=type(exc).__name__)
            results.append({
                "id": qid,
                "question": question,
                "answer": f"ERROR: {type(exc).__name__}",
                "sources": [],
                "contexts": [],
                "keyword_hit_rate": 0.0,
                "answer_overlap_rate": 0.0,
                "question_overlap_rate": 0.0,
                "source_match": None,
                "page_match": None,
                "missing_keywords": expected_kw,
            })
            continue

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        contexts = result.get("meta", {}).get("contexts", [])

        # Retrieval metrics
        khr = keyword_hit_rate(contexts, expected_kw)
        sm = source_match(sources, exp_source)
        pm = page_match(sources, exp_pages)

        # Overlap metrics (offline — no API calls)
        ao = answer_context_overlap(answer, contexts)
        qo = question_context_overlap(question, contexts)

        # Missing keywords
        joined_ctx = " ".join(contexts).lower()
        missing_kw = [kw for kw in expected_kw if kw.lower() not in joined_ctx]

        entry: dict[str, Any] = {
            "id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "keyword_hit_rate": khr,
            "answer_overlap_rate": ao,
            "question_overlap_rate": qo,
            "source_match": sm,
            "page_match": pm,
            "missing_keywords": missing_kw,
        }
        results.append(entry)

        # Collect for RAGAS — validate sample first
        if ragas_available:
            if not contexts:
                invalid_samples.append({"id": qid, "reason": "Empty contexts (EVAL_MODE may be off or retrieval returned nothing)"})
            elif not ground_truth.strip():
                invalid_samples.append({"id": qid, "reason": "Empty ground_truth in qa.yaml"})
            else:
                ragas_questions.append(question)
                ragas_answers.append(answer)
                ragas_contexts.append(contexts)
                ragas_ground_truths.append(ground_truth)
                ragas_id_map.append(qid)

    # RAGAS evaluation
    ragas_scores: dict[str, float] | None = None

    if ragas_available and ragas_questions:
        log.info("running_ragas", questions=len(ragas_questions))
        try:
            hf_dataset = HFDataset.from_dict({
                "question": ragas_questions,
                "answer": ragas_answers,
                "contexts": ragas_contexts,
                "ground_truth": ragas_ground_truths,
            })
            ragas_result = ragas_evaluate(hf_dataset, metrics=[faithfulness, answer_relevancy])

            # EvaluationResult API changed across ragas versions:
            # - older: dict-like with .get()
            # - newer: object with subscript access or attributes
            def _extract(obj: Any, key: str, default: float = 0.0) -> float:
                """Safely extract a metric from a RAGAS EvaluationResult."""
                try:
                    return float(obj[key])
                except (TypeError, KeyError):
                    pass
                try:
                    return float(getattr(obj, key, default))
                except (TypeError, ValueError):
                    return default

            ragas_scores = {
                "faithfulness": _extract(ragas_result, "faithfulness"),
                "answer_relevancy": _extract(ragas_result, "answer_relevancy"),
            }
            log.info("ragas_complete", scores=ragas_scores)

            # Merge per-row RAGAS scores back into results
            if hasattr(ragas_result, "to_pandas"):
                df = ragas_result.to_pandas()
                nan_relevancy_count = 0
                for row_idx, qid in enumerate(ragas_id_map):
                    if row_idx >= len(df):
                        break
                    entry = next((r for r in results if r["id"] == qid), None)
                    if entry is None:
                        continue
                    faith_val = df.iloc[row_idx].get("faithfulness", float("nan"))
                    rel_val = df.iloc[row_idx].get("answer_relevancy", float("nan"))
                    entry["faithfulness"] = float(faith_val) if _is_valid_score(faith_val) else None
                    entry["answer_relevancy"] = float(rel_val) if _is_valid_score(rel_val) else None
                    if not _is_valid_score(rel_val):
                        nan_relevancy_count += 1

                if nan_relevancy_count > 0:
                    log.warning(
                        "answer_relevancy_nan",
                        count=nan_relevancy_count,
                        hint="RAGAS answer_relevancy returned NaN — likely a langchain-openai embed_query incompatibility.",
                    )

        except Exception as exc:
            log.error("ragas_failed", error=str(exc))
            ragas_scores = None

    elapsed = time.time() - t0

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # JSON results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "questions_evaluated": len(results),
                "elapsed_seconds": round(elapsed, 1),
                "ragas_scores": ragas_scores,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log.info("results_saved", path=str(RESULTS_FILE))

    # Markdown report
    report = generate_report(results, ragas_scores, elapsed, invalid_samples or None)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("report_saved", path=str(REPORT_FILE))

    # NaN-safe terminal summary
    n = len(results)
    avg_khr = sum(r["keyword_hit_rate"] for r in results) / n if n else 0
    faith_avg, faith_valid, faith_total = _nan_safe_avg(
        [r.get("faithfulness") for r in results]
    )
    rel_avg, rel_valid, rel_total = _nan_safe_avg(
        [r.get("answer_relevancy") for r in results]
    )

    print("\n" + "=" * 60)
    print("  RAG Accuracy Evaluation Complete")
    print("=" * 60)
    print(f"  Questions evaluated:    {n}")
    print(f"  Time:                   {elapsed:.1f}s")
    print(f"  Avg keyword hit rate:   {avg_khr:.2%}")
    ao_avg_t, ao_v, ao_t = _nan_safe_avg([r.get("answer_overlap_rate") for r in results])
    qo_avg_t, qo_v, qo_t = _nan_safe_avg([r.get("question_overlap_rate") for r in results])
    if ao_avg_t is not None:
        print(f"  Avg answer overlap:     {ao_avg_t:.4f}  ({ao_v}/{ao_t} valid)")
    if qo_avg_t is not None:
        print(f"  Avg question overlap:   {qo_avg_t:.4f}  ({qo_v}/{qo_t} valid)")
    if faith_avg is not None:
        print(f"  Avg faithfulness:       {faith_avg:.4f}  ({faith_valid}/{faith_total} valid)")
    else:
        print(f"  Avg faithfulness:       —  (0/{faith_total} valid)")
    if rel_avg is not None:
        print(f"  Avg answer relevancy:   {rel_avg:.4f}  ({rel_valid}/{rel_total} valid)")
    else:
        print(f"  Avg answer relevancy:   —  (0/{rel_total} valid — see report)")
    if invalid_samples:
        print(f"  RAGAS excluded:         {len(invalid_samples)} sample(s)")
    print(f"\n  Results: {RESULTS_FILE}")
    print(f"  Report:  {REPORT_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
