"""RAGAS-based evaluation for the RAG pipeline.

Measures faithfulness and answer relevancy on a small Q/A dataset.
Requires EVAL_MODE=true and valid API keys.

Usage:
    set EVAL_MODE=true
    python scripts/eval.py
"""

from __future__ import annotations

import os
import sys

import structlog
from dotenv import load_dotenv

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

# Evaluation Dataset

EVAL_DATASET = [
    {
        "question": "What is Saudi Vision 2030?",
        "ground_truth": "Saudi Vision 2030 is a strategic framework to reduce Saudi Arabia's dependence on oil and diversify its economy.",
    },
    {
        "question": "What are the main goals of Vision 2030?",
        "ground_truth": "The main goals include a vibrant society, a thriving economy, and an ambitious nation.",
    },
    {
        "question": "How does Vision 2030 plan to diversify the economy?",
        "ground_truth": "Vision 2030 plans to diversify through tourism, entertainment, technology, and reducing oil dependency.",
    },
    {
        "question": "What role does education play in Vision 2030?",
        "ground_truth": "Education is central to Vision 2030, aiming to improve quality and align with labor market needs.",
    },
    {
        "question": "What is the Public Investment Fund?",
        "ground_truth": "The Public Investment Fund (PIF) is the sovereign wealth fund of Saudi Arabia, a key driver of Vision 2030.",
    },
]


# Main


def main() -> None:
    load_dotenv()

    # Check required keys
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
            hint="Set EVAL_MODE=true in .env so the RAG pipeline returns contexts for RAGAS.",
        )
        sys.exit(0)

    # Import RAG + RAGAS (only when keys are available)
    try:
        from app.rag import ask  # noqa: WPS433
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
        from datasets import Dataset
    except ImportError as exc:
        log.error("import_error", error=str(exc), hint="pip install ragas datasets")
        sys.exit(1)

    # Run the pipeline for each question
    log.info("eval_starting", questions=len(EVAL_DATASET))

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in EVAL_DATASET:
        question = item["question"]
        log.info("evaluating", question=question[:60])

        try:
            result = ask(question, history=[])
        except Exception as exc:
            log.error("pipeline_error", question=question[:40], error=type(exc).__name__)
            continue

        answer = result.get("answer", "")
        ctx = result.get("meta", {}).get("contexts", [])

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx if ctx else ["No context retrieved."])
        ground_truths.append(item["ground_truth"])

    if not questions:
        log.error("no_results", hint="All questions failed. Check API keys and Pinecone index.")
        sys.exit(1)

    # Build RAGAS dataset
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Evaluate
    log.info("running_ragas_metrics")

    try:
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    except Exception as exc:
        log.error("ragas_error", error=type(exc).__name__, detail=str(exc)[:200])
        sys.exit(1)

    # Print results
    print()
    print("=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    for metric_name, score in result.items():
        print(f"  {metric_name}: {score:.4f}")
    print("=" * 60)
    print()

    log.info("eval_complete", results=dict(result))


if __name__ == "__main__":
    main()
