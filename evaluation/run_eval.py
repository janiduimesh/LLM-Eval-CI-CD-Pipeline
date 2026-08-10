"""
Main evaluation runner.
Loads the golden dataset, runs each question through the RAG chatbot,
computes metrics, checks thresholds, and generates reports.
"""

import json
import sys

from src.rag_chatbot import RAGChatbot
from src.config import GOLDEN_DATASET_PATH, EVAL_RESULTS_PATH, HISTORY_CSV_PATH, THRESHOLDS

from evaluation.metrics import (
    accuracy_score,
    hallucination_score,
    latency_metric,
    cost_metric,
    is_failed,
)
from evaluation.threshold_checker import check, format_report
from evaluation.report_generator import (
    generate_json_report,
    append_to_history,
    print_summary,
)


def load_golden_dataset(filepath: str = None) -> list[dict]:
    """Load the golden dataset from JSON file."""
    filepath = filepath or GOLDEN_DATASET_PATH
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Eval] Loaded {len(data)} questions from {filepath}")
    return data


def run(dataset_path: str = None) -> int:
    """
    Run the full evaluation pipeline.

    Args:
        dataset_path: Optional path to a custom golden dataset.

    Returns:
        Exit code: 0 for pass, 1 for fail.
    """
    print("=" * 60)
    print("  LLM EVALUATION PIPELINE — STARTING")
    print("=" * 60)
    print()

    # Load dataset
    dataset = load_golden_dataset(dataset_path)

    # Initialize RAG chatbot
    chatbot = RAGChatbot()

    # Evaluate each question
    per_question_results = []
    total_accuracy = 0.0
    total_hallucination = 0.0
    total_latency = 0.0
    total_cost = 0.0
    failed_count = 0

    for i, item in enumerate(dataset):
        question_id = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "unknown")

        print(f"[Eval] [{i + 1}/{len(dataset)}] Evaluating {question_id}: {question[:60]}...")

        try:
            # Get chatbot answer
            result = chatbot.answer(question)

            # Compute metrics
            acc = accuracy_score(result["answer"], ground_truth)
            hall = hallucination_score(result["answer"], result["context_used"])
            lat = latency_metric(result["latency"])
            cst = result["cost"]  # Already calculated by LLMClient
            failed = is_failed(acc, hall, lat, cst, THRESHOLDS)

            if failed:
                failed_count += 1

            # Accumulate
            total_accuracy += acc
            total_hallucination += hall
            total_latency += lat
            total_cost += cst

            question_result = {
                "id": question_id,
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": result["answer"],
                "context_used": result["context_used"],
                "sources": result["sources"],
                "metrics": {
                    "accuracy": acc,
                    "hallucination": hall,
                    "latency": lat,
                    "cost": cst,
                },
                "tokens": result["tokens"],
                "failed": failed,
            }

            per_question_results.append(question_result)

            status = "❌ FAIL" if failed else "✅ PASS"
            print(f"         {status}  Acc={acc:.2%}  Hall={hall:.2%}  Lat={lat:.2f}s  Cost=${cst:.6f}")

        except Exception as e:
            print(f"         ⚠️  ERROR: {e}")
            failed_count += 1
            per_question_results.append({
                "id": question_id,
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": f"ERROR: {str(e)}",
                "context_used": "",
                "sources": [],
                "metrics": {
                    "accuracy": 0.0,
                    "hallucination": 1.0,
                    "latency": 0.0,
                    "cost": 0.0,
                },
                "tokens": {},
                "failed": True,
            })

    # Calculate averages
    n = len(dataset)
    summary = {
        "total_questions": n,
        "avg_accuracy": total_accuracy / n if n > 0 else 0,
        "avg_hallucination": total_hallucination / n if n > 0 else 0,
        "avg_latency": total_latency / n if n > 0 else 0,
        "avg_cost": total_cost / n if n > 0 else 0,
        "failed_count": failed_count,
        "total_cost": total_cost,
        "total_latency": total_latency,
    }

    # Check thresholds
    passed, threshold_report = check(summary)
    summary["passed"] = passed

    # Generate reports
    generate_json_report(per_question_results, EVAL_RESULTS_PATH)
    append_to_history(summary, HISTORY_CSV_PATH)
    print_summary(summary, threshold_report)
    print(format_report(threshold_report))

    # Return exit code
    if passed:
        print("🎉 Evaluation PASSED — pipeline may proceed.")
        return 0
    else:
        print("🚫 Evaluation FAILED — merge blocked.")
        return 1


if __name__ == "__main__":
    exit_code = run()
    sys.exit(exit_code)
