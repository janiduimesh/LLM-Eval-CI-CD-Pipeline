"""
Report generator for the evaluation pipeline.
Handles JSON result output, CSV history logging, and stdout summary printing.
"""

import os
import json
import csv
from datetime import datetime, timezone


def generate_json_report(results: list[dict], filepath: str) -> None:
    """
    Write detailed per-question evaluation results to a JSON file.

    Args:
        results: List of per-question result dicts.
        filepath: Path to the output JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[ReportGenerator] JSON report saved to {filepath}")


def append_to_history(summary: dict, filepath: str) -> None:
    """
    Append a summary row to the history CSV file.

    Creates the file with headers if it doesn't exist.

    Args:
        summary: Aggregate metrics summary dict.
        filepath: Path to the history CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    headers = [
        "timestamp",
        "accuracy",
        "hallucination_rate",
        "avg_latency",
        "avg_cost",
        "failed_count",
        "total_questions",
        "pass_fail",
    ]

    file_exists = os.path.isfile(filepath)

    # Check if file exists but is empty or only has headers
    write_header = not file_exists
    if file_exists:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                write_header = True

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy": round(summary.get("avg_accuracy", 0), 4),
            "hallucination_rate": round(summary.get("avg_hallucination", 0), 4),
            "avg_latency": round(summary.get("avg_latency", 0), 4),
            "avg_cost": round(summary.get("avg_cost", 0), 8),
            "failed_count": summary.get("failed_count", 0),
            "total_questions": summary.get("total_questions", 0),
            "pass_fail": "PASS" if summary.get("passed", False) else "FAIL",
        }
        writer.writerow(row)

    print(f"[ReportGenerator] History appended to {filepath}")


def print_summary(summary: dict, threshold_report: dict) -> None:
    """
    Pretty-print the evaluation summary to stdout.

    Args:
        summary: Aggregate metrics summary dict.
        threshold_report: Per-metric pass/fail report from threshold_checker.
    """
    passed = summary.get("passed", False)
    status_emoji = "✅" if passed else "❌"
    status_text = "PASSED" if passed else "FAILED"

    print()
    print("╔" + "═" * 62 + "╗")
    print(f"║  {status_emoji} LLM EVALUATION RESULTS — {status_text}".ljust(63) + "║")
    print("╠" + "═" * 62 + "╣")
    print(f"║  Total Questions:    {summary.get('total_questions', 0):<10}".ljust(63) + "║")
    print(f"║  Failed Questions:   {summary.get('failed_count', 0):<10}".ljust(63) + "║")
    print("╠" + "═" * 62 + "╣")

    for metric_name, details in threshold_report.items():
        status = "✅" if details["passed"] else "❌"
        value = details["value"]

        if metric_name == "cost":
            value_str = f"${value:.6f}"
        elif metric_name in ("accuracy", "hallucination"):
            value_str = f"{value:.2%}"
        elif metric_name == "latency":
            value_str = f"{value:.4f}s"
        else:
            value_str = str(value)

        line = f"║  {status} {metric_name:<20s} {value_str:<15s}"
        print(line.ljust(63) + "║")

    print("╚" + "═" * 62 + "╝")
    print()
