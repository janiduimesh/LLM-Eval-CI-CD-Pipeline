from src.config import THRESHOLDS


def check(summary: dict, thresholds: dict = None) -> tuple[bool, dict]:
    
    thresholds = thresholds or THRESHOLDS

    report = {
        "accuracy": {
            "value": summary.get("avg_accuracy", 0),
            "threshold": thresholds["accuracy"],
            "operator": ">=",
            "passed": summary.get("avg_accuracy", 0) >= thresholds["accuracy"],
        },
        "hallucination": {
            "value": summary.get("avg_hallucination", 1),
            "threshold": thresholds["hallucination"],
            "operator": "<=",
            "passed": summary.get("avg_hallucination", 1) <= thresholds["hallucination"],
        },
        "latency": {
            "value": summary.get("avg_latency", 0),
            "threshold": thresholds["latency"],
            "operator": "<=",
            "passed": summary.get("avg_latency", 0) <= thresholds["latency"],
        },
        "cost": {
            "value": summary.get("avg_cost", 0),
            "threshold": thresholds["cost"],
            "operator": "<=",
            "passed": summary.get("avg_cost", 0) <= thresholds["cost"],
        },
        "failed_questions": {
            "value": summary.get("failed_count", 0),
            "threshold": thresholds["failed_questions"],
            "operator": "<=",
            "passed": summary.get("failed_count", 0) <= thresholds["failed_questions"],
        },
    }

    all_passed = all(metric["passed"] for metric in report.values())

    return all_passed, report


def format_report(report: dict) -> str:
   
    lines = []
    lines.append("=" * 60)
    lines.append("THRESHOLD CHECK REPORT")
    lines.append("=" * 60)

    for metric_name, details in report.items():
        status = "✅ PASS" if details["passed"] else "❌ FAIL"
        operator = details["operator"]
        value = details["value"]
        threshold = details["threshold"]

        if isinstance(value, float):
            if metric_name == "cost":
                value_str = f"${value:.6f}"
                threshold_str = f"${threshold:.6f}"
            elif metric_name in ("accuracy", "hallucination"):
                value_str = f"{value:.2%}"
                threshold_str = f"{threshold:.2%}"
            else:
                value_str = f"{value:.4f}s"
                threshold_str = f"{threshold:.4f}s"
        else:
            value_str = str(value)
            threshold_str = str(threshold)

        lines.append(
            f"  {status}  {metric_name:<20s}  "
            f"Value: {value_str:<12s}  {operator} Threshold: {threshold_str}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
