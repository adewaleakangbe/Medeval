"""
evaluation/metrics.py
=====================
Aggregates scored predictions into a per-model summary dict.

Computes:
    - Overall accuracy and exact-match score.
    - Per-answer-option accuracy breakdown (useful for detecting answer bias).
    - Extraction failure rate.
    - Combines with latency stats from LatencyTracker.summary().

The output of compute_metrics() is the canonical per-model result record
that gets written to JSON/CSV by results/writer.py.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: list[dict],
    latency_summary: dict,
    dataset_name: str,
) -> dict:
    """
    Compute evaluation metrics for a single model's predictions on one dataset.

    Expects each prediction dict to contain:
        "correct"          (bool) : Set by evaluation/exact_match.score_predictions().
        "predicted_answer" (str)  : Extracted letter or "".
        "answer"           (str)  : Ground-truth letter.
        "model_alias"      (str)  : Model name.

    Args:
        predictions:     Scored prediction dicts from exact_match.score_predictions().
        latency_summary: Dict from LatencyTracker.summary().
        dataset_name:    Short dataset alias from eval_config.yaml.

    Returns:
        Dict containing all accuracy, per-option, and latency metrics.
        Suitable for direct serialisation to JSON.
    """
    if not predictions:
        logger.warning("compute_metrics() called with empty predictions list.")
        return {}

    model_alias: str = predictions[0].get("model_alias", "unknown")
    n_total = len(predictions)
    n_correct = sum(1 for p in predictions if p.get("correct", False))
    n_extraction_failed = sum(1 for p in predictions if p.get("predicted_answer", "") == "")

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    extraction_failure_rate = n_extraction_failed / n_total if n_total > 0 else 0.0

    # -------------------------------------------------------------------------
    # Per ground-truth option accuracy breakdown
    # e.g. "How accurate is the model when the correct answer is A?"
    # High variance here can indicate answer-selection bias.
    # -------------------------------------------------------------------------
    per_option_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})
    for pred in predictions:
        gt = pred.get("answer", "").upper()
        if gt in {"A", "B", "C", "D"}:
            per_option_counts[gt]["total"] += 1
            if pred.get("correct", False):
                per_option_counts[gt]["correct"] += 1

    per_option_accuracy = {}
    for option in ["A", "B", "C", "D"]:
        counts = per_option_counts.get(option, {"total": 0, "correct": 0})
        t = counts["total"]
        c = counts["correct"]
        per_option_accuracy[option] = {
            "total":    t,
            "correct":  c,
            "accuracy": round(c / t, 4) if t > 0 else None,
        }

    # -------------------------------------------------------------------------
    # Prediction distribution (what letters does the model actually output?)
    # -------------------------------------------------------------------------
    pred_distribution: dict[str, int] = defaultdict(int)
    for pred in predictions:
        pred_distribution[pred.get("predicted_answer", "") or "FAILED"] += 1

    # -------------------------------------------------------------------------
    # Assemble final summary
    # -------------------------------------------------------------------------
    summary = {
        # Identity
        "model_alias":              model_alias,
        "dataset_name":             dataset_name,
        # Core accuracy
        "n_total":                  n_total,
        "n_correct":                n_correct,
        "accuracy":                 round(accuracy, 4),
        "accuracy_pct":             round(accuracy * 100, 2),
        # Answer extraction quality
        "n_extraction_failed":      n_extraction_failed,
        "extraction_failure_rate":  round(extraction_failure_rate, 4),
        # Per-option breakdown
        "per_option_accuracy":      per_option_accuracy,
        # Prediction distribution (bias check)
        "prediction_distribution":  dict(pred_distribution),
        # Latency (merged from LatencyTracker)
        **latency_summary,
    }

    logger.info(
        f"[{model_alias}] Metrics: accuracy={accuracy:.1%} "
        f"({n_correct}/{n_total}) | "
        f"extraction_failures={n_extraction_failed} | "
        f"throughput={latency_summary.get('throughput_sps', 'N/A')} sps"
    )

    return summary