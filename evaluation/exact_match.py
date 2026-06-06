"""
evaluation/exact_match.py
=========================
Extracts a predicted answer letter from raw model output and scores it
against the ground truth using exact match.

Strategy (configurable via eval_config.yaml → evaluation.extract_strategy):

    "first_token"
        Take the first non-whitespace character of the output.
        Works well when the model is properly instructed to reply with a
        single letter. Fast and robust for instruction-tuned models.

    "regex"
        Search for the first occurrence of a standalone A/B/C/D in the
        output using a regex. More tolerant of models that output
        "The answer is A." instead of just "A".

Adding a new strategy:
    1. Write a function `_extract_<name>(raw_output: str) -> str`.
    2. Register it in EXTRACTION_STRATEGIES.
    3. Reference the name in eval_config.yaml.
"""

import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Extraction strategies
# =============================================================================

def _extract_first_token(raw_output: str) -> str:
    """
    Return the first non-whitespace character of the output, uppercased.

    Suitable for models that reliably output "A", "B", "C", or "D" as the
    first token when prompted correctly.
    """
    stripped = raw_output.strip()
    if not stripped:
        return ""
    return stripped[0].upper()


def _extract_regex(raw_output: str) -> str:
    """
    Find the first standalone A/B/C/D letter in the output.

    Handles outputs like:
        "A"
        "The answer is B."
        "Based on the information, C is correct."
        "(A) Metformin"

    Returns "" if no valid answer letter is found.
    """
    # Match a letter A-D that is either at the start of the string,
    # preceded by a non-letter character, or inside parentheses.
    pattern = r'(?:^|(?<=[^a-zA-Z]))([ABCD])(?=[^a-zA-Z]|$)'
    match = re.search(pattern, raw_output.strip().upper())
    return match.group(1) if match else ""


EXTRACTION_STRATEGIES = {
    "first_token": _extract_first_token,
    "regex":       _extract_regex,
}


# =============================================================================
# Public API
# =============================================================================

def extract_answer(raw_output: str, strategy: str = "first_token") -> str:
    """
    Extract a predicted answer letter from a model's raw output string.

    Args:
        raw_output: Raw text returned by model.generate().
        strategy:   Extraction strategy key (see module docstring).

    Returns:
        Extracted letter ("A", "B", "C", "D") or "" if extraction fails.
    """
    if strategy not in EXTRACTION_STRATEGIES:
        raise ValueError(
            f"Unknown extraction strategy '{strategy}'. "
            f"Available: {list(EXTRACTION_STRATEGIES.keys())}"
        )
    extractor = EXTRACTION_STRATEGIES[strategy]
    predicted = extractor(raw_output)

    if predicted not in {"A", "B", "C", "D"}:
        logger.debug(
            f"Extraction returned non-standard answer '{predicted}' "
            f"from output: {raw_output!r}"
        )
        return ""   # Treat as a missed prediction

    return predicted


def score_predictions(
    predictions: list[dict],
    extract_strategy: str = "first_token",
    valid_answers: set[str] | None = None,
) -> list[dict]:
    """
    Add extracted prediction and correctness flag to each prediction dict.

    Mutates each dict in-place by adding:
        "predicted_answer" (str)  : Extracted letter or "" on failure.
        "correct"          (bool) : True if predicted_answer == answer.

    Args:
        predictions:      List of prediction dicts from inference/pipeline.py.
        extract_strategy: Answer extraction strategy.
        valid_answers:    Set of valid answer letters. Defaults to {"A","B","C","D"}.

    Returns:
        The same list with "predicted_answer" and "correct" added to each dict.
    """
    if valid_answers is None:
        valid_answers = {"A", "B", "C", "D"}

    n_correct = 0
    n_missing = 0

    for pred in predictions:
        raw_output: str = pred.get("raw_output", "")
        ground_truth: str = pred.get("answer", "").strip().upper()

        predicted = extract_answer(raw_output, strategy=extract_strategy)
        is_correct = (predicted == ground_truth) and (predicted != "")

        pred["predicted_answer"] = predicted
        pred["correct"] = is_correct

        if is_correct:
            n_correct += 1
        if predicted == "":
            n_missing += 1

    n_total = len(predictions)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    logger.info(
        f"Scoring complete: {n_correct}/{n_total} correct "
        f"({accuracy:.1%} accuracy) | {n_missing} extraction failures"
    )

    return predictions