"""
data/loader.py
==============
Loads and prepares evaluation datasets from HuggingFace Hub.

Responsibilities:
    - Download and cache datasets via the `datasets` library.
    - Apply optional sample limits or fractional sampling.
    - Return a consistent list-of-dicts structure for the rest of the pipeline.

Two sample schemas are supported:

MCQ schema (used by medqa_usmle_4opt, ultramedical_mcq):
    {
        "id":           str   — unique sample identifier,
        "question":     str   — the question text,
        "options":      dict  — e.g. {"A": "...", "B": "...", "C": "...", "D": "..."},
        "answer":       str   — ground-truth answer letter, e.g. "A",
        "dataset_name": str   — short dataset alias from config,
        "eval_type":    str   — "mcq",
    }

Preference schema (used by ultramedical_preference):
    {
        "id":           str   — unique sample identifier,
        "question":     str   — the prompt/question text,
        "chosen":       str   — the preferred (higher quality) response,
        "rejected":     str   — the dispreferred (lower quality) response,
        "dataset_name": str   — short dataset alias from config,
        "eval_type":    str   — "preference",
        "options":      dict  — empty dict (keeps pipeline compatible),
        "answer":       str   — empty string (keeps pipeline compatible),
    }

Adding a new dataset:
    1. Add its entry to configs/eval_config.yaml.
    2. Add a parser function `_parse_<dataset_name>(row)` in this file.
    3. Register it in DATASET_PARSERS at the bottom of this file.
"""

import logging
import random
import re

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Per-dataset row parsers
# =============================================================================
# Each parser receives a raw HuggingFace dataset row (dict) and returns a
# normalised sample dict matching one of the schemas described above.
# =============================================================================

def _parse_medqa_usmle_4opt(row: dict, idx: int) -> dict:
    """
    Parse a single row from GBaker/MedQA-USMLE-4-options.

    Expected HF columns: 'question', 'options' (dict), 'answer_idx' or 'answer'.
    We normalise the answer to an uppercase letter ("A", "B", "C", "D").
    """
    options: dict = row.get("options", {})

    # The dataset stores the answer as a letter string e.g. "A"
    raw_answer: str = str(row.get("answer_idx", row.get("answer", ""))).strip().upper()

    # Guard: if the answer is a full-text answer rather than a letter, map it back
    if raw_answer not in {"A", "B", "C", "D"} and raw_answer in options.values():
        raw_answer = next(
            (letter for letter, text in options.items() if text == raw_answer),
            raw_answer,
        )

    return {
        "id": f"medqa_usmle_4opt_{idx}",
        "question": str(row.get("question", "")).strip(),
        "options": {k: str(v).strip() for k, v in options.items()},
        "answer": raw_answer,
        "dataset_name": "medqa_usmle_4opt",
        "eval_type": "mcq",
    }


def _parse_ultramedical_mcq(row: dict, idx: int) -> dict:
    """
    Parse a single Exam row from TsinghuaC3I/UltraMedical.

    Confirmed schema (as of 2026):
        id            : str  — e.g. "MedQA,0"
        type          : str  — "Exam" for all MCQ rows
        answer        : str  — direct answer letter e.g. "A", "B", "C", "D"
        score         : str  — GPT-4 difficulty justification (ignored)
        conversations : list — [
            {"from": "human", "value": "<question text with options embedded>"},
            {"from": "gpt",   "value": "<full explanation — NOT the answer>"},
          ]

    The answer comes from the top-level 'answer' field directly.
    The human turn contains the question + options which we parse via regex.
    The gpt turn is a long explanation — we do NOT use it for the answer.
    """
    conversations: list = row.get("conversations", [])

    if not conversations:
        raise ValueError(f"Row {idx} has no conversation turns.")

    human_turn: str = str(conversations[0].get("value", "")).strip()

    # -------------------------------------------------------------------------
    # Split question text from embedded options.
    # Options appear as "\nA. text\nB. text..." in the human turn.
    # We split on newlines followed by a letter and period.
    # -------------------------------------------------------------------------
    option_pattern = re.compile(r'\n([A-E])\.\s')
    parts = option_pattern.split(human_turn)

    question_text = parts[0].strip()

    options: dict = {}
    i = 1
    while i + 1 < len(parts):
        letter = parts[i].strip().upper()
        text = parts[i + 1].strip()
        # Clean any trailing content from the next option bleeding in
        text = re.split(r'\n[A-E]\.', text)[0].strip()
        options[letter] = text
        i += 2

    # -------------------------------------------------------------------------
    # Answer comes directly from the top-level 'answer' field — clean and simple.
    # -------------------------------------------------------------------------
    answer_letter: str = str(row.get("answer", "")).strip().upper()

    # Validate — skip rows with no parseable answer
    if answer_letter not in {"A", "B", "C", "D", "E"}:
        raise ValueError(
            f"Row {idx} has unrecognised answer '{answer_letter}'. Skipping."
        )

    return {
        "id": str(row.get("id", f"ultramedical_mcq_{idx}")),
        "question": question_text,
        "options": options,
        "answer": answer_letter,
        "dataset_name": "ultramedical_mcq",
        "eval_type": "mcq",
    }


def _parse_ultramedical_preference(row: dict, idx: int) -> dict:
    """
    Parse a single row from TsinghuaC3I/UltraMedical-Preference.

    Expected HF columns:
        'prompt_id'  : unique identifier
        'prompt'     : the question / instruction text
        'chosen'     : the preferred response (higher quality, human/GPT-4 ranked)
        'rejected'   : the dispreferred response (lower quality)

    The chosen/rejected fields may be either:
        - A plain string response.
        - A list of message dicts [{"role": ..., "content": ...}].
          In that case we extract the assistant turn content.

    Evaluation metric: Preference Alignment Score (see evaluation/preference.py).
    The model generates a response to 'prompt'; we measure whether that response
    is more similar to 'chosen' than 'rejected' using ROUGE-L.
    """

    def _extract_text(field) -> str:
        """Extract plain text from either a string or a message list."""
        if isinstance(field, str):
            return field.strip()
        if isinstance(field, list):
            # Find the assistant turn
            for msg in field:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return str(msg.get("content", "")).strip()
            # Fallback: concatenate all content
            return " ".join(str(m.get("content", "")) for m in field).strip()
        return str(field).strip()

    prompt_id = str(row.get("prompt_id", f"ultramedical_pref_{idx}"))
    prompt = _extract_text(row.get("prompt", ""))
    chosen = _extract_text(row.get("chosen", ""))
    rejected = _extract_text(row.get("rejected", ""))

    if not prompt or not chosen or not rejected:
        raise ValueError(
            f"Row {idx} is missing prompt, chosen, or rejected field."
        )

    return {
        "id": prompt_id,
        "question": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "options": {},       # Empty — keeps inference pipeline compatible
        "answer": "",        # Empty — scoring handled by preference evaluator
        "dataset_name": "ultramedical_preference",
        "eval_type": "preference",
    }


# =============================================================================
# Registry — maps dataset alias → parser function
# =============================================================================

DATASET_PARSERS = {
    "medqa_usmle_4opt":        _parse_medqa_usmle_4opt,
    "ultramedical_mcq":        _parse_ultramedical_mcq,
    "ultramedical_preference": _parse_ultramedical_preference,
}


# =============================================================================
# Public API
# =============================================================================

def load_eval_dataset(
    dataset_name: str,
    hf_path: str,
    split: str,
    max_samples: int | None = None,
    sample_fraction: float | None = None,
    mcq_type_filter: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Load a dataset from HuggingFace Hub and return normalised samples.

    Args:
        dataset_name:     Short alias matching a key in DATASET_PARSERS.
        hf_path:          HuggingFace dataset path.
        split:            Dataset split to use (e.g. "train", "test").
        max_samples:      If set, cap the number of samples returned.
        sample_fraction:  If set (0.0–1.0), randomly sample this fraction of
                          the split. Applied before max_samples.
        mcq_type_filter:  If set, pre-filter rows where row["type"] == this value
                          before parsing. Used for UltraMedical where type == "mc"
                          selects only MCQ rows from the mixed dataset.
        seed:             Random seed for reproducible sampling.

    Returns:
        List of normalised sample dicts.

    Raises:
        ValueError: If dataset_name has no registered parser.
        RuntimeError: If the HuggingFace download fails.
    """
    if dataset_name not in DATASET_PARSERS:
        raise ValueError(
            f"No parser registered for dataset '{dataset_name}'. "
            f"Available: {list(DATASET_PARSERS.keys())}"
        )

    parser = DATASET_PARSERS[dataset_name]

    logger.info(f"Loading dataset '{dataset_name}' from '{hf_path}' (split='{split}')...")

    try:
        hf_dataset: Dataset = load_dataset(hf_path, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{hf_path}' from HuggingFace Hub: {exc}"
        ) from exc

    logger.info(f"  Raw size: {len(hf_dataset)} samples")

    # -------------------------------------------------------------------------
    # Optional type filter — used to extract MCQ rows from UltraMedical
    # -------------------------------------------------------------------------
    if mcq_type_filter is not None:
        hf_dataset = hf_dataset.filter(
            lambda row: row.get("type", "") == mcq_type_filter
        )
        logger.info(
            f"  After type filter (type=='{mcq_type_filter}'): "
            f"{len(hf_dataset)} samples"
        )

    # -------------------------------------------------------------------------
    # Optional fractional sampling
    # -------------------------------------------------------------------------
    if sample_fraction is not None:
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1]; got {sample_fraction}")
        n_fraction = max(1, int(len(hf_dataset) * sample_fraction))
        random.seed(seed)
        indices = random.sample(range(len(hf_dataset)), n_fraction)
        hf_dataset = hf_dataset.select(indices)
        logger.info(f"  After fraction ({sample_fraction:.0%}): {len(hf_dataset)} samples")

    # -------------------------------------------------------------------------
    # Optional hard cap
    # -------------------------------------------------------------------------
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        logger.info(f"  After max_samples cap: {len(hf_dataset)} samples")

    # -------------------------------------------------------------------------
    # Parse each row into normalised format
    # -------------------------------------------------------------------------
    samples: list[dict] = []
    for idx, row in enumerate(hf_dataset):
        try:
            sample = parser(row, idx)
            samples.append(sample)
        except Exception as exc:
            logger.warning(f"  Skipping row {idx} due to parse error: {exc}")

    logger.info(f"  Final sample count: {len(samples)}")
    return samples