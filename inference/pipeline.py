"""
inference/pipeline.py
=====================
Orchestrates batched inference for a single model over a dataset.

Responsibilities:
    - Split samples into batches of configurable size.
    - Build prompts for each batch via data/formatter.py.
    - Call model.generate() and record timing via LatencyTracker.
    - Return a flat list of prediction dicts, one per sample.

Each prediction dict contains:
    {
        "id":           str   — sample ID from loader,
        "question":     str   — original question,
        "options":      dict  — answer choices,
        "answer":       str   — ground-truth answer letter,
        "dataset_name": str   — dataset alias,
        "raw_output":   str   — raw text from model.generate(),
        "model_alias":  str   — model alias,
        "prompt_style": str   — prompt style used,
    }

Answer extraction (predicted_answer) is handled downstream in
evaluation/exact_match.py so that extraction logic stays testable
in isolation from generation.
"""

import logging
import math

from data.formatter import build_prompts_batch
from inference.tracker import LatencyTracker
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Runs batched inference for one model over one dataset split.

    Args:
        model:            A loaded BaseModel instance.
        batch_size:       Number of samples per generation call.
        prompt_style:     Key into PROMPT_FORMATTERS (from eval_config.yaml).
    """

    def __init__(self, model: BaseModel, batch_size: int, prompt_style: str):
        self.model = model
        self.batch_size = batch_size
        self.prompt_style = prompt_style
        self.tracker = LatencyTracker(model_alias=model.model_alias)

    def run(self, samples: list[dict]) -> list[dict]:
        """
        Run inference over all samples and return prediction dicts.

        Args:
            samples: List of normalised sample dicts from data/loader.py.

        Returns:
            List of prediction dicts with raw_output added.
            Order matches the input samples exactly.
        """
        n_samples = len(samples)
        n_batches = math.ceil(n_samples / self.batch_size)

        logger.info(
            f"[{self.model.model_alias}] Starting inference: "
            f"{n_samples} samples | batch_size={self.batch_size} | "
            f"{n_batches} batches | prompt_style='{self.prompt_style}'"
        )

        predictions: list[dict] = []

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n_samples)
            batch_samples = samples[start:end]
            actual_batch_size = len(batch_samples)

            # -----------------------------------------------------------------
            # Build prompts for this batch
            # -----------------------------------------------------------------
            prompts = build_prompts_batch(batch_samples, self.prompt_style)

            # -----------------------------------------------------------------
            # Generate with timing
            # -----------------------------------------------------------------
            self.tracker.start_batch(batch_size=actual_batch_size)
            try:
                raw_outputs: list[str] = self.model.generate(prompts)
            except Exception as exc:
                # Log and fill blanks so the rest of the run continues
                logger.error(
                    f"[{self.model.model_alias}] Generation failed on batch "
                    f"{batch_idx + 1}/{n_batches}: {exc}"
                )
                raw_outputs = [""] * actual_batch_size
            elapsed = self.tracker.end_batch()

            # -----------------------------------------------------------------
            # Assemble prediction dicts
            # -----------------------------------------------------------------
            for sample, raw_output in zip(batch_samples, raw_outputs):
                pred = {
                    **sample,                               # id, question, options, answer, dataset_name
                    "raw_output":   raw_output.strip(),
                    "model_alias":  self.model.model_alias,
                    "prompt_style": self.prompt_style,
                }
                predictions.append(pred)

            logger.info(
                f"[{self.model.model_alias}] Batch {batch_idx + 1}/{n_batches} "
                f"done in {elapsed:.2f}s "
                f"({elapsed / actual_batch_size:.3f}s/sample)"
            )

        logger.info(
            f"[{self.model.model_alias}] Inference complete. "
            f"Total: {len(predictions)} predictions."
        )
        return predictions