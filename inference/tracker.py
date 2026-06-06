"""
inference/tracker.py
====================
Tracks latency and throughput metrics during inference.

Records:
    - Per-batch wall-clock time.
    - Per-sample latency (batch_time / batch_size).
    - Aggregate statistics: mean, median, P95, total time, samples/sec.

Usage:
    tracker = LatencyTracker(model_alias="gemma3_4b")
    tracker.start_batch(batch_size=8)
    # ... run inference ...
    tracker.end_batch()
    stats = tracker.summary()
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchRecord:
    """Record for a single inference batch."""
    batch_size: int
    elapsed_seconds: float

    @property
    def per_sample_latency(self) -> float:
        """Average latency per sample in this batch (seconds)."""
        return self.elapsed_seconds / max(self.batch_size, 1)


class LatencyTracker:
    """
    Accumulates per-batch timing records and computes summary statistics.

    Attributes:
        model_alias: Short model name — included in all summary outputs.
        records:     List of BatchRecord objects accumulated during a run.
    """

    def __init__(self, model_alias: str):
        """
        Args:
            model_alias: Short model name used for logging and output labelling.
        """
        self.model_alias = model_alias
        self.records: list[BatchRecord] = []
        self._batch_start: float | None = None
        self._current_batch_size: int = 0

    def start_batch(self, batch_size: int) -> None:
        """
        Mark the start of a new inference batch.

        Args:
            batch_size: Number of samples in this batch.
        """
        self._current_batch_size = batch_size
        self._batch_start = time.perf_counter()

    def end_batch(self) -> float:
        """
        Mark the end of the current batch and record elapsed time.

        Returns:
            Elapsed time for this batch in seconds.

        Raises:
            RuntimeError: If end_batch() is called without a preceding start_batch().
        """
        if self._batch_start is None:
            raise RuntimeError("end_batch() called without a preceding start_batch().")

        elapsed = time.perf_counter() - self._batch_start
        record = BatchRecord(
            batch_size=self._current_batch_size,
            elapsed_seconds=elapsed,
        )
        self.records.append(record)
        self._batch_start = None

        logger.debug(
            f"[{self.model_alias}] Batch of {record.batch_size} samples: "
            f"{elapsed:.2f}s ({record.per_sample_latency:.3f}s/sample)"
        )
        return elapsed

    def summary(self) -> dict:
        """
        Compute aggregate latency and throughput statistics.

        Returns:
            Dict with keys:
                model_alias       (str)   : Model name.
                total_samples     (int)   : Total number of samples processed.
                total_time_s      (float) : Total wall-clock time in seconds.
                throughput_sps    (float) : Samples per second (total).
                latency_mean_s    (float) : Mean per-sample latency.
                latency_median_s  (float) : Median per-sample latency.
                latency_p95_s     (float) : 95th percentile per-sample latency.
                latency_min_s     (float) : Minimum per-sample latency.
                latency_max_s     (float) : Maximum per-sample latency.
        """
        if not self.records:
            logger.warning(f"[{self.model_alias}] No timing records — summary is empty.")
            return {"model_alias": self.model_alias}

        per_sample_latencies = [r.per_sample_latency for r in self.records]
        total_samples = sum(r.batch_size for r in self.records)
        total_time = sum(r.elapsed_seconds for r in self.records)

        return {
            "model_alias":      self.model_alias,
            "total_samples":    total_samples,
            "total_time_s":     round(total_time, 3),
            "throughput_sps":   round(total_samples / total_time, 3) if total_time > 0 else 0.0,
            "latency_mean_s":   round(float(np.mean(per_sample_latencies)), 4),
            "latency_median_s": round(float(np.median(per_sample_latencies)), 4),
            "latency_p95_s":    round(float(np.percentile(per_sample_latencies, 95)), 4),
            "latency_min_s":    round(float(np.min(per_sample_latencies)), 4),
            "latency_max_s":    round(float(np.max(per_sample_latencies)), 4),
        }

    def reset(self) -> None:
        """Clear all accumulated records (e.g. between model evaluations)."""
        self.records = []
        self._batch_start = None
        logger.debug(f"[{self.model_alias}] LatencyTracker reset.")