"""
results/writer.py
=================
Saves evaluation outputs to disk incrementally.

Outputs per model+dataset run:
    results/runs/<timestamp>/<model_alias>_<dataset>_predictions.json
    results/runs/<timestamp>/<model_alias>_<dataset>_predictions.csv
    results/runs/<timestamp>/<model_alias>_<dataset>_summary.json

Saving is incremental: per-sample predictions are written as soon as a model
finishes, so a crash mid-run does not lose completed models' results.

A final combined summary is written at the end of the full run:
    results/runs/<timestamp>/all_models_summary.json
    results/runs/<timestamp>/all_models_summary.csv
"""

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ResultsWriter:
    """
    Writes prediction and summary files for a single evaluation run.

    Args:
        results_dir: Root output directory (from eval_config.yaml).
        formats:     List of output formats: ["json"], ["csv"], or ["json", "csv"].
    """

    def __init__(self, results_dir: str, formats: list[str]):
        self.formats = formats

        # Create a timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(results_dir) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Results will be saved to: {self.run_dir}")

    # -------------------------------------------------------------------------
    # Per-model output
    # -------------------------------------------------------------------------

    def save_predictions(
        self,
        predictions: list[dict],
        model_alias: str,
        dataset_name: str,
    ) -> None:
        """
        Save per-sample predictions for one model + dataset pair.

        Args:
            predictions:  List of scored prediction dicts.
            model_alias:  Short model name.
            dataset_name: Short dataset alias.
        """
        stem = f"{model_alias}_{dataset_name}_predictions"

        if "json" in self.formats:
            path = self.run_dir / f"{stem}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved predictions JSON: {path}")

        if "csv" in self.formats:
            path = self.run_dir / f"{stem}.csv"
            self._write_csv(predictions, path)
            logger.info(f"Saved predictions CSV: {path}")

    def save_model_summary(
        self,
        summary: dict,
        model_alias: str,
        dataset_name: str,
    ) -> None:
        """
        Save the aggregate metric summary for one model + dataset pair.

        Args:
            summary:      Output of evaluation/metrics.compute_metrics().
            model_alias:  Short model name.
            dataset_name: Short dataset alias.
        """
        path = self.run_dir / f"{model_alias}_{dataset_name}_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved model summary: {path}")

    # -------------------------------------------------------------------------
    # Combined run-level output (written at the end of all models)
    # -------------------------------------------------------------------------

    def save_all_summaries(self, all_summaries: list[dict]) -> None:
        """
        Save a combined summary table for all models evaluated in this run.

        Args:
            all_summaries: List of summary dicts, one per model+dataset pair.
        """
        if "json" in self.formats:
            path = self.run_dir / "all_models_summary.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved combined summary JSON: {path}")

        if "csv" in self.formats:
            # Flatten nested dicts (per_option_accuracy, prediction_distribution)
            # into top-level keys for CSV compatibility
            flat_summaries = [_flatten_summary(s) for s in all_summaries]
            path = self.run_dir / "all_models_summary.csv"
            self._write_csv(flat_summaries, path)
            logger.info(f"Saved combined summary CSV: {path}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _write_csv(records: list[dict], path: Path) -> None:
        """Write a list of flat dicts to a CSV file."""
        if not records:
            logger.warning(f"No records to write to {path}.")
            return

        # Collect all keys across all records to handle variable fields
        all_keys: list[str] = []
        seen: set[str] = set()
        for record in records:
            for key in record.keys():
                if key not in seen:
                    all_keys.append(key)
                    seen.add(key)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def _flatten_summary(summary: dict) -> dict:
    """
    Flatten nested dicts in a summary for CSV output.

    E.g. per_option_accuracy["A"]["accuracy"] → per_option_accuracy_A_accuracy
    """
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        flat[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
                else:
                    flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat