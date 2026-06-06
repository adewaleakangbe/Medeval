"""
run_eval.py
===========
Main entry point for the MedEval evaluation pipeline.

Usage:
    # Full run with all models from config
    python run_eval.py

    # Limit to specific models (by alias)
    python run_eval.py --models gemma3_4b medgemma_4b

    # Override max_samples for a quick debug run
    python run_eval.py --max-samples 50

    # Use a custom config file
    python run_eval.py --config configs/eval_config.yaml

Flow:
    1. Load and validate config (YAML).
    2. Load dataset(s).
    3. For each model:
        a. Instantiate and load the model.
        b. Run inference (InferencePipeline).
        c. Score predictions (exact match).
        d. Compute metrics (accuracy + latency).
        e. Save predictions and summary.
        f. Unload the model to free VRAM before the next one.
    4. Save combined summary across all models.
    5. Print a final leaderboard to stdout.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from data.loader import load_eval_dataset
from evaluation.exact_match import score_predictions
from evaluation.metrics import compute_metrics
from evaluation.preference import score_preference_predictions, compute_preference_metrics
from inference.pipeline import InferencePipeline
from models.hf_model import HuggingFaceModel
from results.writer import ResultsWriter

# =============================================================================
# Logging setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("medeval_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Config helpers
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from: {config_path}")
    return config


def resolve_models(config: dict, requested: list[str] | None) -> dict:
    """
    Return the subset of models to evaluate.

    Args:
        config:    Full config dict.
        requested: List of model aliases from CLI, or None to use all.

    Returns:
        Dict of {alias: model_config} to evaluate.
    """
    all_models: dict = config.get("models", {})
    if not requested:
        return all_models
    selected = {}
    for alias in requested:
        if alias not in all_models:
            raise ValueError(
                f"Requested model '{alias}' not found in config. "
                f"Available: {list(all_models.keys())}"
            )
        selected[alias] = all_models[alias]
    return selected


# =============================================================================
# Leaderboard printer
# =============================================================================

def print_leaderboard(all_summaries: list[dict]) -> None:
    """Print a formatted leaderboard to stdout, split by eval_type."""
    if not all_summaries:
        return

    mcq_summaries = [s for s in all_summaries if s.get("eval_type", "mcq") != "preference"]
    pref_summaries = [s for s in all_summaries if s.get("eval_type") == "preference"]

    if mcq_summaries:
        print(
            f"\n{'='*72}\n"
            f"{'MedEval Results — MCQ Accuracy':^72}\n"
            f"{'='*72}\n"
            f"{'Model':<20} {'Dataset':<22} {'Accuracy':>10} {'Throughput':>12} {'P95 Lat':>10}\n"
            f"{'-'*72}"
        )
        for s in sorted(mcq_summaries, key=lambda x: x.get("accuracy", 0), reverse=True):
            print(
                f"{s.get('model_alias',''):<20} "
                f"{s.get('dataset_name',''):<22} "
                f"{s.get('accuracy_pct', 0):>9.2f}% "
                f"{s.get('throughput_sps', 0):>10.2f}/s "
                f"{s.get('latency_p95_s', 0):>10.3f}s"
            )
        print("=" * 72)

    if pref_summaries:
        print(
            f"\n{'='*80}\n"
            f"{'MedEval Results — Preference Alignment':^80}\n"
            f"{'='*80}\n"
            f"{'Model':<20} {'Dataset':<22} {'ROUGE PAS':>10} {'BERT PAS':>10} {'Combined':>10}\n"
            f"{'-'*80}"
        )
        for s in sorted(pref_summaries, key=lambda x: x.get("preference_alignment_score", 0), reverse=True):
            bert_pas = s.get("bert_pas_pct")
            bert_str = f"{bert_pas:>9.2f}%" if bert_pas is not None else f"{'N/A':>10}"
            print(
                f"{s.get('model_alias',''):<20} "
                f"{s.get('dataset_name',''):<22} "
                f"{s.get('rouge_pas_pct', 0):>9.2f}% "
                f"{bert_str} "
                f"{s.get('pas_pct', 0):>9.2f}%"
            )
        print("=" * 80 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedEval — LLM evaluation on medical QA benchmarks."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Space-separated list of model aliases to evaluate (default: all).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples for all datasets (useful for debugging).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1. Load config
    # -------------------------------------------------------------------------
    config = load_config(args.config)

    global_seed: int = config.get("seed", 42)
    inference_cfg: dict = config.get("inference", {})
    eval_cfg: dict = config.get("evaluation", {})
    output_cfg: dict = config.get("output", {})

    writer = ResultsWriter(
        results_dir=output_cfg.get("results_dir", "results/runs"),
        formats=output_cfg.get("formats", ["json", "csv"]),
    )

    models_to_eval = resolve_models(config, args.models)
    datasets_cfg: dict = config.get("datasets", {})
    extract_strategy: str = eval_cfg.get("extract_strategy", "first_token")

    logger.info(
        f"Evaluation plan: {len(models_to_eval)} model(s) × "
        f"{len(datasets_cfg)} dataset(s) = "
        f"{len(models_to_eval) * len(datasets_cfg)} run(s)"
    )

    all_summaries: list[dict] = []

    # -------------------------------------------------------------------------
    # 2. Loop over datasets
    # -------------------------------------------------------------------------
    for dataset_alias, dataset_cfg in datasets_cfg.items():
        max_samples = args.max_samples or dataset_cfg.get("max_samples")
        prompt_style: str = dataset_cfg.get("prompt_style", "mcq_4opt")
        eval_type: str = dataset_cfg.get("eval_type", "mcq")
        mcq_type_filter: str | None = dataset_cfg.get("mcq_type_filter")

        samples = load_eval_dataset(
            dataset_name=dataset_alias,
            hf_path=dataset_cfg["hf_path"],
            split=dataset_cfg.get("split", "train"),
            max_samples=max_samples,
            sample_fraction=dataset_cfg.get("sample_fraction"),
            mcq_type_filter=mcq_type_filter,
            seed=global_seed,
        )

        if not samples:
            logger.warning(f"No samples loaded for '{dataset_alias}'. Skipping.")
            continue

        # ---------------------------------------------------------------------
        # 3. Loop over models
        # ---------------------------------------------------------------------
        for model_alias, model_cfg in models_to_eval.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {model_alias} on {dataset_alias} (eval_type={eval_type})")
            logger.info(f"{'='*60}")

            # 3a. Load model
            model = HuggingFaceModel(
                model_alias=model_alias,
                config=model_cfg,
                inference_config=inference_cfg,
            )
            try:
                model.load()
            except Exception as exc:
                logger.error(f"Failed to load model '{model_alias}': {exc}. Skipping.")
                continue

            # 3b. Run inference (same for both eval types)
            pipeline = InferencePipeline(
                model=model,
                batch_size=inference_cfg.get("batch_size", 8),
                prompt_style=prompt_style,
            )
            predictions = pipeline.run(samples)
            latency_summary = pipeline.tracker.summary()

            # 3c & 3d. Score + compute metrics — branched by eval_type
            if eval_type == "preference":
                # Preference alignment evaluation (ROUGE-L chosen vs rejected)
                predictions = score_preference_predictions(predictions)
                summary = compute_preference_metrics(
                    predictions=predictions,
                    latency_summary=latency_summary,
                    dataset_name=dataset_alias,
                )
            else:
                # Default: MCQ exact-match accuracy
                predictions = score_predictions(
                    predictions,
                    extract_strategy=extract_strategy,
                )
                summary = compute_metrics(
                    predictions=predictions,
                    latency_summary=latency_summary,
                    dataset_name=dataset_alias,
                )

            # 3e. Save outputs
            if output_cfg.get("save_per_sample", True):
                writer.save_predictions(predictions, model_alias, dataset_alias)
            if output_cfg.get("save_summary", True):
                writer.save_model_summary(summary, model_alias, dataset_alias)

            all_summaries.append(summary)

            # 3f. Unload model — free VRAM before next model
            model.unload()

    # -------------------------------------------------------------------------
    # 4. Save combined summary and print leaderboard
    # -------------------------------------------------------------------------
    writer.save_all_summaries(all_summaries)
    print_leaderboard(all_summaries)

    logger.info("Run complete. All results saved.")


if __name__ == "__main__":
    main()