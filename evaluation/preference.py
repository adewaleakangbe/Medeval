"""
evaluation/preference.py
========================
Evaluates model outputs against UltraMedical-Preference pairs using two
complementary metrics:

    1. ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
       - Measures longest common subsequence overlap between model output
         and chosen/rejected responses.
       - Fast, no GPU needed, pure Python.
       - Captures surface-level lexical similarity.

    2. BERTScore
       - Uses contextual BERT embeddings to measure semantic similarity.
       - Captures meaning even when wording differs (paraphrases score high).
       - Recommended by supervisor; standard in NLG evaluation literature.
       - Requires: pip install bert-score
       - Model: microsoft/deberta-xlarge-mnli (best F1 for English medical text)

Both metrics are used to compute a Preference Alignment Score (PAS):

    For each sample:
        - Compute similarity(output, chosen)  → sim_chosen
        - Compute similarity(output, rejected) → sim_rejected
        - If sim_chosen > sim_rejected → preference = 1.0 (aligned with human preference)
        - If sim_chosen < sim_rejected → preference = 0.0 (misaligned)
        - If equal                    → preference = 0.5 (tie)

    PAS = mean(preference scores across all samples)
        PAS = 1.0 → always aligns with human preference
        PAS = 0.5 → random baseline
        PAS < 0.5 → systematically prefers lower-quality responses

BERTScore gracefully degrades to ROUGE-L only if bert-score is not installed,
with a clear warning so you know which metric is active.
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BERTScore availability check
# =============================================================================

try:
    from bert_score import score as _bert_score_fn
    BERTSCORE_AVAILABLE = True
    logger.info("BERTScore is available and will be used for preference evaluation.")
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning(
        "bert-score not installed. BERTScore will be skipped. "
        "Install with: pip install bert-score"
    )


# =============================================================================
# ROUGE-L implementation (pure Python, no dependencies)
# =============================================================================

def _lcs_length(a: list, b: list) -> int:
    """
    Compute the Longest Common Subsequence length of two token lists.
    Uses space-optimised O(min(|a|, |b|)) dynamic programming.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b):
            if token_a == token_b:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev = curr
    return prev[len(b)]


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return re.findall(r'\b\w+\b', text.lower())


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between a hypothesis and reference string.

    Args:
        hypothesis: Generated text (model output).
        reference:  Reference text (chosen or rejected response).

    Returns:
        ROUGE-L F1 in [0.0, 1.0]. Returns 0.0 if either string is empty.
    """
    hyp_tokens = _tokenize(hypothesis)
    ref_tokens = _tokenize(reference)

    if not hyp_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return round((2 * precision * recall) / (precision + recall), 6)


# =============================================================================
# BERTScore computation (batched for efficiency)
# =============================================================================

def compute_bertscore_batch(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "roberta-large",
) -> list[float]:
    """
    Compute BERTScore F1 for a batch of hypothesis-reference pairs.

    Uses DeBERTa-xlarge-mnli by default — highest correlation with human
    judgements for English text per the BERTScore paper. Falls back to
    distilbert-base-uncased if the large model fails (e.g. low VRAM).

    Args:
        hypotheses:  List of generated texts (model outputs).
        references:  List of reference texts (chosen or rejected responses).
        model_type:  HuggingFace model ID for BERTScore computation.

    Returns:
        List of BERTScore F1 values in [0.0, 1.0], one per pair.
        Returns list of 0.0s if BERTScore is unavailable or fails.
    """
    if not BERTSCORE_AVAILABLE:
        return [0.0] * len(hypotheses)

    if not hypotheses or not references:
        return []

    # Replace empty strings with a placeholder to avoid BERTScore errors
    safe_hyps = [h if h.strip() else "[EMPTY]" for h in hypotheses]
    safe_refs = [r if r.strip() else "[EMPTY]" for r in references]

    try:
        _, _, F1 = _bert_score_fn(
            cands=safe_hyps,
            refs=safe_refs,
            model_type=model_type,
            lang="en",
            verbose=False,
            device=None,   # Auto-detect GPU/CPU
        )
        return [round(float(f), 6) for f in F1.tolist()]

    except Exception as exc:
        logger.warning(
            f"BERTScore failed with model '{model_type}': {exc}. "
            "Falling back to distilbert-base-uncased."
        )
        try:
            _, _, F1 = _bert_score_fn(
                cands=safe_hyps,
                refs=safe_refs,
                model_type="distilbert-base-uncased",
                lang="en",
                verbose=False,
            )
            return [round(float(f), 6) for f in F1.tolist()]
        except Exception as exc2:
            logger.error(f"BERTScore fallback also failed: {exc2}. Returning zeros.")
            return [0.0] * len(hypotheses)


# =============================================================================
# Preference scoring — ROUGE-L + BERTScore combined
# =============================================================================

def _preference_vote(sim_chosen: float, sim_rejected: float) -> float:
    """
    Convert similarity scores to a preference vote.

    Returns:
        1.0 if sim_chosen > sim_rejected (model aligns with human preference)
        0.0 if sim_chosen < sim_rejected (model misaligns)
        0.5 if equal (tie)
    """
    if sim_chosen > sim_rejected:
        return 1.0
    elif sim_chosen < sim_rejected:
        return 0.0
    return 0.5


def score_preference_predictions(
    predictions: list[dict],
    bertscore_model: str = "roberta-large",
) -> list[dict]:
    """
    Score model outputs against chosen/rejected pairs using ROUGE-L and BERTScore.

    Adds the following fields to each prediction dict in-place:

        ROUGE-L fields:
            "rouge_chosen"        (float) : ROUGE-L F1 vs chosen response.
            "rouge_rejected"      (float) : ROUGE-L F1 vs rejected response.
            "rouge_preference"    (float) : 1.0 / 0.5 / 0.0 ROUGE-L alignment.
            "rouge_aligned"       (bool)  : True if rouge_chosen > rouge_rejected.

        BERTScore fields (if bert-score is installed):
            "bert_chosen"         (float) : BERTScore F1 vs chosen response.
            "bert_rejected"       (float) : BERTScore F1 vs rejected response.
            "bert_preference"     (float) : 1.0 / 0.5 / 0.0 BERTScore alignment.
            "bert_aligned"        (bool)  : True if bert_chosen > bert_rejected.

        Combined field:
            "preference"          (float) : Mean of rouge_preference and bert_preference
                                            (or just rouge_preference if BERTScore unavailable).
            "aligned"             (bool)  : True if preference > 0.5.

    Args:
        predictions:     List of prediction dicts with "raw_output", "chosen", "rejected".
        bertscore_model: HuggingFace model ID for BERTScore. Override to use a
                         lighter model if VRAM is limited.

    Returns:
        The same list with all preference fields added.
    """
    n_total = len(predictions)
    if n_total == 0:
        return predictions

    outputs  = [p.get("raw_output", "") for p in predictions]
    chosens  = [p.get("chosen", "")     for p in predictions]
    rejecteds = [p.get("rejected", "")  for p in predictions]

    # -------------------------------------------------------------------------
    # ROUGE-L — computed per-sample (fast, no batching needed)
    # -------------------------------------------------------------------------
    logger.info("Computing ROUGE-L scores...")
    rouge_chosen_scores   = [rouge_l_f1(o, c) for o, c in zip(outputs, chosens)]
    rouge_rejected_scores = [rouge_l_f1(o, r) for o, r in zip(outputs, rejecteds)]

    # -------------------------------------------------------------------------
    # BERTScore — batched for efficiency
    # -------------------------------------------------------------------------
    if BERTSCORE_AVAILABLE:
        logger.info(f"Computing BERTScore (model: {bertscore_model})...")
        logger.info("  This may take a few minutes on first run (model download).")
        bert_chosen_scores   = compute_bertscore_batch(outputs, chosens,   bertscore_model)
        bert_rejected_scores = compute_bertscore_batch(outputs, rejecteds, bertscore_model)
    else:
        bert_chosen_scores   = [0.0] * n_total
        bert_rejected_scores = [0.0] * n_total

    # -------------------------------------------------------------------------
    # Assign scores to each prediction dict
    # -------------------------------------------------------------------------
    n_rouge_aligned = 0
    n_bert_aligned  = 0
    n_combined_aligned = 0

    for i, pred in enumerate(predictions):
        rc = rouge_chosen_scores[i]
        rr = rouge_rejected_scores[i]
        bc = bert_chosen_scores[i]
        br = bert_rejected_scores[i]

        rouge_pref = _preference_vote(rc, rr)
        bert_pref  = _preference_vote(bc, br)

        # Combined preference — average of both metrics
        if BERTSCORE_AVAILABLE:
            combined_pref = (rouge_pref + bert_pref) / 2.0
        else:
            combined_pref = rouge_pref

        rouge_aligned = rouge_pref == 1.0
        bert_aligned  = bert_pref == 1.0
        combined_aligned = combined_pref > 0.5

        # ROUGE-L fields
        pred["rouge_chosen"]     = rc
        pred["rouge_rejected"]   = rr
        pred["rouge_preference"] = rouge_pref
        pred["rouge_aligned"]    = rouge_aligned

        # BERTScore fields
        pred["bert_chosen"]      = bc
        pred["bert_rejected"]    = br
        pred["bert_preference"]  = bert_pref
        pred["bert_aligned"]     = bert_aligned

        # Combined
        pred["preference"] = combined_pref
        pred["aligned"]    = combined_aligned

        if rouge_aligned:    n_rouge_aligned    += 1
        if bert_aligned:     n_bert_aligned     += 1
        if combined_aligned: n_combined_aligned += 1

    rouge_pas = sum(p["rouge_preference"] for p in predictions) / n_total
    bert_pas  = sum(p["bert_preference"]  for p in predictions) / n_total if BERTSCORE_AVAILABLE else None

    logger.info(
        f"Preference scoring complete: "
        f"ROUGE-L PAS={rouge_pas:.3f} ({n_rouge_aligned}/{n_total} aligned) | "
        + (f"BERTScore PAS={bert_pas:.3f} ({n_bert_aligned}/{n_total} aligned)"
           if BERTSCORE_AVAILABLE else "BERTScore: not available")
    )

    return predictions


# =============================================================================
# Aggregate metrics
# =============================================================================

def compute_preference_metrics(
    predictions: list[dict],
    latency_summary: dict,
    dataset_name: str,
) -> dict:
    """
    Compute aggregate preference alignment metrics across all predictions.

    Args:
        predictions:     Scored preference dicts from score_preference_predictions().
        latency_summary: Dict from LatencyTracker.summary().
        dataset_name:    Short dataset alias from eval_config.yaml.

    Returns:
        Summary dict with ROUGE-L PAS, BERTScore PAS, combined PAS, and latency.
    """
    if not predictions:
        return {}

    model_alias: str = predictions[0].get("model_alias", "unknown")
    n_total = len(predictions)

    # ROUGE-L aggregates
    rouge_chosen_scores   = [p.get("rouge_chosen", 0.0)   for p in predictions]
    rouge_rejected_scores = [p.get("rouge_rejected", 0.0) for p in predictions]
    rouge_prefs           = [p.get("rouge_preference", 0.0) for p in predictions]
    rouge_pas = float(np.mean(rouge_prefs))
    n_rouge_aligned = sum(1 for p in predictions if p.get("rouge_aligned", False))

    # BERTScore aggregates
    bert_chosen_scores   = [p.get("bert_chosen", 0.0)   for p in predictions]
    bert_rejected_scores = [p.get("bert_rejected", 0.0) for p in predictions]
    bert_prefs           = [p.get("bert_preference", 0.0) for p in predictions]
    bert_pas = float(np.mean(bert_prefs)) if BERTSCORE_AVAILABLE else None
    n_bert_aligned = sum(1 for p in predictions if p.get("bert_aligned", False))

    # Combined
    combined_prefs = [p.get("preference", 0.0) for p in predictions]
    combined_pas = float(np.mean(combined_prefs))
    n_combined_aligned = sum(1 for p in predictions if p.get("aligned", False))
    n_tied    = sum(1 for p in predictions if p.get("preference", 0) == 0.5)
    n_rejected = sum(1 for p in predictions if p.get("preference", 0) == 0.0)

    summary = {
        # Identity
        "model_alias":                  model_alias,
        "dataset_name":                 dataset_name,
        "eval_type":                    "preference",
        "n_total":                      n_total,

        # ROUGE-L preference metrics
        "rouge_pas":                    round(rouge_pas, 4),
        "rouge_pas_pct":                round(rouge_pas * 100, 2),
        "rouge_n_aligned":              n_rouge_aligned,
        "mean_rouge_chosen":            round(float(np.mean(rouge_chosen_scores)), 4),
        "mean_rouge_rejected":          round(float(np.mean(rouge_rejected_scores)), 4),
        "mean_rouge_delta":             round(
            float(np.mean(rouge_chosen_scores)) - float(np.mean(rouge_rejected_scores)), 4
        ),

        # BERTScore preference metrics
        "bertscore_available":          BERTSCORE_AVAILABLE,
        "bert_pas":                     round(bert_pas, 4) if bert_pas is not None else None,
        "bert_pas_pct":                 round(bert_pas * 100, 2) if bert_pas is not None else None,
        "bert_n_aligned":               n_bert_aligned,
        "mean_bert_chosen":             round(float(np.mean(bert_chosen_scores)), 4),
        "mean_bert_rejected":           round(float(np.mean(bert_rejected_scores)), 4),
        "mean_bert_delta":              round(
            float(np.mean(bert_chosen_scores)) - float(np.mean(bert_rejected_scores)), 4
        ),

        # Combined metric
        "preference_alignment_score":   round(combined_pas, 4),
        "pas_pct":                      round(combined_pas * 100, 2),
        "n_aligned_with_chosen":        n_combined_aligned,
        "n_aligned_with_rejected":      n_rejected,
        "n_tied":                       n_tied,

        # Latency
        **latency_summary,
    }

    logger.info(
        f"[{model_alias}] Preference summary: "
        f"ROUGE-L PAS={rouge_pas:.1%} | "
        + (f"BERTScore PAS={bert_pas:.1%} | " if bert_pas is not None else "")
        + f"Combined PAS={combined_pas:.1%}"
    )

    return summary