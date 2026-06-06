"""
data/formatter.py
=================
Converts normalised sample dicts into prompt strings ready for model input.

Each formatter function receives a sample dict (as produced by data/loader.py)
and returns a single prompt string.

Adding a new prompt style:
    1. Write a function `format_<style_name>(sample: dict) -> str`.
    2. Register it in PROMPT_FORMATTERS at the bottom of this file.
    3. Reference the style name in configs/eval_config.yaml under
       `datasets.<name>.prompt_style`.

Design philosophy:
    - Prompts are explicit and unambiguous: the model is told exactly what
      format its answer should take ("Reply with only the letter A, B, C, or D").
    - No chain-of-thought by default (we want greedy single-token answers for
      accurate exact-match evaluation).
    - Chat-style models receive a system message + user turn via apply_chat_template.
      Base/causal models receive a plain completion prompt.
"""

import logging
from typing import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt formatters
# =============================================================================

def format_mcq_4opt(sample: dict) -> str:
    """
    Format a 4-option multiple choice question as a plain completion prompt.

    Used for base/causal models that do not use a chat template.
    The prompt ends just before the answer token so the model completes it.

    Example output:
        The following is a medical multiple-choice question.
        Choose the single best answer. Reply with only the letter A, B, C, or D.

        Question: A 45-year-old man presents with...

        A. Metformin
        B. Insulin
        C. Glipizide
        D. Acarbose

        Answer:
    """
    question = sample["question"]
    options = sample["options"]

    option_lines = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(options.items())
    )

    prompt = (
        "The following is a medical multiple-choice question.\n"
        "Choose the single best answer. Reply with only the letter A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
        f"{option_lines}\n\n"
        "Answer:"
    )
    return prompt


def format_mcq_4opt_chat(sample: dict) -> list[dict]:
    """
    Format a 4-option MCQ as a chat message list.

    Used with tokenizer.apply_chat_template() for instruction-tuned models
    (e.g. Gemma-IT, MedGemma-IT, OpenBioLLM).

    Returns a list of message dicts in the OpenAI-compatible format:
        [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
        ]

    The InferencePipeline will call tokenizer.apply_chat_template(messages)
    when it detects this format.
    """
    question = sample["question"]
    options = sample["options"]

    option_lines = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(options.items())
    )

    system_message = (
        "You are a medical expert. Answer the following multiple-choice question "
        "by replying with only the single letter of the correct answer: A, B, C, or D. "
        "Do not explain your reasoning."
    )

    user_message = (
        f"Question: {question}\n\n"
        f"{option_lines}\n\n"
        "Answer:"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]


def format_open_ended(sample: dict) -> str:
    """
    Format an open-ended question as a plain completion prompt.

    Used for base/causal models on preference or free-text datasets.
    The model is asked to answer the medical question directly.
    """
    question = sample["question"]
    return (
        "You are a medical expert. Answer the following question accurately and concisely.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def format_open_ended_chat(sample: dict) -> list[dict]:
    """
    Format an open-ended question as a chat message list.

    Used with instruction-tuned models for preference evaluation.
    Returns a list of message dicts for tokenizer.apply_chat_template().
    """
    question = sample["question"]

    return [
        {
            "role": "system",
            "content": (
                "You are a medical expert. Answer questions accurately and concisely. "
                "Provide a clear, informative response."
            ),
        },
        {
            "role": "user",
            "content": f"{question}",
        },
    ]


# =============================================================================
# Registry
# =============================================================================

PROMPT_FORMATTERS: dict[str, Callable] = {
    "mcq_4opt":           format_mcq_4opt,
    "mcq_4opt_chat":      format_mcq_4opt_chat,
    "open_ended":         format_open_ended,
    "open_ended_chat":    format_open_ended_chat,
}


# =============================================================================
# Public API
# =============================================================================

def build_prompt(sample: dict, prompt_style: str) -> str | list[dict]:
    """
    Build a prompt for a single sample using the specified style.

    Args:
        sample:       Normalised sample dict from data/loader.py.
        prompt_style: Key into PROMPT_FORMATTERS (from eval_config.yaml).

    Returns:
        A prompt string (for base models) or a list of message dicts
        (for chat/instruction-tuned models).

    Raises:
        ValueError: If the prompt_style is not registered.
    """
    if prompt_style not in PROMPT_FORMATTERS:
        raise ValueError(
            f"Unknown prompt style '{prompt_style}'. "
            f"Available: {list(PROMPT_FORMATTERS.keys())}"
        )
    return PROMPT_FORMATTERS[prompt_style](sample)


def build_prompts_batch(
    samples: list[dict],
    prompt_style: str,
) -> list[str | list[dict]]:
    """
    Build prompts for a batch of samples.

    Args:
        samples:      List of normalised sample dicts.
        prompt_style: Key into PROMPT_FORMATTERS.

    Returns:
        List of prompts in the same order as the input samples.
    """
    return [build_prompt(sample, prompt_style) for sample in samples]