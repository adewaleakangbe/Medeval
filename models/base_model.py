"""
models/base_model.py
====================
Abstract base class defining the contract every model must satisfy.

All concrete model implementations (HuggingFace, API-based, etc.) inherit
from BaseModel and implement the abstract methods below.

This design means:
    - The inference pipeline only depends on BaseModel, not any specific loader.
    - Adding a new model type (e.g. an OpenAI API model) requires only a new
      subclass — zero changes to the pipeline or evaluation code.
    - Type checkers and IDEs can surface missing implementations immediately.

Usage:
    Do not instantiate BaseModel directly. Use a concrete subclass such as
    HuggingFaceModel from models/hf_model.py.
"""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract interface for all evaluation models.

    Subclasses must implement:
        - load()
        - generate(prompts)
        - unload()

    Subclasses should also set self.model_alias in their __init__ so that
    results can be attributed correctly in the output files.
    """

    def __init__(self, model_alias: str, config: dict):
        """
        Args:
            model_alias: Short name for this model (e.g. "gemma3_4b").
                         Used as a key in result files. Should match the alias
                         in eval_config.yaml.
            config:      The per-model config dict from eval_config.yaml
                         (e.g. {"hf_id": "...", "dtype": "bfloat16", ...}).
        """
        self.model_alias = model_alias
        self.config = config
        self._is_loaded = False

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # -------------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and tokenizer into memory.

        Should set self._is_loaded = True on success.
        Should log the model size, device placement, and dtype.
        Raises RuntimeError if loading fails.
        """
        ...

    @abstractmethod
    def generate(self, prompts: list[str | list[dict]]) -> list[str]:
        """
        Run inference on a batch of prompts and return raw generated text.

        Args:
            prompts: List of prompts. Each element is either:
                     - A plain string (for base/completion models).
                     - A list of message dicts (for chat/IT models), which
                       the implementation should pass through
                       tokenizer.apply_chat_template().

        Returns:
            List of generated strings in the same order as the input prompts.
            The strings should be the raw model output — answer extraction
            happens in evaluation/exact_match.py, not here.

        Raises:
            RuntimeError: If generate() is called before load().
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """
        Release model weights from GPU/CPU memory.

        Should delete the model and tokenizer objects and call
        torch.cuda.empty_cache() where applicable.
        Sets self._is_loaded = False.
        """
        ...

    # -------------------------------------------------------------------------
    # Shared helpers — available to all subclasses
    # -------------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Raise RuntimeError if the model has not been loaded yet."""
        if not self._is_loaded:
            raise RuntimeError(
                f"Model '{self.model_alias}' has not been loaded. "
                "Call model.load() before model.generate()."
            )

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"{self.__class__.__name__}(alias='{self.model_alias}', status={status})"