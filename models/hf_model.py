"""
models/hf_model.py
==================
Concrete BaseModel implementation for HuggingFace transformers models.

Handles:
    - Auto device mapping across available GPUs / CPU fallback.
    - Optional 4-bit and 8-bit quantisation via bitsandbytes.
    - Chat template formatting for instruction-tuned models.
    - Batched generation with configurable decoding parameters.
    - Clean memory release after evaluation.

Supported model families:
    Gemma 3, MedGemma, OpenBioLLM (Llama-3 based), OpenLLaMA, and any other
    model loadable via AutoModelForCausalLM + AutoTokenizer.
"""

import logging
import time
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseModel):
    """
    HuggingFace-backed model for evaluation.

    Config keys (from eval_config.yaml → models.<alias>):
        hf_id           (str)  : HuggingFace model repository ID.
        dtype           (str)  : "bfloat16" | "float16" | "float32".
        load_in_4bit    (bool) : Enable 4-bit quantisation (requires bitsandbytes).
        load_in_8bit    (bool) : Enable 8-bit quantisation (requires bitsandbytes).
        device_map      (str)  : "auto" | "cuda:0" | "cpu".
        max_new_tokens  (int)  : Maximum tokens to generate per prompt.
        trust_remote_code (bool): Pass to from_pretrained for models with custom code.
    """

    def __init__(self, model_alias: str, config: dict, inference_config: dict):
        """
        Args:
            model_alias:      Short name matching the key in eval_config.yaml.
            config:           Per-model config dict from eval_config.yaml.
            inference_config: Shared inference config dict (batch_size, do_sample, etc.).
        """
        super().__init__(model_alias, config)
        self.inference_config = inference_config

        # These are set during load()
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # -------------------------------------------------------------------------
    # BaseModel interface
    # -------------------------------------------------------------------------

    def load(self) -> None:
        """
        Download (or load from cache) the model and tokenizer.

        Applies quantisation config if requested. Logs device placement,
        parameter count, and dtype after loading.
        """
        hf_id: str = self.config["hf_id"]
        dtype_str: str = self.config.get("dtype", "bfloat16")
        load_in_4bit: bool = self.config.get("load_in_4bit", False)
        load_in_8bit: bool = self.config.get("load_in_8bit", False)
        device_map: str = self.config.get("device_map", "auto")
        trust_remote_code: bool = self.config.get("trust_remote_code", False)

        logger.info(f"[{self.model_alias}] Loading tokenizer from '{hf_id}'...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            trust_remote_code=trust_remote_code,
            padding_side="left",   # Left-pad for batch generation
        )

        # Most models lack a pad token; reuse eos_token to avoid errors
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.debug(f"[{self.model_alias}] pad_token set to eos_token.")

        # ---------------------------------------------------------------------
        # Quantisation config (optional)
        # ---------------------------------------------------------------------
        bnb_config = None
        if load_in_4bit:
            logger.info(f"[{self.model_alias}] Applying 4-bit quantisation (bitsandbytes).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            logger.info(f"[{self.model_alias}] Applying 8-bit quantisation (bitsandbytes).")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # ---------------------------------------------------------------------
        # Resolve torch dtype
        # ---------------------------------------------------------------------
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # ---------------------------------------------------------------------
        # Load model weights
        # ---------------------------------------------------------------------
        logger.info(f"[{self.model_alias}] Loading model weights from '{hf_id}'...")
        t0 = time.perf_counter()

        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": hf_id,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        self._model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        self._model.eval()   # Disable dropout etc. for deterministic inference

        elapsed = time.perf_counter() - t0
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e9
        logger.info(
            f"[{self.model_alias}] Loaded in {elapsed:.1f}s | "
            f"~{n_params:.1f}B params | dtype={torch_dtype}"
        )

        self._is_loaded = True

    def generate(self, prompts: list[str | list[dict]]) -> list[str]:
        """
        Run batched generation on a list of prompts.

        Handles both plain string prompts (base models) and message-list
        prompts (chat/IT models) transparently.

        Args:
            prompts: List of prompt strings or message-list dicts.

        Returns:
            List of raw generated strings (model output only, no prompt text).
        """
        self._require_loaded()

        max_new_tokens: int = self.config.get("max_new_tokens", 16)
        do_sample: bool = self.inference_config.get("do_sample", False)
        temperature: float = self.inference_config.get("temperature", 1.0)
        top_p: float = self.inference_config.get("top_p", 1.0)

        # ---------------------------------------------------------------------
        # Convert each prompt to a token string via chat template or plain
        # ---------------------------------------------------------------------
        formatted_prompts: list[str] = []
        for prompt in prompts:
            if isinstance(prompt, list):
                # Chat-template format — apply the model's template
                try:
                    text = self._tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as exc:
                    # Fallback: concatenate role + content if template fails
                    logger.warning(
                        f"[{self.model_alias}] apply_chat_template failed ({exc}); "
                        "falling back to plain concatenation."
                    )
                    text = "\n".join(
                        f"{msg['role'].upper()}: {msg['content']}" for msg in prompt
                    ) + "\nASSISTANT:"
            else:
                text = prompt
            formatted_prompts.append(text)

        # ---------------------------------------------------------------------
        # Tokenise (batched, left-padded)
        # ---------------------------------------------------------------------
        inputs = self._tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        # Move to the same device as the model (handles multi-GPU gracefully)
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ---------------------------------------------------------------------
        # Generate
        # ---------------------------------------------------------------------
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # ---------------------------------------------------------------------
        # Decode — strip prompt tokens, return only generated text
        # ---------------------------------------------------------------------
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]   # Slice off the prompt

        decoded: list[str] = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return decoded

    def unload(self) -> None:
        """
        Delete model and tokenizer from memory and free GPU cache.
        """
        logger.info(f"[{self.model_alias}] Unloading model from memory.")
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"[{self.model_alias}] CUDA cache cleared.")

        self._is_loaded = False