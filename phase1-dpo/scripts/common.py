"""Shared helpers for Phase 1 (SFT + DPO) training scripts.

Keeps config loading, LoRA setup, and seeding in one place so the SFT and DPO
scripts stay small and the *hyperparameters* (the actual lessons) live in YAML.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducibility (a Phase 1 principle)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lora_config(cfg: dict[str, Any]):
    """Build a PEFT LoraConfig from the YAML config, or return None if disabled."""
    if not cfg.get("use_lora", False):
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_quantization_config(cfg: dict[str, Any]):
    """Return a 4-bit BitsAndBytesConfig for QLoRA, or None."""
    if not cfg.get("load_in_4bit", False):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(cfg: dict[str, Any]):
    """Load the tokenizer with a sane pad token (Qwen needs this handled)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=cfg.get("trust_remote_code", True)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@dataclass
class ResolvedPaths:
    """Convenience for resolving config paths relative to phase1-dpo/."""

    root: Path

    def out(self, cfg: dict[str, Any]) -> Path:
        return self.root / cfg["output_dir"]
