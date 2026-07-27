"""Phase 1 — Supervised Fine-Tuning (SFT) with LoRA on a small instruct model.

This is the baseline every RL post-training run starts from. Goal: a clean SFT
checkpoint with a sensible loss curve, then use it as the DPO starting point.

Usage:
    uv run python phase1-dpo/scripts/train_sft.py --config phase1-dpo/configs/sft.yaml

Notes:
    - Requires a CUDA GPU. Small model (1.5B) fits comfortably on one 24GB+ card.
    - Set report_to: none in the config to run without W&B.
"""

from __future__ import annotations

import argparse

from common import (
    get_lora_config,
    get_quantization_config,
    load_config,
    load_tokenizer,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training (Phase 1).")
    parser.add_argument("--config", required=True, help="Path to sft.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # Imports here so the file can be inspected without the ML stack installed.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    tokenizer = load_tokenizer(cfg)

    # Datasets
    ds = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])
    max_samples = cfg.get("max_samples")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    def to_text(example):
        """Render chat-format examples into a single training string.

        ultrachat_200k provides a `messages` list; adapt if your dataset differs.
        """
        messages = example.get("messages")
        if messages is not None:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback: expect a plain `text` field.
            text = example["text"]
        return {"text": text}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    # Models
    quant_config = get_quantization_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else "auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    # Training
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=cfg.get("bf16", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        seed=cfg["seed"],
        max_seq_length=cfg["max_seq_length"],
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name"),
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=get_lora_config(cfg),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"SFT complete. Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
