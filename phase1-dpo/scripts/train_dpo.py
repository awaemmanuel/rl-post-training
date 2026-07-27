"""Phase 1: Direct Preference Optimization (DPO) from an SFT checkpoint.

Lesson focus: beta and the reference model are the whole game. Watch the reward
margin (chosen minus rejected logps) AND the KL to the reference together. A rising
reward margin with a controlled KL is a healthy run; a runaway KL means the policy
is drifting into fluent nonsense.

Usage:
    uv run python phase1-dpo/scripts/train_dpo.py --config phase1-dpo/configs/dpo.yaml

Notes:
    Requires a CUDA GPU. Set report_to: none in the config to run without W&B.
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
    parser = argparse.ArgumentParser(description="DPO training (Phase 1).")
    parser.add_argument("--config", required=True, help="Path to dpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # Imports here so the file can be inspected without the ML stack installed.
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    tokenizer = load_tokenizer(cfg)

    # Datasets
    # ultrafeedback_binarized already provides `chosen`/`rejected` as message
    # lists. DPOTrainer accepts this conversational format directly.
    train_ds = load_dataset(cfg["dataset_name"], split=cfg["dataset_split"])
    max_samples = cfg.get("max_samples")
    if max_samples:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

    eval_ds = None
    if cfg.get("eval_split"):
        eval_ds = load_dataset(cfg["dataset_name"], split=cfg["eval_split"])

    # Models
    # With LoRA, TRL uses the frozen base weights as the implicit reference model,
    # so we do not load a separate ref model here.
    quant_config = get_quantization_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else "auto",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )

    # Training
    dpo_config = DPOConfig(
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
        eval_strategy=cfg.get("eval_strategy", "no"),
        eval_steps=cfg.get("eval_steps"),
        save_strategy=cfg["save_strategy"],
        seed=cfg["seed"],
        beta=cfg["beta"],
        loss_type=cfg["loss_type"],
        max_prompt_length=cfg["max_prompt_length"],
        max_length=cfg["max_length"],
        report_to=cfg.get("report_to", "none"),
        run_name=cfg.get("run_name"),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=get_lora_config(cfg),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"DPO complete. Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
