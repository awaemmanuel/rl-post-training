# Phase 1: SFT + DPO

Fine-tune a small instruct model with SFT, then run Direct Preference Optimization,
and measure a held-out win-rate lift over the SFT baseline.

Model: `Qwen/Qwen2.5-1.5B-Instruct` (LoRA). Small on purpose so iterations are cheap;
the skills and failure modes transfer identically at larger scale.

## Lesson focus

- The reference model and KL behavior are the whole game in DPO.
- Watch the reward margin (chosen minus rejected logps) AND the KL together.
- A "win" is only real against the correct baseline on a held-out prompt set.

## Layout

```
phase1-dpo/
├── configs/
│   ├── sft.yaml           # SFT hyperparameters
│   └── dpo.yaml           # DPO hyperparameters (ablate beta, learning_rate first)
├── scripts/
│   ├── common.py          # config loader, LoRA/quant setup, seeding
│   ├── train_sft.py       # SFT training
│   ├── train_dpo.py       # DPO training
│   └── eval_winrate.py    # head-to-head win-rate eval (SFT vs DPO)
└── eval_prompts.jsonl     # small held-out prompt set for eval
```

## Setup (on a CUDA GPU host)

```bash
# From the repo root. Installs the base training stack.
uv sync

# Optional: log in to Weights & Biases (or set report_to: none in the configs).
wandb login
```

Note: `torch`, `bitsandbytes`, and (later) `vllm` need a CUDA machine. The repo can
be edited on a non-GPU machine, but training must run on a GPU.

## Run

```bash
# 1. SFT baseline
uv run python phase1-dpo/scripts/train_sft.py --config phase1-dpo/configs/sft.yaml

# 2. DPO from the SFT checkpoint
#    (dpo.yaml's model_name points at the SFT output_dir by default)
uv run python phase1-dpo/scripts/train_dpo.py --config phase1-dpo/configs/dpo.yaml

# 3. Win-rate eval: SFT baseline vs DPO policy
#    Set OPENAI_API_KEY to enable LLM-as-judge; otherwise pairs are written for
#    manual labeling.
uv run python phase1-dpo/scripts/eval_winrate.py \
    --baseline outputs/sft-qwen2.5-1.5b \
    --policy   outputs/dpo-qwen2.5-1.5b \
    --prompts  phase1-dpo/eval_prompts.jsonl \
    --out      outputs/eval
```

## Ablations (Week 3)

Change one knob at a time and log the result:

- `beta`: try 0.05, 0.1, 0.5. Lower beta lets the policy drift further from the
  reference (higher reward margin, higher KL, more collapse risk).
- `learning_rate`: try 1e-6 to 1e-5. DPO wants a smaller lr than SFT.
- `loss_type`: `sigmoid` (DPO) vs `ipo` vs `hinge`. SimPO/ORPO use their own
  trainers; add them as separate configs if comparing.

## What good looks like

- Reproducible win-rate lift over the SFT baseline on the held-out prompts.
- Reward margin rises steadily; KL to the reference stays controlled.
- You can explain each hyperparameter's effect from your own curves.

## Failure modes to document

- beta too low: fluent nonsense, runaway KL.
- learning_rate too high: reward margin spikes then collapses.
- wrong reference model: an illusory "win" measured against the wrong baseline.
