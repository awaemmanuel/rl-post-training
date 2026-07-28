# Reinforcement Learning Post-Training & Agentic RL

Hands-on projects building **RL post-training** and **agentic RL** skills, with the
goal of capturing lessons learned along the way.

The work is organized as a progressive series of real training runs — SFT →
preference optimization → reward modeling + PPO → GRPO/RLVR → agentic RL — each
producing a runnable repo, honest evaluation, and documented lessons learned.

See **[ROADMAP.md](./ROADMAP.md)** for the full 4-phase, week-by-week plan.

## Motivation

RL post-training is where a capable base model becomes genuinely useful — aligned,
instruction-following, and able to reason and act. It is also notoriously hard to
get right: reward hacking, KL blowups, policy collapse, and rollout throughput all
conspire against you. The fastest way to build real intuition is to run the methods
end-to-end on small models, break things, and write down *why* they broke and what
fixed them. That is the entire point of this repo.

## Approach

- **Depth over breadth.** A few real, working training runs beat many toy demos.
- **Reproducible by default.** Fixed seeds, clean configs, honest evaluation with
  baselines and learning curves — never "vibes".
- **Small models on purpose.** Runs use 0.5B–3B models so iterations are cheap and
  fast; the skills and failure modes transfer identically at 70B.
- **Document the failures.** Reward-hacking and collapse war stories are the most
  valuable lessons learned and the most useful thing to be able to explain.
- **One project, many lessons.** Each phase yields a reproduction, an ablation, an
  optional upstream contribution, and a writeup.

## Phases

The roadmap builds difficulty progressively, roughly mirroring how the field itself
evolved from offline preference methods to online RL and agentic training.

| Phase | Focus | Key methods | Deliverable |
|-------|-------|-------------|-------------|
| 1 | SFT + first preference win | LoRA/QLoRA SFT, DPO/SimPO/ORPO | Writeup + `phase1-dpo/` |
| 2 | Reward modeling + PPO | Bradley-Terry RM, PPO | Repro + `phase2-ppo/` |
| 3 | GRPO + RLVR | GRPO, verifiable rewards (R1-style) | Repro + ablation + OSS PR |
| 4 | Agentic RL | Tool-use env, multi-turn RL | Capstone + `phase4-agentic/` |

### Phase 1 — SFT + first preference win
Fine-tune a small instruct model, then run Direct Preference Optimization. Learn
the roles of the reference model, KL behavior, and the chosen/rejected gap. Ablate
DPO vs. SimPO vs. ORPO. *Lesson focus:* what actually moves a held-out win-rate.

### Phase 2 — Reward modeling + PPO
Train a Bradley-Terry reward model, then a full PPO loop with vLLM rollouts. Battle
KL control, value-function stability, reward normalization, and reward hacking.
*Lesson focus:* diagnosing and fixing an unstable online RL run.

### Phase 3 — GRPO + RLVR
The current frontier: RL with verifiable rewards on math/code (R1-Zero style). Run
GRPO on GSM8K/MATH, measure pass@1 gains and emergent longer reasoning, and add an
original ablation. *Lesson focus:* group-relative advantage and grader design.

### Phase 4 — Agentic RL
Train a model to use tools over multi-turn trajectories with a verifiable task
reward. The least-crowded, most-differentiating area. *Lesson focus:* multi-turn
credit assignment and environment/reward design.

## Repository layout

```
rl-post-training/
├── README.md            # this file
├── ROADMAP.md           # detailed 4-phase, week-by-week plan
├── pyproject.toml       # uv-managed Python project
├── main.py
└── phase{1..4}-*/       # one repo per phase (added as work progresses)
```

> Note: a private `.claude/` folder holds personal memory and draft notes. It is
> `.gitignore`d and is **never** part of this public repo.

## Stack

PyTorch · TRL · verl · OpenRLHF · vLLM · Weights & Biases · `uv`

- **TRL** — accessible DPO/PPO/GRPO on small models (Phases 1–3 starting point).
- **verl** — throughput-oriented RLVR/reasoning scale work (Phase 3+).
- **OpenRLHF** — clean PPO/GRPO reference implementation to read and compare.
- **vLLM** — fast rollout generation during online RL (throughput is half the battle).
- **W&B** — experiment tracking, curves, and reproducibility.

## Compute

External **H100/A100 on-demand** (Lambda / Runpod / Vast.ai). Small models
(0.5B–3B) by design — the skills transfer identically at larger scale. Phases 1–3
fit on a single 80GB GPU; multi-GPU is only needed for later scale-ups.

## Getting started

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies (once phase deps are added to pyproject.toml)
uv sync

# Run the entry point
uv run main.py
```

Per-phase setup, exact commands, datasets, and success criteria live in
**[ROADMAP.md](./ROADMAP.md)**.

## Datasets (defaults)

- **Preference:** UltraFeedback, HH-RLHF, Nectar.
- **RLVR:** GSM8K, MATH, Countdown (R1-Zero style).
- **Agentic:** ToolBench-style, BFCL, or a custom small tool environment.

## Status

Phase 0 (planning/setup) complete. Next: scaffold Phase 1 (SFT + DPO on
Qwen2.5-1.5B). See ROADMAP.md for the current plan.
