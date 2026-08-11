# RL Post-Training & Agentic — Hands-On Roadmap

A 3–6 month, ~10–15 hrs/week plan to build **frontier-lab-grade** hands-on skills
in RL post-training and agentic RL, and to capture the lessons learned along the way.
A parallel **Inference Engineering Track** (see below) targets roles that combine
inference and post-training (e.g. Baseten's "AI Inference & Post-Training").

> **North star:** The real skill is *making RL training actually work* —
> diagnosing reward hacking, KL blowups, collapse, and throughput bottlenecks,
> not just completing a course. Every project here is designed to produce durable,
> documented lessons learned.

---

## Guiding Principles

1. **Depth over breadth.** A few *real, working* training runs beat many toy demos.
2. **Everything is reproducible.** Fixed seeds, clean repos, honest eval.
   (See note on Google-internal compute below — it's a *learning supplement*.)
3. **Small models on purpose.** The skills transfer identically from 1.5B to 70B.
   Small = more iterations = more learning per dollar.
4. **Document the failures.** Your reward-hacking/collapse war stories are the
   most valuable lessons learned.
5. **One project → four kinds of lessons:** reproduce a result, add an ablation
   (original experimentation), contribute upstream (OSS), and write it up.

## Compute Strategy

- **Primary:** Rent **H100 80GB on-demand** (Lambda / Runpod / Vast.ai) for heavier
  runs; dev/debug on a cheaper single GPU. Rollout generation is the bottleneck —
  H100 throughput lets you iterate.
- **Sizing:** Phases 1–3 fit on **1× H100 (or 1× A100 80GB)** with 0.5B–3B models.
  Multi-GPU only for Phase 3/4 scale-ups.
- **Google-internal (TPU/JAX):** Use ONLY as a cheap experimentation supplement or
  if you want JAX/TPU fluency. Note it is invisible outside Google, so keep the
  reproducible, documented lessons in the external stack.

## Framework & Tooling

- **PyTorch** ecosystem (max transferability, where the jobs are).
- **TRL** (HuggingFace) — accessible DPO/PPO/GRPO on small models.
- **verl** (ByteDance) — serious RLVR/reasoning scale work.
- **OpenRLHF** — clean PPO/GRPO reference implementation.
- **vLLM** — fast rollout generation during RL.
- **Eval/logging:** Weights & Biases, `lm-eval-harness`, custom evals.

## Models & Datasets (defaults)

- **Base models:** `Qwen2.5-0.5B/1.5B/3B-Instruct`, `Llama-3.2-1B/3B`.
- **Preference:** `UltraFeedback`, `HH-RLHF`, `Nectar`.
- **RLVR:** `GSM8K`, `MATH`, `Countdown` (for R1-Zero style).
- **Agentic:** `ToolBench`-style, `BFCL`, or a custom small tool environment.

---

## Phase 1 — SFT + First Preference Win (Weeks 1–3)

**Goal:** Get a *measurable* preference-tuned improvement; understand every knob.

- **Week 1:** Environment + SFT.
  - Set up `uv` deps (transformers, trl, peft, datasets, vllm, wandb, accelerate).
  - LoRA/QLoRA SFT of Qwen2.5-1.5B on an instruction dataset. Confirm loss curves,
    checkpointing, and a working generation eval.
- **Week 2:** DPO.
  - Run DPO on UltraFeedback from your SFT checkpoint. Track reward margins,
    KL to reference, chosen/rejected logps. Get a win-rate improvement on a held-out
    eval (LLM-as-judge or a small preference eval).
- **Week 3:** Ablations + writeup.
  - Ablate beta, learning rate, and DPO vs. SimPO vs. ORPO. Plot learning curves.
  - **Deliverable:** Blog post #1 + clean repo (`phase1-dpo/`).

**Success criteria:** Reproducible win-rate lift over SFT baseline; you can explain
every hyperparameter's effect from your own curves.

---

## Phase 2 — Reward Modeling + PPO (Weeks 4–7)

**Goal:** Learn the heart of RLHF and how to diagnose it.

- **Week 4:** Reward model.
  - Train a Bradley-Terry RM (Qwen2.5-1.5B backbone) on UltraFeedback. Validate
    RM accuracy on held-out pairs; inspect reward distributions.
- **Weeks 5–6:** PPO.
  - Full PPO loop (TRL or OpenRLHF) using your RM. Wire vLLM for rollout generation.
  - Battle the classics: KL controller tuning, value function stability, reward
    normalization, and **reward hacking** (watch length exploitation, RM gaming).
- **Week 7:** Reproduce + ablate.
  - Reproduce a known RLHF result direction; ablate KL coefficient and RM quality.
  - **Deliverable:** Reproduction + ablation writeup (blog #2), `phase2-ppo/` repo.
    Document at least two failure modes you hit and fixed.

**Success criteria:** A PPO run that improves RM-scored quality *without* obvious
reward hacking; a written diagnosis of a failure you overcame.

---

## Phase 3 — GRPO + RLVR (Weeks 8–12)  ← highest hiring signal now

**Goal:** Current frontier — RL with verifiable rewards (R1-Zero / o1 style).

- **Week 8:** GRPO fundamentals.
  - Implement/run GRPO (TRL or verl) on GSM8K with a verifiable reward (exact-match
    / grader). Understand group-relative advantage vs. PPO's value function.
- **Weeks 9–10:** RLVR scale-up.
  - Move to verl for throughput. Train Qwen2.5-3B on GSM8K/MATH; measure pass@1
    improvement and emergent longer reasoning. Watch for reward hacking of the grader.
- **Week 11:** Original ablation.
  - E.g., reward shaping, KL on/off, group size, format rewards, or a small
    "aha-moment" reproduction (Countdown). This is your original-experiment candidate.
- **Week 12:** OSS + writeup.
  - Contribute a fix/feature/doc to TRL or verl encountered along the way.
  - **Deliverable:** Reproduction + original ablation (blog #3), `phase3-rlvr/` repo,
    linked upstream PR.

**Success criteria:** Measurable reasoning-benchmark lift from RLVR you ran yourself,
plus one merged (or credible) OSS contribution.

---

## Phase 4 — Agentic RL (Weeks 13–20)  ← least crowded, most differentiating

**Goal:** Train a model to use tools / do multi-step tasks via RL.

- **Weeks 13–14:** Environment design.
  - Build a small tool-use environment (e.g., calculator/search/code tools) with a
    verifiable task reward. Multi-turn rollout plumbing (this is the hard part).
- **Weeks 15–17:** Train.
  - GRPO/PPO over multi-turn trajectories. Handle credit assignment across turns,
    trajectory-level rewards, and tool-call formatting rewards.
- **Weeks 18–19:** Evaluate + iterate.
  - Task success rate, tool-call validity, efficiency. Ablate reward design.
- **Week 20:** Capstone writeup.
  - **Deliverable:** Original mini-research project (blog #4) + `phase4-agentic/`
    repo. This is your headline lessons-learned piece.

**Success criteria:** An agent whose task success improved via RL, with an honest
analysis of what the RL learned and where it breaks.

---

## Inference Engineering Track (parallel to Phases 1-4)

Motivation: real capability in this field spans **inference AND post-training**.
The post-training phases above are inference-light; this track fills that gap. It
also reinforces the RL work directly, because in online RL (Phases 2-3) rollout
generation is the throughput bottleneck. Understanding inference makes you faster
at training, not just serving.

Run this track alongside the phases (a few hours per week), not after them.

### Track A: Serving fundamentals (with Phase 1-2)
- Latency vs throughput, and why they trade off.
- Why generation is far more expensive than prompt processing (autoregressive,
  stateless transformers). See the PyTorch primer's generation-cost appendix.
- KV cache: what it stores, why it dominates memory, how it bounds batch size.
- Continuous/dynamic batching (the core idea behind vLLM throughput).
- Deliverable: a short latency/throughput benchmark of one small model with a
  writeup of the tradeoff curve.

### Track B: vLLM internals (with Phase 2-3)
- You already use vLLM for rollouts; now read how it works.
- PagedAttention, the KV cache manager, scheduler, and how batching is done.
- Measure tokens/sec vs batch size and sequence length; find the knee.
- Deliverable: annotate your Phase 3 rollout throughput; explain the bottleneck.

### Track C: Optimization techniques (with Phase 3-4)
- Quantization (INT8/FP8/AWQ/GPTQ): quality vs speed vs memory.
- Speculative decoding: draft model + verify, when it helps.
- TensorRT-LLM and compiled kernels (awareness level; deep-dive optional).
- Cost-per-token economics: how serving decisions map to dollars.
- Deliverable: quantize one trained checkpoint, measure quality/latency/memory.

### Track D: Packaging and deployment (capstone, optional)
- Package a trained checkpoint for serving (e.g. Baseten Truss or a vLLM server).
- Autoscaling, cold starts, scale-to-zero tradeoffs (concepts).
- Deliverable: deploy one of your fine-tuned models behind an API and document it.

The book itself frames inference as three layers, which map onto the tracks above:
Runtime (single-GPU: KV cache, kernels, vLLM/SGLang/TensorRT-LLM, quantization,
speculative decoding, parallelism) -> Infrastructure (multi-GPU: autoscaling,
routing, multi-cloud, zero-downtime deploy) -> Tooling (benchmarking, observability,
clients). Its modalities chapter (VLM, ASR/Whisper, TTS, video) overlaps directly
with the ClipForge project.

**Primary reference (confirmed):** *Inference Engineering* by Philip Kiely
(Baseten Books, 2026). https://www.baseten.co/library/inference-engineering/
Recommended by a friend working in inference + post-training. The goal here is to
genuinely learn the material by doing the experiments, not to prep for interviews.

**Community outline (inspiration, do not copy):**
github.com/elizabetht/100-days-of-inference is a 100-day, runnable study log built
around the same book (chapter-by-day mapping). Great for structure/ideas; do your
own work and write your own notes.

---

## Lessons-Learned Checklist (what "frontier-grade" looks like)

- [ ] 4 clean, runnable repos with READMEs, fixed seeds, and one-command repro.
- [ ] 4 writeups explaining *why* things worked (not just that they did).
- [ ] ≥1 merged/credible OSS contribution (TRL / verl / OpenRLHF).
- [ ] Documented failure war stories (reward hacking, KL blowup, collapse).
- [ ] Proper eval: baselines, seeds, learning curves, not vibes.
- [ ] Throughput awareness demonstrated (vLLM rollouts, batching).
- [ ] Inference track: a latency/throughput benchmark + a quantized checkpoint.
- [ ] Inference track: one fine-tuned model deployed behind an API (capstone).

## Learning Resources (to consult as you go)

- Papers: InstructGPT, DPO, SimPO, DeepSeek-R1, GRPO (DeepSeekMath), RLOO,
  Constitutional AI, Tulu 3 (excellent recipes report).
- Codebases to read: TRL, verl, OpenRLHF (read the trainers, not just run them).
- Eval: `lm-evaluation-harness`, AlpacaEval, MT-Bench.
- Inference: the Philip Kiely (Baseten) inference-engineering book (confirm title),
  the PyTorch primer's generation-cost appendix, and the vLLM docs/paper
  (PagedAttention).

## Stretch / "Stay Ahead" Ideas

- Process reward models (PRMs) vs. outcome rewards.
- Multi-turn RL credit assignment experiments.
- Reward-hacking taxonomy blog (very interview-friendly).
- Efficiency: async rollouts, sequence packing, throughput benchmarking writeup.
