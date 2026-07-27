"""Phase 1: head-to-head win-rate eval (SFT baseline vs DPO policy).

Generates responses from two checkpoints on a shared set of prompts, then scores
them pairwise. The default judge is an LLM-as-judge via an OpenAI-compatible API;
if no API key is set, it falls back to writing the pairs to a JSONL file for
manual labeling (honest eval beats a fake automated number).

Lesson focus: a "win" is only real against the correct baseline on a held-out set
of prompts the model did not train on. Randomize which response is shown as A vs B
to the judge to avoid position bias.

Usage:
    uv run python phase1-dpo/scripts/eval_winrate.py \\
        --baseline outputs/sft-qwen2.5-1.5b \\
        --policy   outputs/dpo-qwen2.5-1.5b \\
        --prompts  phase1-dpo/eval_prompts.jsonl \\
        --out      outputs/eval

Prompt file: JSONL with one object per line: {"prompt": "..."}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def load_prompts(path: str, limit: int | None) -> list[str]:
    prompts: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line)["prompt"])
    if limit:
        prompts = prompts[:limit]
    return prompts


def generate_all(model_path: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Generate one response per prompt from a checkpoint."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    outputs: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
            )
        completion = tok.decode(
            gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        outputs.append(completion.strip())
    return outputs


JUDGE_PROMPT = """You are comparing two AI assistant responses to a user prompt.
Pick the response that is more helpful, correct, and well written.

User prompt:
{prompt}

Response A:
{a}

Response B:
{b}

Reply with exactly one token: "A" or "B"."""


def judge_llm(prompt: str, a: str, b: str) -> str | None:
    """LLM-as-judge via an OpenAI-compatible API. Returns 'A', 'B', or None."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(prompt=prompt, a=a, b=b),
                }
            ],
            temperature=0.0,
            max_tokens=1,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return verdict if verdict in ("A", "B") else None
    except Exception as e:  # noqa: BLE001
        print(f"Judge call failed ({e}); falling back to manual labeling.")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Win-rate eval (Phase 1).")
    parser.add_argument("--baseline", required=True, help="SFT checkpoint path")
    parser.add_argument("--policy", required=True, help="DPO checkpoint path")
    parser.add_argument("--prompts", required=True, help="JSONL of {'prompt': ...}")
    parser.add_argument("--out", default="outputs/eval", help="Output directory")
    parser.add_argument("--limit", type=int, default=100, help="Max prompts")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts, args.limit)
    print(f"Loaded {len(prompts)} prompts.")

    print("Generating baseline responses...")
    base_resp = generate_all(args.baseline, prompts, args.max_new_tokens)
    print("Generating policy responses...")
    pol_resp = generate_all(args.policy, prompts, args.max_new_tokens)

    records = []
    wins = ties = losses = judged = 0
    for prompt, base, pol in zip(prompts, base_resp, pol_resp):
        # Randomize position to avoid the judge favoring A or B by slot.
        policy_is_a = random.random() < 0.5
        a, b = (pol, base) if policy_is_a else (base, pol)
        verdict = judge_llm(prompt, a, b)

        result = "unlabeled"
        if verdict is not None:
            judged += 1
            policy_won = (verdict == "A") == policy_is_a
            result = "policy" if policy_won else "baseline"
            wins += int(policy_won)
            losses += int(not policy_won)

        records.append(
            {
                "prompt": prompt,
                "baseline": base,
                "policy": pol,
                "policy_is_a": policy_is_a,
                "verdict": verdict,
                "winner": result,
            }
        )

    (out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    if judged:
        win_rate = wins / judged
        print(f"\nJudged {judged}/{len(prompts)} pairs.")
        print(f"Policy wins: {wins} | baseline wins: {losses}")
        print(f"Policy win-rate: {win_rate:.1%}")
    else:
        print(
            "\nNo automated judging (set OPENAI_API_KEY to enable LLM-as-judge).\n"
            f"Pairs written to {out_dir / 'pairs.jsonl'} for manual labeling."
        )


if __name__ == "__main__":
    main()
