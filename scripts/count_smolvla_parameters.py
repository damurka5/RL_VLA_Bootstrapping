#!/usr/bin/env python3
"""Report SmolVLA parameter counts to size an unfreeze / fine-tune decision.

Prints the total parameter count, a breakdown by top-level submodule, keyword
buckets (vision / language / action-expert / other), and LoRA trainable-count
estimates for a few common target sets. Loads on CPU by default so it needs no
GPU. Run on the training server:

    python scripts/count_smolvla_parameters.py --base-checkpoint lerobot/smolvla_base
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

# Same public-model policy as scripts/huggingface_public_models.sh: drop any
# stale token so the anonymous download does not 401.
if os.environ.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1") == "1":
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime

VISION_HINTS = ("vision", "siglip", "vit", "image", "patch", "visual")
LANGUAGE_HINTS = ("text", "lm", "language", "llm", "embed_tokens", "vlm")
ACTION_HINTS = ("action", "expert", "flow", "state_proj", "action_out")
ATTENTION_HINTS = ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "attn")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--group-depth", type=int, default=2)
    parser.add_argument(
        "--lora-ranks", nargs="+", type=int, default=[16, 32]
    )
    return parser.parse_args(argv)


def _millions(value: int) -> str:
    return f"{value / 1e6:8.2f}M"


def _bucket(name: str) -> str:
    lower = name.lower()
    if any(h in lower for h in ACTION_HINTS):
        return "action_expert"
    if any(h in lower for h in VISION_HINTS):
        return "vision"
    if any(h in lower for h in LANGUAGE_HINTS):
        return "language"
    return "other"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"[count] loading {args.base_checkpoint} on {args.device}")
    runtime = load_smolvla_runtime(
        checkpoint=str(args.base_checkpoint),
        device=str(args.device),
        mixed_precision="fp32",
        compile_model=False,
    )
    policy = runtime.policy
    named = list(policy.named_parameters())
    total = sum(p.numel() for _, p in named)
    print(f"\n=== total parameters: {_millions(total)} ({total:,}) ===")

    depth = max(1, int(args.group_depth))
    groups: dict[str, int] = collections.defaultdict(int)
    for name, param in named:
        key = ".".join(name.split(".")[:depth]) or name
        groups[key] += int(param.numel())
    print(f"\n=== by submodule (first {depth} name components) ===")
    for key, count in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"    {_millions(count)}  {100.0 * count / total:5.1f}%  {key}")

    buckets: dict[str, int] = collections.defaultdict(int)
    for name, param in named:
        buckets[_bucket(name)] += int(param.numel())
    print("\n=== keyword buckets (best-effort) ===")
    for key in ("vision", "language", "action_expert", "other"):
        count = buckets.get(key, 0)
        print(f"    {_millions(count)}  {100.0 * count / total:5.1f}%  {key}")

    linears = [
        (name, module)
        for name, module in policy.named_modules()
        if isinstance(module, nn.Linear)
    ]

    def lora_count(filter_hints: tuple[str, ...], rank: int) -> int:
        total_added = 0
        for name, module in linears:
            lower = name.lower()
            if any(h in lower for h in filter_hints):
                total_added += rank * (module.in_features + module.out_features)
        return total_added

    action_linears = [
        n for n, _ in linears if any(h in n.lower() for h in ACTION_HINTS)
    ]
    print("\n=== fine-tuning scenarios (trainable parameters) ===")
    print(f"    residual MLP only (current)        ~1-2M  (frozen VLA: {_millions(total)})")
    action_expert_params = buckets.get("action_expert", 0)
    print(
        f"    unfreeze action expert only        {_millions(action_expert_params)}"
        f"  ({len(action_linears)} Linear layers matched)"
    )
    for rank in args.lora_ranks:
        lora_action = lora_count(ACTION_HINTS, rank)
        lora_attn = lora_count(ATTENTION_HINTS, rank)
        print(
            f"    LoRA r={rank:<3} action-expert linears  {_millions(lora_action)}"
        )
        print(
            f"    LoRA r={rank:<3} all attention linears  {_millions(lora_attn)}"
        )
    print(f"    full fine-tune                     {_millions(total)}")
    print(
        "\n    Note: buckets are keyword heuristics -- read the by-submodule table"
        " above for the authoritative split, and grep the printed names to pick"
        " exact LoRA targets."
    )


if __name__ == "__main__":
    main()
