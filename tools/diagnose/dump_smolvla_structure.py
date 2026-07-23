#!/usr/bin/env python3
"""Introspect the frozen SmolVLA model structure for the vision-aware residual.

We want to feed SmolVLA's frozen *visual* features into the trainable residual.
The residual currently sees only proprioception + the collapsed 40-dim prior
action, so it has no spatial vision signal. To tap the pre-collapse features we
must know (a) which submodule produces the image/vision tokens, (b) the tensor
shape/hidden dim of that output, and (c) a stable module name to hook.

This script loads the frozen SmolVLA runtime exactly as training does, registers
forward hooks on every submodule, runs ONE dummy forward, and prints the module
names + output shapes for the vision-relevant modules, plus the config hidden
sizes. Run it once on the remote GPU box; paste the output back.

Usage (remote, cdpr-mjlab env):
    conda run --no-capture-output -n cdpr-mjlab python3 \
        tools/diagnose/dump_smolvla_structure.py \
        --checkpoint lerobot/smolvla_base --device cuda:0
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--model-image-size", type=int, default=256)
    ap.add_argument("--state-dim", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--action-dim", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-rows", type=int, default=400)
    args = ap.parse_args()

    # SmolVLA + its SmolVLM2 backbone are public. A stale HF_TOKEN inherited from
    # the shell turns an anonymous download into a 401, so force anonymous access
    # (mirrors scripts/huggingface_public_models.sh). Must precede the HF import.
    import os

    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

    import torch
    from rl_vla_bootstrapping.policy.smolvla_cdpr import load_smolvla_runtime

    runtime = load_smolvla_runtime(
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        mixed_precision="bf16",
        image_size=int(args.image_size),
        state_dim=int(args.state_dim),
        include_wrist=True,
        include_aux_camera=True,
        mask_empty_aux_camera=True,
        chunk_size=int(args.chunk_size),
        action_dim=int(args.action_dim),
        action_normalization="tanh",
        model_image_size=int(args.model_image_size),
        compile_model=False,
    )
    policy = runtime.policy

    # ---- config hidden sizes -------------------------------------------------
    cfg = getattr(policy, "config", None)
    print("=" * 70)
    print("CONFIG hidden-size-ish attributes:")
    for name in sorted(dir(cfg)):
        if name.startswith("_"):
            continue
        low = name.lower()
        if any(k in low for k in ("hidden", "dim", "size", "token", "expert", "vlm", "image", "width", "layers")):
            try:
                val = getattr(cfg, name)
            except Exception:
                continue
            if isinstance(val, (int, float, str, bool, tuple, list)) and not callable(val):
                print(f"  cfg.{name} = {val!r}")

    # ---- module tree (types) -------------------------------------------------
    print("=" * 70)
    print("TOP-LEVEL module tree under policy.model (depth<=3):")
    model = getattr(policy, "model", policy)
    for name, mod in model.named_modules():
        depth = name.count(".")
        if depth <= 2 and name:
            print(f"  {name}  <{type(mod).__name__}>")

    # ---- forward-hook capture of output shapes -------------------------------
    shapes: dict[str, str] = {}

    def make_hook(nm: str):
        def hook(_m, _inp, out):
            def shp(o):
                if isinstance(o, torch.Tensor):
                    return "x".join(str(s) for s in tuple(o.shape))
                return None
            s = shp(out)
            if s is None and isinstance(out, (tuple, list)):
                s = ",".join(str(shp(o)) for o in out if shp(o))
            if s:
                shapes[nm] = f"{type(_m).__name__}:{s}"
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in policy.named_modules() if n]

    B = int(args.batch)
    H = W = int(args.image_size)
    overview = torch.rand(B, H, W, 3, device=args.device)
    wrist = torch.rand(B, H, W, 3, device=args.device)
    states = torch.zeros(B, int(args.state_dim), device=args.device)
    instructions = ["move to the apple"] * B
    try:
        with torch.inference_mode():
            runtime.sample_cdpr_chunks_from_tensors(
                primary_images=overview,
                wrist_images=wrist,
                states=states,
                instructions=instructions,
                microbatch_size=0,
            )
    finally:
        for h in handles:
            h.remove()

    print("=" * 70)
    print("VISION-RELEVANT module output shapes (name contains vision/image/"
          "connector/projector/patch/embed):")
    keys = ("vision", "image", "connector", "projector", "patch", "embed", "modality")
    rows = [(n, s) for n, s in shapes.items() if any(k in n.lower() for k in keys)]
    for n, s in rows[: int(args.max_rows)]:
        print(f"  {n}  ->  {s}")

    print("=" * 70)
    print("ALL captured output shapes (first N, for context):")
    for n, s in list(shapes.items())[: int(args.max_rows)]:
        print(f"  {n}  ->  {s}")

    print("=" * 70)
    print(f"captured {len(shapes)} module outputs total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
