#!/usr/bin/env python3
"""Diagnose the frozen SmolVLA action prior that feeds the CDPR residual.

Run this on the training server to answer three questions that decide whether
the residual-on-frozen-prior design can ever learn "move to <object>":

1. Scale: is the raw ``predict_action_chunk`` output in a sane [-1, 1]-ish
   range, or in SmolVLA's native (un-normalized) action units (values >> 1)?
2. Saturation: after the ``tanh`` action normalization, is the prior a soft
   value the 0.30-scale residual can move, or a bang-bang +/-1 whose sign the
   residual physically cannot reverse (``tanh(prior + 0.30*tanh(residual))``)?
3. Conditioning: does the prior actually change with the language instruction
   and the camera image, or does the frozen VLA emit essentially the same
   chunk regardless of the task (i.e. no usable vision/language signal)?

The images are synthetic random frames on purpose: the output *scale* and
*saturation* are properties of the checkpoint's normalization, not of any
specific pixels, and varying the random frames is exactly how we measure
whether the prior is conditioned on the image at all.

Example:
    python scripts/diagnose_smolvla_prior_scale.py \
        --base-checkpoint lerobot/smolvla_base --device cuda --batch-size 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Mirror scripts/huggingface_public_models.sh: SmolVLA and its SmolVLM2 backbone
# are public, so drop any stale credential inherited from the remote shell that
# would otherwise turn an anonymous download into a 401. Must run before any
# huggingface_hub import triggers implicit-token auth.
if os.environ.get("RLVLA_HF_PUBLIC_MODELS_ONLY", "1") == "1":
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from rl_vla_bootstrapping.policy.smolvla_cdpr import (
    SmolVLAActionAdapterSpec,
    adapt_smolvla_action_tensors_to_cdpr,
    load_smolvla_runtime,
)

AXIS_NAMES = ("x", "y", "z", "yaw", "gripper")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model-image-size", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument(
        "--action-normalization",
        default="tanh",
        help="Must match the training config (default: tanh).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--instructions",
        nargs="+",
        default=[
            "move to banana",
            "move to apple",
            "move to blue plate",
            "move to bowl",
        ],
        help="Distinct instructions used to measure language conditioning.",
    )
    return parser.parse_args(argv)


def _random_images(
    *, batch: int, height: int, width: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Synthetic BCHW uint8 RGB frames on the target device."""

    return torch.randint(
        0,
        256,
        (batch, 3, height, width),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )


def _plausible_states(
    *, batch: int, state_dim: int, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """[ee_x, ee_y, ee_z, ee_yaw, gripper, xy_distance] in training ranges."""

    state = torch.zeros((batch, state_dim), dtype=torch.float32, device=device)
    if state_dim >= 3:
        state[:, 0] = (torch.rand(batch, generator=generator, device=device) - 0.5) * 0.5
        state[:, 1] = (torch.rand(batch, generator=generator, device=device) - 0.5) * 0.5
        state[:, 2] = 0.27
    if state_dim >= 4:
        state[:, 3] = (torch.rand(batch, generator=generator, device=device) - 0.5) * 6.28
    if state_dim >= 5:
        state[:, 4] = 1.0
    if state_dim >= 6:
        state[:, 5] = 0.1 + torch.rand(batch, generator=generator, device=device) * 0.2
    return state


def _fmt_row(label: str, values: list[float]) -> str:
    body = "  ".join(f"{name}={value:+.3f}" for name, value in zip(AXIS_NAMES, values))
    return f"    {label:<14} {body}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    print(f"[diagnose] loading frozen SmolVLA replica: {args.base_checkpoint} on {device}")
    runtime = load_smolvla_runtime(
        checkpoint=str(args.base_checkpoint),
        device=str(device),
        mixed_precision=str(args.mixed_precision),
        image_size=int(args.image_size),
        state_dim=int(args.state_dim),
        include_wrist=True,
        include_aux_camera=True,
        chunk_size=int(args.chunk_size),
        action_dim=int(args.action_dim),
        action_normalization=str(args.action_normalization),
        model_image_size=(
            None if int(args.model_image_size) <= 0 else int(args.model_image_size)
        ),
        compile_model=False,
    )
    adapter_spec = SmolVLAActionAdapterSpec(
        action_dim=int(args.action_dim),
        chunk_size=int(args.chunk_size),
        normalization=str(args.action_normalization),
    )

    batch = int(args.batch_size)
    overview = _random_images(
        batch=batch,
        height=int(args.render_height),
        width=int(args.render_width),
        device=device,
        generator=generator,
    )
    wrist = _random_images(
        batch=batch,
        height=int(args.render_height),
        width=int(args.render_width),
        device=device,
        generator=generator,
    )
    states = _plausible_states(
        batch=batch, state_dim=int(args.state_dim), device=device, generator=generator
    )
    instruction = str(args.instructions[0])
    instructions = [instruction] * batch

    raw = runtime.sample_actions_from_tensors(
        primary_images=overview,
        wrist_images=wrist,
        states=states,
        instructions=instructions,
    ).float()  # [B, H, D_source]
    prior = adapt_smolvla_action_tensors_to_cdpr(raw, spec=adapter_spec).float()  # [B, H, 5]

    print("\n=== 1. RAW predict_action_chunk output scale ===")
    print(f"    shape={tuple(raw.shape)}  (batch, horizon, source_action_dim)")
    print(f"    abs mean = {raw.abs().mean().item():.4f}")
    print(f"    abs max  = {raw.abs().max().item():.4f}")
    print(f"    fraction |raw| > 1.0 : {(raw.abs() > 1.0).float().mean().item():.3f}")
    print(f"    fraction |raw| > 3.0 : {(raw.abs() > 3.0).float().mean().item():.3f}")
    print(
        "    -> if abs max >> 1 and a large fraction exceed 1, the tanh "
        "normalization below saturates."
    )

    print("\n=== 2. Adapted prior (post-tanh) that feeds the residual ===")
    first = prior[:, 0, :]  # first chunk step, the one acted on after replan
    print(_fmt_row("mean", [first[:, i].mean().item() for i in range(5)]))
    print(_fmt_row("std", [first[:, i].std(unbiased=False).item() for i in range(5)]))
    sat_090 = [(first[:, i].abs() > 0.90).float().mean().item() for i in range(5)]
    sat_099 = [(first[:, i].abs() > 0.99).float().mean().item() for i in range(5)]
    print(_fmt_row("frac>0.90", sat_090))
    print(_fmt_row("frac>0.99", sat_099))
    print(
        "    -> residual authority is +/-0.30 pre-tanh; any axis with a high "
        "frac>0.90 has its sign locked by the frozen prior."
    )

    print("\n=== 3. Does the prior depend on the instruction and the image? ===")
    # Same images, different instructions: language conditioning.
    per_instruction_first = []
    for text in args.instructions:
        raw_i = runtime.sample_actions_from_tensors(
            primary_images=overview,
            wrist_images=wrist,
            states=states,
            instructions=[text] * batch,
        ).float()
        prior_i = adapt_smolvla_action_tensors_to_cdpr(raw_i, spec=adapter_spec).float()
        per_instruction_first.append(prior_i[:, 0, :].mean(dim=0))  # [5]
    lang_stack = torch.stack(per_instruction_first, dim=0)  # [n_instr, 5]
    lang_spread = (lang_stack.max(dim=0).values - lang_stack.min(dim=0).values).tolist()
    print(f"    instructions tested: {list(args.instructions)}")
    print(_fmt_row("lang spread", lang_spread))
    print(
        "    -> spread ~0 means the frozen VLA ignores the language target; the "
        "residual is then the ONLY source of direction, and it has no target "
        "vector in its state."
    )

    # Same instruction, different random images: image conditioning.
    per_image_first = []
    for _ in range(4):
        ov = _random_images(
            batch=batch,
            height=int(args.render_height),
            width=int(args.render_width),
            device=device,
            generator=generator,
        )
        wr = _random_images(
            batch=batch,
            height=int(args.render_height),
            width=int(args.render_width),
            device=device,
            generator=generator,
        )
        raw_j = runtime.sample_actions_from_tensors(
            primary_images=ov,
            wrist_images=wr,
            states=states,
            instructions=instructions,
        ).float()
        prior_j = adapt_smolvla_action_tensors_to_cdpr(raw_j, spec=adapter_spec).float()
        per_image_first.append(prior_j[:, 0, :].mean(dim=0))
    img_stack = torch.stack(per_image_first, dim=0)
    img_spread = (img_stack.max(dim=0).values - img_stack.min(dim=0).values).tolist()
    print(_fmt_row("image spread", img_spread))
    print(
        "    -> spread ~0 means the frozen VLA ignores the camera; vision cannot "
        "drive approach until the base is adapted or a curriculum binds it."
    )

    print("\n=== verdict hints ===")
    print(
        "    * abs max >> 1 AND high frac>0.99  -> prior is bang-bang; raise "
        "residual_scale or unfreeze/adapt SmolVLA."
    )
    print(
        "    * lang spread ~0 AND image spread ~0 -> prior is a fixed bias; the "
        "target-relative state vector + directional curriculum are the fixes."
    )


if __name__ == "__main__":
    main()
