#!/usr/bin/env python3
"""How far checkpoints have drifted from a reference, on the anchor's bank.

The number the reference anchor penalizes (``anchor/kl_bank`` in training),
computed offline for saved checkpoints. Run it on an UNANCHORED lineage to set
``ANCHOR_COEF``: the drift the LR 1e-5 continuation built up from step_56072006
to step_66086572 came with the measured lift loss, so the anchor has to push
back at about that scale.

Only the residual is rebuilt, on CPU by default; no SmolVLA, no simulator.
The reported value is the mean-term KL(reference || checkpoint) in nats per
supervised action dim, split by stage, destination and object.

Usage::

    python3 tools/audit/reference_anchor_drift.py \\
      --bank runs/strict_success_dataset_step_56072006_<stamp>/retention_dataset/demonstrations.npz \\
      --reference runs/three_stage_sparse_grpo_20260925_105132/rl/step_56072006 \\
      --checkpoints runs/<run>/rl/step_63525522 runs/<run>/rl/step_66086572 \\
      --rl-grad-norm 5.78

``coef_for_ratio`` is the ANCHOR_COEF at which the anchor's gradient at that
checkpoint's drift is the given fraction of the RL gradient. The anchor's pull
grows with the drift, so a pilot at that coefficient pushes back at the size of
the drift the unanchored run reached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_vla_bootstrapping.policy.reference_anchor import (  # noqa: E402
    STAGE_NAMES,
    ReferenceAnchor,
)


def _adapter(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "smolvla_grpo_adapter.pt" if path.is_dir() else path


def build_residual(payload: dict[str, Any], torch: Any) -> Any:
    from rl_vla_bootstrapping.policy.smolvla_grpo_finetune_cdpr import SmolVLAGRPOPolicy

    saved = dict(payload.get("args") or {})
    policy = SmolVLAGRPOPolicy(
        state_dim=int(payload["state_dim"]),
        chunk_size=int(payload["chunk_size"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        residual_scale=float(payload["residual_scale"]),
        init_log_std=float(saved.get("init_log_std", -1.2)),
        min_log_std=float(saved.get("min_log_std", -5.0)),
        max_log_std=float(saved.get("max_log_std", 1.0)),
    )
    policy.load_state_dict(payload["policy"])
    return policy.eval()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--stages", default=",".join(STAGE_NAMES),
                        help="Default: all three, so placement drift is visible too.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rl-grad-norm", type=float, default=None,
        help="Typical RL gradient norm per optimizer step (gradient_norm_mean; "
        "median 5.78 over the LR 1e-5 continuation).",
    )
    parser.add_argument(
        "--target-ratio", type=float, nargs="+", default=[0.1, 0.25, 0.5],
        help="Anchor gradient as this fraction of --rl-grad-norm at each checkpoint's drift.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch

    device = torch.device(args.device)
    reference_path = _adapter(args.reference)
    payload = torch.load(reference_path, map_location=device, weights_only=False)
    template = build_residual(payload, torch).to(device)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]
    anchor = ReferenceAnchor.build(
        torch=torch, actor=template, bank_path=args.bank, reference_checkpoint=reference_path,
        stages=stages, coef=0.0, batch_size=256, seed=0, device=device,
    )
    from rl_vla_bootstrapping.policy.reference_anchor import load_anchor_rows

    rows = load_anchor_rows(Path(args.bank).expanduser().resolve(), stages=stages)
    result: dict[str, Any] = {"census": anchor.census, "checkpoints": {}}
    for checkpoint in args.checkpoints:
        path = _adapter(checkpoint)
        other = torch.load(path, map_location=device, weights_only=False)
        policy = build_residual(other, torch).to(device)
        row: dict[str, Any] = {"global_step": other.get("global_step"), "kl_all": round(anchor.full_kl(policy), 8)}
        # The anchor's gradient norm at coef 1 at this drift, over the whole
        # bank: the per-step pull the anchor would exert at this distance.
        parameters = [p for p in policy.parameters() if p.requires_grad]
        kl = anchor.kl(policy, torch.arange(anchor.rows, device=device))
        grads = torch.autograd.grad(kl, parameters, allow_unused=True)
        grad_norm = float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)).item())
        row["anchor_grad_norm_at_coef_1"] = round(grad_norm, 8)
        if args.rl_grad_norm is not None and grad_norm > 0.0:
            row["coef_for_ratio"] = {
                str(ratio): round(float(ratio) * float(args.rl_grad_norm) / grad_norm, 4)
                for ratio in args.target_ratio
            }
        for column in ("stage_name", "destination", "target_catalog"):
            if column not in rows:
                continue
            values = np.asarray(rows[column]).astype(str)
            for name in np.unique(values):
                index = torch.as_tensor(np.flatnonzero(values == name), device=device)
                with torch.no_grad():
                    row[f"kl_{column}_{name}"] = round(float(anchor.kl(policy, index).item()), 8)
        result["checkpoints"][str(path)] = row
        print(json.dumps({str(checkpoint): row}), flush=True)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
