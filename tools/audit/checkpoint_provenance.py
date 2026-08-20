#!/usr/bin/env python3
"""Where did this checkpoint come from, and has it ever been through SFT?

Written because the question "can I see that the self-recorded demonstrations
were applied?" had no answer anywhere. The information exists -- sil_sft stamps
every checkpoint it writes -- but nothing surfaced it: not the training log, not
TensorBoard, not the run directory. A 22-hour run could be, and was, read as
"the loop trained on its demonstrations" when the launcher that started it
refuses resume checkpoints by construction and cannot consume an SFT result at
all.

Reads any adapter and prints:

* whether it carries a ``sil_sft`` stamp, and if so which dataset and which
  epoch produced it, and whether the adapter was trained or only carried over;
* the approach-curriculum caps in ``extra_state``, which is what a resume
  restores and a warm start throws away;
* the global step, and whether LoRA tensors are present at all.

Usage::

    python tools/audit/checkpoint_provenance.py runs/<run>/rl/latest.pt
    python tools/audit/checkpoint_provenance.py runs/*/rl/latest.pt --brief
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_provenance(path: Path) -> dict[str, Any]:
    """Everything a checkpoint says about its own history."""

    import torch

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch < 2.6
        payload = torch.load(Path(path), map_location="cpu")

    stamp = payload.get("sil_sft")
    extra = dict(payload.get("extra_state") or {})
    approach = extra.get("approach_curriculum") or {}
    caps: dict[str, float] = {}
    if isinstance(approach, Mapping):
        for name, entry in approach.items():
            if isinstance(entry, Mapping) and "cap" in entry:
                caps[str(name)] = float(entry["cap"])
            elif name == "cap":
                caps["(legacy, all instructions)"] = float(entry)
    lora = payload.get("vla_lora") or {}
    return {
        "path": str(path),
        "global_step": int(payload.get("global_step", 0)),
        "gradient_step": int(payload.get("gradient_step", 0)),
        # The whole point of the tool. Absent means this policy has never been
        # through sil_sft, whatever else the run was called.
        "sil_sft": dict(stamp) if isinstance(stamp, Mapping) else None,
        "approach_caps": caps,
        "vla_lora_tensors": len(lora) if isinstance(lora, Mapping) else 0,
        # A resume restores these; a warm start discards them. Their presence
        # is what separates "continued the curriculum" from "started it over".
        "has_extra_state": bool(extra),
        "has_optimizer": "optimizer" in payload,
        "has_vla_lora_optimizer": "vla_lora_optimizer" in payload,
    }


def describe(record: Mapping[str, Any]) -> str:
    stamp = record["sil_sft"]
    if stamp is None:
        return (
            "NO SFT STAMP -- this policy has never been through sil_sft. "
            "Any run started from it trained on its own rollouts only."
        )
    trained = str(stamp.get("trained", "?"))
    where = str(stamp.get("dataset", "?"))
    if trained == "residual_only":
        return (
            f"SFT on {where}: residual only, best epoch "
            f"{stamp.get('epoch', '?')} at val_mse {stamp.get('val_mse', '?')}. "
            "The action-expert LoRA was carried over untouched."
        )
    return (
        f"SFT on {where}: {trained}, LoRA epoch "
        f"{stamp.get('lora_best_epoch', '?')} of {stamp.get('lora_epochs', '?')}, "
        f"KL coefficient {stamp.get('kl_coef', '?')}."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    records = []
    for path in args.checkpoints:
        record = read_provenance(path.expanduser().resolve())
        records.append(record)
        if args.json:
            continue
        print(f"\n{record['path']}")
        print(f"  global step        {record['global_step']}")
        print(f"  provenance         {describe(record)}")
        if args.brief:
            continue
        caps = record["approach_caps"]
        print(
            "  approach caps      "
            + (
                ", ".join(f"{k}={v:.3f}" for k, v in sorted(caps.items()))
                or "(none -- a warm start discards them)"
            )
        )
        print(f"  vla_lora tensors   {record['vla_lora_tensors']}")
        print(
            "  optimizer state    "
            + (
                "present"
                if record["has_optimizer"]
                else "absent (an SFT return payload drops it on purpose)"
            )
        )
    if args.json:
        print(json.dumps(records, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
