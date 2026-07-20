"""Minimal, dependency-free LoRA for fine-tuning a frozen SmolVLA in RL.

Hand-rolled (no ``peft``) so the custom grad-through-VLA GRPO update stays fully
under our control. ``LoRALinear`` wraps a frozen ``nn.Linear`` with a low-rank
update ``scaling * (x A^T) B^T``; ``B`` starts at zero so the adapter is an exact
no-op at init and the prior is unchanged until training moves it (important for
a KL/trust-region leash to the frozen prior).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch-less env (tests skip).
    torch = None
    nn = None
    F = None


if nn is not None:

    class LoRALinear(nn.Module):
        """Frozen base ``nn.Linear`` plus a trainable low-rank residual."""

        def __init__(
            self,
            base: "nn.Linear",
            *,
            rank: int,
            alpha: float,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if rank <= 0:
                raise ValueError("LoRA rank must be positive.")
            self.base = base
            for param in self.base.parameters():
                param.requires_grad_(False)
            self.rank = int(rank)
            self.scaling = float(alpha) / float(rank)
            in_features = int(base.in_features)
            out_features = int(base.out_features)
            self.lora_a = nn.Parameter(
                torch.zeros(self.rank, in_features)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(out_features, self.rank)
            )
            # A ~ Kaiming, B = 0 => the adapter starts as an exact no-op.
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.dropout = (
                nn.Dropout(float(dropout))
                if float(dropout) > 0.0
                else nn.Identity()
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            base_out = self.base(x)
            update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
            return base_out + self.scaling * update

    def attach_lora(
        root: "nn.Module",
        *,
        target_leaf_names: Sequence[str],
        name_contains: Sequence[str] | None = None,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> list[str]:
        """Replace matching ``nn.Linear`` leaves in-place with ``LoRALinear``.

        A linear is wrapped iff its child attribute name is in
        ``target_leaf_names`` (e.g. ``q_proj``/``v_proj``) AND, when
        ``name_contains`` is given, its qualified path contains at least one of
        those substrings (e.g. the action-expert prefix, to exclude the VLM).
        Returns the qualified names that were wrapped.
        """

        targets = set(target_leaf_names)
        contains = tuple(name_contains or ())
        replaced: list[str] = []
        for module_name, module in list(root.named_modules()):
            for child_name, child in list(module.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                if child_name not in targets:
                    continue
                qualified = (
                    f"{module_name}.{child_name}"
                    if module_name
                    else child_name
                )
                if contains and not any(token in qualified for token in contains):
                    continue
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        child, rank=rank, alpha=alpha, dropout=dropout
                    ),
                )
                replaced.append(qualified)
        return replaced

    def lora_parameters(root: "nn.Module") -> list["torch.Tensor"]:
        """Every trainable LoRA parameter tensor under ``root``."""

        return [
            param
            for name, param in root.named_parameters()
            if "lora_" in name and param.requires_grad
        ]

    def freeze_all_but_lora(root: "nn.Module") -> None:
        """Freeze every parameter except the LoRA adapters."""

        for name, param in root.named_parameters():
            param.requires_grad_("lora_" in name)

    def count_trainable(root: "nn.Module") -> int:
        return int(
            sum(
                param.numel()
                for param in root.parameters()
                if param.requires_grad
            )
        )

else:  # pragma: no cover - torch-less env.

    class LoRALinear:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("LoRALinear requires PyTorch.")

    def attach_lora(*args: Any, **kwargs: Any) -> list[str]:  # type: ignore[misc]
        raise RuntimeError("attach_lora requires PyTorch.")
