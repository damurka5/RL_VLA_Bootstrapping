"""Re-head SmolVLA's action expert for a specific robot's degrees of freedom.

SmolVLA ships with `max_action_dim = 32` and `max_state_dim = 32`: every
embodiment it was trained on is zero-padded into a 32-wide slot, and which
columns mean what differs per dataset. Two layers carry that width --
``action_in_proj`` (32 -> expert hidden) and ``action_out_proj`` (expert hidden
-> 32) -- and everything between them, the expert transformer itself, is
width-agnostic.

The repo's established path keeps all 32 and slices five columns out of the
output (``smolvla_cdpr._resolve_action_indices``). That is the right default
when the prior is worth something. On this stack it is measurably not: the
frozen prior behaves as a near-constant action offset (per-step sd <= 0.04),
so the residual head is doing all the work and the 32-wide head is 27 columns
of arithmetic that never reach an actuator.

This module does the other thing: narrow the two projections to the robot's own
action width and re-initialize them. The expert transformer keeps its
pretrained weights -- this is re-heading, not retraining from scratch -- but the
map between its hidden space and the action vector is new, so:

**The prior is gone the moment you call this.** ``action_out_proj`` no longer
decodes anything; until it is trained, the policy emits noise. That is the
trade: a 5-wide head that can be trained to mean something, against a 32-wide
head whose output you slice and whose prior is already nearly constant. It is a
reasonable trade on this stack and a bad one on a stack where the prior works,
so it is opt-in and never the default.

Nothing here imports LeRobot. It duck-types the SmolVLA module layout, so the
tests can exercise it against a stand-in with the same attribute names on a
host with no LeRobot and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ActionExpertSurgery",
    "SMOLVLA_ACTION_PROJECTIONS",
    "reinitialize_action_expert",
    "trainable_action_expert_parameters",
]

# The two layers whose width is the action dimension, and the direction each
# faces. `action_in_proj` consumes a (padded) action vector during flow
# matching; `action_out_proj` emits the velocity field that becomes the action.
SMOLVLA_ACTION_PROJECTIONS: tuple[tuple[str, str], ...] = (
    ("action_in_proj", "in"),
    ("action_out_proj", "out"),
)


@dataclass
class ActionExpertSurgery:
    """What was changed, so a run can record it and a resume can check it."""

    action_dim: int
    previous_action_dim: int
    state_dim: int | None = None
    previous_state_dim: int | None = None
    replaced: tuple[str, ...] = field(default_factory=tuple)
    trainable_parameters: int = 0

    @property
    def prior_preserved(self) -> bool:
        """False whenever ``action_out_proj`` was replaced.

        Worth checking before anything in the pipeline leans on the prior --
        a KL leash to it, a warm start that assumes it, a residual head
        initialized to be a small correction on top of it.
        """

        return "action_out_proj" not in self.replaced

    def to_metadata(self) -> dict[str, Any]:
        return {
            "action_expert/action_dim": int(self.action_dim),
            "action_expert/previous_action_dim": int(self.previous_action_dim),
            "action_expert/state_dim": self.state_dim,
            "action_expert/previous_state_dim": self.previous_state_dim,
            "action_expert/replaced": list(self.replaced),
            "action_expert/trainable_parameters": int(self.trainable_parameters),
            "action_expert/prior_preserved": bool(self.prior_preserved),
        }


def _flow_model(policy: Any) -> Any:
    """The ``VLAFlowMatching`` that owns the projections."""

    model = getattr(policy, "model", None)
    if model is None:
        raise AttributeError(
            "Expected a SmolVLA policy exposing `.model` (VLAFlowMatching); got "
            f"{type(policy).__name__}."
        )
    for name, _ in SMOLVLA_ACTION_PROJECTIONS:
        if not hasattr(model, name):
            raise AttributeError(
                f"SmolVLA model has no `{name}`. The action-expert layout has "
                "changed; re-check it against the pinned LeRobot before "
                "re-heading, rather than silently skipping a layer."
            )
    return model


def _replace_linear(
    module: Any,
    name: str,
    *,
    in_features: int | None = None,
    out_features: int | None = None,
) -> int:
    import torch.nn as nn

    old = getattr(module, name)
    new = nn.Linear(
        int(in_features if in_features is not None else old.in_features),
        int(out_features if out_features is not None else old.out_features),
        bias=old.bias is not None,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    setattr(module, name, new)
    return sum(parameter.numel() for parameter in new.parameters())


def reinitialize_action_expert(
    policy: Any,
    action_dim: int,
    *,
    state_dim: int | None = None,
    freeze_rest: bool = True,
) -> ActionExpertSurgery:
    """Narrow SmolVLA's action head to ``action_dim`` and re-initialize it.

    ``action_dim`` is the robot's own action width -- 5 for the WidowX-200 and
    the CDPR (``[x, y, z, yaw, gripper]``). ``state_dim`` optionally narrows
    ``state_proj`` the same way; leaving it ``None`` keeps the 32-wide state
    projection, which costs nothing because LeRobot zero-pads the state anyway
    and keeps that part of the pretrained model intact.

    ``freeze_rest`` freezes every other parameter, so only the new head trains.
    Pass ``False`` when attaching LoRA to the expert as well.

    Returns what changed. Record it in the checkpoint: a run that re-headed and
    a run that sliced a 32-wide output are not the same policy, and resuming one
    from the other silently trains a different problem.
    """

    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")

    model = _flow_model(policy)
    config = getattr(policy, "config", None)
    previous_action_dim = int(
        getattr(config, "max_action_dim", 0)
        or getattr(model, "action_out_proj").out_features
    )
    previous_state_dim = (
        int(getattr(config, "max_state_dim", 0) or 0) if config is not None else None
    )

    if freeze_rest:
        parameters = getattr(policy, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)

    replaced: list[str] = []
    trainable = 0
    for name, direction in SMOLVLA_ACTION_PROJECTIONS:
        if direction == "in":
            trainable += _replace_linear(model, name, in_features=action_dim)
        else:
            trainable += _replace_linear(model, name, out_features=action_dim)
        replaced.append(name)

    if state_dim is not None:
        state_dim = int(state_dim)
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}.")
        if hasattr(model, "state_proj"):
            trainable += _replace_linear(model, "state_proj", in_features=state_dim)
            replaced.append("state_proj")

    # The padding helpers and the sampling shape both read the config, so it has
    # to move with the layers. Leaving it at 32 would make `prepare_action` pad
    # a 5-wide action to 32 and feed a 32-wide vector to a 5-wide Linear.
    if config is not None:
        if hasattr(config, "max_action_dim"):
            config.max_action_dim = action_dim
        if state_dim is not None and hasattr(config, "max_state_dim"):
            config.max_state_dim = state_dim

    for name in replaced:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)

    return ActionExpertSurgery(
        action_dim=action_dim,
        previous_action_dim=previous_action_dim,
        state_dim=state_dim,
        previous_state_dim=previous_state_dim,
        replaced=tuple(replaced),
        trainable_parameters=int(trainable),
    )


def trainable_action_expert_parameters(policy: Any) -> list[Any]:
    """The parameters an optimizer should receive after re-heading.

    Returns them in a stable order so two ranks build identical optimizer state
    -- the same reason the DDP schedule is synchronized elsewhere in this stack.
    """

    model = _flow_model(policy)
    out: list[Any] = []
    for name in ("state_proj", "action_in_proj", "action_out_proj",
                 "action_time_mlp_in", "action_time_mlp_out"):
        layer = getattr(model, name, None)
        if layer is None:
            continue
        for parameter in layer.parameters():
            if parameter.requires_grad:
                out.append(parameter)
    return out
