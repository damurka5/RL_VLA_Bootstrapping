"""Frozen-reference logit correction and the latent-Gaussian action likelihood.

Implements docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_IMPLEMENTATION.md.

Architecture ``frozen_reference_logit_correction_v1``
-----------------------------------------------------

The legacy actor (``ResidualChunkActor``, tagged ``bounded_residual_v0``) is::

    x         = concat(state, flatten(prior))
    logit_ref = prior + reference_scale * tanh(reference_net(x))
    mean      = tanh(logit_ref)

Its bounded residual spends most of its range cancelling the prior's nearly
constant offset. The new actor keeps that network, frozen, and adds an
unbounded correction in the final action's pre-tanh coordinates::

    correction = correction_net(x)            # linear output, no tanh
    mean       = tanh(logit_ref + correction)

``correction_net`` is a deep copy of the reference MLP with only its final
linear layer zeroed, so at initialization ``correction == 0`` exactly for every
finite input and the new mean equals the old one bit for bit. Only that final
layer gets an actor gradient on the first update; its zero weights block the
hidden layers until they move. The reference's PARAMETERS are frozen; its
OUTPUT is recomputed from every observation and prior.

Likelihood ``latent_gaussian_conditional_offset_v1``
----------------------------------------------------

For every sampled action slot::

    b_t        = realized_episode_offset * gate_t     # exogenous, recorded
    u_t        = mu_t + b_t + sigma * eps_t           # the policy sample
    a_t        = clamp(u_t, -1, 1)                    # sent to the controller
    old_logp_t = sum_dims Normal(mu_t + b_t, sigma).log_prob(u_t)

The controller only ever sees ``a_t``; the policy gradient scores ``u_t``.
Clipping is a deterministic map from the latent action to the environment
action, so this is an exact policy-gradient formulation over latent actions.
The legacy path instead scores the CLIPPED action under a Gaussian density, and
the clipped action has a point mass at each bound that density does not
describe.

The persistent offset is an exogenous episode-level random variable whose
density has no policy parameters, so conditioning on its realized value is
exact: the offset density cancels in every likelihood ratio, and the
trajectory score is ``sum_t (u_t - mu_t - b_t) / sigma^2 * d mu_t``. Learning
from the offset happens through the returns and later states it changes,
exactly as for any other exogenous noise. The earlier claim that a conditional
likelihood makes the offset invisible came from a toy whose advantage was the
offset itself, which no return-dependent process produces; see
tests/test_latent_correction_likelihood.py for the check against a stochastic
control problem with action-dependent returns.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional local dependency
    torch = None
    nn = None


POLICY_ARCHITECTURE_LEGACY = "bounded_residual_v0"
POLICY_ARCHITECTURE_CORRECTION = "frozen_reference_logit_correction_v1"
POLICY_ARCHITECTURES = (POLICY_ARCHITECTURE_LEGACY, POLICY_ARCHITECTURE_CORRECTION)

ACTION_LIKELIHOOD_LEGACY = "clipped_action_v0"
ACTION_LIKELIHOOD_LATENT = "latent_gaussian_conditional_offset_v1"
ACTION_LIKELIHOODS = (ACTION_LIKELIHOOD_LEGACY, ACTION_LIKELIHOOD_LATENT)
# Stored per record, so a mixed or mislabelled buffer fails at update time.
ACTION_LIKELIHOOD_CODES = {ACTION_LIKELIHOOD_LEGACY: 0, ACTION_LIKELIHOOD_LATENT: 1}

# Fields every latent-mode record set must carry. "action" is deliberately NOT
# among them: latent records store the clipped controller command under
# "executed_action", so the legacy path (which scores "action") cannot read a
# latent buffer by accident, and the latent path can never score the clip.
LATENT_RECORD_FIELDS = (
    "state",
    "prior",
    "executed_action",
    "policy_sample",
    "behavior_mean_offset",
    "action_index",
    "old_log_prob",
    "likelihood_version",
    "advantage",
)

ACTION_AXES = ("x", "y", "z", "yaw", "gripper")


def normal_log_prob(value: Any, mean: Any, log_std: Any) -> Any:
    """Diagonal Gaussian log density summed over the last dimension."""

    var = torch.exp(2.0 * log_std)
    return (
        -0.5 * (((value - mean).pow(2) / var) + 2.0 * log_std + float(np.log(2.0 * np.pi)))
    ).sum(dim=-1)


def latent_gaussian_log_prob(
    policy_sample: Any, policy_mean: Any, behavior_offset: Any, log_std: Any
) -> Any:
    """``log N(u; mu + b, sigma^2)`` for the latent sample ``u``.

    ``behavior_offset`` is the realized effective offset recorded at sampling
    time (zeros where the gate was off). It is conditioned on, never re-drawn,
    and never inferred from the clipped action.
    """

    return normal_log_prob(policy_sample, policy_mean + behavior_offset, log_std)


def checkpoint_policy_architecture(payload: Mapping[str, Any]) -> str:
    """The architecture a checkpoint was saved with; legacy when untagged.

    An untagged state that nonetheless holds correction weights is refused
    rather than guessed at: a silent reinterpretation in either direction
    evaluates a different policy.
    """

    tagged = payload.get("policy_architecture")
    policy = payload.get("policy") or {}
    has_correction = any(str(key).startswith("actor.correction_net.") for key in policy)
    if tagged is None:
        if has_correction:
            raise RuntimeError(
                "Checkpoint holds correction-network weights but no "
                "policy_architecture tag; refusing to guess its architecture."
            )
        return POLICY_ARCHITECTURE_LEGACY
    tagged = str(tagged)
    if tagged not in POLICY_ARCHITECTURES:
        raise RuntimeError(
            f"Unknown policy_architecture {tagged!r}; known: {list(POLICY_ARCHITECTURES)}."
        )
    if (tagged == POLICY_ARCHITECTURE_CORRECTION) != has_correction and policy:
        raise RuntimeError(
            f"Checkpoint is tagged {tagged!r} but its policy state "
            f"{'has' if has_correction else 'lacks'} correction-network weights."
        )
    return tagged


def checkpoint_action_likelihood(payload: Mapping[str, Any]) -> str:
    tagged = str(payload.get("action_likelihood") or ACTION_LIKELIHOOD_LEGACY)
    if tagged not in ACTION_LIKELIHOODS:
        raise RuntimeError(
            f"Unknown action_likelihood {tagged!r}; known: {list(ACTION_LIKELIHOODS)}."
        )
    return tagged


def require_legacy_policy_checkpoint(payload: Mapping[str, Any], consumer: str) -> None:
    """Fail loudly in utilities that only understand the legacy actor.

    Loading a correction checkpoint's reference branch alone would evaluate
    the starting policy and report it as the trained one.
    """

    architecture = checkpoint_policy_architecture(payload)
    if architecture != POLICY_ARCHITECTURE_LEGACY:
        raise RuntimeError(
            f"{consumer} only supports {POLICY_ARCHITECTURE_LEGACY!r} checkpoints; "
            f"this one is {architecture!r}. Use a version-aware consumer "
            "(SmolVLAGRPOTrainer built from the checkpoint's args, e.g. "
            "tools/audit/evaluate_cdpr_full_put_into.py) instead of loading "
            "partial weights."
        )


def _linear_indices(state: Mapping[str, Any], prefix: str) -> list[int]:
    indices = set()
    for key in state:
        key = str(key)
        if key.startswith(prefix) and key.endswith(".weight"):
            indices.add(int(key[len(prefix):].split(".")[0]))
    return sorted(indices)


def legacy_actor_geometry(policy_state: Mapping[str, Any]) -> dict[str, int]:
    """Input, hidden and output widths of a legacy SmolVLAGRPOPolicy state."""

    prefix = "actor.net.net."
    indices = _linear_indices(policy_state, prefix)
    if len(indices) != 3:
        raise RuntimeError(
            f"Expected a three-linear legacy residual MLP under {prefix!r}, found "
            f"linear indices {indices}."
        )
    first = policy_state[f"{prefix}{indices[0]}.weight"]
    middle = policy_state[f"{prefix}{indices[1]}.weight"]
    last = policy_state[f"{prefix}{indices[-1]}.weight"]
    return {
        "input_dim": int(first.shape[1]),
        "hidden_dim": int(first.shape[0]),
        "hidden_dim_2": int(middle.shape[0]),
        "output_dim": int(last.shape[0]),
        "final_linear_index": int(indices[-1]),
    }


def check_legacy_geometry(
    policy_state: Mapping[str, Any],
    *,
    state_dim: int,
    chunk_size: int,
    action_dim: int,
    hidden_dim: int,
) -> dict[str, int]:
    """Shapes in the state must match the runtime's observation contract."""

    geometry = legacy_actor_geometry(policy_state)
    expected = {
        "input_dim": int(state_dim) + int(chunk_size) * int(action_dim),
        "hidden_dim": int(hidden_dim),
        "hidden_dim_2": int(hidden_dim),
        "output_dim": int(chunk_size) * int(action_dim),
    }
    differences = {
        key: (geometry[key], value)
        for key, value in expected.items()
        if geometry[key] != value
    }
    if differences:
        details = ", ".join(
            f"{key}: checkpoint={old}, runtime={new}"
            for key, (old, new) in differences.items()
        )
        raise RuntimeError(f"Legacy checkpoint geometry does not match the runtime: {details}.")
    log_std = policy_state.get("log_std")
    if log_std is None or tuple(log_std.shape) != (int(chunk_size), int(action_dim)):
        raise RuntimeError(
            f"Legacy checkpoint log_std has shape "
            f"{None if log_std is None else tuple(log_std.shape)}, expected "
            f"{(int(chunk_size), int(action_dim))}."
        )
    return geometry


def convert_legacy_policy_state(policy_state: Mapping[str, Any]) -> dict[str, Any]:
    """Legacy ``SmolVLAGRPOPolicy`` state -> correction-architecture state.

    The reference branch is an exact copy. The correction branch copies the
    hidden layers and zeroes only the final linear layer's weight and bias.
    Every tensor is cloned, so no storage is shared between the branches or
    with the source state. ``log_std`` is copied unchanged.
    """

    geometry = legacy_actor_geometry(policy_state)
    final = int(geometry["final_linear_index"])
    allowed = {"log_std"}
    converted: dict[str, Any] = {}
    for key, value in policy_state.items():
        key = str(key)
        if key in allowed:
            converted[key] = value.detach().clone()
            continue
        if not key.startswith("actor.net."):
            raise RuntimeError(f"Unexpected legacy policy key {key!r}; refusing conversion.")
        suffix = key[len("actor.net."):]
        converted[f"actor.reference_net.{suffix}"] = value.detach().clone()
        layer = int(suffix.split(".")[1])
        correction = value.detach().clone()
        if layer == final:
            correction.zero_()
        converted[f"actor.correction_net.{suffix}"] = correction
    return converted


def tensor_fingerprint(tensors: Mapping[str, Any]) -> str:
    """SHA-256 over names, shapes, dtypes and raw bytes, in sorted key order."""

    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().to("cpu").contiguous()
        digest.update(str(key).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b"")
    return digest.hexdigest()


if nn is not None:
    from rl_vla_bootstrapping.policy.octo_finetune_cdpr import MLP

    class FrozenReferenceCorrectionActor(nn.Module):
        """``tanh(prior + scale * tanh(reference_net(x)) + correction_net(x))``."""

        def __init__(
            self,
            *,
            state_dim: int,
            chunk_size: int,
            action_dim: int,
            hidden_dim: int,
            residual_scale: float,
        ) -> None:
            super().__init__()
            self.chunk_size = int(chunk_size)
            self.action_dim = int(action_dim)
            # The inherited reference scale; the correction's scale is 1.0.
            self.residual_scale = float(residual_scale)
            input_dim = int(state_dim) + int(chunk_size) * int(action_dim)
            output_dim = int(chunk_size) * int(action_dim)
            dims = (input_dim, int(hidden_dim), int(hidden_dim), output_dim)
            # The reference MLP consumes the initialization RNG exactly as the
            # legacy actor's single MLP does; the correction MLP is built under
            # a forked RNG. A converted candidate therefore leaves the global
            # RNG where a legacy trainer would, so the control and candidate
            # arms start their rollouts from the same streams.
            self.reference_net = MLP(dims)
            with torch.random.fork_rng(devices=[]):
                self.correction_net = MLP(dims)
            self.reference_net.requires_grad_(False)
            self.reference_net.eval()
            self.zero_correction_output()

        # ----------------------------------------------------------------- setup

        def correction_output_layer(self) -> Any:
            return [m for m in self.correction_net.net if isinstance(m, nn.Linear)][-1]

        def zero_correction_output(self) -> None:
            layer = self.correction_output_layer()
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()

        def train(self, mode: bool = True):  # type: ignore[override]
            super().train(mode)
            # Mode changes never touch requires_grad, and the reference stays
            # in inference mode whatever the caller asks for.
            self.reference_net.eval()
            return self

        def reference_parameters(self) -> list[Any]:
            return list(self.reference_net.parameters())

        def assert_reference_frozen(self) -> None:
            live = [name for name, p in self.reference_net.named_parameters() if p.requires_grad]
            if live:
                raise RuntimeError(f"Frozen reference parameters require grad: {live}.")

        # --------------------------------------------------------------- forward

        def components(self, state: Any, prior_chunk: Any) -> dict[str, Any]:
            """Every term of the action, computed directly (never via atanh)."""

            prior = prior_chunk.reshape(prior_chunk.shape[0], self.chunk_size, self.action_dim)
            features = torch.cat([state, prior.reshape(prior.shape[0], -1)], dim=-1)
            reference_residual = torch.tanh(self.reference_net(features)).reshape_as(prior)
            reference_logit = prior + self.residual_scale * reference_residual
            correction = self.correction_net(features).reshape_as(prior)
            logit = reference_logit + correction
            return {
                "prior": prior,
                "reference_residual": reference_residual,
                "reference_logit": reference_logit,
                "correction": correction,
                "logit": logit,
                "mean": torch.tanh(logit),
            }

        def forward(self, state: Any, prior_chunk: Any) -> Any:
            return self.components(state, prior_chunk)["mean"]

        def action_at(self, state: Any, prior_chunk: Any, action_index: Any) -> Any:
            chunk = self.forward(state, prior_chunk)
            idx = action_index.reshape(-1).long().clamp(0, self.chunk_size - 1)
            return chunk[torch.arange(chunk.shape[0], device=chunk.device), idx]

else:  # pragma: no cover - dependency guard

    class FrozenReferenceCorrectionActor:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for FrozenReferenceCorrectionActor")


def actor_components(actor: Any, state: Any, prior_chunk: Any) -> dict[str, Any]:
    """Direct action components for either architecture.

    The legacy actor has no correction, so its correction is reported as an
    exact zero and its logit is the reference logit.
    """

    if hasattr(actor, "components"):
        return actor.components(state, prior_chunk)
    prior = prior_chunk.reshape(prior_chunk.shape[0], actor.chunk_size, actor.action_dim)
    reference_residual, reference_logit = actor.reference_terms(state, prior_chunk)
    return {
        "prior": prior,
        "reference_residual": reference_residual,
        "reference_logit": reference_logit,
        "correction": torch.zeros_like(reference_logit),
        "logit": reference_logit,
        "mean": torch.tanh(reference_logit),
    }


class FrozenParameterGuard:
    """Detects any change to parameters that must stay fixed during a run."""

    def __init__(self, parameters: Mapping[str, Any]) -> None:
        self.names = sorted(parameters)
        self.snapshot = {name: parameters[name].detach().clone() for name in self.names}

    def max_abs_change(self, parameters: Mapping[str, Any]) -> float:
        if sorted(parameters) != self.names:
            raise RuntimeError("Frozen parameter set changed shape or names.")
        worst = 0.0
        with torch.no_grad():
            for name in self.names:
                current = parameters[name].detach()
                if current.shape != self.snapshot[name].shape:
                    raise RuntimeError(f"Frozen parameter {name} changed shape.")
                diff = (current.to(self.snapshot[name].device) - self.snapshot[name]).abs()
                worst = max(worst, float(diff.max().item()) if diff.numel() else 0.0)
        return worst
