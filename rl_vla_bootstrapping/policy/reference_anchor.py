"""A frozen-reference anchor for the residual actor during GRPO.

Why this exists
---------------

Every continuation of the three-stage lineage has traded skills against each
other: grasp and lift erode, and plate and bowl swap. The protections the
trainer already has act on the RL signal (the achieved-negative scale zeroes
some negative advantages). They do not stop OTHER updates to the shared
residual from moving what it does on states where it was already good. The
per-update PPO clip bounds each step but not their sum: the LR 1e-5
continuation averaged 0.009 sampled KL per update and still lost lift retention
over 113 updates.

This adds a weak penalty that pulls the residual's mean back toward a FROZEN
reference policy's mean on a fixed bank of states that reference produced when
it succeeded, balanced over object x destination x stage.

What makes it cheap and exact here
----------------------------------

In the three-stage config the SmolVLA LoRA is frozen for GRPO
(``vla_lora_updates_enabled: false``), so the prior the residual is conditioned
on never changes during a run. A bank row's stored ``(state, prior)`` therefore
stays a valid input for the whole run. The reference's mean on it is computed
ONCE, and afterwards the anchor is one residual forward per optimizer step: no
VLA forward, no frames, no reference network in memory.

The target is the reference's mean on the SAME input, not the recorded action.
The recorded action carries the prior's noise draw from recording time, and
fitting it is exactly the null target the self-imitation SFT arms measured:
the loss would pull the residual toward ignoring the prior rather than toward
the reference. Against the reference's own mean the anchor is exactly zero at
the start and only acts once the policy has moved.

The loss is the mean term of KL(reference || current) at the reference's
std: ``(mu - mu_ref)^2 / (2 sigma_ref^2)``, averaged over supervised slots and
action dims. So it reads in nats per action dimension and does not fight the
entropy bonus through log_std.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STAGE_NAMES = ("move_to", "pick_up", "placement")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_anchor_rows(
    bank_path: Path, *, stages: Sequence[str]
) -> dict[str, np.ndarray]:
    """The bank rows the anchor will use, with a census for the log."""

    unknown = sorted(set(stages) - set(STAGE_NAMES))
    if unknown:
        raise ValueError(f"Unknown anchor stages {unknown}; known: {list(STAGE_NAMES)}.")
    with np.load(bank_path, allow_pickle=False) as data:
        required = {"state", "prior", "action", "action_mask", "stage_name"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(
                f"{bank_path} lacks {missing}; the anchor needs a bank written by "
                "build_cdpr_full_put_into_dataset.py."
            )
        keep = np.isin(np.asarray(data["stage_name"]).astype(str), list(stages))
        if not keep.any():
            raise ValueError(f"{bank_path} has no rows in stages {list(stages)}.")
        rows = {
            name: np.asarray(data[name])[keep]
            for name in (
                "state", "prior", "action", "action_mask", "stage_name",
                "destination", "target_catalog", "source_checkpoint_sha256",
            )
            if name in data.files
        }
    return rows


class ReferenceAnchor:
    """Precomputed reference means on a fixed bank and the loss against them."""

    def __init__(
        self,
        *,
        torch: Any,
        rows: Mapping[str, np.ndarray],
        reference_mean: Any,
        reference_log_std: Any,
        coef: float,
        batch_size: int,
        seed: int,
        device: Any,
        census: Mapping[str, Any],
    ) -> None:
        self.torch = torch
        self.device = device
        self.coef = float(coef)
        self.batch_size = int(batch_size)
        self.state = torch.as_tensor(rows["state"], dtype=torch.float32, device=device)
        self.prior = torch.as_tensor(rows["prior"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(rows["action_mask"], device=device)
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], mask.shape[1], -1).any(dim=-1)
        self.mask = mask.to(torch.float32)
        self.slots = int(self.mask.shape[1])
        self.reference_mean = reference_mean.to(device=device, dtype=torch.float32)
        # Per-slot, per-dim inverse variance of the reference: the KL's scale.
        self.inv_two_var = (
            0.5 * torch.exp(-2.0 * reference_log_std[: self.slots].to(device=device, dtype=torch.float32))
        )
        self.rows = int(self.state.shape[0])
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.census = dict(census)

    @classmethod
    def build(
        cls,
        *,
        torch: Any,
        actor: Any,
        bank_path: Path,
        reference_checkpoint: Path,
        stages: Sequence[str],
        coef: float,
        batch_size: int,
        seed: int,
        device: Any,
        runtime_lora_state: Mapping[str, Any] | None = None,
        eval_batch: int = 1024,
    ) -> "ReferenceAnchor":
        """Load the bank, load the reference into a COPY of ``actor``, precompute.

        ``actor`` is the unwrapped residual module (the trainer's policy). It is
        deep-copied, so the live policy is never touched. The reference is read
        from an explicit checkpoint, never from whatever the run resumed from:
        a pilot resumed mid-way must still be anchored to its starting policy.
        """

        import copy

        bank_path = Path(bank_path).expanduser().resolve()
        reference_checkpoint = Path(reference_checkpoint).expanduser().resolve()
        rows = load_anchor_rows(bank_path, stages=stages)
        try:
            payload = torch.load(reference_checkpoint, map_location=device, weights_only=False)
        except TypeError:  # PyTorch < 2.6
            payload = torch.load(reference_checkpoint, map_location=device)
        if "policy" not in payload:
            raise KeyError(f"{reference_checkpoint} has no 'policy' weights.")
        from rl_vla_bootstrapping.policy.latent_correction_policy import (
            require_legacy_policy_checkpoint,
        )

        require_legacy_policy_checkpoint(payload, "ReferenceAnchor")
        reference = copy.deepcopy(actor)
        reference.load_state_dict(payload["policy"])
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

        # The bank's priors were produced by the reference's LoRA. With a frozen
        # LoRA the run's prior network must be that same LoRA, or the stored
        # priors are not what this run's residual will ever be conditioned on.
        lora_max_abs_diff = None
        reference_lora = payload.get("vla_lora")
        if reference_lora and runtime_lora_state is not None:
            diffs = [
                float((reference_lora[key].to(device=device, dtype=torch.float32)
                       - runtime_lora_state[key].to(device=device, dtype=torch.float32)).abs().max().item())
                for key in reference_lora
                if key in runtime_lora_state
            ]
            if not diffs:
                raise ValueError("No LoRA tensor of the reference matches the run's runtime.")
            lora_max_abs_diff = max(diffs)
            if lora_max_abs_diff > 1.0e-6:
                raise ValueError(
                    f"The run's LoRA differs from the anchor reference's by "
                    f"{lora_max_abs_diff:.3g}; the bank's priors would not be "
                    "this run's priors."
                )

        state = torch.as_tensor(rows["state"], dtype=torch.float32, device=device)
        prior = torch.as_tensor(rows["prior"], dtype=torch.float32, device=device)
        action = torch.as_tensor(rows["action"], dtype=torch.float32, device=device)
        mask = torch.as_tensor(rows["action_mask"], device=device)
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], mask.shape[1], -1).any(dim=-1)
        slots = int(action.shape[1])
        means = []
        with torch.no_grad():
            for start in range(0, int(state.shape[0]), int(eval_batch)):
                stop = start + int(eval_batch)
                try:
                    out = reference(state[start:stop], prior[start:stop])
                except RuntimeError as error:
                    raise ValueError(
                        f"The residual cannot consume the anchor bank's rows "
                        f"(state {tuple(state.shape)}, prior {tuple(prior.shape)}); "
                        "it was recorded under a different observation layout."
                    ) from error
                means.append(out[:, :slots])
            reference_mean = torch.cat(means, dim=0)
            weight = mask.to(torch.float32).unsqueeze(-1)
            # Integrity readout: on a bank recorded by the reference itself with
            # a deterministic residual, the recorded action IS this mean up to
            # action clipping. A large value means the bank came from another
            # policy or its priors were refreshed with fresh noise.
            recorded_mse = float(
                (((reference_mean - action) * weight) ** 2).sum().item()
                / max(1.0, float(weight.sum().item()) * float(action.shape[-1]))
            )
            reference_log_std = reference.clamped_log_std().detach().clone()

        census: dict[str, Any] = {
            "bank": str(bank_path),
            "bank_sha256": _sha256(bank_path),
            "reference_checkpoint": str(reference_checkpoint),
            "reference_sha256": _sha256(reference_checkpoint),
            "stages": list(stages),
            "rows": int(state.shape[0]),
            "by_stage": {
                str(name): int((rows["stage_name"] == name).sum())
                for name in np.unique(rows["stage_name"])
            },
            "recorded_action_mse_vs_reference_mean": round(recorded_mse, 10),
            "reference_lora_max_abs_diff": lora_max_abs_diff,
        }
        for column in ("destination", "target_catalog"):
            if column in rows:
                census[f"by_{column}"] = {
                    str(name): int((rows[column] == name).sum())
                    for name in np.unique(rows[column])
                }
        if "source_checkpoint_sha256" in rows:
            census["bank_source_checkpoint_sha256"] = sorted(
                {str(value) for value in np.unique(rows["source_checkpoint_sha256"])}
            )
        del reference
        return cls(
            torch=torch,
            rows=rows,
            reference_mean=reference_mean,
            reference_log_std=reference_log_std,
            coef=coef,
            batch_size=batch_size,
            seed=seed,
            device=device,
            census=census,
        )

    def sample(self) -> Any:
        return self.torch.randint(
            0, self.rows, (min(self.batch_size, self.rows),),
            generator=self.generator, device=self.device,
        )

    def kl(self, actor: Any, index: Any) -> Any:
        """Mean-term KL(reference || current) in nats per supervised action dim."""

        mean = actor(self.state[index], self.prior[index])[:, : self.slots]
        weight = self.mask[index].unsqueeze(-1)
        per_dim = (mean - self.reference_mean[index]) ** 2 * self.inv_two_var
        denominator = (weight.sum() * float(mean.shape[-1])).clamp_min(1.0)
        return (per_dim * weight).sum() / denominator

    def full_kl(self, actor: Any, *, batch: int = 2048) -> float:
        """The anchor KL over the whole bank: the drift readout, logged every update."""

        torch = self.torch
        total = 0.0
        count = 0.0
        with torch.no_grad():
            for start in range(0, self.rows, int(batch)):
                index = torch.arange(start, min(self.rows, start + int(batch)), device=self.device)
                weight = self.mask[index].sum().item()
                total += float(self.kl(actor, index).item()) * weight
                count += weight
        return total / max(count, 1.0)
