# Action-module initialization audit — 22 October 2026

## Conclusion for the evaluated SmolVLA controller

The implementation retains pretrained `lerobot/smolvla_base`, including its action expert and action projections. The newly initialized component is a separate residual MLP, with learned exploration parameters. Later training stages load the learned residual and, where present, inherited LoRA weights. The selected bounded-residual continuation freezes VLA/LoRA and updates the residual and exploration parameters. The 61.43% result therefore supports learning an embodiment-specific action module over a retained VLA; it does not demonstrate recovery after resetting the entire pretrained expert.

This conclusion is based on local source and campaign reports. Repository HEAD at the end of inspection: `d6e0332c0e45f549f983647d36fdc6d0dea9c662`. No model download, training, or tensor-by-tensor comparison with the remote checkpoint was performed. Exact historical resolved configurations and the complete checkpoint tree remain reproducibility inputs.

## Component evidence

Source paths below are relative to the nested code repository, `../../RL_VLA_Bootstrapping/` from this document. Line numbers refer to the inspected files. Abbreviated `policy/` paths are under `rl_vla_bootstrapping/`.

| Component | Initialization / inheritance | Evidence |
|---|---|---|
| SmolVLA base, expert and native projections | Loaded using `SmolVLAPolicy.from_pretrained`; base parameters frozen. No local replacement/reset of native expert/projections occurs in the inspected loading path. | `policy/smolvla_cdpr.py:674–680`; base config `configs/examples/cdpr_smolvla_three_stage_put_into_latent_correction.yaml:880` |
| Target action mapping | Selects five coordinates, pads/truncates when necessary, normalizes and adjusts the chunk horizon; not a learned replacement projection. | `policy/smolvla_cdpr.py:400–424,466–520` |
| Residual MLP | Fresh `nn.Linear` layers with the library's default initialization at scratch start. Receives state/visual features and mapped prior; predicts a bounded residual. | `policy/octo_finetune_cdpr.py:765–817`; scratch config `configs/examples/cdpr_smolvla_move_to_distance_grpo_mjlab_scratch.yaml:18–19` |
| Exploration | New `log_std` parameter initialized to a configured constant; subsequent loading restores learned policy state, including this parameter. | `policy/smolvla_grpo_finetune_cdpr.py:110–142,2773` |
| LoRA | Retains existing base linear layer; creates low-rank A/B matrices, with A Kaiming-initialized and B zero, so the initial additive update is zero. Earlier phases can train these adapters without resetting the expert. | `policy/lora.py:28–57`; history summary `docs/reports/history/OPENVLA_OCTO_ERA_SUMMARY.md:112` |
| Warm starts | Load residual policy weights and any saved LoRA. Resetting counters/optimizer state does not reset these learned weights. | `policy/smolvla_grpo_finetune_cdpr.py:2773–2785` |
| Selected continuation | Copies existing bounded-residual weights during legacy conversion; restores LoRA and freezes it. Selected architecture: `bounded_residual_v0`. | `policy/smolvla_grpo_finetune_cdpr.py:2866–2900,1116–1128`; `scripts/train_cdpr_latent_correction_pilot_remote.sh:5–15,43–46`; `docs/reports/campaign/CDPR_40M_CONTROL_REVIEW_20261019.md`, promoted final40m and continuation sections |

The current configuration has six proprioceptive dimensions, 512 visual features, eight slots, five actions and 1,024-wide hidden layers. Together with the actor code, this specifies a residual MLP of 558 -> 1024 -> 1024 -> 40: 518 observation features plus the 40-dimensional prior. This is a code/configuration inference; the manuscript still requests the selected run's resolved configuration to establish exact runtime dimensions and controller timing.

## Checkpoint-history corroboration

`docs/reports/campaign/CDPR_SMOLVLA_CAMPAIGN_REPORT.md:134–153` identifies the move-to scratch adapter as an ancestor of pickup and later training, and distinguishes the inherited early branch from the end of the entire move-to campaign. Lines 231–232 name the pickup initializer:

`cdpr_smolvla_move_to_scratch_mjwarp_w512_20260719_081705/rl/step_5000081`.

The report states that some historical hops were recorded from launchers rather than verified against model bytes. The manuscript states the supported initialization mechanism without presenting a completely audited tensor lineage.

The selected final checkpoint is recorded as:

`runs/latent_control_to40m_20261012_230538/rl/step_40056193/smolvla_grpo_adapter.pt`

with report-supplied SHA-256 `63aff785cc0202ecea2f9c1e713f58648f14a0695a593975fa3022b4f2505e91`. Its inherited LoRA is explicitly frozen in the selected continuation. `train_vla_lora: true` in the shared configuration attaches adapters so their weights can be loaded; accompanying `vla_lora_updates_enabled: false` disables updates. The former flag alone is not evidence that LoRA trained in this phase.

## Branches that must remain separate

1. **Historical OpenVLA/CDPR experiments.** `docs/reports/history/OPENVLA_OCTO_ERA_SUMMARY.md:29–38` describes an OFT continuous action head, a PPO value head and LoRA; line 112 distinguishes this architecture from the later SmolVLA residual. This historical CDPR branch is also separate from the supplied OpenVLA-OFT/LIBERO PDF. The presence of an OpenVLA action head in historical reports cannot establish that the SmolVLA expert was reset.
2. **Zero-initialized correction candidate.** The latent pilot launcher distinguishes a candidate that adds a zero-initialized correction over a frozen reference from a control that continues training the existing bounded residual. The promoted final40m result belongs to the control. The shared YAML's candidate default is overridden by `ARM=control`; reading the YAML alone would misidentify the selected architecture.
3. **Fresh training counter.** Weights-only warm starts and latent conversion can start a new step counter and optimizer while retaining learned weights. “Step zero” is not sufficient evidence of weight reinitialization.

## Manuscript wording

Use: **“We learn a new embodiment-specific action module over a pretrained VLA using dense-reward skill learning, binary refinement and self-imitation, followed by staged RL for the complete task.”**

For the evaluated implementation, name the residual module and retained SmolVLA expert. General motivation can include replacement or reinitialization of an action interface, but resetting a complete expert remains an untested extension. The abstract, introduction, method, limitations and data checklist have been aligned with this distinction.
