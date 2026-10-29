# DAPO attribution and sampling description

Checked 29 October 2026 for the Introduction / Related Work revision.

## Primary literature

- DAPO, Yu et al., arXiv:2503.14476, §3.2: oversample and remove groups with all-correct or all-incorrect binary outcomes, refilling an informative batch. This is one component of DAPO, which also includes other algorithmic changes. The manuscript attributes the sampling component as **DAPO-style dynamic sampling**, without implying that every part of DAPO is reproduced. https://arxiv.org/html/2503.14476v1
- SimpleVLA-RL, arXiv:2509.09674, §3.3–3.4: applies dynamic sampling, asymmetric clipping and exploration changes to group-relative VLA RL. §6.1 reports behavior absent from supervised demonstrations; §6.2 discusses limitations of weak initialization. The Related Work therefore describes task-specific learning and behavioral discovery, without claiming arbitrary skill acquisition from zero competence. https://arxiv.org/html/2509.09674v1
- iRe-VLA, arXiv:2501.16664: online RL with a frozen VLM and trainable action head alternates with supervised updates on expert and successful online data. https://arxiv.org/html/2501.16664v1
- PLD, arXiv:2511.00091: residual specialists, policy-aligned data collection, and distillation into a generalist. https://arxiv.org/html/2511.00091v1

## Implementation evidence

Paths below are relative to the code repository containing this manuscript.

- `rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py`, near lines 4770–4805: dense-reward branches check reward span; the binary branch checks success fraction; a separate `min_group_reward_std` masks low-dispersion returns. With split credit the masks operate per return stream.
- `rl_vla_bootstrapping/policy/smolvla_grpo_mjwarp_cdpr.py`, near lines 2639–2705: bounded collection/refill loop uses informative-group or record targets, a maximum round count and a time limit.
- `configs/examples/cdpr_smolvla_pick_up_dense_grpo_mjlab_warmstart.yaml`, near lines 871–898: dynamic sampling enabled; a minimum reward standard deviation of 0.05 is set for the dense-reward setting.
- `configs/examples/cdpr_smolvla_three_stage_put_into.yaml`, near lines 1125–1143: explains why group and collection budgets must permit another round; sets an informative-group refill target.
- `docs/reports/campaign/CDPR_SMOLVLA_CAMPAIGN_REPORT.md`, §4.13, and `CDPR_CONSOLIDATED_PROGRESS_REPORT.md`, near line 694: explain low-dispersion dense-return masking and noise amplification.

## Wording boundaries

The author reports that dynamic sampling helped training leave its initial zero-success regime during dense acquisition. The implementation supports a distinction between binary-success homogeneity and dense-return homogeneity: an all-failure group can still contain informative dense-return differences.

Do not describe the dense implementation as unconditionally rejecting every group without a complete success. Do not imply every configuration performs multiple collection rounds: filtering and actual refill depend on the collector path and budgets. A filtering flag alone does not prove extra rollout collection. No isolated numeric DAPO gain is added to the manuscript from this observation.

## Edits in this revision

- Removed the final LIBERO qualification sentence from Introduction and replaced its generic contribution sentence with concrete schedule, milestone-credit and empirical contributions.
- Replaced model-centered Related Work with three method-centered paragraphs: group-relative RL/dynamic sampling, RL/supervised alternation, and residual RL/distillation.
- Added DAPO and iRe-VLA primary references.
- Added one dynamic-sampling paragraph to Methods. Reward definitions and staged-reward credit sections are preserved.
- Backup: `archive/manuscript_20261029_before_rl_related_work/`.
