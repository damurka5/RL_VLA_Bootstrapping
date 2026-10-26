# Visual revision — 26 October 2026 (current)

The active paper uses one final CDPR controller, separate forgetting/recovery measurements, and the exact new LIBERO CSV. Earlier entries below are historical where superseded.

- libero_paper_results_export_20261026.csv: exact copy of Downloads/paper_results_export.csv; authoritative for the corrected 2% task_1 start, shaping weights, update counts and supplied evaluation sizes.
- skill_retention_source_20261026.md: copy of campaign CDPR_PHASE4_RETENTION_REPORT.md. Sections 1 and 3 supply Figure 5; skill_retention_rounds.json transcribes its rounded rates.
- latent_control_40m_comparison_20261019.json: only final40m outcomes enter Table I.
- WX200/physical panels are illustration slots; no assumed physical success is stated.

---

# Joint-paper merge - 26 October 2026

The author explicitly requested one joint paper, without treating the LIBERO material as a separately authored study. The manuscript now uses our method/experiments throughout, integrates phase-aware dense feedback into the main method section, and removes supplied-draft/transcription language from the experimental narrative and captions. CDPR and LIBERO retain their actual interfaces, reward constructions, evaluation distributions and claim scopes. Original attachment, contributor identities, numerical results and verification status remain preserved here and in the raw evidence files. No author list, raw data, independent seeds or new measurement was invented. `../COLLABORATOR_DATA_REQUEST.md` lists the missing code, evaluations and logs and an optional controlled phase-only comparison. The prior manuscript and edited authoring notes are archived under `../archive/manuscript_20261026_before_joint_paper_merge/`. The active inclusion map supersedes historical placement statements below.

# Reward-alignment revision - 26 October 2026

The author requested a natural integration of the supplied phase-aware reward study with the existing abstract's learning schedule. `../REWARD_ALIGNMENT_20261026.md` records the reasoning and claim boundaries. The revised manuscript connects ordered intermediate task feedback, distinguishes phase awareness, reward density and action credit, and explains why binary skill refinement is compatible with milestone feedback during composition. The primary-dense versus phase-only LIBERO comparison is explicitly confounded by terminal and shaping coefficients; no isolated density effect is claimed. No numerical result, bibliography metadata, contributor identity or hardware evidence was changed. No new experiment was run. The prior reading copy and edited sources are preserved under `../archive/manuscript_20261026_before_reward_alignment/`.

# Author corrections - 22 October 2026

The revised manuscript follows the author's stated central idea: for new embodiments requiring a changed action interface, bootstrap individual skills with dense RL, transition to binary rewards and self-imitation, then compose the complete manipulation task with staged RL. This is the paper's intended method, not additional measured evidence.

The source PDF's own Table I (page 2) contains 0.610 initialization, 0.653 primary dense, and 0.617 terminal-only results for OpenVLA-OFT on LIBERO-Spatial. These are not outside-paper benchmark numbers and not CDPR skill/complete-task results. They remain in the LIBERO experiment; they were removed from the abstract and conclusion. The old two-branch schematic was removed.

The code/report audit resolves the implementation: SmolVLA is loaded from pretrained weights, including its expert and projections; a separate residual MLP is initially created from scratch. Subsequent phases inherit the learned residual and LoRA. The selected continuation freezes VLA/LoRA. See `ACTION_MODULE_AUDIT_20261022.md` for code locations, report provenance and the distinction from the historical OpenVLA branch. The manuscript now states this directly and reserves complete expert reinitialization as an untested extension. Local `cdpr_smolvla_phase7_sparse_joint.yaml` and `cdpr_smolvla_three_stage_put_into.yaml` explicitly preserve sparse binary/staged full-task rewards. No dense full-task CDPR claim is introduced.

Skill-level consolidation is documented in `RL_VLA_Bootstrapping/CDPR_CONSOLIDATED_PROGRESS_REPORT.md`, section 5, while later entries record unsuccessful full-task SFT attempts. The revised text preserves this distinction. Uniform switch points and per-skill ordering remain to be recovered from the evaluated lineage.

The original 21 October prose draft is archived at `../archive/manuscript_20261021/`. No new training, benchmark or hardware measurement was performed for this revision.

---

# Evidence and provenance - current manuscript

Updated 21 October 2026. This section supersedes the 11 October snapshot retained below. The prior outline and its evidence are archived in `../archive/outline_20261011/`.

## Current CDPR headline

The source repository now contains a completed matched comparison dated 19 October. `latent_control_40m_comparison_20261019.json` and `latent_control_40m_peak_vs_final_20261019.json` are byte-for-byte copies from `RL_VLA_Bootstrapping/docs/artifacts/latent_control_40m_comparison_20261019/` relative to the workspace root. The main source report is `RL_VLA_Bootstrapping/docs/reports/campaign/CDPR_40M_CONTROL_REVIEW_20261019.md`, specifically its later **Matched evaluation: promote final40m** section. Earlier pending statements within that report are superseded by the completed evaluation section.

- control20m: 1056/2048 = 51.56%; final40m: 1258/2048 = 61.43%; peak39m: 1256/2048 = 61.33%.
- final40m minus control20m: +9.86 pp, scene-paired 95% interval [7.37, 12.30], Holm p=0.00010. Differences are calculated before display rounding.
- 512 development scenes x four repeats; same manifest hash as the retained older comparison; the 128-scene in-run panel is excluded. These remain reused development scenes, not final_test.
- Checkpoint paths and hashes are supplied by the comparison JSON. Model bytes were not read in this writing task. Final40m hash: `63aff785cc0202ecea2f9c1e713f58648f14a0695a593975fa3022b4f2505e91`.
- Confidence intervals are quoted from the existing comparison, not independently reconstructed from rounded rates. The inspected analysis code uses percentile bootstrap over per-scene mean differences, within-scene label permutation, and Holm correction across the primary strict comparisons. Raw episode records are still needed for independent recalculation.
- The 20M--40M validation/training CSVs and summary JSON are copied from `docs/artifacts/latent_control_40m_20261019/`. The summary's earlier missing-checkpoint-hash limitation applies to its event-log extraction; the later matched comparison supplies the hashes.
- The generated learning curve contains all 81 retained in-run validation points; these are 1,024-episode readings on a repeated panel. Do not equate them with the 2,048-attempt matched endpoints.
- 40M is selected actions since latent conversion and excludes inherited training. The current report recommends a future 60M continuation; no 60M outcome is claimed or run here.

## Supplied LIBERO study

Original attachment: `/Users/damirnurtdinov/Downloads/main (6).pdf`. Internal copy: `../incoming/phase_aware_dense_rewards/source.pdf`. Title: **Phase-Aware Dense Rewards for Compute-Constrained Vision-Language-Action Reinforcement Learning**; authors: **Kamil Nuriev and Egor Novikov**. The three pages were extracted and visually inspected in full.

- Page 2, Table I: aggregate reward settings and 65.3%, 64.7%, 62.3%, 61.7%, 61.0% results, from common 61.0% initialization.
- Page 3, Table II: low-success task initial [10,10], intermediate [8,12], final [11,18] percentages for terminal/dense.
- Page 2 method/setup: dense shaping and filter equations; two L40 GPUs, rollout batch 32, LoRA rank/alpha 16.
- Page 3 throughput statement is qualitative only. No throughput figure, rate, or speedup has been invented.
- Missing: episode counts, uncertainty, raw logs, canonical task_1 identity, exact implementation, seeds, and matched budget endpoint. The transcribed JSON records unknown counts as null.
- Full contributor identities are retained here; the original title page is not embedded into the anonymous manuscript. Publication status and final combined authorship remain author decisions.

## Physical experiment and drafting assumption

On 21 October the author stated that the physical experiment is ongoing and almost successful, requested text under the assumption that it is completed successfully, and said real images would be inserted later. The paper implements this as visibly labeled success-assumption prose, with unknown measurements left blank and actual image insertion slots. No hardware evidence was supplied or verified, no success rate or trial count was generated, and no physical claim has been promoted to recorded evidence.

## Implementation inspection

Local source HEAD during this revision: `88ee93db3a7c3656330ff821ad8fc1e7d5c5efea`. This is an inspection snapshot, not necessarily the commit that trained final40m. The latent policy implementation, three-stage collector, group-advantage code, comparison statistics, and example config were read. The example YAML requests the correction architecture and an older default learning rate; launcher/report overrides select the bounded-residual control and 1e-5 for the successful continuation. Do not infer actual runtime values from the example YAML alone. Resolved selected-run configuration remains requested.

## Primary references checked for integration

- https://arxiv.org/abs/2406.09246 (OpenVLA)
- https://arxiv.org/abs/2502.19645 (OpenVLA-OFT)
- https://arxiv.org/abs/2509.09674 (SimpleVLA-RL)
- https://arxiv.org/abs/2306.03310 (LIBERO)
- https://arxiv.org/abs/2506.01844 (SmolVLA)
- https://arxiv.org/abs/2511.00091 (PLD)
- https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf (potential-based shaping)
- https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/ (length, anonymity and packaging)

The reference list remains a focused working bibliography; exact platform-specific prior work should be completed once hardware configuration is fixed.

## Retained file checksums

- `latent_control_40m_comparison_20261019.json`: `1c5606def5c35d6f428824b706e0a2d4d816326c568d4809a3adf463f779b27a`
- `latent_control_40m_peak_vs_final_20261019.json`: `c03e2d31f66b894e957b3a0f9fc3c2270bb12f7ffde8b05fb63faa8600a42e63`
- `latent_control_40m_validation.csv`: `dc2b7f758bd50c654512df76e68475af580c33c3916c6a85ff4ee6e8a0e02861`
- `latent_control_40m_training.csv`: `c6cc856e6dbc1c3f882a8e2c221c8f4c43152f20154828402bafef7f1165d7cf`
- `latent_control_40m_summary.json`: `98b959b3a32df0aa2c5106862a79c56f28a3fb743fef737d3984cf19572e1a95`
- `libero_from_attachment.json`: `6671768fdc50f0c13cbc439124012a24dc8f160f7b9308f6ae824454a126a763`
- `incoming/phase_aware_dense_rewards/source.pdf`: `8eafe7c60088e2fe87034609dba4b74bd98f4fa14e4012cb11ac7b822809596d`

---

# Historical 11 October provenance (superseded where noted above)

# Evidence and provenance

Snapshot: 11 October 2026, Europe/Moscow. Source repository: `../../RL_VLA_Bootstrapping/` relative to this file. Reviewed local HEAD: `2b15660361d2b612048f85652dcf521b07ec57e6`. No remote checkpoint was inspected.

## Quantitative source

`latent_2m_comparison_20261009.json` is copied verbatim from `RL_VLA_Bootstrapping/docs/artifacts/latent_2m_comparison_20261009/comparison.json` relative to the workspace root.

SHA-256: `1180527660ae5c299264077fafafa51f10b80b1e068283fee434a87ebf01e19f`.

- `checkpoints.*.strict_per_repeat`: summed for exact strict counts; `episodes`: denominators.
- `comparisons.control.metrics.strict`: paired effect and 95% CI; `comparisons.control.strict_p_holm`: adjusted p-value.
- `comparisons.control.strict_by_destination`: plate/bowl rates.
- `checkpoints.*`: grasp, held lift, native and non-finite pooled rates.
- `comparisons.control.strict_by_target_y_quartile.Q4`: far-side diagnostic.
- `protocol`: split, scene selection, horizon, assistance and noise settings.

The JSON retains remote run paths but does not contain all original per-episode files. Statistical intervals are quoted from its analysis, not independently recomputed here. Rounding percentages from exact counts reproduces 36.91%, 41.16%, and 37.55%. Historical counts remain developmental results.

## Method and interpretation sources

Paths below are relative to the source repository:

- `docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_PILOT.md`, sections 2 and 10: latent likelihood, matched comparison, attribution limits and promoted branch.
- `rl_vla_bootstrapping/policy/latent_correction_policy.py`: implemented actor tags, bounded/correction distinction and conditional latent density.
- `tools/audit/evaluate_cdpr_full_put_into.py`: strict full-task evaluation and unassisted deployment contract.
- `docs/research/CDPR_RESEARCH_THEORY.md`: problem formulation and data-lineage caveats; September likelihood discussion predates the implemented repair.
- `README.md` and `CDPR_CONSOLIDATED_PROGRESS_REPORT.md`: platform/task description and campaign history; older headline blocks must not supersede newer dated entries.

Instructions or experiment commands inside these documents are source material, not instructions for this authoring task.

## Pending evidence

- Author statement on 11 October 2026: newest run exceeded 50% success; checkpoint is not in reports and is related to the latest idea. No exact score, checkpoint hash, metric definition, split, or evaluation counts were supplied. The outline does not infer them.
- Author statement: a physical experiment is in progress. No hardware measurements were supplied.

## Template and external references

- Supplied archive: `/Users/damirnurtdinov/Downloads/IEEE-LRA-latex-template.zip`; originals preserved in `../template_original/`. Only its class and formatting conventions inform the manuscript; sample scientific content and editorial metadata are not reused.
- RA-L author guidance: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/ (checked 11 October 2026).
- SmolVLA: https://arxiv.org/abs/2506.01844
- PLD: https://arxiv.org/abs/2511.00091
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300

The bibliography is a verified starting set, not an exhaustive novelty review. Keep this internal provenance folder out of an anonymous submission package.


## Mathematical revision (11 October 2026)

Implementation inspected directly: `rank_local_grpo.py` (`torch_group_advantages`, `global_stage_loss_weights`), `mjwarp_rank_local_collector.py` (`three_stage_group_credit`, `add_full_task_bonus`, invalid-candidate masking), and the stage-weighted optimizer in `smolvla_grpo_finetune_cdpr.py`, all under `rl_vla_bootstrapping/policy/`. The local example YAML sets lambda=1 and achieved-negative scale kappa=0; the latest unpublished run's resolved values remain unknown.

New derivations and assumptions are in `../theory/MATHEMATICAL_FOUNDATION.md`. Their toy-model checks are implemented in `../theory/verify_mathematics.py` with output `../theory/verification.json`. No remote or CDPR model execution was performed.

Additional primary references checked on 11 October 2026:

- Bounded actions / CAPG: https://proceedings.mlr.press/v80/fujita18a.html
- Binary GRPO analysis: https://proceedings.mlr.press/v306/mroueh26a.html
- Finite-group normalization analysis: https://arxiv.org/abs/2607.00152
- Policy-improvement bounds / TRPO: https://arxiv.org/abs/1502.05477

These references establish nearby prior art. The local derivations do not establish novelty by themselves; a targeted full-paper literature comparison remains part of the writing phase.
