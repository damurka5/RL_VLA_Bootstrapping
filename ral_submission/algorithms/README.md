# Training algorithm snippets

The snippets require algorithm, algpseudocode, amsmath and amssymb.

- staged_pipeline_paper.tex is the method-level Algorithm 1 included in the manuscript (revised 20 November 2026). It uses only the manuscript's vocabulary: all dense skill stages, then all binary skill stages, each followed by saving smoothed and replayed successes and by self-imitation recovery when a skill falls below its best rate; then relabeling under the full-task instruction, full-task RL with milestone rewards, and the frozen-VLA continuation. A short TrainRL procedure gives the GRPO loop (dynamic sampling, group-relative advantage, DAPO-style clipped objective, starting-distance schedule). The notes give the rewards, the balanced self-imitation loss, and the justification from Eq. (1) and Eq. (2): with ordered milestones, P(std R_ms > 0) >= max_m P_mixed(prod_{i<=m} p_i), whose m = M term is the binary full-task value. Implementation details (log-std bounds, LoRA anchor, offset-conditioned likelihood, CDPR per-milestone protected credit, numerical gates) stay in the two snippets below and the pipeline config.
- staged_bootstrapping.tex describes the current automated WX200 controller: all dense skill stages, all binary skill stages, then joint staged RL. SIL is conditional after individual skill stages.
- cdpr_group_update.tex describes the selected CDPR continuation's latent likelihood and three-stage credit.

The implementations differ. The default WX200 config still uses legacy clipped-action likelihood, permits LoRA updates, and uses milestone-fraction rewards rather than the CDPR independent three-stage advantage streams. Do not claim the two snippets describe one identical executed configuration.

The full Russian mathematical audit and transition thresholds are in ../METHOD_AUDIT_20261026_RU.md. The two detailed implementation-specific snippets remain supporting material; the compact paper procedure is included through sections/04_method.tex.
