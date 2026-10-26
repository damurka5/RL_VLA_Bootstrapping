# Training algorithm snippets

The snippets require algorithm, algpseudocode, amsmath and amssymb.

- staged_pipeline_paper.tex is the compact method-level Algorithm 1 now included in the manuscript. It gives qualitative stage stopping and retention criteria, with all dense skills preceding all binary skills. The paper distinguishes automated stopping from historical CDPR manual transitions; numerical WX200 gates remain in this supporting material.
- staged_bootstrapping.tex describes the current automated WX200 controller: all dense skill stages, all binary skill stages, then joint staged RL. SIL is conditional after individual skill stages.
- cdpr_group_update.tex describes the selected CDPR continuation's latent likelihood and three-stage credit.

The implementations differ. The default WX200 config still uses legacy clipped-action likelihood, permits LoRA updates, and uses milestone-fraction rewards rather than the CDPR independent three-stage advantage streams. Do not claim the two snippets describe one identical executed configuration.

The full Russian mathematical audit and transition thresholds are in ../METHOD_AUDIT_20261026_RU.md. The two detailed implementation-specific snippets remain supporting material; the compact paper procedure is included through sections/04_method.tex.
