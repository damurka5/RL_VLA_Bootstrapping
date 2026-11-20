# RA-L visual draft

Updated 11 November 2026.

Figure 1 uses the original pipeline artwork in `figures/abstract_overview.png`. Figure 2 shows the mixed-group probability. Figure 3 shows skill measurements for both robots. Rebuild Fig. 2 with `python3 scripts/generate_task_figures.py` (FFmpeg, Pillow, NumPy and Matplotlib) and Fig. 3 with `python3 scripts/generate_retention_figure.py`. Ready-to-include figures are in `figures/`. The LaTeX entry point is `main.tex`, and `main.pdf` is the reading copy.

## Revision of 4 November 2026

The current reading copy reports **69.14% strict CDPR success (1,416/2,048)**
on 512 off-panel development scenes with four repeats. The matched gain over the contemporaneously
evaluated earlier controller is +8.35 pp, paired 95% CI [+6.05, +10.69], Holm
p=0.00005. The WidowX-200 simulation result is expanded into a transition audit: 7.13%
aligned empty-hand composition, skill loss before composition, weak ordinary
pickup signal, and no demonstrated rollout gain from composition SFT. A compact
theoretical rationale now opens Methods immediately after the pipeline overview
and algorithm input. It explains signal, retention, composition, and the limits
of ideal convergence results with short proof sketches. The pre-integration PDF
and edited sources are preserved in
`archive/manuscript_20261104_before_theory_and_widowx_integration/`.
`THEORY_WIDOWX_INTEGRATION_20261104.md` maps the new claims to their sources.

main.pdf is the current reading copy; main.tex is the entry point.

The focused 29 October revision restored the structure and two results tables from `archive/manuscript_20261029_before_checklist_revision/`. A compact four-row skill-by-stage matrix accompanies them, reporting three skill-specific RL references and the shared policy after consolidation. It clarifies adaptation without collecting human demonstrations, the newly initialized residual over the retained SmolVLA expert, and the relationship between the LIBERO feedback comparison and the complete CDPR pipeline. Automatically generated supervised data are described briefly in the experimental setup. Detailed training-history tables and editorial notes remain in supporting records. The preceding expanded version is preserved in `archive/manuscript_20261029_before_focused_revision/`.

## Current structure

1. Introduction
2. Related Work
3. Methods
4. Experiments (setup and results together)
5. Conclusion
6. Future Work

The article centers on one adaptation pipeline. Dense reward is used only at the beginning of adaptation to learn the base policy on simple, separately reset skills. Binary refinement and balanced self-imitation consolidate that policy; complete multi-stage tasks are trained with staged binary milestone reward. The LIBERO comparison includes manually defined dense-and-phase feedback as a reward variant, rather than as the reward used for full-task training in the adaptation pipeline. The LIBERO results table contains three full-suite reward configurations: Terminal only (SimpleVLA-RL), Dense + phase, and Phase-only, without lambda notation. Reward definitions remain in Methods; the LIBERO training setup and measured success rates appear in Experiments. Floats cannot precede their source location, keeping the LIBERO results table after the start of Experiments. Numerical provenance follows the author-corrected 26 October CSV; implementation details are audited at remote commit 44ed954. There is no separate Methods subsection for LIBERO. Experiments present CDPR base-policy skill learning and retention first (A), the LIBERO reward comparison second (B), complete-task CDPR adaptation third (C), and preliminary WidowX-200 simulation fourth (D); the scope of evidence follows (E). Methods A explains the theoretical rationale; Methods B lists the exact skill prompts and held-object placement resets; Methods D distinguishes the full-task prompt. The plate skill uses "on the plate", while full-task training uses "into plate", following the recorded instruction templates. The task is described as multi-stage pick-and-place throughout. The forgetting comparison and the balanced-SIL recovery panel are separate measurements, as recorded in evidence/skill_retention_rounds.json and sections 1 and 3 of evidence/skill_retention_source_20261026.md. The manuscript contains six displayed expressions: mixed-outcome group probability, a skill-retention bound, a composition bound, self-imitation loss, the compact clipped RL loss, and the latent-action sampling rule. The ideal mirror-descent rate is stated inline with its assumptions. The 40-dimensional action prior is described in prose. Methods E describes the CDPR continuation sampling and likelihood evaluation. The WidowX audit states that its archived run instead updated LoRA and used the historical clipped-action likelihood.

Algorithm 1 is sourced from algorithms/staged_pipeline_paper.tex (revised 20 November 2026). It keeps the order of all dense skill stages before all binary stages, then relabeling under the full-task instruction, full-task RL with milestone rewards, and the continuation with frozen VLA and adapters. A short TrainRL procedure states the GRPO loop with dynamic sampling and the clipped objective. Its notes give the rewards and the balanced self-imitation loss, and justify the stage order and the milestone reward from Eqs. (1)–(2). It uses the manuscript's own terms; the offset-conditioned likelihood, LoRA anchor and numerical gates remain in supporting audit notes.

## Editable figures

Three figures are included in the current paper:

1. Author-supplied graphical abstract, figures/abstract_overview.png, on the first manuscript page.
2. SmolVLA, LoRA, 518d observation features, and the residual action path.
3. Skill success before and after balanced SIL, using only the recovery comparison, with individual evaluation rounds. Generate SVG/PDF/PNG exports with scripts/generate_retention_figure.py from evidence/skill_retention_rounds.json.

WidowX-200 simulation is reported in Experiments with a compact six-row
transition table (`tables/widowx200_handoffs.tex`) and discussed in Future Work;
its placeholder illustration is not included in the paper. The graphical
abstract and architecture figure retain the corrected descriptions of the new
residual and pretrained expert.

Open figures/editable/ral_figures.xml in diagrams.net: File → Open from → Device. It retains six original source diagrams; individual XML files are in the same folder. The graphical abstract PNG and retention SVG supersede the corresponding XML designs. The former milestone-credit and feedback-resolution figures are omitted from the manuscript; their editable assets remain available.

The graphical abstract is included directly as PNG; other figures use PDF exports. See figures/README.md for editing instructions. The original combined XML and individual diagrams are packaged in figures/editable/ral_figures.zip.

## Evidence

- evidence/libero_paper_results_export_20261026.csv supplies the exact endpoint values for the LIBERO reward-comparison table.
- evidence/skill_retention_rounds.json contains the round rates for Figure 3 and the reaching reference in the skill matrix. The remaining matrix entries come from the instruction-success table in ../CDPR_CONSOLIDATED_PROGRESS_REPORT.md; the consolidated row is the Cycle 3 shared policy. Full-task rates remain in their separate table.
- evidence/latent_control_40m_comparison_20261019.json supports the earlier 40M controller evaluation (61.43%, 1,258/2,048).
- evidence/latent_control_60m_comparison_20261104.json supplies the current matched result: 1,416/2,048 = 69.14%, against the same comparison's 1,245/2,048 = 60.79% baseline. It records the selected adapter hash, scene exclusion and paired intervals. Per-episode outcomes were not supplied, so the interval and p-value are retained as reported.
- evidence/latent_control_60m_review_summary_20261104.json records checked integer counts; tables/results_inventory.csv records the development evaluations.
- evidence/widowx200_final_report_20261103.json and evidence/widowx200_snapshot_audit_20261104.json supply the preliminary arm results and diagnostics. tables/widowx200_results.csv retains each reset protocol and denominator. The full snapshot ZIP and 435 verified source files are indexed in ../docs/artifacts/widowx200_snapshot_20261104/.
- theory/rl_only_justification_ru/main.tex supplies the detailed Russian theoretical source; it remains unchanged. Its verify_theorems.py checks finite-model identities and bounds, not neural-run assumptions.
- evidence/ACTION_MODULE_AUDIT_20261022.md documents the architecture and LoRA implementation.

The manuscript reports 69.14% strict CDPR success on development scenes outside
the fixed training-validation panel, and descriptive LIBERO comparisons. The
CDPR table uses all 2,048 attempts, including numerical failures, and reports
grasp/lift, carry slip and wrong placement. The text gives the paired strict
gain without a checkpoint-history table. Four rollout repeats do not provide
independent training seeds, and the final-test split remains unused. WidowX
scores come from simulation; its aligned empty-hand composition omits a distant
approach and is not directly comparable with CDPR strict completion. The arm's
weak result does not establish reliable cross-embodiment adaptation or hardware
performance. Source data, training history, and audits remain in supporting records.

## Build

From this folder:

    python3 scripts/create_editable_figures.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    cp build/main.pdf main.pdf

Python needs matplotlib; exporting the supplied retention SVG also needs rsvg-convert. Figure assets are already included, so LaTeX can compile without Python. The default figure command renders existing individual XML except for skill_retention, whose supplied SVG is preserved and exported to PDF/PNG. The --rebuild-layouts flag recreates initial XML layouts and overwrites manual XML edits, but does not replace the supplied PNG or SVG.

The local renderer supports the original draft's basic shapes/styles. After adding new diagrams.net features, export a vector PDF directly from the editor using the same basename.

STRUCTURE_REVISION_20261026.md records the current editorial revision. Earlier dated integration notes are historical where they conflict with this version. The manuscript preceding the visual revision is preserved in archive/manuscript_20261026_before_visual_revision/.

REVISION_NOTES_20261028.md records the figure removals, shorter LIBERO account, retention-claim provenance, and plateau-continuation evidence.
