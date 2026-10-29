# RA-L visual draft

Updated 29 October 2026.

main.pdf is the current reading copy; main.tex is the entry point.

The 29 October revision makes the Introduction contribution paragraph concrete and centers Related Work on GRPO/DAPO, SimpleVLA-RL, iRe-VLA, and PLD. DAPO-style filtering and bounded resampling are described in Methods, including the distinction between binary outcomes and dense-return dispersion; provenance is recorded in evidence/DAPO_SAMPLING_AUDIT_20261029.md.

## Current structure

1. Introduction
2. Related Work
3. Methods
4. Experiments (setup and results together)
5. Conclusion
6. Future Work

The article centers on one adaptation pipeline. Methods C integrates the LIBERO-Spatial feedback comparison as empirical motivation for full-task composition, followed by milestone credit in the adapted controller. There is no separate Methods E for LIBERO. Experiments present the LIBERO comparison first, then CDPR adaptation and retention, and the ongoing WX200 evaluation. The manuscript contains four displayed expressions: self-imitation loss, mixed-outcome group probability, the compact clipped RL loss with entropy regularization, and the latent-action sampling rule. The 40-dimensional action prior is described in prose. Methods D describes current sampling and likelihood evaluation without historical implementation comparisons or clipping examples.

Algorithm 1 is sourced from algorithms/staged_pipeline_paper.tex. It preserves the order of all dense skill stages before all binary stages, then full-task composition. After a composition plateau, it includes the CDPR continuation with a retained bounded residual, frozen VLA/LoRA, fresh optimizer, and offset-conditioned pre-clipping likelihoods. The text gives qualitative transition criteria and distinguishes automated stopping from historical CDPR manual transitions. Exact configuration values and implementation provenance remain in supporting audit notes.

## Editable figures

Four figures are included in the current paper:

1. Author-supplied graphical abstract, figures/abstract_overview.png, on the first manuscript page.
2. SmolVLA, LoRA, 518d observation features, and the residual action path.
3. Skill success before and after balanced SIL, using only the recovery comparison. Generate SVG/PDF/PNG exports with scripts/generate_retention_figure.py from evidence/skill_retention_rounds.json.
4. WX200 apparatus and five-frame rollout slots.

Open figures/editable/ral_figures.xml in diagrams.net: File → Open from → Device. It retains six original source diagrams; individual XML files are in the same folder. The graphical abstract PNG and retention SVG supersede the corresponding XML designs. The former milestone-credit and feedback-resolution figures are omitted from the manuscript; their editable assets remain available.

The graphical abstract is included directly as PNG; other figures use PDF exports. See figures/README.md for editing instructions. The original combined XML and individual diagrams are packaged in figures/editable/ral_figures.zip.

## Evidence

- evidence/libero_paper_results_export_20261026.csv supplies the exact endpoint values for the LIBERO reward-comparison table.
- evidence/skill_retention_rounds.json contains the round rates for Figure 3. The figure uses the balanced-SIL recovery comparison.
- evidence/latent_control_40m_comparison_20261019.json supports the earlier 40M controller evaluation (61.43%, 1,258/2,048).
- evidence/latent_control_60m_tensorboard_20261028.json records the 60M endpoint's strict success (70.61%, 723/1,024), peak (71.09%, 728/1,024), source hash and measurement details. The complete CSV exports and extraction script are in ../docs/artifacts/latent_control_60m_20261028/.
- evidence/ACTION_MODULE_AUDIT_20261022.md documents the architecture and LoRA implementation.

The manuscript reports 70.61% strict success for the CDPR controller on development validation used during training, and descriptive LIBERO comparisons. Table I uses all 1,024 CDPR attempts and includes milestones, carry slip, and wrong placement. Numerical failures remain in the denominator but have no separate table row. Training-step counters, checkpoint identifiers, intermediate checkpoint scores, and the history of logged runs are omitted from the reader-facing text. WX200 illustrations and quantitative results remain reserved for the ongoing evaluation. Source data, training history, and detailed mathematical audits remain in the supporting records.

## Build

From this folder:

    python3 scripts/create_editable_figures.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    cp build/main.pdf main.pdf

Python needs matplotlib; exporting the supplied retention SVG also needs rsvg-convert. Figure assets are already included, so LaTeX can compile without Python. The default figure command renders existing individual XML except for skill_retention, whose supplied SVG is preserved and exported to PDF/PNG. The --rebuild-layouts flag recreates initial XML layouts and overwrites manual XML edits, but does not replace the supplied PNG or SVG.

The local renderer supports the original draft's basic shapes/styles. After adding new diagrams.net features, export a vector PDF directly from the editor using the same basename.

STRUCTURE_REVISION_20261026.md records the current editorial revision. Earlier dated integration notes are historical where they conflict with this version. The manuscript preceding the visual revision is preserved in archive/manuscript_20261026_before_visual_revision/.

REVISION_NOTES_20261028.md records the figure removals, shorter LIBERO account, retention-claim provenance, and plateau-continuation evidence.
