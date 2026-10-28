# RA-L visual draft

Updated 28 October 2026.

main.pdf is the current reading copy; main.tex is the entry point.

## Current structure

1. Introduction
2. Related Work
3. Methods
4. Experiments (setup and results together)
5. Conclusion
6. Future Work

The article centers on the adaptation pipeline. CDPR simulation, OpenVLA-OFT on LIBERO-Spatial, and the ongoing WX200 study are evaluation settings. Problem motivation is integrated into Methods. The manuscript contains four displayed expressions: self-imitation loss, the 40d action prior, the compact clipped RL loss with entropy regularization, and mixed-outcome group probability.

Algorithm 1 is sourced from algorithms/staged_pipeline_paper.tex. It preserves the order of all dense skill stages before all binary stages, then full-task composition. The text gives qualitative transition criteria and distinguishes automated stopping from historical CDPR manual transitions. Exact configuration values and implementation provenance remain in supporting audit notes.

## Editable figures

Open figures/editable/ral_figures.xml in diagrams.net: File → Open from → Device. It contains six pages of native editable shapes, labels, arrows, bars, and evaluation marks. Individual XML files are in the same folder.

1. Pipeline overview on the first manuscript page.
2. SmolVLA, LoRA, 518d observation features, and the residual action path.
3. Milestone credit after an achieved prefix and failed placement.
4. Terminal, phase, and within-phase feedback.
5. Measured forgetting and balanced-SIL recovery.
6. WX200 apparatus and five-frame rollout slots.

PDF exports are included in the paper; SVG and PNG exports are alongside them. See figures/README.md for editing instructions. The combined XML and individual diagrams are packaged in figures/editable/ral_figures.zip.

## Evidence

- evidence/libero_paper_results_export_20261026.csv supplies the exact endpoint values for Table II.
- evidence/skill_retention_rounds.json contains the round rates for Figure 5. The panels are separate experiments.
- evidence/latent_control_40m_comparison_20261019.json supports the earlier 40M controller evaluation (61.43%, 1,258/2,048).
- evidence/latent_control_60m_tensorboard_20261028.json records the 60M endpoint's strict success (70.61%, 723/1,024), peak (71.09%, 728/1,024), source hash and measurement details. The complete CSV exports and extraction script are in ../docs/artifacts/latent_control_60m_20261028/.
- evidence/ACTION_MODULE_AUDIT_20261022.md documents the architecture and LoRA implementation.

The manuscript reports 70.61% strict success for the 60M CDPR controller and descriptive LIBERO endpoints. Table I uses all 1,024 CDPR attempts and includes milestones and failure events. The source log was recorded October 19–23 and reviewed October 28. WX200 illustrations and quantitative results remain reserved for the ongoing evaluation. Source data and detailed mathematical audits stay outside the reader-facing text.

## Build

From this folder:

    python3 scripts/create_editable_figures.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    cp build/main.pdf main.pdf

Python needs matplotlib. Vector PDFs are already included, so LaTeX can compile without Python. The default figure command renders existing individual XML and preserves manual edits. The --rebuild-layouts flag recreates initial layouts and overwrites manual XML edits.

The local renderer supports the original draft's basic shapes/styles. After adding new diagrams.net features, export a vector PDF directly from the editor using the same basename.

STRUCTURE_REVISION_20261026.md records the current editorial revision. Earlier dated integration notes are historical where they conflict with this version. The manuscript preceding the visual revision is preserved in archive/manuscript_20261026_before_visual_revision/.
