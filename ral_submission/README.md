# RA-L visual draft

Updated 26 October 2026.

main.pdf is the current reading copy; main.tex is the entry point. This revision reduces the draft from eight to six pages and roughly halves the text. Training-snapshot comparisons and generic training curves are removed.

## Editable figures

Open figures/editable/ral_figures.xml in diagrams.net: File → Open from → Device. It contains six pages of native editable shapes, labels, arrows, bars and evaluation marks. Individual XML files are in the same folder.

1. Pipeline overview on the first manuscript page.
2. SmolVLA, LoRA, 518d observation features and the residual action path.
3. Milestone credit after an achieved prefix and failed placement.
4. Terminal, phase and within-phase feedback.
5. Measured forgetting and balanced-SIL recovery.
6. WX200 apparatus and five-frame rollout slots.

PDF exports are included in the paper; SVG and PNG exports are alongside them. See figures/README.md for editing instructions.

## Evidence

- evidence/libero_paper_results_export_20261026.csv: exact copy of the new author-supplied export found in Downloads. Table II uses its final rows. It corrects task_1 initialization to 2% and primary dense weights to w_p = w_c = 0.05.
- evidence/skill_retention_rounds.json and skill_retention_source_20261026.md: reported round rates for Figure 5. Its panels are separate measurements.
- evidence/latent_control_40m_comparison_20261019.json: only the final controller's evaluation is included.
- evidence/ACTION_MODULE_AUDIT_20261022.md: architecture and LoRA evidence.

The paper reports development evaluation. WX200 and physical results remain unfilled. Two explicit method placeholders remain: the training manifest and exact phase-reward implementation.

## Build

From this folder:

    python3 scripts/create_editable_figures.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
    cp build/main.pdf main.pdf

Python needs matplotlib. Vector PDFs are already included, so LaTeX can compile without Python. The default figure command renders existing individual XML and preserves manual edits. The --rebuild-layouts flag recreates initial layouts and overwrites manual XML edits.

The local renderer supports the original draft's basic shapes/styles. After adding new diagrams.net features, export a vector PDF directly from the editor using the same basename.

VISUAL_REVISION_20261026.md records changes and evidence limits. Earlier active sources and PDF are preserved in archive/manuscript_20261026_before_visual_revision/. Other dated notes are historical and are superseded where they conflict with this revision.
