# Editable manuscript figures

The combined file is editable/ral_figures.xml. Each page is also saved separately.

| Paper | Individual XML | Purpose |
|---|---|---|
| Fig. 1 | pipeline_overview.xml | Dense acquisition → binary refinement and balanced SIL → milestone composition |
| Fig. 2 | smolvla_residual_lora.xml | Retained expert, LoRA, 512 + 6 = 518d observation, 40d prior and residual |
| Fig. 3 | milestone_credit.xml | Local and terminal credit after failed placement |
| Fig. 4 | phase_feedback.xml | Terminal / phase / within-phase feedback |
| Fig. 5 | skill_retention.xml | Observed forgetting and recovery with reported rounds |
| Fig. 6 | wx200_slots.xml | Apparatus and actual task images to insert later |

## Editing

Open an XML file in diagrams.net using File → Open from → Device. Blocks, chart bars, marks, labels and arrows are individual objects; no screenshots are embedded.

Use individual XML files as the source of truth. After editing the combined file, save the edited page back into its individual XML before running the local renderer. Otherwise the combined file is regenerated from the unchanged individual sources.

Run python3 scripts/create_editable_figures.py from ral_submission/ to render existing XML and rebuild the combined file and CSV-based Table II. This preserves individual XML edits. The renderer supports the draft's basic shapes and styles; use diagrams.net's vector PDF export for added shape types/effects.

The --rebuild-layouts flag resets all initial layouts. Do not use it after manual design work unless you intend to reset them.

Each figure has PDF, SVG and PNG exports. Keep the PDF basename when replacing it.

## Numerical integrity

Figure 5 uses evidence/skill_retention_rounds.json (relative to the paper folder). Bars are means of rounded reported rates; dots show individual rounds. No confidence intervals are invented. Forgetting and balanced-SIL recovery are separate measurements. The reaching bank includes an earlier policy's experience.

Figure 4 is a schematic reward sequence, not empirical telemetry. Figure 3 assumes usable groups, a negative terminal advantage and kappa = 0.

## WX200 insertion

Replace the layout with a complete vector panel at figures/wx200/panel.pdf. LaTeX inserts it automatically. Preserve approximately the 1000 × 287 aspect ratio. Add actual apparatus or verified CAD, five labeled joints, workspace/camera views, and reach/grasp/lift/transport/release frames with timestamps. Record instruction, trial ID and deployment condition.

The old CDPR photo slots and training curves remain in the revision archive.
