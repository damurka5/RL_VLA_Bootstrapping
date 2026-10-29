# Editable manuscript figures

The graphical abstract is the author-supplied abstract_overview.png. The retention chart shows only the balanced-SIL recovery comparison, generated from evidence/skill_retention_rounds.json by scripts/generate_retention_figure.py, with SVG, PDF and PNG exports. The original diagrams remain in editable/ral_figures.xml and individual XML files.

| Paper | Current source | Purpose |
|---|---|---|
| Fig. 1 | abstract_overview.png | Author-supplied pipeline and SmolVLA graphical abstract |
| Fig. 2 | smolvla_residual_lora.xml | Retained expert, LoRA, 512 + 6 = 518d observation, 40d prior and residual |
| Fig. 3 | skill_retention.svg | Recovery-only comparison before and after balanced SIL |
| Fig. 4 | wx200_slots.xml | Apparatus and actual task images to insert later |
| Omitted | milestone_credit.xml | Former Fig. 3; explained in Methods C |
| Omitted | phase_feedback.xml | Former Fig. 4; replaced by LIBERO methods prose |

The six original XML diagrams remain editable. Their page names are asset identifiers, not current manuscript figure numbers. The pipeline_overview and skill_retention XML designs are earlier versions, superseded by the supplied PNG and SVG.

## Editing

Open an XML file in diagrams.net using File → Open from → Device. Blocks, chart bars, marks, labels and arrows are individual objects; no screenshots are embedded.

Use individual XML files as the source of truth for XML-based figures. Regenerate the retention chart with python3 scripts/generate_retention_figure.py; its values come from evidence/skill_retention_rounds.json. Manual SVG edits are retained by the general renderer but overwritten by this chart generator. After editing the combined file, save the edited page back into its individual XML before running the local renderer. Otherwise the combined file is regenerated from the unchanged individual sources.

Run python3 scripts/create_editable_figures.py from ral_submission/ to render existing XML, export the supplied retention SVG with rsvg-convert, and rebuild the combined file and CSV-based Table II. This preserves individual XML edits and the supplied SVG. The renderer supports the draft's basic shapes and styles; use diagrams.net's vector PDF export for added shape types/effects.

The --rebuild-layouts flag resets all initial layouts. Do not use it after manual design work unless you intend to reset them.

XML-based figures and the retention chart have PDF, SVG and PNG exports. Keep the PDF basename when replacing them. The graphical abstract is included directly from its PNG.

## Numerical integrity

Figure 3 uses only the recovery entries in evidence/skill_retention_rounds.json (relative to the paper folder). Bars are means of rounded reported rates; individual rounds are not plotted. Reaching has three rounds per condition and each placement skill has two. The reaching bank includes an earlier policy's experience. The previous two-panel figure is preserved in archive/manuscript_20261029_before_single_panel_fig3/figures/.

The omitted phase-feedback asset is a schematic reward sequence, not empirical telemetry. The omitted milestone-credit asset assumes informative milestone groups and a failed placement. Its plain-language labels show positive local prefix credit and a blocked negative completion bonus.

## WX200 insertion

Replace the layout with a complete vector panel at figures/wx200/panel.pdf. LaTeX inserts it automatically. Preserve approximately the 1000 × 287 aspect ratio. Add actual apparatus or verified CAD, five labeled joints, workspace/camera views, and reach/grasp/lift/transport/release frames with timestamps. Record instruction, trial ID and deployment condition.

The old CDPR photo slots and training curves remain in the revision archive.
