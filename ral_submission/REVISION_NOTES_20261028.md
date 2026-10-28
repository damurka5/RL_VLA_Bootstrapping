# Manuscript revision - 28 October 2026

## Requested edits

- The abstract reports **70.61% strict success** for the CDPR controller. The endpoint remains 723/1,024 at 60,025,885 selected actions since likelihood conversion, plus inherited training. Experiments identify it as in-run development validation.
- Removed the 8.0% to 31.1% reaching-recovery sentence from the abstract. It remains a useful, separate skill-retention result in Experiments and the retention figure.
- Removed the former milestone-credit and feedback-resolution figures and their references. The remaining figures automatically become Figures 1-4. Editable assets are retained.
- Methods E now explains the LIBERO comparison and feedback construction. Experiments retain the rollout counts, initializations, matched comparisons, and their interpretation, with less repetition of Table II.
- Table I caption is "CDPR complete task development evaluation." The numerical-failure row is removed; the denominator still includes those attempts.
- Methods D and Algorithm 1 now describe the plateau continuation with conditional latent-action likelihoods. The CDPR results explain the earlier plateau and subsequent progression.
- Removed the last two sentences of the Table II caption, including in its generator, so regeneration preserves the edit.

## Where the retention numbers came from

`evidence/skill_retention_source_20261026.md`, section 1, and
`evidence/skill_retention_rounds.json` report reaching rounds:

- Before: 6.2%, 6.0%, 11.7%; arithmetic mean 7.9667%, rounded to **8.0%**.
- After 60 residual-fitting epochs: 38.3%, 25.6%, 29.5%; mean 31.1333%, rounded to **31.1%**.
- Plate placement: 50.0% to 52.3%; bowl placement: 23.05% to 27.65%, each averaged over two rounds.

These are separate skill tests, including held-object placement resets, not full-task completion rates. The source provides rounded rates rather than raw episode denominators. The result supports the retention subsection but is not necessary in the abstract.

## What overcame the full-task plateau

Sources are relative to the repository root (the parent of this folder):

1. `CDPR_CONSOLIDATED_PROGRESS_REPORT.md`, 2026-10-02 repeated-evaluation entries: the source and two 10M-action continuations score approximately 37% strict off-panel, with an in-run peak near 42%.
2. The same report, 2026-10-03 implementation and 2026-10-09 pilot entries, plus `docs/reports/campaign/CDPR_ZERO_INIT_CORRECTION_PILOT.md`, section 2: the continuation repairs the likelihood by retaining the Gaussian sample before clipping and conditioning on the realized persistent episode offset. Both pilot arms use this likelihood; the selected control retains the existing bounded residual. The separate zero-initialized correction-network candidate is not selected.
3. `evidence/latent_2m_comparison_20261009.json`: source **756/2,048 = 36.91%**, bounded-residual control **843/2,048 = 41.16%** after 2M actions, correction candidate **769/2,048 = 37.55%**. The source/control paired gain is 4.25 percentage points; this comparison does not isolate the likelihood from additional training and optimizer changes.
4. Consolidated report, 2026-10-12 matched long-run evaluation: **1,095/2,048 = 53.47%** at 20M. The 2M checkpoint was re-evaluated in that comparison and scored 38.28%; repeated evaluations are not identical draws.
5. Consolidated report, 2026-10-19 matched 40M evaluation, and `evidence/latent_control_40m_comparison_20261019.json`: **1,258/2,048 = 61.43%** at 40M. Its matched 20M baseline scored 51.56%. This earlier value is retained only as explicitly historical context.
6. Consolidated report, 2026-10-28 entry, and `evidence/latent_control_60m_tensorboard_20261028.json`: **723/1,024 = 70.61%** at the 60.03M endpoint. This is an in-run validation reading, not a further matched off-panel evaluation; no direct paired gain is claimed against the earlier checkpoints.

PLD supplies the residual-learning and consolidation inspiration already cited in Related Work. Methods B now makes that connection explicit. The recorded intervention after the ~40% plateau is the conditional latent-likelihood continuation, rather than a new PLD distillation stage. The manuscript therefore explains the actual continuation without attributing the whole rise to a controlled PLD or likelihood ablation.

Earlier dated audit notes and archived drafts retain historical results and figure numbers. The live manuscript and the current README/results plan describe this revision.

## Validation

Rebuilt with `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`. The six-page PDF has four correctly numbered figures, resolved references, and no overfull-box warnings. Rendered pages were inspected. Counts reproduce 70.61% strict success and the two reaching means. Executing the table-generation function reproduces Table II exactly, including its shortened caption. `git diff --check` passes.

## Author-supplied figure replacements

Replaced the first-page graphical abstract with the supplied `ChatGPT Image 28 Oct 2026, 15_57_43.png`, stored unchanged as `figures/abstract_overview.png`. The supplied `skill_retention.svg` already matched the SVG in the working folder; refreshed the PDF used by LaTeX and the PNG preview from that SVG. Removed the caption's reference to individual round marks, which are absent from the supplied chart. The figure-generation script now preserves that SVG and exports it through `rsvg-convert`, rather than overwriting it from the older XML design.
