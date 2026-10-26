# RA-L working manuscript

Updated 22 October 2026. The author's request to start writing supersedes the old outline-only stop instruction. The original outline is preserved under `archive/outline_20261011/`; the first prose draft is preserved under `archive/manuscript_20261021/`.

**Working title:** Bootstrapping Vision-Language-Action Models for New Robot Embodiments with Staged Reinforcement Learning.

The current working draft is eight pages, including references and reserved photo space. The paper is centered on adaptation of VLA-based control to new embodiments: dense-reward skill learning, transition to binary rewards and self-imitation, then staged RL for the complete task. The CDPR is an unconventional embodiment test case. The separate OpenVLA-OFT/LIBERO study tests stage-aware feedback on an already trained manipulator. It does not share CDPR scenes or reproduce the complete adaptation curriculum. The prior two-branch overview figure has been removed. Two data-backed simulation plots and hardware photo slots remain. It remains a working manuscript: missing evidence is visible, and hardware results are not represented as verified.

## Start here

- `main.pdf`: compiled reading copy; `main.tex`: manuscript entry point.
- `REVISION_NOTES_20261022.md`: response to the six author notes and the resolved action-module initialization question.
- `evidence/ACTION_MODULE_AUDIT_20261022.md`: code/report evidence for pretrained SmolVLA retention, fresh residual initialization, LoRA, and continuation weights.
- `DATA_NEEDED.md`: prioritized tables, graphs, raw data, and exact fields needed to finish.
- `INTEGRATION_MAP.md`: where the supplied PDF enters the combined paper and how to replace/extend it.
- `data_templates/`: empty CSV intake schemas and a run-manifest template; no fabricated sample outcomes.
- `evidence/SOURCES.md`: provenance and boundaries for every result group.
- `tables/results_inventory.csv`: checkpoint counts/hashes and evidence status.
- `scripts/generate_figures.py`: reproducible plots from copied source logs.

## Current evidence

The 19 October matched CDPR report supersedes the old unverified >50% placeholder: final40m achieves **1258/2048 = 61.43%**, versus **1056/2048 = 51.56%** for control20m, on 512 development scenes with four repeats. The paired increase is **9.86 percentage points [7.37, 12.30]**. This is development evidence from one continuing training lineage, not a locked final test or a likelihood-only ablation. The older 2M comparison remains supporting context.

The supplied LIBERO PDF reports 65.3% versus 61.7% on the aggregate benchmark and 18% versus 11% in its low-success task. Its percentages are transcribed with source status. Trial counts, raw logs, seeds, uncertainty, and exact compute matching are missing.

## Hardware writing convention

`\assumed{...}` visibly labels text written as if the requested successful experiment has completed. The setup, results, abstract, and conclusion use this convention. `\pending{...}` marks unknown measurements or implementation details. No trial count, success rate, baseline advantage, zero-shot claim, or hardware photograph has been fabricated.

Add real images as `figures/hardware/setup.jpg` and `figures/hardware/sequence.jpg`; these automatically replace the boxes on recompilation. Follow `figures/README.md` for panel contents. Confirm the actual protocol, replace pending values from the complete trial log, and rewrite/remove the assumption labels only once supported. A single successful video establishes an example, not a reliability percentage.

## Build

From `ral_submission/`:

```sh
python3 scripts/generate_figures.py   # requires matplotlib; plots already included
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

Without latexmk, run pdflatex once, `bibtex build/main`, then pdflatex twice (each pdflatex invocation uses `-output-directory=build main.tex`). The checked-in PDF figures allow TeX compilation without Python. The existing multifile project can also be opened in Overleaf with `main.tex` selected.

## Length and submission packaging

RA-L is a journal with conference-presentation options. Its current guidance allows six pages including references, with at most two charged additional pages, and uses double-anonymous review: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/ (checked 21 October 2026).

Keep central evidence in the paper; an additional text appendix outside the eight-page limit is not a substitute. The current integration uses a compact complementary LIBERO study instead of appending the original three-page article verbatim. Preserve its contributors in the internal source record and settle the combined author list before final packaging; the review manuscript contains no identifying byline.

`incoming/`, `archive/`, `evidence/`, local paths, internal notes, and original templates are authoring materials. They are not an anonymous submission bundle. Do not submit this working draft while assumption labels or missing evidence remain.
