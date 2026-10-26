# RA-L submission outline - awaiting author review

**Working title:** Bootstrapping Vision-Language-Action Control for Cable-Driven Manipulation with Staged Reinforcement Learning

Created 11 October 2026. This is a research plan and bullet-point manuscript skeleton, not final prose or a submission-ready paper. Stop at this outline until the author approves the direction. No experiments or submissions are launched by this folder.

## Mathematical grounding added for review

Start with [the mathematical foundation](theory/MATHEMATICAL_FOUNDATION.md). It contains proofs, explicit assumptions, a code correspondence, and the measurements required to connect theory to the neural controller. The new `sections/04b_mathematical_analysis.tex` keeps the manuscript in outline form; `sections/06d_theory_validation.tex` distinguishes finite-model checks from pending robot evidence. Reproduce the finite-model checks with `python3 theory/verify_mathematics.py`; results are in `theory/verification.json`.

The main theoretical benchmark proves nondecreasing success for the achieved-stage protection rule in a separable sequential-gate model. Applying it to the real shared network requires studying interference and the actual update direction. Correct latent scoring and mixed-group probability explain two further mechanisms. No generic theorem is represented as a certificate for the unpublished >50% checkpoint.

## Proposed paper story

Adapt SmolVLA-based control to a cable-driven robot through skill bootstrapping and staged full-task residual RL. Explain the corrected latent-action likelihood, and test which parts account for the gains. Make the latest >50% run the intended headline once its exact evaluation is recorded; include the ongoing physical experiment in the main Results section.

The latest inspected local commit is `2b15660361d2b612048f85652dcf521b07ec57e6` (9 October 2026): it promotes the bounded-residual latent-likelihood control. The separate zero-initialized correction network did not pass promotion. The author confirms that the new >50% checkpoint is not in the reports and relates to this latest direction; its exact architecture and metric still need provenance. Do not invent its run name or assume the suggested 5M continuation is the actual run.

The strongest defensible framing is an experimentally tested adaptation method on CDPR. Neither residual RL nor self-generated SFT nor scoring latent actions should be labeled novel in isolation. The related-work section reserves a direct comparison with PLD. If the prior-control ablations show no benefit from the VLA action prior, revise the framing around the observed role of pretrained features.

Alternative shorter title for review: **Staged Residual Reinforcement Learning for Cable-Driven Manipulation**.

## Files and reading order

- `main.tex`: article entry point; IEEEtran two-column letterpaper journal format.
- `sections/00_abstract.tex`: bullet-point abstract.
- `sections/01_introduction.tex`: motivation, research question, proposed contributions.
- `sections/02_related_work.tex`: comparison axes and literature gaps.
- `sections/03_problem_formulation.tex`: robot/task contract and success definitions.
- `sections/04_method.tex`: residual actor, initialization, protected staged credit, latent likelihood.
- `sections/04b_mathematical_analysis.tex`: propositions, proof sketches, assumptions and update-alignment condition.
- `sections/05_experimental_setup.tex`: simulation comparisons and pending physical protocol.
- `sections/06_results.tex`: includes separate simulation, ablation/failure, and real-world result files (`06a` through `06c`), plus mathematical mechanism checks in `06d`.
- `sections/07_discussion.tex`: interpretation and limitations.
- `sections/08_conclusion.tex`: claims to fill only after evidence is complete.
- `references.bib`: seven verified seed references; intentionally not a completed literature review.
- `RESULTS_PLAN.md`: prioritized experiments, tables, and evidence requirements.
- `evidence/SOURCES.md`: provenance, caveats, and source map.
- `tables/results_inventory.csv`: recorded counts and blank pending rows.
- `template_original/`: unchanged files from the supplied archive; `IEEEtran.cls` is also copied beside `main.tex`.
- `main.pdf`: compiled review copy; `build/` holds disposable compilation/rendering files.

RA-L uses article sections, so each requested chapter is a separate `.tex` file included with `\input`, without forced chapter page breaks.

## Result labels

- **Recorded:** backed by retained local artifacts, with its evaluation scope stated.
- **Author-reported:** latest >50% result; exact score, checkpoint, and strict/native/in-run/standalone definition pending.
- **Proposed:** experiment or analysis to run, not a completed finding.
- **Pending:** no result yet; in particular every physical-robot result.

The older 41.16% result is supporting development evidence, not a replacement for the author's newer result. No final-test score is available in the reviewed evidence. A point estimate above 50% does not itself establish a confidence interval above 50%.

## Proposed final page budget

| Content | Pages |
|---|---:|
| Title, abstract, introduction | 0.7 |
| Related work and formulation | 0.5 |
| Method | 0.8 |
| Mathematical analysis with short proofs | 1.2 |
| Experimental setup | 0.5 |
| Mechanism tests, simulation and ablations | 1.1 |
| Physical results | 0.5 |
| Discussion and conclusion | 0.2 |
| References | 0.5 |
| Total, including figures and tables in their sections | 6.0 |

This is a writing budget, not a requirement that the review bullets have identical pagination. Combine planned figures/tables where needed instead of shrinking fonts. RA-L's official guidance specifies six pages including figures/references, with up to two charged extra pages, and double-anonymous review: [information for authors](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/), checked 11 October 2026. The outline omits identifying author metadata and the sample's publication dates/copyright banner. The internal evidence files and original template are working materials, not files to send to reviewers.

## Build

From this folder, with a TeX distribution including IEEEtran bibliography style:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf main.pdf
```

The project uses multiple source files; upload the folder to Overleaf and select `main.tex`, or compile locally. The original template is retained for reference, not imported as scientific content.

## Decisions for the author's review

1. Approve the title and the staged-RL/latent-likelihood emphasis.
2. Approve the proposed propositions and their assumptions, the mechanism tests, and the core matched comparisons including the VLA-prior ablation.
3. Confirm the physical apparatus, transfer/adaptation condition, baseline, and feasible trial count. The proposed 80 trials per policy is a draft design, not an assertion about completed work.
4. Supply the latest checkpoint/protocol/counts when available; leave their placeholders until then.
5. After approval, expand bullets into prose, complete literature coverage, and generate final figures/tables from evidence.
