# Joint-paper method and experiment map

Updated 26 October 2026. CDPR adaptation and LIBERO reward construction are experiments within one paper. The manuscript uses our method and our experiments, without presenting the LIBERO material as a separately authored article.

## Common argument

The central contribution remains dense skill acquisition -> binary refinement and self-imitation -> staged full-task composition. Ordered task progress supplies intermediate feedback at appropriate points in this sequence. We instantiate it with phase-aware dense rewards for LIBERO reward experiments and discrete milestones with stage-specific credit for CDPR composition.

Our LIBERO experiments examine reward construction with a trained OpenVLA-OFT policy. Our CDPR experiments instantiate embodiment adaptation and complete-task learning with a residual over SmolVLA. They answer different experimental questions within the same paper. The complete schedule was not run on LIBERO, and CDPR full-task training does not use the LIBERO dense reward equation.

The reward definition is now part of the main method section, alongside milestone composition and stage credit. The introduction, setup, results, discussion and conclusion use this shared structure. The original three-page article is not appended.

## Active LaTeX inclusion map

| Content | File | Inclusion and role |
|---|---|---|
| Training sequence, task composition and stage credit | `sections/04_method.tex` | Main method section, included directly by `main.tex` |
| Phase-aware progress, transition reward and group filtering | `sections/04c_libero_dense_rewards.tex` | Included once by `04_method.tex` as a method subsection |
| LIBERO model, compute, configurations and evaluation | `sections/05b_libero_setup.tex` | Included by `05_experimental_setup.tex` |
| Aggregate reward comparison and low-success learning | `sections/06e_libero_results.tex` | Included by `06_results.tex` |
| LIBERO numerical outcomes | `tables/libero_results.tex`, `tables/libero_low_success.tex` | Included by the LIBERO results section |
| Shared interpretation and unresolved attribution | `sections/07_discussion.tex` | Distinguishes feedback granularity, schedule and embodiment evidence |

The original attachment remains byte-for-byte preserved at `incoming/phase_aware_dense_rewards/source.pdf`. Internal provenance and contributor identities remain in `evidence/SOURCES.md`. Original materials, archives and internal notes are not manuscript content. Joint-paper wording does not change the evidence verification status.

## Claim boundaries

- Phase-aware dense feedback yields a larger observed LIBERO endpoint than terminal-only under the recorded settings. Counts, seeds, uncertainty and compute matching must still be recovered.
- The primary-dense versus phase-only rows change multiple coefficients. Their difference does not isolate within-phase progress.
- CDPR continuation demonstrates development improvement within a training lineage, not a controlled staged-versus-terminal effect.
- The results support a shared intermediate-feedback rationale. They do not establish the superiority of every step in the schedule or its generalization to additional embodiments.

## Data collection

Use [COLLABORATOR_DATA_REQUEST.md](COLLABORATOR_DATA_REQUEST.md) for a prioritized request and a message ready to send. Recover exact code/configuration, raw evaluation outcomes, training/budget logs, model likelihood details and editable sources first. A matched phase-only run with the primary dense terminal/phase coefficients is the most useful optional LIBERO extension.

`REWARD_ALIGNMENT_20261026.md` explains the mechanism and claim-to-evidence reasoning. `DATA_NEEDED.md` also lists CDPR lineage, schedule ablations and hardware requirements. Publication status and contributor roles belong in internal project records and the final combined author list.

## Layout

Retain the existing IEEE layout and verify the compiled reading copy and figures after edits. Preserve the previous draft at `archive/manuscript_20261026_before_joint_paper_merge/`. Compress repetition before changing formatting or omitting principal evidence. The earlier removed overview figure has not been reintroduced.
