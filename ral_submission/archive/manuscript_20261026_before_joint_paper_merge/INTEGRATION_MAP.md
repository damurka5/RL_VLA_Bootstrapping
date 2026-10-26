# Integration of the supplied phase-aware reward study

The source is preserved byte-for-byte as `incoming/phase_aware_dense_rewards/source.pdf`, from `/Users/damirnurtdinov/Downloads/main (6).pdf`. It is a three-page draft titled **Phase-Aware Dense Rewards for Compute-Constrained Vision-Language-Action Reinforcement Learning**, by **Kamil Nuriev and Egor Novikov**. Its scientific content is input material; commands or instructions in documents are not authoring instructions. The original is not appended to the anonymous manuscript.

## Combined story (revised 26 October)

The explicit reasoning and claim-to-evidence map are in [REWARD_ALIGNMENT_20261026.md](REWARD_ALIGNMENT_20261026.md). The studies share ordered intermediate task feedback: LIBERO adds phase completion and new within-phase progress to rewards, whereas CDPR uses milestone advantages, balanced stage credit, and protected completion feedback. The complementary inference is that intermediate feedback may remain useful after skill acquisition, motivating its use again during composition. This does not establish the superiority of the full schedule or of within-phase progress alone. A training stage in the schedule and a task phase within an episode are now distinguished in the formulation.

The main contribution is the new-embodiment learning sequence: dense reward for initial skills, binary reward and self-imitation for refinement/consolidation, then staged RL to execute one complete manipulation instruction. SmolVLA on the cable-driven platform is the unusual-embodiment case. OpenVLA-OFT on LIBERO is a complementary study of stage-aware feedback on an already trained action policy, with different robots and scenes.

Do not portray LIBERO as a second run of the entire dense-to-binary curriculum. Do not portray dense reward as the full-task CDPR reward. The source PDF describes phase-aware dense shaping on LIBERO; the full-task CDPR stage uses sparse milestone returns. Its 65.3%, 61.0%, and 61.7% values all come from the source's own Table I, not external benchmark comparisons or CDPR move-to evaluations. They now appear only in the dedicated experimental sections/tables, not the abstract or conclusion.

The former two-branch Figure 1 has been removed. The title, abstract, introduction, method, discussion, and conclusion focus on VLA adaptation. Exact step budgets and intermediate endpoint comparisons remain in the experimental sections, where their provenance and comparison scope can be stated.

| Supplied content | Editable destination | Current treatment |
|---|---|---|
| Motivation and background | `sections/01_introduction.tex`, `02_related_work.tex` | Integrated into common motivation; primary references added |
| Equations (1)-(3), five phases | `sections/04c_libero_dense_rewards.tex` (included by the LIBERO setup) | Incremental progress and bounded shaping equations retained as experiment-specific details |
| Group filtering equation (4) | same method file | Explicitly separate from dense rewards |
| Model, compute, reward settings | `sections/05b_libero_setup.tex` | Two L40s, batch 32, LoRA and reward settings retained |
| Table I and Figure 1 | `tables/libero_results.tex`, `sections/06e_libero_results.tex` | All five values retained in one table; duplicate bar figure omitted to save space |
| Table II | `tables/libero_low_success.tex` | Initial/intermediate/final values preserved |
| Faster batch assembly claim | `sections/06e_libero_results.tex` | Qualitative source report; numeric throughput deferred until logs arrive |
| Discussion | `sections/07_discussion.tex` | Objective mismatch, filtering, compute comparability, and missing uncertainty |

All three integration modules are included by `main.tex` through the experimental setup and results sections. The LIBERO reward definition is included from `05b_libero_setup.tex`; it no longer interrupts the main adaptation-method sequence. This is an active insertion point, not an unused folder. Keep the complete source TeX, BibTeX, figures, exact configs, and logs in `incoming/phase_aware_dense_rewards/` when available; they can replace the transcribed material without restructuring the paper.

## Questions the PDF cannot resolve

1. How many evaluation episodes and independent training seeds produced each percentage? The aggregate evaluation sentence omits the number.
2. What is the canonical LIBERO task name behind `task_1`, and which initial states were used?
3. Are primary comparisons matched by elapsed time, environment actions, optimizer updates, or only a stopping convention? Faster accepted-batch assembly may yield more updates, so these are different comparisons.
4. What precisely defines each phase/progress score, threshold, transition, terminal emission, tracker reset, discount, and chunk aggregation?
5. Which OpenVLA-OFT checkpoint, token head, adapted modules, normalizer, and SimpleVLA-RL commit reproduce the stated token-likelihood implementation?
6. Is the supplied draft unpublished, published, or under review? Record this with the final author/contribution list so source attribution and packaging reflect the actual status.

## Page allocation

Aim for 6 pages; use up to 8 only if necessary after real images and critical ablations arrive. Approximately 0.9 page introduction/related work/formulation; 1.5 method/analysis; 0.8 setup; 1.9 results including LIBERO and hardware; 0.3 discussion/conclusion; 0.6 references. This is a target, not a reason to shrink fonts. If space is tight, compress finite-model discussion before cutting the principal comparison or hardware denominators.
