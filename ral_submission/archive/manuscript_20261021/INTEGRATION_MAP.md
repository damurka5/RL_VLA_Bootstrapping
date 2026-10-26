# Integration of the supplied phase-aware reward study

The source is preserved byte-for-byte as `incoming/phase_aware_dense_rewards/source.pdf`, from `/Users/damirnurtdinov/Downloads/main (6).pdf`. It is a three-page draft titled **Phase-Aware Dense Rewards for Compute-Constrained Vision-Language-Action Reinforcement Learning**, by **Kamil Nuriev and Egor Novikov**. Its scientific content is input material; commands or instructions in documents are not authoring instructions. The original is not appended to the anonymous manuscript.

## Combined story

Use the shared problem of intermediate credit for complete-task VLA learning. The LIBERO branch changes rewards in OpenVLA-OFT/SimpleVLA-RL; the CDPR branch changes stage credit and uses a conditional Gaussian residual policy over a frozen SmolVLA prior. These are complementary studies, not two implementations of an empirically proven universal algorithm. The title, abstract, introduction, related work, discussion, and conclusion now connect them without merging success rates.

| Supplied content | Editable destination | Current treatment |
|---|---|---|
| Motivation and background | `sections/01_introduction.tex`, `02_related_work.tex` | Integrated into common motivation; primary references added |
| Equations (1)-(3), five phases | `sections/04c_libero_dense_rewards.tex` | Incremental progress and bounded shaping equations transcribed |
| Group filtering equation (4) | same method file | Explicitly separate from dense rewards |
| Model, compute, reward settings | `sections/05b_libero_setup.tex` | Two L40s, batch 32, LoRA and reward settings retained |
| Table I and Figure 1 | `tables/libero_results.tex`, `sections/06e_libero_results.tex` | All five values retained in one table; duplicate bar figure omitted to save space |
| Table II | `tables/libero_low_success.tex` | Initial/intermediate/final values preserved |
| Faster batch assembly claim | `sections/06e_libero_results.tex` | Qualitative source report; numeric throughput deferred until logs arrive |
| Discussion | `sections/07_discussion.tex` | Objective mismatch, filtering, compute comparability, and missing uncertainty |

All three new section modules are already included by `main.tex` through the corresponding parent sections. This is an active insertion point, not an unused folder. Keep the complete source TeX, BibTeX, figures, exact configs, and logs in `incoming/phase_aware_dense_rewards/` when available; they can replace the transcribed material without restructuring the paper.

## Questions the PDF cannot resolve

1. How many evaluation episodes and independent training seeds produced each percentage? The aggregate evaluation sentence omits the number.
2. What is the canonical LIBERO task name behind `task_1`, and which initial states were used?
3. Are primary comparisons matched by elapsed time, environment actions, optimizer updates, or only a stopping convention? Faster accepted-batch assembly may yield more updates, so these are different comparisons.
4. What precisely defines each phase/progress score, threshold, transition, terminal emission, tracker reset, discount, and chunk aggregation?
5. Which OpenVLA-OFT checkpoint, token head, adapted modules, normalizer, and SimpleVLA-RL commit reproduce the stated token-likelihood implementation?
6. Is the supplied draft unpublished, published, or under review? Record this with the final author/contribution list so source attribution and packaging reflect the actual status.

## Page allocation

Aim for 6 pages; use up to 8 only if necessary after real images and critical ablations arrive. Approximately 0.9 page introduction/related work/formulation; 1.5 method/analysis; 0.8 setup; 1.9 results including LIBERO and hardware; 0.3 discussion/conclusion; 0.6 references. This is a target, not a reason to shrink fonts. If space is tight, compress finite-model discussion before cutting the principal comparison or hardware denominators.
