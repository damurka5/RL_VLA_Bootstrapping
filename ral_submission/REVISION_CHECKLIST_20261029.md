# Revision checklist after critical review

29 October 2026. This checklist incorporates the author's clarification that the **new residual controller is initialized from scratch while SmolVLA's pretrained action expert is retained**. It is a revision plan, not evidence that any additional experiment has been completed.

The intended central contribution is adaptation to a target robot without collecting human demonstrations for that embodiment, beginning with RL and subsequently consolidating successful experience. Whether the selected result satisfies that exact data contract must be established from its training ancestry.

1. **Priority 0 — Fix the architecture claim everywhere.**

   - [ ] Describe a newly initialized embodiment-specific residual controller over a retained pretrained SmolVLA expert. Do not describe the evaluated implementation as reinitializing the native action expert or training the whole VLA from scratch.
   - [ ] Add a compact module table: pretrained source, initialization, trainable stages, frozen stages, and parameter count for the VLM, action expert, LoRA, residual, and exploration parameters.
   - [ ] Make the title, abstract, Methods, Algorithm 1, and Figures 1–2 use consistent terminology. In Figure 1, qualify “from scratch” as applying to the residual controller.
   - [ ] Record the actual initial checkpoint, initial residual weights or initialization seed, selected checkpoint, resolved configuration, and code revision.

   Targets: `sections/00_abstract.tex`, `sections/01_introduction.tex`, `sections/04_method.tex`, `algorithms/staged_pipeline_paper.tex`, and the graphical abstract. Evidence: `evidence/ACTION_MODULE_AUDIT_20261022.md` and `../rl_vla_bootstrapping/policy/smolvla_cdpr.py:674`.

2. **Priority 0 — Establish exactly what “without a manually generated dataset” means.**

   - [ ] Trace the selected checkpoint from its initial target-robot training, including every inherited SFT checkpoint and data bank.
   - [ ] Report human teleoperation demonstrations, scripted-oracle trajectories, learned-policy trajectories, assisted starts, and human interventions separately. Distinguish trajectories used as supervised action targets from trajectories used only to create starting states.
   - [ ] Confirm the human demonstration count before writing “zero human demonstrations.” Describe pretrained robot data separately from new target-embodiment data.
   - [ ] Investigate whether the oracle-generated composition seed and assisted stages documented in `../CDPR_CONSOLIDATED_PROGRESS_REPORT.md` are ancestors of the reported result. The report includes a Phase 6 oracle-SFT seed; the manuscript already acknowledges inherited assistance.
   - [ ] If automated oracle data were used, describe their role. Such data can be compatible with zero human teleoperation, but they do not support “only the learner's own RL trajectories” or “no demonstration dataset of any kind.”

   Deliverable: one lineage table with checkpoint, parent, training objective, data origin, assistance, and interaction budget. This is the central evidence for the revised contribution, not an optional implementation detail.

3. **Priority 0 — Rewrite the problem and contribution around the data requirement.**

   - [ ] Put the cost of collecting target-embodiment demonstrations in the first motivation paragraph.
   - [ ] Define the alternative as RL being the first target-embodiment adaptation step, followed by successful-experience consolidation and complete-task learning.
   - [ ] Distinguish human data collection from simulator development, reward design, reset design, oracle engineering, and RL compute. Eliminating manual demonstration collection does not automatically establish lower total time or cost.
   - [ ] Present the pipeline as a method and the CDPR result as its empirical instantiation. Avoid claims that any robot can be adapted or that all data collection is unnecessary.

   Candidate wording, conditional on the lineage audit: “We investigate adaptation of a pretrained VLA to a target robot without collecting human demonstrations for that embodiment. A newly initialized residual controller first acquires component skills through reinforcement learning, after which successful-experience consolidation and milestone-based training produce a policy for complete manipulation instructions.” Add any automated teacher-data requirement explicitly.

4. **Priority 1 — Position the advance relative to earlier work, including the earlier CDPR preprint.**

   - [ ] Cite and distinguish the August 2026 [RL Bootstrapping of OpenVLA-OFT for a Novel Robot Embodiment](https://arxiv.org/abs/2608.01013), authored by Damir Nurtdinov, Alexei Kornaev, and Alexander Maloletov. Its abstract already presents zero-demo CDPR adaptation using RL and reports motion/approach tasks. The journal contribution must explain the additional advance to complete manipulation, consolidation, and the evaluated controller.
   - [ ] Separate the historical OpenVLA/CDPR work from the current OpenVLA-OFT/LIBERO reward experiment. They have different evidential roles.
   - [ ] Compare close methods by target human demonstrations, synthetic/teacher data, initial target-task competence, trainable action module, embodiment change, and complete-task evaluation. Do not infer these attributes from the label “RL.”
   - [ ] Include [iRe-VLA](https://arxiv.org/abs/2501.16664), [PLD](https://arxiv.org/abs/2511.00091), [SimpleVLA-RL](https://arxiv.org/abs/2509.09674), and relevant synthetic adaptation such as [ET-VLA](https://arxiv.org/abs/2511.01224). ET-VLA describes synthetic continued pretraining followed by target-embodiment fine-tuning; verify each data source before assigning it a zero-demo label.
   - [ ] Avoid a “first” claim unless a focused prior-art review supports the precisely defined setting.

5. **Priority 1 — Define the unseen-embodiment setting precisely.**

   - [ ] Identify the CDPR mechanism, action semantics, control dimensions, control frequency, gripper, observations, and differences from documented pretraining robots.
   - [ ] Support “absent from the documented robotic pretraining data” using the pretrained model's data documentation. Avoid the broader, unverifiable statement “absolutely unseen” across all visual/language pretraining.
   - [ ] Explain which low-level controller translates the policy's Cartesian commands into CDPR actuation. Make the learned action-interface contribution clear.
   - [ ] Measure the initial unadapted system under the same full-task evaluator; do not infer zero competence merely from the new embodiment or randomly initialized residual.

6. **Priority 1 — Clarify the LIBERO experiment and its budget.**

   - [ ] State that terminal-only SimpleVLA-RL is your own implementation/run under the experiment's budget, not a success rate copied from the original publication.
   - [ ] Keep initial and final performance separate. The current CSV records a shared **61.0% initial policy**, terminal-only **61.7% after 160 updates**, and dense-plus-phase **65.3% after 160 updates**. Check against the original logs if the intended meaning of “61%” differs; correct all affected text and tables consistently.
   - [ ] Give the initial checkpoint and its training/data provenance, shared settings, independent seeds, task-level counts, rollout counts, and uncertainty.
   - [ ] Report environment interactions and preferably GPU-hours alongside updates. Dynamic sampling can make equal updates consume different amounts of interaction.
   - [ ] Treat the phase-only result as a separate configuration until terminal coefficient, phase weight, filtering, and budget are matched.
   - [ ] Explain that LIBERO is an auxiliary feedback study. SmolVLA/CDPR evaluates the full schedule. LIBERO does not demonstrate the full pipeline across two backbones or isolate the protected CDPR credit rule.
   - [ ] Motivate binary full-task milestones by their completion-predicate interface and reduced need for continuous progress functions. Do not claim that the existing dense-plus-phase result proves binary milestones are superior.

   Targets: `sections/01_introduction.tex`, `sections/05b_libero_setup.tex`, `sections/06e_libero_results.tex`, and `tables/libero_results.tex`.

7. **Priority 1 — Add a contextual literature comparison, with explicit metric differences.**

   - [ ] Collect related pick-and-place results and target-robot demonstration counts from primary papers and their supplements. Prefer exact plate/bowl tasks where available, but also retain relevant embodiment-adaptation studies rather than selecting only convenient task names.
   - [ ] Use columns for model, robot, task, simulation/hardware, human target demonstrations, other training data, reported metric, trial count, and evaluation conditions. Keep pretrained data separate from adaptation data; distinguish per-task from total multitask demonstrations.
   - [ ] Label the table as contextual evidence across different experimental settings. Do not rank it as a controlled benchmark or use it to claim superior performance or sample efficiency.
   - [ ] Check whether “success rate” includes partial credit. For example, the [SmolVLA paper](https://arxiv.org/html/2506.01844v1) describes 50 demonstrations per real-world task dataset and assigns half credit for pickup and half for placement. These scores are not directly interchangeable with CDPR strict binary completion. Its SO100 evaluation also includes multitask training, so the full adaptation-data total matters.

   Similar instruction wording does not equal the same task distribution. Different objects, containers, resets, horizons, robots, and success criteria prevent attributing cross-paper score differences to the training procedure.

8. **Priority 1 — Build the skill-by-stage matrix from the consolidated report.**

   - [ ] Start with the existing “Instruction success by phase and retention cycle” table in `../CDPR_CONSOLIDATED_PROGRESS_REPORT.md` around line 307.
   - [ ] Use rows for actual checkpoints/stages and columns for reaching, pickup, held-object plate placement, held-object bowl placement, and empty-gripper full-task plate/bowl completion.
   - [ ] Include successes/attempts, evaluation rounds, protocol identifiers, and uncertainty where reconstructable. Mark unmeasured cells as unavailable, not zero.
   - [ ] Separate dedicated skill references, bank-harvest evaluations, mixed-start validations, and strict full-task evaluations. Do not connect incompatible rows into a single learning curve or treat references from other branches as consecutive ancestors.
   - [ ] Distinguish the existence of all behaviors from evidence that replay balancing improves later composition. The matrix documents development and retention; an appropriate control is needed for causal attribution.
   - [ ] Replace the claim that each isolated skill must exceed a positive threshold before even one full-task success is possible. For nested full-task events, success is a product of conditional probabilities; low nonzero probabilities permit occasional complete successes. Isolated skill tests additionally use different prompts and state distributions. Treat thresholds as practical transition criteria and test their predictive value if they are a central claim.

9. **Priority 1 — Obtain at least one controlled comparison on the target setting.**

   - [ ] Prioritize the full schedule versus direct full-task RL with the same pretrained initialization, newly initialized residual architecture, available information, evaluation protocol, and total interaction budget. A direct milestone-reward baseline is informative because it tests whether skill acquisition/consolidation adds value beyond intermediate feedback.
   - [ ] Count all acquisition, consolidation-data generation, and composition interactions for the proposed schedule. Report differences in assistance or teacher data; do not silently grant them to only one arm.
   - [ ] If protected milestone credit is presented as an independent algorithmic contribution, compare it with unprotected credit from the same consolidated checkpoint.
   - [ ] If instruction balancing is presented as an independently effective mechanism, compare it against ordinary successful replay using the same data and optimization budget.

   These are priorities, not a demand to run every combination. A narrower feasibility claim needs strong provenance and independent evaluation; claims that particular components improve learning need corresponding controls. Literature scores from other robots cannot replace these comparisons.

10. **Priority 1 — Strengthen the final evaluation.**

    - [ ] Freeze checkpoint selection and evaluate on untouched scenes, with explicit object/destination distributions and all failures in the denominator.
    - [ ] Report counts, uncertainty, and the number of distinct scenes versus stochastic repetitions. Use paired or scene-aware analysis where applicable.
    - [ ] Repeat central training comparisons across independent seeds where feasible; evaluation repetitions are not independent training runs.
    - [ ] Preserve “development validation” for the current 70.61% endpoint until a separate evaluation is available. Archive the selected model, resolved configuration, and per-episode outcomes.
    - [ ] Include same-scene instruction changes if claiming that the resulting controller uses language to select objects and destinations.

11. **Priority 2 — Make the method and resource requirements reproducible.**

    - [ ] Define stage predicates, assignment of actions to stages, group advantages, terminal bonuses, protection masks, and the actual clipped surrogate.
    - [ ] Specify acquisition rewards, skill reset distributions, transition thresholds/patience, actual manual decisions, replay quotas, and consolidation acceptance criteria.
    - [ ] Report LoRA targets/rank, chunk length/executed prefix, learning rates, group size, rollout filtering, normalization, exploration, and stopping budgets.
    - [ ] Report the total adaptation interactions and compute, including inherited stages. Explain simulator-state access for rewards and any training-only assistance separately from policy observations at evaluation.
    - [ ] Quantify human collection effort avoided only to the extent measured. Describe remaining simulator, predicate, reset, and teacher-engineering work; avoid a measured total-time-saving claim without a cost comparison.

12. **Priority 2 — Complete hardware evidence and revise the presentation.**

    - [ ] Finish WX200 trials with failures, interventions, resets, training-data sources, and the exact adaptation/deployment setting recorded.
    - [ ] Say whether this tests the same schedule on another robot or transfers an existing controller; these are different claims.
    - [ ] Replace placeholders with actual apparatus/rollout images and a quantitative table. Keep simulation and hardware results distinct.
    - [ ] Add uncertainty or individual points to the retention figure and prioritize the data-lineage/module table and controlled results over repeated schedule descriptions.
    - [ ] Rewrite the abstract and conclusion after the above evidence is fixed. State the demonstrated scope and retain only claims supported by the selected experiment.

Recommended execution order: resolve data ancestry; fix the central claim and architecture wording; extract the matrix and LIBERO protocol; define the matched comparison; archive and independently evaluate the final controller; finish hardware evidence; then polish the narrative and figures.
