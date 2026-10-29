# Dense, phase and terminal reward — code audit, 29 October 2026

Repository: https://github.com/StarlightProg/SimpleVLA-RL-DenseReward

Remote HEAD checked directly: `44ed954482e1a23795c0c65f2fcc52496290e0d4`. The sibling local checkout is older (`5755c3ba5940d6e7574b4a99391274fb78906438`). This audit uses the pinned remote source files. Copies, hashes and the repository tree are in `../research/dense_reward_code_20261029/`; downloaded repository code was read, not executed.

## Reward construction

- `verl/utils/subgoal_reward/dense_reward.py`: weighted within-phase progress increment plus weighted completed-phase count, with optional action-change penalty. Their sum is clipped to the configured shaping range. An optional internal terminal-success component is then added outside this clip.
- `tracker.py`: with best-progress tracking enabled, positive progress is the improvement over the maximum already reached in the active phase; the baseline resets when the phase advances. The dense increment and completion event are distinct signals. Final-phase completion handling is not generally a once-per-episode bonus: the last phase can remain done on subsequent steps before environment success. No stronger anti-farming guarantee is claimed here.
- `phases.py`: reach uses gripper–object distance; grasp combines proximity and gripper closure/grasp state; lift uses height change from the initial object height; transport uses object–target XY distance; placement combines proximity, release and native success. These require geometric scales/thresholds and reliable state extraction.
- `task_specs.py`: constructs a five-phase pick-and-place sequence using task text and available object/target positions. It does not define arbitrary long-horizon instructions. Unsupported tasks can fall back to terminal-only behavior through `engine.py`.
- `verl/trainer/main_ppo.py`: dense module outputs are assigned to action-chunk token positions. `add` combines them with the external verifier reward, `replace` substitutes them, and `log_only` preserves the original reward. Native task success is used separately for validation and group filtering.

Pinned source links:

- https://github.com/StarlightProg/SimpleVLA-RL-DenseReward/blob/44ed954482e1a23795c0c65f2fcc52496290e0d4/verl/utils/subgoal_reward/dense_reward.py
- https://github.com/StarlightProg/SimpleVLA-RL-DenseReward/blob/44ed954482e1a23795c0c65f2fcc52496290e0d4/verl/utils/subgoal_reward/phases.py
- https://github.com/StarlightProg/SimpleVLA-RL-DenseReward/blob/44ed954482e1a23795c0c65f2fcc52496290e0d4/verl/utils/subgoal_reward/task_specs.py
- https://github.com/StarlightProg/SimpleVLA-RL-DenseReward/blob/44ed954482e1a23795c0c65f2fcc52496290e0d4/verl/trainer/main_ppo.py

## Dense versus phase-only

The combined dense-and-phase condition has both within-phase progress and completion bonuses. Phase-only shaping sets the progress weight to zero; it still retains phase-completion feedback and a terminal objective. It must not be described as a comparison between a stage-unaware dense reward and an entirely different staged method.

The author-corrected CSV gives progress/phase weights 0.05/0.05 for the combined configurations and 0/0.20 for phase-only. At terminal coefficient 1 and 110 updates the reported endpoints are 61.0% and 62.3%, respectively. Since phase weights differ, this is a configuration comparison rather than an isolated removal of continuous progress. The 65.3% endpoint also uses a different external terminal coefficient from the phase-only condition.

## Why staged composition is chosen

The design rationale follows from the implementation requirements: extending dense shaping beyond the supplied pick-and-place sequence requires identifying relevant entities, defining continuous progress for new actions, and selecting scales and state conditions. A binary milestone interface retains intermediate feedback while requiring completion predicates rather than a continuous progress function throughout each phase.

This motivates the author's choice of binary milestones for full-task composition and dense feedback for acquiring the individual skills. It does not prove universal performance superiority, eliminate task-specific predicate engineering, or establish the full CDPR credit rule from the LIBERO comparison. The CDPR optimizer also assigns stage-specific credit rather than only summing phase bonuses into one scalar return.

## Role of the 2% task

The low-success task in the author-corrected CSV compares terminal-only feedback (11%) and dense feedback (18%) after 20 updates, starting from 2%. Both use the same terminal-outcome group filter. There is no phase-only result for this task in the supplied data or the inspected repository reports. Its evidential role is initial dense-feedback acquisition, not selection of the final composition reward. The manuscript and table now identify that role explicitly.

## Provenance and configuration limits

The manuscript's numerical source remains the author-supplied `libero_paper_results_export_20261026.csv`, already designated authoritative in `SOURCES.md`. The repository's older study README/results describe a 58% initial checkpoint and a terminal endpoint at update 170, whereas the corrected manuscript CSV specifies 61% and update 160. This audit does not replace the author's corrected numerical export with that older archive summary. Matching-budget descriptions refer to the corrected export and are not a reconstruction of missing historical training commands.

The repository defaults include an internal terminal-success weight of 1. In `add` mode it can be added to the external verifier coefficient, so a nominal external coefficient of 5 need not be the entire effective terminal weight. The study README itself notes incomplete run configurations. The table now calls lambda the external terminal verifier coefficient. Exact internal-terminal settings cannot be inferred for each archived run from current defaults. No isolated density or universally best reward claim is made from these configurations.

## Manuscript edits

- Methods C explains continuous progress, phase events and their combination, then motivates the milestone-based composition choice through reduced task-specific reward design.
- Experiments A distinguishes the full-suite configuration comparison from the low-success acquisition diagnostic.
- Table labels identify dense-plus-phase feedback and retain every reported success rate/update count from the same authoritative CSV.
- Abstract/conclusion reflect the engineering rationale without claiming an empirically established universal generalization advantage.
- Original files: `archive/manuscript_20261029_before_staged_reward_rationale/`.
