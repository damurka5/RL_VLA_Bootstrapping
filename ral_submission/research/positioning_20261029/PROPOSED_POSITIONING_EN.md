# Proposed positioning — author review copy

29 October 2026. These are candidate paragraphs for the current manuscript. They have not been inserted into the author's Introduction or Related Work. Scientific priority of the full schedule and milestone-credit rule has not yet been established by an exhaustive search.

## Central message

We study VLA adaptation as a progression from acquiring individual manipulation skills to retaining and composing them under a complete task instruction. Our procedure uses dense rewards for initial acquisition, binary success for refinement, instruction-balanced self-imitation for retention, and milestone-based RL for full-task composition.

## Candidate contribution paragraph

The contributions of this work are:

1. **A staged adaptation procedure** that progresses from dense-reward skill acquisition to binary refinement and milestone-based full-task learning, with instruction-balanced self-imitation to consolidate earlier and current successful experience.
2. **A stage-specific credit rule** that retains local milestone credit for successful approach and pickup, while suppressing their negative completion bonuses when subsequent placement fails. The rule specifies which training signal is protected; shared network parameters can still introduce interference.
3. **An empirical study of skill retention and task execution** that documents forgetting and partial recovery in CDPR simulation, reports complete-task performance of the adapted controller, and separately examines intermediate feedback with OpenVLA-OFT on LIBERO-Spatial.

Editorial note: item 1 states the procedure; item 2 states a concrete implementation contribution whose priority and isolated benefit still need checking; item 3 states the evidence actually present. WX200 should enter the evidence claim when its evaluation is complete. If the credit rule is not a distinct contribution after prior-art review, combine items 1–2 instead of insisting on three bullets.

## Candidate Related Work

**VLA adaptation and self-improvement.** OpenVLA, SmolVLA, and OpenVLA-OFT provide pretrained control policies and adaptation methods [1–3]. Online interaction and successful-trajectory supervision are established tools for improving these policies. iRe-VLA alternates RL on a lightweight action head with supervised updates using successful experience [4]. Probe, Learn, Distill (PLD) combines residual RL specialists, policy-aligned data collection, and distillation into a VLA generalist [5]. We build on these ingredients to study acquisition, retention, and full-task composition when adapting the action interface.

**Feedback for VLA reinforcement learning.** Intermediate feedback has also been explored in VLA fine-tuning: RFTF obtains dense rewards from a value model trained with temporal information [6]. SimpleVLA-RL studies group-relative optimization for VLA rollouts [7]. Our procedure uses task-specific dense feedback during initial skill learning, binary success during refinement, and milestones during full-task RL. The contribution under study is this feedback schedule together with consolidation and stage-specific credit assignment.

**Skill retention and composition.** Our consolidation step fits an instruction-balanced bank of earlier and current successful trajectories. We evaluate the resulting changes in retained skill success before training the complete instruction. During full-task learning, milestone credit distinguishes completed approach and pickup from a later placement failure. Final evaluation measures the entire instruction from an empty-gripper start, keeping skill retention and complete-task success as distinct outcomes.

Editorial note: the third paragraph is currently a positioning bridge, not a sufficient survey of continual imitation, curriculum learning, or hierarchical RL. Add direct prior-art comparisons from those areas before treating this as submission-ready Related Work. A citation to a general shaping theorem does not replace them. Use the three headings as an organizing proposal, not a journal requirement.

## Verified source mapping

- [1] OpenVLA: https://arxiv.org/abs/2406.09246 — already in the manuscript bibliography.
- [2] SmolVLA: https://arxiv.org/abs/2506.01844 — already in the manuscript bibliography.
- [3] OpenVLA-OFT: https://arxiv.org/abs/2502.19645 — already in the manuscript bibliography.
- [4] iRe-VLA: https://arxiv.org/html/2501.16664v1 — Introduction and §IV-B; add a verified bibliography entry when editing the manuscript.
- [5] PLD: https://arxiv.org/html/2511.00091v1 — §1 and §3; existing bibliography key `pld`.
- [6] RFTF: https://arxiv.org/abs/2505.19767 — dense-reward/value-model mechanism verified from the primary arXiv record; title/venue metadata should follow the selected final version.
- [7] SimpleVLA-RL: https://arxiv.org/abs/2509.09674 — existing bibliography key `simplevlarl`.

## Claims to keep out of this wording

- First use of RL + imitation, residual RL, or dense rewards for VLAs.
- Proven prevention of neural-policy forgetting by milestone-credit protection.
- A causal attribution of the final 70.61% to self-imitation or any one stage.
- Completed quantitative WX200/hardware validation while it remains ongoing.
- Demonstration-free acquisition without accounting for inherited assisted training.
- General applicability established across arbitrary robot embodiments.

