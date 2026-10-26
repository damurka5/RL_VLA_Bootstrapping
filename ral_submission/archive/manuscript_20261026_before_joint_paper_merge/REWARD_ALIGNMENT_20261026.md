# Integrating the phase-aware reward study into the RA-L argument

26 October 2026. This note explains the revised manuscript argument. It is authoring material, not part of the submission. The previous reading copy and edited sources are preserved in `archive/manuscript_20261026_before_reward_alignment/`.

## Recommended argument

Keep the abstract's main contribution: **dense skill acquisition -> binary refinement and self-imitation -> milestone-based composition on the new embodiment**. Use the LIBERO study to examine the intermediate-feedback mechanism that motivates this sequence. It supports a component of the explanation rather than functioning as a second evaluation of the complete schedule.

The connection is substantive: complete-task failure hides differences between never reaching an object, lifting and dropping it, and nearly completing placement. Both implementations expose some of those differences. CDPR uses discrete milestones and controls stage-specific action credit. LIBERO combines phase completion with rewards for previously unachieved within-phase progress.

Suggested spoken explanation:

> Our main contribution is a learning sequence for acquiring skills on a new embodiment and composing them under one final instruction. The LIBERO study examines a shared ingredient: exposing manipulation progress when terminal success gives insufficient feedback. It shows an observed benefit of phase-aware dense feedback with an existing action policy. Our CDPR implementation uses milestone feedback and stage-specific credit for composition. Together, the studies motivate structured intermediate feedback at different points in learning; they do not establish that the same reward formula or the complete curriculum was tested on both robots.

## Three useful inferences

### 1. Intermediate feedback can remain useful after skill acquisition

The original abstract can sound as though dense feedback is useful only while the target action module is new. LIBERO starts from a trained policy with 61.0% benchmark success, yet primary dense feedback ends at 65.3% versus 61.7% for terminal-only. This supports the narrower claim that intermediate feedback can help after usable action behavior exists. It makes the return to milestone feedback during CDPR composition coherent.

This is an observed comparison under the supplied settings. It is not an independent-seed significance claim, an equal-interaction efficiency claim, or proof of a universal advantage.

### 2. Phase awareness, reward density, and action credit are different choices

- **Phase awareness:** use ordered reach/grasp/lift/carry/place achievements.
- **Reward density:** pay only at boundaries, or also for progress inside a phase.
- **Action credit:** choose whether actions receive a trajectory-level return or stage-specific advantages, and how full-task outcome affects earlier stages.

The first is shared. The LIBERO ablations primarily explore reward construction. CDPR's separate milestone streams, balanced stage losses, and protected completion bonus also intervene in action credit. Consequently, the paper can explain a shared mechanism without claiming algorithmic equivalence.

The three CDPR segments map conceptually to LIBERO reach, grasp/lift, and move/place. Exact predicates cannot be equated without the source implementation. CDPR's 0/1/2/3 telemetry is not a complete description of its optimizer objective.

### 3. Useful contrast, rather than reward density alone, connects the results

For a binary event with probability p and group size G, the existing analysis gives `P(mixed) = 1 - p^G - (1-p)^G`. An intermediate event can create contrast when full success is rare. If all candidates achieve the same milestone, its binary stream also loses contrast. Within-phase progress may distinguish those candidates, provided their progress differs and the filter retains them.

This is a mechanism hypothesis with an existing finite-model basis, not a measured explanation of the neural runs. In the LIBERO low-success study, the terminal-accuracy filter excludes all-failure groups even if dense returns differ. Dense shaping cannot provide gradients from a group that never reaches the optimizer. Measured return spreads, stage usability, and accepted-group logs would test this connection.

## Why binary refinement is compatible with their result

Binary refinement optimizes completion of an individual skill under its declared starts. Composition changes both the instruction and reset distribution and introduces a longer chain of dependencies. Intermediate feedback can therefore be useful again. The proposed schedule need not assert that binary reward is globally preferable after a fixed point in training.

The present evidence does not prove that switching from dense to binary skill reward beats continued dense training. That claim still needs a matched branch from the same skill checkpoint. LIBERO does not supply it, and the manuscript now says so explicitly.

## Claim-to-evidence map

| Claim | Evidence available | Defensible interpretation |
|---|---|---|
| Trained VLA behavior can benefit from intermediate task feedback | LIBERO primary dense 65.3% vs terminal-only 61.7%; low-success 18% vs 11% | Observed support under the reported configurations |
| Within-phase progress alone improves over phase-only reward | Primary dense 65.3% vs phase-only 62.3%, but terminal/shaping coefficients differ | Not isolated; require matched coefficients and budgets |
| CDPR composes the whole instruction with milestone learning | Reported empty-start strict outcomes; current 61.43% development endpoint | Feasibility and continuing-lineage performance; not a staged-vs-terminal ablation |
| Stage-specific credit is better than adding the same rewards globally | Implementation and finite-model rationale | Mechanism proposal; need an advantage-construction ablation |
| Binary refinement and self-imitation improve the complete schedule | Historical skill consolidation plus unsuccessful later full-task SFT | Recover matched skill evidence; no blanket guarantee |
| The full schedule generalizes across embodiments | Full schedule on CDPR; component study on LIBERO | Not yet established across embodiments |

## Manuscript changes

- Abstract/introduction: explain what the complementary study contributes to the main argument.
- Formulation: distinguish a training stage from an episode phase.
- Method/analysis: connect progress granularity and stage credit through reward contrast.
- LIBERO definition/results: separate phase-only and dense feedback and reject an isolated-density interpretation of the confounded rows.
- Discussion/conclusion: explain the return to intermediate feedback for composition and preserve limits on schedule and cross-embodiment claims.

Existing numerical results and hardware assumption labels are preserved. The original study's authors and source remain recorded in `evidence/SOURCES.md`. Publication status and the combined author list remain unresolved; no citation metadata or authorship has been invented.

## Most informative additional comparisons

1. **For the central RA-L claim:** same consolidated CDPR initializer, ordinary full-task starts, matched scenes and budget; compare terminal-only, milestone reward with common trajectory credit, and the implemented stage-specific credit. The latter two separate reward construction from credit assignment.
2. **To connect the two implementations:** add phase-aware within-phase progress to the CDPR comparison while holding the terminal objective and optimizer/credit construction fixed. This is an extension to test, not a performed experiment.
3. **For the LIBERO density claim:** phase-only versus phase-plus-progress with the same terminal coefficient, phase coefficient, clipping, filters, initializer and budget. Report both wall-time and interaction/update counts. Clipping can couple components, so a matched configuration estimates the operational effect of adding progress, not an unclipped additive contribution.
4. **For the schedule:** branch from the same dense-trained skill checkpoint into continued dense versus binary refinement, then evaluate with/without compatible skill consolidation before the same composition procedure.

Priority is the first comparison because it tests the paper's central composition claim. Existing logs and resolved configs should be recovered before launching new experiments; this writing task launches none.
