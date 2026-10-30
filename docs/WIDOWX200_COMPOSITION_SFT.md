# Composition SFT before joint WidowX-200 RL

The updated automated schedule is:

1. Dense skill RL (stages 1–3), with successful experience harvesting and conditional retention SFT.
2. Binary skill RL (stages 4–6), with the same harvesting/retention mechanism.
3. Final-prompt **composition SFT**, configured by `widowx200_pipeline.composition_sft`.
4. Joint staged-reward RL (stage 7), warm-started from the composition SFT adapter.

## Existing production run

Stop the old launcher with Ctrl+C and wait for its shell prompt before running
the updated launcher. Stopping just after `=== stage 7: s7_joint_staged` is fine.
The preceding stage must finish its harvest and retention, as well as its RL.
If stopped earlier, the driver resumes that work before composition SFT.

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only
PIPELINE_DIR=/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201 \
MEMORY_CHECK_FIRST=0 SMOKE_FIRST=0 RESTART_JOINT_FROM_SFT=1 \
bash scripts/train_widowx200_pipeline_remote.sh
```

If the old stage 7 has already created a run directory, this option preserves
that directory under `__before_composition_sft_<timestamp>`. Its old pipeline
record is retained in `joint_restarts`. Only the interrupted joint stage is
reset; completed stages 1–6 and their banks are reused. The old `latest.pt` must
not be resumed after changing the initializer. Restart intent is saved before
the directory rename so an interruption during preservation can be reconciled.
A completed old joint experiment is not replaced in place.

For future launches, `STOP_BEFORE_STAGE=s7_joint_staged` exits cleanly after the
skill stages, before composition SFT/joint RL. These production controls do not
short-circuit the launcher's separate smoke pass.

### Ending skill RL early at a saved checkpoint

Interrupt the launcher with Ctrl+C and wait until it exits. To finish stage 6
without more RL updates, then harvest, check retention, prepare composition SFT
and continue to joint RL:

```bash
PIPELINE_DIR=/root/repo/RL_VLA_Bootstrapping/runs/widowx200_pipeline_20261024_131201 \
MEMORY_CHECK_FIRST=0 SMOKE_FIRST=0 RESTART_JOINT_FROM_SFT=1 \
FINISH_STAGE_AT_CHECKPOINT=s6_put_into_binary=latest \
bash scripts/train_widowx200_pipeline_remote.sh
```

`latest` chooses the highest numbered saved `step_*/smolvla_grpo_adapter.pt`,
not an in-memory update or the mutable `latest.pt`. It requires the matching
fixed-distance validation for the stage metric. An explicit absolute path to
one of that stage's saved adapters can replace `latest`. The option refuses to
replace an already harvested stage or change an earlier stage after a later
one has started. Keep the same option when resuming interrupted harvest/SFT;
the initial checkpoint choice is saved and is not selected again on resume.

The state records a `user_checkpoint` stop, the selected RL checkpoint and its
validation score separately from the actual best validated checkpoint. Harvest
uses the selected checkpoint; retention may subsequently select an SFT policy.
The exporter preserves all three identities. Stages 1–5 are reused, and stage 6
still completes its data and retention work before stage 7.

Changing object geometry does not certify old scores or banks under the new
setup. The new harvest uses the current simulator. Existing banks may retain
earlier scene settings; inspect source provenance when comparing results.

## Data and training contract

- The retained bank for each skill normally names a best validated checkpoint
  from a skill stage; an explicit manual finish uses the chosen checkpoint.
  A weaker or empty later harvest can leave an earlier bank in
  place. The controller records exact source stages/checkpoints/paths.
- Missing replay/frame pairs are repaired through the existing harvest/replay
  path using available best skill checkpoints. No production oracle fallback
  is allowed. Every required skill must supply filmed successful episodes.
- Legacy banks are re-filmed with the scene-spreading frame selection before
  pooling. GRPO copies of one reset carry a shared `scene_uid`; the source pool
  must contain that column. The earlier frame budget can otherwise be consumed
  by eight near-identical candidates from only a few scenes.
- Banks are recorded from `sil.harvest_first_round` (100), past every
  evaluation round. Legacy banks recorded from round 0 shared scenes with the
  retention and candidate evaluations, so they are re-recorded from their own
  checkpoint at the held-out rounds before pooling.
- The source pool is built with `--require-frames`, keeping complete filmed
  episodes and executed-action masks. Non-graspable reach targets are excluded
  from composition supervision.
- Reach/pick source episodes are assigned to plate/bowl final instructions as
  whole episodes. Existing placement destinations are preserved. Original
  teacher prompts/ids are retained alongside the student prompts/ids.
- A disjoint original-label slice remains for reaching, pickup and placement.
  Default sampling exposure is 80% final-prompt stage supervision and 20%
  original-label retention, in both the residual and LoRA passes. Within each
  view, sampling balances stage/destination/object. Repeated sampling changes
  exposure, not the number of distinct demonstrations.
- The row quota keeps whole source-scene groups and episodes. Each sampling
  cell has train and held-out scenes. `sft_validation` is enforced by
  `--split-by provided`; every GRPO copy of one recorded scene receives the
  same view, final destination and holdout assignment. No cross-teacher scene
  deduplication is claimed. LoRA's frame budget covers every available cell on
  both split sides.
- Relabelling marks priors/features stale. They are recomputed from the recorded
  frames under the student initializer and new prompts, with 100% frame coverage
  required. `--final-prompt-prefix ''` is used because this is a declared mixed
  bank containing original-label retention as well as final prompts.
- The existing SFT tool fits the residual and action-expert LoRA, selects epochs
  against its held-out loss, and writes provenance. Vision LoRA is left off.
  Defaults are 20 residual epochs, 4 LoRA epochs, an 8192-row frame budget and a
  4-row LoRA microbatch. If no improved SFT adapter is written, joint RL is held
  rather than silently pretending that SFT was applied.
- Original-skill and empty-start composed placement success are evaluated before
  and after SFT. These results are recorded, not claimed from SFT loss alone.
  Joint RL uses the selected SFT adapter with fresh optimizer/curriculum/step
  state and the existing staged reward and stopping rules.

## Scope of the supervision

These are pooled **skill-stage samples under a final instruction**, not newly
executed continuous grasp-carry-release trajectories. Recorded scenes are not
altered, and destination presence is not certified for reach/pick scenes. No
action/state/frame is invented to bridge independently collected episodes.
Complete-task competence is measured by the composed rollout evaluations.

The CDPR continuous-chain builder remains a separate collection/data contract.
This transition reuses the already-recorded WidowX skill banks; it does not
claim that those files satisfy the CDPR continuous-chain format.

## Artifacts and resumption

Artifacts live under `<pipeline>/composition_sft/<identity>/`: original pool,
modified mixed bank, refreshed bank, SFT report/adapter, and composed evaluations.
The identity includes settings, initializer and retained bank provenance.
`pipeline_state.json.composition_sft` records progress, source banks, before/after
scores and the selected policy. The joint stage records `composition_sft_id`.
Completed preparation is reused when resuming corrected joint RL; keeping
`RESTART_JOINT_FROM_SFT=1` does not archive that corrected run again.

Use `export_widowx200_pipeline_results.py` to export this evidence for analysis.
