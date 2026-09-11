# Three-stage demonstrations for full `put_into`, followed by SFT

2026-09-10. Design prepared against local repository `e56b7fb` and the
consolidated report, including the September 10 pilot attachment review.
**Status, 2026-09-11.** Every module, flag and config field
named in §12 exists and is unit-tested; the commands below are real. Nine
initial teacher screens were floored by collection-harness faults. The repaired
single-pass destination-prompt screen accepted 3/128 continuous chains. Its
selected triple then produced the first collection bank: 5/512 accepted chains
over five unique scenes, 365 full-task decision rows with 100% frame coverage,
plus 12 successful pickup prefixes stored separately. The exact pooled ladder
from both shard reports is 79 reaches, 20 aligned handoffs, 17 pickups and five
placements. This is a valid pipeline smoke artifact, not yet a sufficient SFT
corpus: the full bank has only five scenes and no successful potato chain.
Future recordings retain frames from alignment success onward and the builder
also emits a strictly gated final-prompt transition view: at this yield, 20
move/alignment transitions, 17 pickup transitions and five placement
transitions. The original full-chain view remains unchanged. The full ledger is
§7.16b and §14 of the consolidated report; the summary is below.

**Stage-by-stage status, measured on 64 scenes per screen:**

| stage | status | measured |
|---|---|---|
| scene manifest | done | 1024 scenes, splits disjoint, 0 start inside the goal |
| yaw calibration (§5) | done | 0.000 rad; fixed-angle residual mean 10.7°, max 25.3° |
| move-to → reach | **the ceiling** | 23/128 in selection; 79/512 in collection |
| alignment tail → pickup | second ceiling | 7/23 promoted in selection; 20/79 in collection |
| alignment tail — yaw | done after damping | 0.0 rad median; 20.5% of steps outside the 5° band |
| pickup under final destination prompt | reliable after handoff | 7/7 in selection; 17/20 in collection |
| placement: carry/release | usable, still sparse | 3/7 in selection; 5/17 in collection |
| dataset | first smoke bank built | 365 rows, 1,454 supervised actions, five scenes, bowl and plate present, potato absent |
| refresh / SFT / evaluation | implemented, not run | bank is intentionally still marked `priors_stale` |

**Two corrections to this document's own assumptions**, both measured:

1. §5 assumed the alignment tail would "keep the gripper open". It has to
   OPEN it: the `move_to` reward has no gripper term under
   `sparse_binary_reward`, so a shared four-instruction policy arrives with the
   hand closed on ~100% of reaches.
2. §6's handoff condition assumed the production `move_to` predicate was a
   sufficient pickup-readiness test. It is not. The reach window is 0.02 m and
   the open gripper's lateral slack is 0.0130 m (apple) to 0.0185 m (others),
   so a chain can satisfy the reach and still hand over a pose from which the
   fingers cannot bracket the object. Readiness now carries its own per-object
   lateral bound derived from the measured aperture, and §5's "explicit
   recorded bridge" is implemented as `--align-xy-centring`.

Existing GRPO remains available for subsequent work; this document specifies
the user's requested demonstration collection and SFT, and §13 is the order to
run it in.

## 1. Result to build

Produce a durable dataset of successful physical episodes:

```text
empty gripper away from object
  → move_to teacher approaches object and establishes pickup yaw
  → pick_up teacher grasps and lifts the same object
  → put_into teacher carries it to the selected receptacle and releases it
  → final placement is verified
```

All three stages share one scene and continuous simulator state. Keep their
original teacher prompts and checkpoint identities as provenance. The SFT
view gives EVERY stage the same final instruction, for example
`put apple into bowl`. At deployment the student receives this instruction
from the beginning and controls the whole episode without teacher switches.

Use the plate wording requested by the user: `put apple into plate`. The
existing teacher can retain its familiar `put apple on the plate` wording
during collection. Normalize the student text consistently in dataset refresh
and evaluation; do not accidentally test a different prompt template.

Different teacher checkpoints are allowed. This is imitation/distillation of
the project's own policies; it is not necessarily on-policy self-imitation of
the final student. Teacher action probabilities are irrelevant to the SFT
loss, and these off-policy demonstrations must not be passed to GRPO as fresh
on-policy records.

## 2. Corrections to the proposed collection order

**Record continuous episodes, then expose their stage slices.** Recording all
move-to stages in one process and pickup stages later is also possible, but
each pickup must resume its own parent's complete state or a verified replay
of that parent. A new randomized reset with similar XYZ is not a continuation.
Same applies to pickup → placement. Prefer one live collector initially;
persistent cross-process state restoration is an optional later optimization.

**Equal row counts are a sampling choice, not a physical requirement.** If a
successful episode contains 12 move-to, 9 pickup and 25 placement decisions,
preserve all 46 in the archive. Do not truncate placement to nine decisions,
pad pickup with fake actions or repeat the last frame to manufacture equal
lengths. A balanced sampler can draw one third of its decision rows from each
stage, with replacement for the smaller stage. Repeated draws are not new
demonstrations and must not increase the reported unique-data count.

**Yaw cannot change invisibly at a join.** Pickup must begin at the actual yaw
reached by move-to plus any recorded alignment motion. Calling
`set_end_effector_poses` between slices changes state without an action and
would make the concatenated demonstration physically discontinuous.

## 3. Teacher checkpoints: shortlist first, select on the new scene contract

There is no supported global winner across these instructions. Historical
scores use different distances, yaw distributions, starts and predicates.
The latest pilot final is below its donor on all four external metrics.

The following recent shortlist has known run/step references. Paths are
relative to `/root/repo/RL_VLA_Bootstrapping` on the server; existence and
hashes must be verified there.

| Role | Provisional candidate | Supporting evidence and limitation |
|---|---|---|
| Move-to teacher | `runs/release_recovery_continue_3m_20260908_102004/rl/step_3416645/smolvla_grpo_adapter.pt` | 324/400 = 81.0% in the September 9 three-checkpoint comparison at move cap 0.08. Not established best at new approach distances or with the yaw controller. |
| Pickup teacher | `runs/release_recovery_continue_3m_20260908_102004/rl/step_3540208/smolvla_grpo_adapter.pt` | Best pickup in that comparison, 90/328 = 27.44%; September 10 baseline 83/328 = 25.30%. Different measurements, not interchangeable estimates. |
| Placement teacher, both destinations initially | `runs/release_recovery_continue_3m_20260908_102004/rl/step_2117145/smolvla_grpo_adapter.pt` | Bowl-peak candidate: 70.18% configured plate / 37.78% bowl in the September 9 comparison. Mostly inside-radius plate starts; not a measured carry-from-real-pickup score. |
| Alternative carry/release teacher | `runs/phase5_placement_iter3_20260828_224948/rl/step_2754052/smolvla_grpo_adapter.pt` | The Cycle 3 script's placement source, with historical caught-placement results. Particularly relevant to a held handoff, but must be tested on the new actual handoff distribution. |
| Alternative longer-distance move-to teacher | `runs/phase4_move_to_iter0_resume_20260818_080928/rl/step_11009573/smolvla_grpo_adapter.pt` | Dedicated reaching reference, 645/1024 at cap 0.19. Stronger provenance at that distance, not directly comparable to cap 0.08. The report says its adapter is missing from the local evidence set. |

Also resolve the final adapter of `release_recovery_pilot_20260909_130003`
from its manifest: the report records 82.75% move-to on its evaluation, though
its manipulation regressed. It is a reach candidate, not an assumed best
shared learner. Resolve historical dedicated pickup donors from the retained
bank manifests if available; do not invent paths or choose the latest file
by timestamp as a substitute for an evaluation result.

Create a teacher manifest with explicit role → checkpoint path, SHA-256,
base-model revision, complete adapter keys/shapes, action normalization,
observation/camera contract, action scales and control timing. Verify that
loading each teacher reconstructs its complete residual and LoRA weights.
Do not average weights. Cache/swap compatible adapters or load separate
runtimes within measured GPU memory limits.

Selection procedure:

1. Create disjoint collection, teacher-selection, student-validation and final
   test scene IDs. Hash the scene definition and split assignment, including
   parent/retry identities. Do not use final test scenes to select teachers.
2. Score reach candidates from the same empty starts, including the yaw
   alignment and pickup-readiness checks. Rank readiness as well as native
   move-to success: XY success alone does not establish a usable pickup pose.
3. Generate a held-out set of real reach endpoints. Compare pickup candidates
   on verified clones/replays of those endpoints, with the same yaw rule.
4. Compare placement candidates on real successful pickup endpoints, separately
   for plate and bowl. No synthetic caught reset or relocation to the bowl.
5. Confirm the chosen combination with full chains. Optimize complete-chain
   yield and diverse usable demonstrations, not just marginal stage maxima.
   If teachers change, regenerate downstream selection endpoints and retest.

An initial selection budget can be 64 distinct scenes per destination, with
two declared rollout seeds each; this is a screening default, not a guarantee
of a precise ranking. Report uncertainty clustered by scene. If close or
poorly covered, expand selection scenes rather than using the final test set.

Use `step_3540208` as the provisional student initialization because it is a
recent complete shared adapter with an audited baseline. Confirm its new-task
baseline first. Teacher choice and student initialization are separate.

## 4. Special scenes and a genuine full-task reset

Add a versioned scene manifest, independent of each checkpoint's stored
curriculum. Initial scenes contain one target object and one plate OR bowl.
Include both destinations in a later ambiguity test, with the named one
explicitly identified. Unused object slots must be disabled consistently.

Use the existing object catalog and model assets; target choices initially
include apple, orange, potato and tomato, then banana/mug as separate coverage
strata if the geometry is feasible. Do not count a visible object as reachable
or a receptacle as large enough without checking collisions and dimensions.

Suggested initial collection settings, subject to geometric feasibility:

- Ordinary empty/open gripper, no grasp history, object resting on the desk.
- EE–target XY start distance 0.06–0.10 m, outside the native reaching success
  region. EE height inside a declared safe band; log both XYZ distance and XY.
- Object–destination XY distance at least the production destination radius
  plus 0.04 m; initially cap it at 0.18 m. Thus plate and bowl have different
  minimum center distances but the same minimum outside-radius margin.
- Also reject initial geometry overlap using object/receptacle extents and
  collision checks. A center-distance predicate alone does not prevent meshes
  overlapping or the object already resting on a receptacle's edge.
- Keep objects in the actual workspace AND relevant camera coverage. Reject
  and resample if realized positions fail; do not clamp into an easier task.
- Preserve current production success radii (plate 0.091 m, bowl 0.057 m).
  Scene design must not loosen success to improve demonstration yield.

The current uncaught container reset overwrites EE XY with target XY and
samples target distance 0.06–0.10 m before clamping. Introduce an explicit
full-task scene/reset route that bypasses that alignment. Reuse it unchanged
for teacher chains and single-prompt student evaluation. Do not silently
change all existing training/evaluation configs.

Store object orientations, support height, camera extrinsics, actual EE yaw,
gripper opening and complete initial state. Initialize the lift datum from
the settled pre-action target pose; do not reuse the stale pre-relocation
`initial_target_positions` value identified in earlier composed recordings.

## 5. Pickup yaw: specification and physical implementation

**User-confirmed choice: one calibrated fixed world yaw** for pickup,
corresponding to the successful pose the user describes as the wrist camera
facing the overview camera. Per-position camera-facing is not selected for
this implementation; the geometric explanation below documents why that
would be a different rule as the object moves around the workspace.

The current resetter samples yaw uniformly in [-π, π]. Backend action dimension
3 increments the yaw controller by `action_step_yaw` (currently 0.08 rad per
normalized unit per executed action). There is no inspected reset rule that
always turns the wrist camera toward the overview camera. The observed
rotation may be a learned behavior; record it before assigning a cause.

Calibration should produce a report, not a guessed `0` or `π`:

- Render successful move-to endpoints from candidate teachers. Log commanded
  yaw, actual joint yaw, EE/world quaternion, camera world pose, object pose
  and image samples. Use actual endpoint measurements rather than the reset
  yaw or the command alone.
- Determine the desired camera-facing pose using the loaded model's camera
  extrinsics. The camera sits on an offset body below `ee_yaw` and has fixed
  tilt. Yaw determines horizontal orientation; yaw alone cannot point its
  full optical axis toward an arbitrary elevated overview-camera position.
- A future per-position facing mode would solve for the horizontal optical-axis bearing
  relative to the overview-camera bearing at the handoff location, accounting
  for the camera's rotated local offset and mounting rotation. Refuse a
  degenerate horizontal direction. For fixed mode, store the calibrated angle
  and calibration pose; it is not an exact camera-to-camera alignment at
  every workspace location.
- Proposed acceptance: actual yaw within 5 degrees of the target for two
  consecutive policy decisions, with EE speed/position and open-gripper
  readiness also satisfied. This is a measurable tolerance, not bitwise
  equality in a dynamical simulator. Validate the tolerance on GPU.

Execution rule: after native reach success, append a bounded **yaw-alignment
tail within the move-to stage**. Hold the safe XYZ setpoint, keep the gripper
open, and apply a bounded yaw servo through the ordinary action interface.
Record all executed alignment actions and observations. Require clearance
for the rotating fingers; if reach ended too low, any lift/repositioning must
also be an explicit recorded bridge, or the chain is rejected.

Pickup starts from that exact live state. Keep a yaw-hold override during
pickup as the initial controlled variant, recording both the raw teacher yaw
command and the applied yaw command. During placement, release that override
by default and let the teacher control yaw. No object orientation reset is
allowed. Respect the yaw joint's hard limits: a wrapped shortest-angle error
must not request motion through an inaccessible ±π joint boundary.

The alignment/override makes this a policy-plus-controller demonstration.
Preserve that provenance. Student evaluation starts with ordinary initial
yaw and must learn the alignment from the recorded tail; adding the same
external yaw servo at evaluation would measure an assisted system and must
be reported as a separate diagnostic arm.

## 6. Continuous three-stage collector

Introduce a separate collector built on the existing backend and policy
interfaces. Avoid reusing a task's `terminated` mask as full-episode completion.
Maintain separate stage status and one full-episode status per world:

| Stage | Teacher instruction | Transition condition |
|---|---|---|
| Move-to, including alignment tail | Existing `move to <object>` template | Native reach success plus actual pickup-ready position/height, open gripper, no premature grasp, and calibrated yaw tolerance. |
| Pickup | Existing `pick up <object>` template | Production pickup success, still holding the target at the handoff, valid 5 cm lift from the true desk datum. |
| Placement | Teacher's familiar destination-specific template | Production placement success, with continuous target grasp history; then verify the full-chain quality conditions. |

At each teacher switch, retain physical state, contact/grasp history and the
global action clock. Clear that world's old policy action queue and
instruction-conditioned caches; render a new observation and sample the
destination teacher using its full adapter. Do not change object slots or
initialize a pre-grasped reset. A checkpoint's curriculum does not own the
live scene after startup.

Use policy-decision boundaries for handoffs in the first implementation.
Native stage success can occur within the four executed actions of a chunk;
keep a stage-event timestamp and check that readiness still holds at the next
decision boundary. Do not mark the world inactive on that intermediate event.
If the grasp is lost before handoff, there is no valid pickup endpoint yet.
On final success/failure, stop recording effective episode actions and mask
unused tail slots. A later implementation may support partial-chunk switches
only with explicit event-driven observations and variable-length masks.

Proposed starting budgets are 32 move/alignment + 32 pickup + 64 placement
decisions, at four executed actions per decision: at most 512 action steps.
These are collection caps, not required stage lengths or tuned optima.
Give the single-prompt student the same total budget for a fair comparison.
Audit all counter conversions: existing `max_env_steps` and curriculum
horizons must not silently truncate the collector to the old 40-decision task.
Also report results at shorter budgets later, rather than attributing a
horizon increase to learning.

For batching, keep each live world advancing every simulation step. Group
worlds by current teacher for inference, scatter their actions, then step the
single backend allocation. Do not wait for every world to complete move-to
before permitting any pickup. Masking actions does not freeze MJWarp physics;
unrecorded waits would corrupt joins. Completed/failed worlds can stop
contributing data while their physics continues, and reset only through a
tested independent-world reset route or at a full batch boundary.

Two GPUs can run independent collection workers with disjoint scene IDs; DDP
is not needed for inference. Keep all stages of one episode on the same
worker initially. Batch inference by teacher role and measure adapter-switch
and frame-recording cost before assuming 512 worlds is the best layout.

## 7. Completion, rejection and retries

Maintain the final placement observer from the first action. Its production
task state preserves true initial target height and target grasp history
across teacher switches. Separate stage evaluators must not mutate each
other's histories by aliasing tensors. A native reach or pickup success is
an event, not a final `put_into` success.

Keep two verdicts: native production placement success, and full-chain data
acceptance. Accept a positive full demonstration only if:

- it began empty with the target on the desk, outside the destination region,
  and with genuine approach distance;
- recorded actions account for the entire approach, yaw bridge, grasp, lift,
  carry and release, without pose writes/reset discontinuities;
- pickup met the production criterion and the object remained physically held
  during transport until intentional release (reject mid-carry loss/regrasp
  for the initial clean dataset; label it separately for later recovery work);
- an intentional release and final production placement success are observed,
  with a short recorded stability check if required by the new data contract;
- frames, proprioception, actions and camera IDs align; all values are finite;
  no world in this trajectory was quarantined for simulator divergence.

Log stability checks explicitly and apply the same checks in student
evaluation. Keep production success reported separately if the full-chain
contract is stricter; do not rewrite old scores under new criteria.

Initial retry limit: one chain per scene/rollout seed; a later bounded retry
mode can replay/restore a parent handoff and vary only the suffix seed. Every
retry retains its parent `scene_uid`; shared prefixes are not independent
episodes for splitting or uncertainty. Report attempts, rejected chains and
conditional stage yields, not only successful survivors.

Failed placement may supply a valid pickup demonstration under `pick_up`,
but not a positive full-placement episode. Store partial material separately
so later recovery/imitation experiments can deliberately use it. The initial
full-task bank contains only verified complete successes.

Additionally derive a separate **final-prompt stage-transition view** from the
same continuous trajectories. Include the move/alignment rows of a world only
when its alignment handoff completed, its pickup rows only when grasp-and-lift
completed, and its placement rows only when strict full-chain acceptance
passed. This implements the distinct-transition proposal without losing the
real continuous handoff state and without relabelling the actions of a failed
stage as success. Keep this view separate from the end-to-end bank so an
experiment chooses it explicitly and reports unique successful-transition
counts by stage.

## 8. Durable recording schema and derived SFT rows

Store immutable, versioned trajectory shards plus a JSON manifest. Required
raw fields:

| Category | Required content |
|---|---|
| Identity | `scene_uid`, `episode_uid`, `parent_episode_uid`, retry/rollout seed, split, target/destination catalog IDs and slots, config/model/code hashes. |
| Reset | Complete reset description, actual pre-action positions/quaternions/velocities, EE yaw/opening, controller state, task history/lift datum; state restoration version if supported. |
| Observations | Every pre-decision overview and wrist RGB frame, proprioception, actual EE yaw/quaternion, object/receptacle poses, camera configuration and auxiliary-camera mask semantics. |
| Actions | Actual executed five-dimensional control, original teacher output, override source/mask, timestamps and global action/decision indices. |
| Stage | Teacher instruction/checkpoint, semantic stage, optional alignment substage, stage-local index, transition events and boundary observations. |
| Outcomes | Per-step grasp/release/active masks, native stage/final success, full-chain acceptance, divergence/failure reason, stage budgets and durations. |

Stream compressed frames/shards to bounded buffers. All retained successful
transitions need frames; a fixed `frame-worlds` subset is not a complete
reusable bank.
Save the terminal post-action observation as well as every pre-action one.
Record frame/action timing explicitly so future per-action windows do not
pretend a frame exists at an unobserved timestep.

Derived SFT fields extend the existing schema:

```text
state, prior, action, action_mask,
instruction_id, instruction_text, episode_uid, decision_index,
scene_uid, parent_episode_uid, split,
stage_id, substage_id, source_checkpoint_sha256,
frame_uid, stage_boundary_distance, full_chain_success
```

One row initially remains one policy decision. The recorded target has four
actually executed five-dimensional actions; the actor emits eight but executes
four. Do not supervise the unexecuted teacher prediction slots 4–7. Padding
after terminal success has mask zero. If extending to eight true future
executed actions, reconstruct them within the SAME physical episode and
document the changed target semantics; do not confuse predicted-but-unused
actions with actual future actions.

Windows may cross a teacher-stage boundary after relabelling if they contain
real consecutive executed actions under the single final goal. They must
never cross a reset, unrelated episode or unrecorded yaw change. Keep the
global frame/decision key unique across all stage files; restarting each stage
at decision zero without a parent key would create silent image/action joins.

## 9. Relabelling and current-policy refresh

The full demonstration's object and actual destination determine its student
label. Change both instruction ID and text for all its rows. Retain teacher
text separately. Relabelling a bowl trajectory as plate is not valid simply
because another receptacle is visible.

For this repository, rewriting text is not sufficient: the residual actor
reads the VLA prior, and `state` contains learned visual features. Run
`sil_refresh_priors.py` over the original frames/proprioception with the
student initialization and final instruction to regenerate these fields.
Extend frame joining to use explicit IDs. Preserve the captured actions.
Require 100% resolvable rows for the new bank; any exclusion is explicit and
removes the corresponding dataset view rather than pairing a wrong frame.

Source-checkpoint smoothing/replay is not necessary for the first bank. Keep
unsmoothed executed actions initially. If smoothing is later introduced, do
not smooth indiscriminately across close/lift/release transitions; replay and
revalidate the entire changed chain. Simulator replay validates physical
continuity; frame-only inference refreshes network inputs. These are different
operations and should remain separate.

## 10. SFT sampling and the OpenVLA comparison

The user's intuition about sampling is substantially correct. Original
OpenVLA's public training data path uses one observation and one action as a
sample; its RLDS pipeline transforms trajectories, flattens them, interleaves
and shuffles samples. It does not require optimizer minibatches to visit an
entire rollout chronologically. Its action-token labels remain paired with
the corresponding image and instruction. See the official
[sample transform](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/datasets.py)
and [RLDS pipeline](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/dataset.py).

This does not license arbitrary observation/action pairings, contradictory
goal labels, failed full-task labels or chunks spanning unrelated episodes.
Our existing residual SFT likewise shuffles decision rows, but each target
is a contiguous executed action chunk. SmolVLA itself uses an action-chunk
expert; see the authors' [architecture description](https://huggingface.co/blog/smolvla).
These details distinguish original OpenVLA, later chunked OpenVLA variants,
and this repository's SmolVLA-plus-residual objective.

**Recommended first sampler:** choose destination equally (plate/bowl), then
semantic stage equally (move/pickup/placement), then object/scene with bounded
imbalance, then a valid decision within that stage. Retain alignment tails
within move-to and report their share. Apply identical sampling logic to the
residual and optional LoRA stages. Require all strata to be present; an empty
stage/family triggers a clear error, not silent replacement with another one.

This gives the user's equal-row exposure without altering recorded lengths.
Keep raw and effective exposure counts separately. Add a measured share of
boundary-near decisions if transitions remain weak; do not discard the
middle of carry/release to force stage balance. Compare with natural-length
sampling under the same optimizer budget as an ablation.

**Continuous data and shuffled training are compatible.** Preserve full
episodes to obtain correct state coverage and transitions, then shuffle valid
rows/chunks for SFT. Merely ordering rows chronologically does not give the
existing residual MLP episodic memory. A sequence/history-conditioned student
would require an explicit model/input change with causal masks and compatible
inference, not just a different DataLoader order. It is outside this first
implementation. An episode-balanced sampler is possible without that model
change and is distinct from history-conditioned training.

Split by `scene_uid` before quota selection, refresh and training, keeping all
eight candidate worlds, retries, stage slices and relabelled views of a scene
together. Existing `_episode_split` only groups individual episode IDs and
must be extended for this bank; it does not by itself prevent shared-scene
leakage. Use training-only data for any newly fitted normalization/statistics.

## 11. SFT objective, schedule and model selection

Retain the current controller and checkpoint format. Existing `sil_sft.py`
supports residual fitting, then an optional image-backed LoRA stage; its
module header's residual-only description is stale relative to the current
`train_lora_stage` implementation. Do not mistake this objective for native
SmolVLA flow-matching pretraining or original OpenVLA action-token training.

The residual predicts actions as `tanh(prior + scale * tanh(net(features)))`.
Before training, measure target reachability under refreshed final-prompt
priors by stage, destination and action axis. A relabelled prefix can be outside
that bounded correction range. A persistent loss floor need not be bad data.
Do not clip the demonstrations to disguise unreachable actions.

Run a bounded comparison from the same student initialization and bank:

| Arm | Trainable components | Purpose |
|---|---|---|
| A | None | Full-task baseline from the chosen shared adapter. |
| B | Residual only, refreshed final-prompt inputs | Verify that the complete chain can be learned through the existing correction head. |
| C | Same residual stage, then action-expert LoRA from images; vision adaptation off initially | Test whether adapting the prior resolves prompt/reachability limitations. Record realized trained modules and gradient coverage. |

Arm C should compute current features/priors while training LoRA and anchor
to priors refreshed under the final prompt, not the old pickup teacher prompt.
If residual fitting follows changed LoRA weights, refresh again. Preserve all
adapter components in the resulting checkpoint and do not carry RL optimizer
moments into the SFT optimizer.

Keep successful original-label move/pickup/placement examples as an explicit
retention slice if maintaining the four instructions remains a requirement.
Proposed initial sampling: 80% full-task relabelled data, 20% original-label
retention. Within the full-task portion use equal stage/destination exposure.
This ratio is a starting experiment, not a proven optimum; report a
composition-only arm if diagnosing a retention tradeoff. Shared scenes still
stay in one split across both views.

Start with small saved intervals (for example one, two and four effective
epochs) and matched validation rollouts, not a long unattended SFT job.
Report actual optimizer updates and sampled valid actions because replacement
sampling makes an "epoch" ambiguous. The historical broad residual-SFT
composition regression in the consolidated report makes rollout checks
essential. Validation imitation loss alone does not establish task success.

Choose checkpoints using unassisted full-chain success on student-validation
scenes, with plate and bowl reported separately and retention as a declared
constraint. Use fixed final prompts, ordinary initial yaw and the same action
budget as the teacher chain. Lock selection before final test. Final test
must not use the stage state machine, yaw servo or teacher assistance.

## 12. Concrete repository changes

New filenames in this table are proposed interfaces. Keep old experiment
defaults unchanged and select the new path explicitly by config.

| File / module | Change |
|---|---|
| New `configs/examples/cdpr_smolvla_three_stage_put_into.yaml` | Scene geometry, yaw mode/calibration, explicit teacher manifest, student initializer, stage/global budgets, success contract, split IDs and sampling settings. Reject unresolved checkpoints or yaw calibration. |
| New `tools/audit/select_cdpr_stage_teachers.py` | Resolve/hash candidates, enforce action/camera compatibility, evaluate stage readiness on shared starts and real upstream handoffs, write selected-teacher manifest with counts and uncertainty. |
| New `rl_vla_bootstrapping/simulation/cdpr_composition_scenes.py` | Generate and validate the explicit full-task scene manifest; separate approach and transport distances; collision/visibility checks and deterministic scene identity. |
| `rl_vla_bootstrapping/policy/mjwarp_rank_local_collector.py` | Expose/reuse reset and observation primitives; add an explicit full-task reset route that avoids forced EE alignment and checkpoint-owned scene geometry. Keep GRPO collection semantics unchanged. |
| New `rl_vla_bootstrapping/policy/cdpr_staged_demonstrations.py` | Per-world stage machine, teacher runtime/cache routing, recorded alignment controller, native stage observers and persistent full-task observer, budget/failure handling. |
| `rl_vla_bootstrapping/simulation/cdpr_backend.py` and `mjlab_mjwarp_backend.py` | Reuse control/pose observations and same-process state broadcast. Expose validated continuation capture/restore only if required by teacher comparison or deferred stage collection; include controller and task state, not just qpos. |
| New `tools/audit/record_cdpr_staged_put_into.py` | CLI for full-chain recording, optional logical stage exports, per-GPU sharding, durable frames/actions/manifests, progress and failure summaries. No optimizer. |
| `tools/audit/sil_record.py` | Factor/reuse recording/frame serialization; support global episode/decision identity and explicit stage metadata in the new schema. Avoid treating intermediate native success as final episode termination. |
| New `tools/audit/build_cdpr_staged_sft_dataset.py` | Verify chain and per-stage transition acceptance, assemble real consecutive action windows, relabel destination ID/text, retain source labels, create strict `demonstrations.npz` and explicitly selected `stage_transitions.npz` views with row/frame mappings. No fake boundary rows. |
| `tools/audit/sil_refresh_priors.py` | Support explicit frame IDs/new schema; require final-prompt student refresh and complete coverage; retain stage/scene metadata. |
| `tools/audit/sil_sft.py` | Scene-level split override, destination/stage-balanced sampler in both training paths, per-stage loss/reachability/gradient reporting, explicit retention mixture and compatible full checkpoints. Update stale module documentation. |
| New `tools/audit/evaluate_cdpr_full_put_into.py` | Same full-task scenes and observer, one student prompt from start, no stage-controller assistance; native and strict full-chain verdicts, phase diagnostics and videos. |
| New `scripts/run_cdpr_three_stage_collection.sh` and `scripts/run_cdpr_three_stage_sft.sh` | Separate remote collection and training entry points. Explicit inputs/manifests, resumable artifacts, bounded budgets, no implicit training while just collecting or auditing. |
| `CDPR_CONSOLIDATED_PROGRESS_REPORT.md` | Link this specification; later append teacher selection, bank yield, SFT and full-task evaluation results with hashes and protocol. |

Do not duplicate production pickup/placement predicates. Reuse
`evaluate_active_sparse_tasks` with separately owned task states and document
additional full-chain data-quality conditions. The existing CPU environment
snapshot tests do not prove a persistent MJWarp continuation is complete.
Use `probe_cdpr_demonstration_handoff.py` as a reference for backend plus
contact/task-history handling, not as a ready three-stage recorder.

## 13. Implementation order and acceptance checks

1. **Scene and yaw contract.** Implement scene manifest/reset, calibration
   recording and yaw-tail controller. Tests: correct object/destination slots,
   realized distance rejection after workspace handling, zero initial grasp,
   yaw mount transform and limits, no hidden mid-episode pose writes. GPU
   inspection must confirm safe yaw motion and visible target/destination.
2. **One live chain.** Implement stage routing and observers. Tests: native
   pickup success does not terminate the full episode; held state/lift datum
   survives switches; queues change per world; boundary loss masks are correct;
   divergence is quarantined. Validate actual continuity on GPU and retain
   video/trace of complete plate and bowl chains.
3. **Teacher screening and bank.** Select donors from their actual handoffs,
   then collect bounded training-only scenes. A useful first target is 50
   unique accepted chains per destination for debugging, then expand by
   object/distance/yaw coverage. This is not enough by itself to claim robust
   learning. Stop at an explicit attempt/GPU-time budget and report missing
   strata rather than filling them with duplicated trajectories.
4. **Dataset and refresh.** Tests: no window crosses an episode/reset;
   teacher-boundary windows reference actual consecutive actions; every
   observation is pre-action; final-prompt IDs/text/priors agree; all frames
   resolve; scene splits are disjoint across retries and original/relabelled
   views; stage balancing changes sampling, not raw episode length.
5. **Bounded SFT.** Verify a small forward/backward batch uses the intended
   modules and valid masks, reports reachability and preserves checkpoint
   compatibility. Compare arms A/B/C on full-task validation at declared
   intervals; do not spend a long run to discover an invalid row/frame join.
6. **Held-out result.** Report counts, scene-level uncertainty, per-object and
   per-destination rates, completion action count, yaw/grasp/lift/release
   diagnostics, retention and videos. Separate teacher chain yield, assisted
   stage scores, native placement success and strict unassisted full-chain
   success. Only the last measures the user's complete requested behavior.

The immediate deliverable is this specification. The next coding work is
the full-task scene/yaw contract and a single continuous three-stage recorder,
followed by teacher screening. Final donor choices and a numeric yaw value
remain measurements to obtain on the remote server, not constants inferred
from historical headlines.
