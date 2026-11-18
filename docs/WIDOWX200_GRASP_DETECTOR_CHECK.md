# Check whether a real grasp is counted

Textures do not enter the grasp predicate. Collision pads, solved contact
forces, relative-pose stability, and persistence do. Visual/collision geometry
differences can make the video misleading. The earlier opening-based veto also
rejected wide objects that physically held the fingers open; contact-v2 removes
that veto. The low aggregate success rate cannot establish whether a remaining
grasp rejection is correct.

The trace audit observes the existing production detector and task predicate,
without recomputing physics or changing any returned result. Every executed
environment step stores left/right target-pad contact, solved forces, relative
position/orientation slip, eligibility, persistence count, physical grasp and
the grasp actually accepted by task scoring. Thresholds and source-file hashes
are saved alongside the measurements. Videos show the corresponding gates in
a footer; the reset and later post-rollout view are explicitly labeled.
The trace also reads the existing contact buffer before the backend's 2 mm
distance filter: minimum reported pad/target contact distance, raw solved pad
force, and force discarded by that filter. This performs no second solver
call. A missing distance entry means no reported contact, rather than a
measurement of the general separation between the surfaces.

## Record a check on the GPU server

This evaluates the existing policy. It starts no training and does not modify
checkpoints, rewards or thresholds. The continuation manifest supplies the
original composed evaluation config and distance cap. Rounds 1000–1001 provide
fresh evaluation resets. All worlds have telemetry; videos sample 32 distinct
reset groups per round, prioritizing suspicious cases within that sample.

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull --ff-only origin main
GRASP_CHECK="runs/widowx_grasp_check_$(date +%Y%m%d_%H%M%S)"
RLVLA_HF_PUBLIC_MODELS_ONLY=1 conda run --no-capture-output -n cdpr-mjlab \
  python tools/audit/grasp_detector_trace.py \
  --run-dir runs/widowx200_s7_contact_v2_20261113_135223 \
  --output "$GRASP_CHECK"
```

The default policy is that run's `rl/latest.pt`. To inspect the earlier selected
best instead, add:

```bash
--checkpoint runs/widowx200_s7_contact_v2_20261113_135223/rl/step_0502404/smolvla_grpo_adapter.pt
```

## Read the evidence

Open `videos/*.mp4` and `grasp_audit/grasp_audit.json`. The report includes:

- `all_gates_passed_but_uncounted_steps`: the recorded production gates imply
  a grasp, but the detector returned false. This indicates a detector mismatch.
- `scoring_mismatch_steps`: scoring differs from the recorded grasp contract.
  Under contact-v2, physical grasp true and scored grasp false is a scoring bug.
- `wide_loaded_counted_steps`: wide physical grasps that were accepted. Along
  with zero mismatches, these demonstrate that opening above 0.94 is not vetoing
  those sampled grasps. `opening_v1_would_reject_steps` shows how many of those
  same frames the old opening rules would reject.
- `motion_proxy_unrecognized_steps`: the object is at least 5 cm above its
  initial height, within 4 cm of the tool, and co-moves for three steps without
  a counted grasp. This is a suspicion flag, not proof of a bilateral grasp.
- `nonfinite_force_steps` and `live_diverged`: distinguish invalid contact data
  and simulation divergence from a valid stable grasp that was mis-scored.
- `loaded_contacts_outside_distance_window_steps`: a solved pad contact carried
  force above threshold outside the backend's contact-distance window while the
  grasp was unrecognized. Inspect its signed gap and rejected force in the
  step CSV/video before changing the distance condition.

`grasp_audit/episodes.csv` covers every world. Full step CSVs are saved for up to
16 suspicious episodes per round. They distinguish missing bilateral contact,
force below threshold, position slip, orientation slip, persistence warm-up,
and legacy opening rejection. Mere lift or visual proximity never relabels a
failure as success. Episodes that diverged while live are excluded from the
missed-grasp evidence; idle divergence after completion is treated separately.

If both loaded pads remain stable for the required two steps and both grasp
flags are true, a subsequent task failure is after grasp detection. If contact
or force is missing despite apparent visual contact, inspect the collision
geometry/alignment. If loaded contact is rejected by slip or rotation, inspect
that trace before changing the thresholds.

## Analyze downloaded files without GPU

Keep `record_*.npz` and `grasp_trace_*.npz` together. Trace fingerprints must
match their recordings; older recordings without gate traces cannot establish
which gate rejected a grasp.

```bash
python tools/audit/grasp_detector_trace.py \
  --recordings '/path/to/downloaded/record_*.npz' --output /tmp/grasp_gate_report
```

This audit measures new deterministic evaluation rollouts under the current
code. It cannot reconstruct the missing contact signals from the historical
training updates with NaN force means, or establish stochastic training
stability from deterministic evaluation alone.
