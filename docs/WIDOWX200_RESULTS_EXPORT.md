# Exporting WidowX-200 pipeline evidence

Run from the repository in another SSH shell while the pipeline continues:

```bash
python3 tools/audit/export_widowx200_pipeline_results.py \
  --pipeline-dir runs/widowx200_pipeline_20261024_131201
```

This uses only the Python standard library. It creates a new timestamped
`results_snapshot_*` directory inside the pipeline directory, plus a ZIP beside
it, and prints the summary tables and archive path. `--output PATH` selects a
different new directory. Existing output directories/archives are not replaced.
No training state or source artifact is modified, and no GPU is used.

The ZIP contains numerical logs and configuration/report files. It excludes
checkpoints, NPZ trajectory/image arrays, videos and TensorBoard event files.

| File | Evidence |
|---|---|
| `report.md` | Stage selection scores, retained banks, retention evaluations |
| `stages.csv` | Best RL checkpoint, handed-off policy, stop rule/reason, SIL attempts/acceptances |
| `validation.csv` | All logged validation metrics and steps |
| `validation_slices.csv` | Instruction/object curves, episode counts, fixed-distance flags, composed validation legs |
| `training.csv` | Logged training/curriculum/throughput diagnostics |
| `retention.csv` | Before-SIL, after-SIL and candidate evaluations, with acceptance flags |
| `banks.csv` | Currently retained source stages, checkpoints, harvest counts and file inventory |
| `record_rounds.csv` | Per-round harvest/evaluation counts; sharded pooled wrappers are not counted twice |
| `replay_survival.csv` | Successful smoothed replays among source successes |
| `raw/` | Captured source state, stage manifests, configs, JSON/JSONL reports and episode CSVs |
| `manifest.json` | Source paths, SHA-256 digests, capture timestamp and parsing warnings |

The snapshot is not globally atomic: training may advance between file reads.
Each file is read once, and the tables use the same bytes stored in `raw/`.
A partially appended JSONL row is excluded from tables and identified in the
manifest; its original bytes remain in the archive. The current source YAML
can differ from launch configuration; use captured per-stage manifests/configs
for the actual stage protocol.

## Interpretation for a paper

- A stage's `best_rl_value` is its validation selection metric. It may be the
  minimum across instructions, rather than overall success.
- The policy handed to the next stage may be an accepted SIL candidate. Its
  evaluation belongs to the retention/candidate results, not the preceding RL
  checkpoint's validation row.
- A retained bank may come from an earlier stage when a later harvest is empty
  or weaker. `banks.csv` describes the retained choice; raw reports also retain
  the other harvests.
- Held-object placement, empty-start composed placement, training success and
  conditional replay survival are distinct quantities.
- Reused validation seeds and successive checkpoints are not independent
  training seeds. Best-checkpoint validation is not a final held-out test.

## Current SIL ordering

The automated driver harvests skill trajectories with executed actions and
camera frames. Conditional retention SFT pools the retained banks with their
**original instruction labels** and refreshes network-dependent priors/features
before fitting the residual. Its candidate is evaluated and accepted/rejected
by the retention gate.

The updated configuration enables a separate **composition SFT** transition
before stage 7. It pools verified filmed skill samples from the retained banks,
assigns final plate/bowl prompts to reach/pick prefixes, preserves actual
placement destinations, and keeps an original-instruction retention slice.
Priors and features are refreshed under the stage-6 initializer before residual
and action-expert LoRA fitting. Before/after skill and composed evaluations are
recorded; stage 7 then starts fresh joint staged-reward RL from the SFT adapter.
Earlier runs may not have this transition; the exported state says whether it
actually ran, rather than inferring it from the current source configuration.

Merely pooling independent skill trajectories does not certify a continuous
successful grasp-carry-release episode. Prefix scenes stay as originally
recorded, and destination visibility is not certified for reach/pick samples.
The builder reports these as stage supervision, not complete-task successes.

See [WIDOWX200_COMPOSITION_SFT.md](WIDOWX200_COMPOSITION_SFT.md) for the stop/resume
procedure and interrupted-joint-run preservation. The export includes the
composition reports and before/after evaluations. Preserved old joint logs are
stored under `raw/archived_joint/`, separate from the corrected joint curves.
