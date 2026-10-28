# 60M continuation: retained TensorBoard evidence

Reviewed October 28, 2026; recorded October 19–23 (Europe/Moscow).
These artifacts describe **in-run validation**, not a separate checkpoint
evaluation. Endpoint strict: **723/1024 = 70.61%**; panel peak:
**728/1024 = 71.09%**, selected from 81 readings.

- `summary.json`: source hash, exact Event.step counters, initial/peak/final
  snapshots, training totals, health coverage and limitations.
- `validation.csv`: all 81 readings, with every `validation/` scalar. Actual
  episode counts are 1,024; the requested-episodes scalar is not the denominator.
- `training.csv`: 323 update readings for optimization and simulator health.
- `training_review.png`: unsmoothed panel curves. Failure-event flags can overlap.
- `extract_metrics.py`: reproducible extraction with scalar sampling disabled
  and consistency assertions across all validation readings.

Source filename: `events.out.tfevents.1792438865.VLAPU.666564.0`.
SHA-256: `7e22bff52be5d0da864e1df940b0aece07265436526f1aac0a6e7ae5cf0e8e73`.
The original event file is user-supplied and is not duplicated in Git.
No checkpoint hash, run directory or scene-level outcomes were supplied.

From the repository root, with TensorBoard and matplotlib installed:

```bash
python3 docs/artifacts/latent_control_60m_20261028/extract_metrics.py \
  /path/to/events.out.tfevents.1792438865.VLAPU.666564.0
```

The [review](../../reports/campaign/CDPR_60M_CONTROL_TENSORBOARD_REVIEW_20261028.md)
explains the measurement scope. The last separate development result remains
the 40M control's 61.43% (1,258/2,048); no matched gain over it is established.
