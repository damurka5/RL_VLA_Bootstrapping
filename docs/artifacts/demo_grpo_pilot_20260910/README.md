# September 10 demonstration-start pilot attachment review

Interpretation and next experiment are in the newest September 10 entry of
[`CDPR_CONSOLIDATED_PROGRESS_REPORT.md`](../../../CDPR_CONSOLIDATED_PROGRESS_REPORT.md).

These are derived CPU analysis artifacts. Original recordings and logs remain
in the supplied Downloads directories; their SHA-256 values are recorded in
`attachment_analysis.json`. No weights, simulator rollouts or remote execution
are included in this review.

From the repository root, reproduce the summary (NumPy required):

```bash
python3 docs/artifacts/demo_grpo_pilot_20260910/analyze.py /Users/damirnurtdinov/Downloads
```

Reproduce the two existing predicate decomposition reports (NumPy and PyYAML):

```bash
python3 tools/audit/placement_failure_decomposition.py \
  --recordings '/Users/damirnurtdinov/Downloads/baseline/record_*.npz' \
  --config configs/examples/cdpr_smolvla_demo_grpo_pilot.yaml \
  --output docs/artifacts/demo_grpo_pilot_20260910/baseline_decomposition
python3 tools/audit/placement_failure_decomposition.py \
  --recordings '/Users/damirnurtdinov/Downloads/final_eval/record_*.npz' \
  --config configs/examples/cdpr_smolvla_demo_grpo_pilot.yaml \
  --output docs/artifacts/demo_grpo_pilot_20260910/final_decomposition
```

Both decompositions reproduced every container success verdict (zero
disagreements). Summary generation verified all CSV/NPZ success counts against
the supplied summaries and exact identity of the compared reset fields.
The local config was used for decomposition; the remote config snapshot and
git revision have not been supplied.

To inspect the missing assisted collection evidence, run this **on the remote
server** and paste the output. It reads existing files and does not train:

```bash
cd /root/repo/RL_VLA_Bootstrapping && python3 - <<'PY'
from collections import Counter, defaultdict
import json
from pathlib import Path

run = Path('runs/demo_grpo_pilot_20260910_110040')
for name in ('pilot_manifest.json', 'pilot_comparison.json'):
    path = run / name
    print(name, path.read_text() if path.exists() else 'MISSING')
paths = sorted((run / 'rl').glob('demonstration_rank*.jsonl'))
if not paths:
    print('MISSING demonstration_rank*.jsonl')
for path in paths:
    totals = defaultdict(Counter)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        group = totals[(row['family'], row['stage'])]
        group['attempts'] += 1
        for key in ('accepted_clean_groups', 'usable_clean_groups',
                    'suffix_loss_rows', 'suffix_vla_nonzero_advantage_rows',
                    'teacher_prefix_loss_rows', 'prefix_replayed_active_actions'):
            group[key] += row.get(key, 0)
    for key, counts in sorted(totals.items()):
        print(path.name, key, dict(counts))
PY
```

Suffix record counts in these collection logs precede some filtering and
optimizer sampling; they are not proof that every reported row entered an
optimizer step. Actual gradient coverage additionally needs training metrics /
TensorBoard events. Missing row fields are printed as zero by this compact
inspection; inspect the raw logs if the schema differs.
