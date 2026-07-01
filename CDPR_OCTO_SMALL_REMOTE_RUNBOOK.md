# CDPR Octo-Small Dense Runbook

This path adds an Octo-Small alternative beside the existing OpenVLA pipeline. It uses pretrained `hf://rail-berkeley/octo-small-1.5`, freezes Octo, and trains only a small Torch residual/readout adapter plus critics against `CDPRLanguageRLEnv`.

The setup follows the upstream Octo install pattern: create an `octo` Python 3.10 environment, install the Octo repo editable, install its requirements, then install JAX GPU wheels. The Octo README shows `OctoModel.load_pretrained(...)`, `create_tasks(...)`, and `sample_actions(...)`, and notes that Octo 1.5 predicts action chunks.

## One-Time Remote Setup

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull origin main

conda env create -f environments/octo-remote.yaml || conda env update -n octo -f environments/octo-remote.yaml --prune

cd /root/repo
git clone https://github.com/octo-models/octo.git || true
cd /root/repo/octo
conda run --no-capture-output -n octo python -m pip install --upgrade pip
conda run --no-capture-output -n octo python -m pip install -e .
conda run --no-capture-output -n octo python -m pip install -r requirements.txt
conda run --no-capture-output -n octo python -m pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n octo python -m rl_vla_bootstrapping.cli.train \
  --config configs/examples/cdpr_octo_small_dense_simple.yaml \
  --stage rl
```

The last command is a dry plan check. It should print the Octo RL command without downloading Octo weights.

## Start a 24-Hour Run

```bash
cd /root/repo/RL_VLA_Bootstrapping
RUN_NAME=cdpr_octo_small_dense_$(date +%Y%m%d_%H%M%S) \
WALLTIME=24h \
bash scripts/train_cdpr_octo_small_dense_remote.sh
```

Logs and checkpoints go under:

```bash
runs/<RUN_NAME>/train.log
runs/<RUN_NAME>/rl/latest.pt
runs/<RUN_NAME>/rl/step_*/octo_cdpr_adapter.pt
```

## Identify Latest Checkpoint

```bash
cd /root/repo/RL_VLA_Bootstrapping
find runs -path '*/rl/latest.pt' -print | sort | tail -n 1
```

## Resume

```bash
cd /root/repo/RL_VLA_Bootstrapping
RESUME_CHECKPOINT=/root/repo/RL_VLA_Bootstrapping/runs/<RUN_NAME>/rl/latest.pt \
RUN_NAME=<RUN_NAME> \
WALLTIME=24h \
bash scripts/train_cdpr_octo_small_dense_remote.sh
```

## Stop

Prefer a graceful interrupt from the terminal running the script. From another shell:

```bash
pkill -INT -f octo_finetune_cdpr.py
```

If the process does not exit, use the remote job manager or a stronger signal.

## Evaluate

```bash
cd /root/repo/RL_VLA_Bootstrapping
CHECKPOINT_DIR=/root/repo/RL_VLA_Bootstrapping/runs/<RUN_NAME>/rl \
EPISODES_PER_INSTRUCTION=20 \
MOVE_TO_OBJECT_EPISODES_PER_TARGET=20 \
bash scripts/evaluate_cdpr_octo_small_dense_remote.sh
```

If `CHECKPOINT_DIR` is omitted, the evaluation launcher uses the newest `runs/*/rl/latest.pt`.

Evaluation writes:

```bash
runs/cdpr_octo_small_dense_evaluations/*/validation_manifest.json
runs/cdpr_octo_small_dense_evaluations/*/episode_results.csv
runs/cdpr_octo_small_dense_evaluations/*/instruction_success_rates.csv
runs/cdpr_octo_small_dense_evaluations/*/normal_scene_canonical_success_rates.csv
runs/cdpr_octo_small_dense_evaluations/*/validation_report.md
```

## Success Threshold

The recent OpenVLA simple baseline to beat is:

- Overall simple success: `50/300 = 16.7%`
- `move_to_object`: `9/100 = 9%`

Octo-Small is meaningfully better if the simple evaluation reports `>16.7%` overall success and `>9%` `move_to_object` success.
