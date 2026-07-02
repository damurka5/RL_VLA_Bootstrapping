# CDPR Octo-Small Dense Runbook

This path adds an Octo-Small alternative beside the existing OpenVLA pipeline. It uses pretrained `hf://rail-berkeley/octo-small-1.5`, freezes Octo, and trains only a small Torch residual/readout adapter plus critics against `CDPRLanguageRLEnv`.

The setup follows the upstream Octo install pattern: create an `octo` Python 3.10 environment, install the Octo repo editable, install its requirements, then install JAX GPU wheels. The Octo README shows `OctoModel.load_pretrained(...)`, `create_tasks(...)`, and `sample_actions(...)`, and notes that Octo 1.5 predicts action chunks.

## One-Time Remote Setup

If an older setup attempt is stuck at `Solving environment`, stop it with `Ctrl+C`. The Octo env file is intentionally minimal so conda only creates Python/pip; GPU frameworks are installed with pip to avoid a large conda SAT solve.

```bash
cd /root/repo/RL_VLA_Bootstrapping
git pull origin main

conda env remove -n octo -y || true
conda env create -f environments/octo-remote.yaml
conda run --no-capture-output -n octo python -m pip install --upgrade pip setuptools wheel
conda run --no-capture-output -n octo python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121 \
  torchaudio==2.3.1+cu121
conda run --no-capture-output -n octo python -m pip install \
  gym==0.26.2 \
  gym-notices==0.1.0 \
  mujoco==3.4.0 \
  opencv-python-headless \
  imageio \
  imageio-ffmpeg \
  huggingface_hub \
  numpy \
  pillow \
  pyyaml \
  tqdm \
  tensorboard

cd /root/repo
git clone https://github.com/octo-models/octo.git || true
cd /root/repo/octo
conda run --no-capture-output -n octo python -m pip install -e .
conda run --no-capture-output -n octo python -m pip install -r requirements.txt
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-deps \
  numpy==1.24.3 \
  ml-dtypes==0.2.0 \
  protobuf==4.25.3 \
  tensorflow-metadata==1.15.0 \
  tensorflow-datasets==4.9.2 \
  scipy==1.11.4 \
  transformers==4.34.1 \
  tokenizers==0.14.1 \
  huggingface-hub==0.17.3
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir \
  "nvidia-cudnn-cu12>=8.9.4,<9"
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-deps \
  numpy==1.24.3 \
  ml-dtypes==0.2.0 \
  scipy==1.11.4

cd /root/repo/RL_VLA_Bootstrapping
OCTO_REPO_PATH=/root/repo/octo conda run --no-capture-output -n octo python -c "import octo, importlib; print(octo, getattr(octo, '__file__', None), getattr(octo, '__path__', None)); import octo.model.octo_model as m; print(m.OctoModel)"
OCTO_REPO_PATH=/root/repo/octo PYTHONPATH=/root/repo/octo JAX_PLATFORMS=cuda \
  env -u LD_LIBRARY_PATH conda run --no-capture-output -n octo python -c "import numpy, jax; print('numpy', numpy.__version__); print(jax.default_backend()); print(jax.devices())"
conda run --no-capture-output -n octo python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
conda run --no-capture-output -n octo python -m rl_vla_bootstrapping.cli.train \
  --config configs/examples/cdpr_octo_small_dense_simple.yaml \
  --stage rl
```

The last command is a dry plan check. It should print the Octo RL command without downloading Octo weights.
For the default remote config, the RL command should include
`torchrun --standalone --nproc-per-node 2`.

The JAX check must print `gpu` and at least one GPU device. Keep `LD_LIBRARY_PATH` unset
for the normal pip-wheel path; a stale system CUDA/cuDNN path can override the wheel
libraries and produce errors such as `Found cuDNN version 0`.

If it prints `cpu` or reports `_ARRAY_API not found`, `Found cuDNN version 0`, or leaves
`numpy 2.x`, first inspect the installed JAX/cuDNN packages:

```bash
conda run --no-capture-output -n octo python -m pip show jax jaxlib nvidia-cudnn-cu12
conda run --no-capture-output -n octo python -c "import glob, site; root=site.getsitepackages()[0]; print(root); print('\n'.join(sorted(glob.glob(root + '/nvidia/cudnn/lib/libcudnn*'))))"
```

Then reinstall JAX and re-pin Octo's NumPy stack:

```bash
cd /root/repo/RL_VLA_Bootstrapping
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir \
  "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir \
  "nvidia-cudnn-cu12>=8.9.4,<9"
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-deps \
  numpy==1.24.3 \
  ml-dtypes==0.2.0 \
  scipy==1.11.4
OCTO_REPO_PATH=/root/repo/octo PYTHONPATH=/root/repo/octo JAX_PLATFORMS=cuda \
  env -u LD_LIBRARY_PATH conda run --no-capture-output -n octo python -c "import numpy, jax; print('numpy', numpy.__version__); print(jax.default_backend()); print(jax.devices())"
```

If the check still reports `Found cuDNN version 0`, try the explicit pip-library fallback
once. This should be needed only on hosts where the loader cannot discover the pip
NVIDIA wheels on its own:

```bash
export LD_LIBRARY_PATH="$(conda run --no-capture-output -n octo python -c 'import glob, os, site; root=site.getsitepackages()[0]; dirs=[root + "/nvidia/cudnn/lib"] + sorted(glob.glob(root + "/nvidia/*/lib")); print(":".join(dict.fromkeys(d for d in dirs if os.path.isdir(d))))')"
OCTO_REPO_PATH=/root/repo/octo PYTHONPATH=/root/repo/octo JAX_PLATFORMS=cuda \
  conda run --no-capture-output -n octo python -c "import numpy, jax; print('numpy', numpy.__version__); print(jax.default_backend()); print(jax.devices())"
```

If `nvidia-smi` reports only a CUDA 11 runtime/driver, replace `cuda12_pip` with
`cuda11_pip` in the JAX install command above. Most A40 servers with recent drivers
should use `cuda12_pip`.

If the JAX check passes but training reports that PyTorch is missing, inspect the real
Torch import error:

```bash
conda run --no-capture-output -n octo python -c '
import traceback
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
except Exception:
    traceback.print_exc()
    raise
'
```

If that import fails, reinstall the Torch CUDA wheels first, then reinstall JAX so its
CUDA 12.2 runtime dependencies win. Do not run an unpinned Torch reinstall after this
JAX step; it can downgrade `nvidia-cuda-runtime-cu12` to CUDA 12.1 and make JAX report
`Found CUDA version 12010`.

```bash
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121 \
  torchaudio==2.3.1+cu121
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir \
  "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-cache-dir \
  "nvidia-cudnn-cu12>=8.9.4,<9"
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-deps \
  numpy==1.24.3 \
  ml-dtypes==0.2.0 \
  scipy==1.11.4
conda run --no-capture-output -n octo python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
OCTO_REPO_PATH=/root/repo/octo PYTHONPATH=/root/repo/octo JAX_PLATFORMS=cuda \
  conda run --no-capture-output -n octo python -c "import numpy, jax; print('numpy', numpy.__version__); print(jax.default_backend()); print(jax.devices())"
```

If the import check says `'octo' is not a package`, Python is seeing a wrong package or module named `octo`. Check and repair with:

```bash
conda run --no-capture-output -n octo python -c "import octo; print(getattr(octo, '__file__', None), getattr(octo, '__path__', None))"
conda run --no-capture-output -n octo python -m pip show octo || true
conda run --no-capture-output -n octo python -m pip uninstall -y octo || true
cd /root/repo/octo
conda run --no-capture-output -n octo python -m pip install -e .
conda run --no-capture-output -n octo python -m pip install --force-reinstall --no-deps \
  numpy==1.24.3 \
  ml-dtypes==0.2.0 \
  protobuf==4.25.3 \
  tensorflow-metadata==1.15.0 \
  tensorflow-datasets==4.9.2 \
  scipy==1.11.4 \
  transformers==4.34.1 \
  tokenizers==0.14.1 \
  huggingface-hub==0.17.3
cd /root/repo/RL_VLA_Bootstrapping
OCTO_REPO_PATH=/root/repo/octo conda run --no-capture-output -n octo python -c "import octo.model.octo_model as m; print(m.OctoModel)"
```

## Start a 24-Hour Run

```bash
cd /root/repo/RL_VLA_Bootstrapping
RUN_NAME=cdpr_octo_small_dense_$(date +%Y%m%d_%H%M%S) \
WALLTIME=24h \
bash scripts/train_cdpr_octo_small_dense_remote.sh
```

By default this launcher uses `CUDA_VISIBLE_DEVICES=0,1`, two torchrun ranks,
`XLA_PYTHON_CLIENT_PREALLOCATE=true`, and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
Each rank sets `JAX_VISIBLE_DEVICES` to its local rank before importing JAX, so the
frozen Octo runtime and Torch residual head are bound to one GPU per process.

The default training config also sets `replan_every: 4` with `chunk_size: 4`, so Octo
samples one action chunk and executes the full chunk before the next Octo prior call.
This reduces camera capture, image preprocessing, JAX dispatch, and CPU/GPU transfer
pressure. The compiled MuJoCo model cache is enabled with
`RLVLA_CDPR_COMPILED_MODEL_CACHE_MAX_SIZE=16` to reduce reset-time model compilation.

This high VRAM allocation is mostly JAX preallocation. It proves the process owns the
GPU memory, but it is not the same as full Octo model fine-tuning. The current training
surface still freezes Octo and trains the Torch residual/readout head plus critics. If
Torch reports OOM, retry with a lower fraction, for example:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 bash scripts/train_cdpr_octo_small_dense_remote.sh
```

Training output is progress-bar first. `RLVLA_CDPR_QUIET=1` and
`RLVLA_CDPR_WRAPPER_LOG=0` suppress cached-wrapper and simulator reset chatter.
Set `RLVLA_CDPR_QUIET=0 RLVLA_CDPR_WRAPPER_LOG=1` if you need verbose wrapper debugging.

The Octo observation path uses live camera frames, not saved videos. The overview camera
is the primary Octo image, and `include_wrist: true` also feeds the end-effector wrist
camera.

The remote wrappers clear `LD_LIBRARY_PATH` by default before importing JAX. If the
explicit pip-library fallback above is the only check that works on a given host, launch
training with `JAX_PREPEND_PIP_NVIDIA_LIBS=1`.

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
