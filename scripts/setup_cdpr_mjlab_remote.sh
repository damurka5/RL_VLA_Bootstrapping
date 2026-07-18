#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
ENV_NAME="${ENV_NAME:-cdpr-mjlab}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/examples/cdpr_smolvla_complex_reverse_frontier_grpo_mjlab.yaml}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repository not found: $REPO_ROOT" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required on the remote CUDA host." >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Install Miniconda/Mambaforge before running this script." >&2
  exit 2
fi

if [[ "$INSTALL_APT_DEPS" == "1" ]]; then
  apt_cmd=(apt-get)
  if [[ "$(id -u)" != "0" ]]; then
    apt_cmd=(sudo apt-get)
  fi
  "${apt_cmd[@]}" update
  "${apt_cmd[@]}" install -y \
    build-essential \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libosmesa6 \
    ninja-build \
    patchelf
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda install -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

python_cmd=(conda run --no-capture-output -n "$ENV_NAME" python3)
"${python_cmd[@]}" -m pip install --upgrade pip setuptools wheel
"${python_cmd[@]}" -m pip install \
  --requirement "$REPO_ROOT/requirements/cdpr-mjlab-cu128.lock.txt"
"${python_cmd[@]}" -m pip install --no-deps -e "$REPO_ROOT"
"${python_cmd[@]}" -m pip check
if [[ ! -f "$REPO_ROOT/assets/externals/robocasa/objects/objaverse/apple/apple_20/visual/model_normalized_0.obj" ]]; then
  "${python_cmd[@]}" "$REPO_ROOT/scripts/stage_cdpr_robocasa_assets.py"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  cd "$REPO_ROOT"
  "${python_cmd[@]}" scripts/preflight_cdpr_mjlab.py \
    --config "$CONFIG" \
    --require-gpus 2 \
    --worlds 16 \
    --output runs/mjlab_preflight.json
fi

echo "Pinned CDPR MJLab environment '$ENV_NAME' is ready."
