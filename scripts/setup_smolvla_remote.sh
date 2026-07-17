#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/repo/RL_VLA_Bootstrapping}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/configs/examples/cdpr_smolvla_dense_2gpu.yaml}"
ENV_NAME="${ENV_NAME:-smolvla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
RUN_ASSET_STAGE="${RUN_ASSET_STAGE:-1}"
RUN_DOCTOR="${RUN_DOCTOR:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/huggingface_public_models.sh
source "$SCRIPT_DIR/huggingface_public_models.sh"
configure_huggingface_public_models

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repo root does not exist: $REPO_ROOT" >&2
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
    libsm6 \
    libxext6 \
    libxrender1 \
    ninja-build \
    patchelf
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required on the remote host. Install Miniconda or Mambaforge first." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda install -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  --index-url "$PYTORCH_INDEX_URL" \
  torch torchvision torchaudio
conda run --no-capture-output -n "$ENV_NAME" python -m pip install \
  "lerobot[smolvla]" \
  accelerate \
  gym \
  gymnasium \
  huggingface_hub \
  "imageio[ffmpeg]" \
  kornia \
  mujoco \
  opencv-python-headless \
  safetensors \
  tensorboard \
  tqdm \
  transformers \
  "PyYAML>=6.0"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$REPO_ROOT"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_ROOT"

huggingface_public_models_preflight "$ENV_NAME"

if [[ "$RUN_ASSET_STAGE" == "1" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" \
    python3 -m rl_vla_bootstrapping.cli.assets --config "$CONFIG_PATH" --stage
fi

if [[ "$RUN_DOCTOR" == "1" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" \
    python3 -m rl_vla_bootstrapping.cli.doctor --config "$CONFIG_PATH"
fi

conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(idx, props.name, round(props.total_memory / 1024**3, 2), "GiB")
PY

echo "SmolVLA remote environment '$ENV_NAME' is ready for $CONFIG_PATH"
