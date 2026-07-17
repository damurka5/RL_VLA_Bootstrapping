#!/usr/bin/env bash

# SmolVLA and its SmolVLM2 backbone are public. By default, prevent a stale
# credential inherited from the remote shell (or cached by huggingface_hub)
# from turning an anonymous model download into an authentication failure.
configure_huggingface_public_models() {
  case "${RLVLA_HF_PUBLIC_MODELS_ONLY:-1}" in
    1)
      unset HF_TOKEN
      unset HUGGING_FACE_HUB_TOKEN
      export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
      ;;
    0)
      ;;
    *)
      echo "RLVLA_HF_PUBLIC_MODELS_ONLY must be 0 or 1." >&2
      return 2
      ;;
  esac
}

huggingface_public_models_preflight() {
  local env_name="${1:-none}"
  local python_cmd=()

  if [[ "${RLVLA_HF_PREFLIGHT:-1}" != "1" ]]; then
    return 0
  fi

  if [[ -z "$env_name" || "$env_name" == "none" ]]; then
    python_cmd=(python3)
  else
    python_cmd=(conda run --no-capture-output -n "$env_name" python3)
  fi

  printf '[huggingface] checking anonymous access to public SmolVLA models\n'
  if ! "${python_cmd[@]}" -c '
from huggingface_hub import HfApi

api = HfApi()
for repo_id in (
    "lerobot/smolvla_base",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
):
    info = api.model_info(repo_id, token=False)
    print(f"[huggingface] anonymous access OK: {info.id}")
'; then
    cat >&2 <<'EOF'
[huggingface] Public-model preflight failed.
Check outbound access to huggingface.co. If you intentionally use a private or
gated checkpoint, set RLVLA_HF_PUBLIC_MODELS_ONLY=0, authenticate with
`hf auth login`, and rerun. Set RLVLA_HF_PREFLIGHT=0 only for an offline run
whose complete model files are already cached.
EOF
    return 1
  fi
}
