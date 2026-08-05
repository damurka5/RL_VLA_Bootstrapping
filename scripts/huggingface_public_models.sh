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

# Run entirely from the local cache. For a box that has lost outbound access to
# huggingface.co but already holds the model files -- the usual case on a
# long-lived training host, where every previous run populated ~/.cache.
#
# Setting RLVLA_HF_OFFLINE=1 exports both offline switches AND skips the
# preflight, because a reachability check is exactly what cannot pass here.
# Without HF_HUB_OFFLINE the preflight can be skipped and the run still dies
# later inside AutoProcessor.from_pretrained, which reaches the network on its
# own and reports the resulting failure as a missing repository.
configure_huggingface_offline() {
  case "${RLVLA_HF_OFFLINE:-0}" in
    0)
      ;;
    1)
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      export RLVLA_HF_PREFLIGHT=0
      printf '[huggingface] offline: using the local cache only\n'
      ;;
    *)
      echo "RLVLA_HF_OFFLINE must be 0 or 1." >&2
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
Check outbound access to huggingface.co. If the host simply has no route out but
the model files are already cached -- the usual case on a long-lived training
box -- rerun with RLVLA_HF_OFFLINE=1, which pins huggingface_hub and transformers
to the local cache and skips this check. If you intentionally use a private or
gated checkpoint, set RLVLA_HF_PUBLIC_MODELS_ONLY=0, authenticate with
`hf auth login`, and rerun. RLVLA_HF_PREFLIGHT=0 alone only silences this check;
it does not stop the loader reaching the network later.
EOF
    return 1
  fi
}
