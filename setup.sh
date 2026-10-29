#!/usr/bin/env bash
set -euo pipefail

MODEL_CHOICE=""
ASSUME_YES=0
while (($#)); do
  case "$1" in
    --model)
      shift
      [[ $# -gt 0 ]] || { echo "--model requires current or aggressive" >&2; exit 2; }
      MODEL_CHOICE="$1"
      ;;
    --yes|-y) ASSUME_YES=1 ;;
    --help|-h)
      cat <<'EOF'
Usage: setup.sh [--model current|aggressive] [--yes]

Installs a local Hugging Face CLI when needed, downloads the selected model
with progress reporting, builds llamAmpere (a llama.cpp fork tuned for the
RTX 3090) with CUDA, and creates a private API key. Existing files are left
untouched.
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${QWEN_TOOLS_DIR:-$HOME/.local/share/qwen3.8}"
LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/llamAmpere-src/v0.6}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-sm86/bin/llama-server}"
LLAMA_REPO="https://github.com/JakeATX/llamAmpere.git"
LLAMA_TAG="v0.6"
KEY_FILE="${QWEN_API_KEY_FILE:-$HOME/.config/qwen3.8/api-key}"

confirm() {
  local prompt="$1" answer
  if (( ASSUME_YES )) || [[ ! -t 0 ]]; then
    return 0
  fi
  printf '%s [Y/n] ' "$prompt"
  read -r answer
  [[ -z "$answer" || "$answer" =~ ^[Yy]$ ]]
}

choose_model() {
  [[ -n "$MODEL_CHOICE" ]] && return
  if [[ -t 0 ]]; then
    printf 'Choose a model to download:\n  1) Qwen3.8 27B · Unsloth UD-Q4_K_XL\n  2) HauhauCS Aggressive · Q4_K_P\n> '
    read -r answer
    case "$answer" in 2) MODEL_CHOICE=aggressive ;; *) MODEL_CHOICE=current ;; esac
  else
    MODEL_CHOICE=current
  fi
}
choose_model

case "$MODEL_CHOICE" in
  current)
    MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
    REPO="unsloth/Qwen3.8-27B-GGUF"
    FILES=(
      "Qwen3.8-27B-UD-Q4_K_XL.gguf"
      "mmproj-F16.gguf"
    )
    ;;
  aggressive)
    MODEL_DIR="${QWEN_AGGRESSIVE_MODEL_DIR:-$HOME/models/Qwen3.8-27B-Aggressive}"
    REPO="HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF"
    FILES=(
      "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
      "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    )
    ;;
  *) echo "Unknown model '$MODEL_CHOICE' (use current or aggressive)." >&2; exit 2 ;;
esac

missing_model=0
for file in "${FILES[@]}"; do
  [[ -r "$MODEL_DIR/$file" ]] || missing_model=1
done

HF="$(command -v hf || true)"
if (( missing_model )); then
  echo
  echo "The selected model is not downloaded yet."
  echo "Repository: $REPO"
  echo "Destination: $MODEL_DIR"
  echo "The download is about 18-20 GB and the Hugging Face CLI will show progress."
  confirm "Install the downloader if needed and start now?" || { echo "Setup cancelled."; exit 1; }

  if [[ -z "$HF" ]]; then
    command -v python3 >/dev/null || { echo "python3 is required to install the Hugging Face CLI." >&2; exit 1; }
    echo
    echo "Installing the Hugging Face CLI in $TOOLS_DIR/venv ..."
    install -d "$TOOLS_DIR"
    if ! python3 -m venv "$TOOLS_DIR/venv"; then
      echo "Could not create a Python virtual environment. Install python3-venv and retry." >&2
      exit 1
    fi
    "$TOOLS_DIR/venv/bin/python" -m pip install --upgrade huggingface_hub
    HF="$TOOLS_DIR/venv/bin/hf"
  fi

  echo
  echo "Downloading model files (safe to resume if interrupted) ..."
  "$HF" download "$REPO" "${FILES[@]}" --local-dir "$MODEL_DIR"
  echo "Model download complete."
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo
  echo "llama-server was not found at $LLAMA_SERVER."
  command -v git >/dev/null || { echo "git is required to build llamAmpere." >&2; exit 1; }
  command -v cmake >/dev/null || { echo "cmake is required to build llamAmpere." >&2; exit 1; }
  command -v nvcc >/dev/null || { echo "The NVIDIA CUDA toolkit (nvcc) is required." >&2; exit 1; }
  echo "The build takes 1-3 hours; one kernel file alone needs about 8.5 GB of RAM."
  confirm "Clone and build llamAmpere $LLAMA_TAG with CUDA now?" || { echo "Setup cancelled."; exit 1; }

  if [[ ! -d "$LLAMA_DIR/.git" && ! -f "$LLAMA_DIR/.git" ]]; then
    git clone --depth 1 -b "$LLAMA_TAG" "$LLAMA_REPO" "$LLAMA_DIR"
  fi
  # CUDA 12.0's ptxas rejects two opt-in kernel instances; harmless on newer toolkits.
  patch_file="$SCRIPT_DIR/patches/llamampere-v0.6-cuda12.0.patch"
  if git -C "$LLAMA_DIR" apply --check "$patch_file" 2>/dev/null; then
    git -C "$LLAMA_DIR" apply "$patch_file"
  fi
  jobs="${QWEN_BUILD_JOBS:-2}"
  echo "Building llama-server with CUDA using $jobs parallel jobs ..."
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build-sm86" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
  cmake --build "$LLAMA_DIR/build-sm86" --target llama-server -j "$jobs"
fi

if [[ ! -s "$KEY_FILE" ]]; then
  echo
  echo "Generating a private API key ..."
  command -v openssl >/dev/null || { echo "openssl is required to generate the API key." >&2; exit 1; }
  install -d -m 700 "$(dirname "$KEY_FILE")"
  umask 077
  openssl rand -hex 32 >"$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "API key saved to $KEY_FILE (do not commit or share it)."
fi

if command -v nvidia-smi >/dev/null; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
  [[ -z "$gpu_name" ]] || echo "Detected GPU: $gpu_name"
else
  echo "Warning: nvidia-smi was not found; the CUDA server may not start." >&2
fi

echo
if (( missing_model )); then
  echo "Onboarding complete."
else
  echo "Setup already complete for the '$MODEL_CHOICE' model."
fi
