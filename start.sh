#!/usr/bin/env bash
set -euo pipefail

MONITOR=1
SHARE=0
MODEL_CHOICE=""
while (($#)); do
  case "$1" in
    --detach|--no-monitor) MONITOR=0 ;;
    --share) SHARE=1 ;;
    --model)
      shift
      [[ $# -gt 0 ]] || { echo "--model requires current or aggressive" >&2; exit 2; }
      MODEL_CHOICE="$1"
      ;;
    --help|-h)
      cat <<'EOF'
Usage: start.sh [--detach] [--model current|aggressive] [--share]

Starts Qwen and opens the live dashboard. Without --model, the model used last
is served (the first run asks). --share creates a public Cloudflare Quick Tunnel
with a fresh temporary API password (cloudflared must be installed).

Models:
  current     Unsloth UD-Q4_K_XL                  (API model ID qwen3.8-27b)
  aggressive  HauhauCS Aggressive Q4_K_P, uncensored (API model ID qwen3.8-27b-aggressive)
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${QWEN_ROOT:-$SCRIPT_DIR}"
LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/llamAmpere-src/v0.6}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build-sm86/bin/llama-server}"
KEY_FILE="${QWEN_API_KEY_FILE:-$HOME/.config/qwen3.8/api-key}"
STATE_DIR="${QWEN_STATE_DIR:-$HOME/.local/state/qwen3.8}"
PID_FILE="$STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/server.log"
PORT="${QWEN_PORT:-2222}"
MONITOR_SCRIPT="$ROOT/monitor.py"

open_monitor() {
  if (( MONITOR )) && [[ -t 0 && -t 1 ]]; then
    exec "$MONITOR_SCRIPT"
  fi
}

install -d -m 700 "$STATE_DIR"
pid_file_is_running() {
  [[ -s "$1" ]] && kill -0 "$(<"$1")" 2>/dev/null
}

# Check this before choosing a model. Re-running start.sh is an attach operation
# while the server is alive; it must not reopen the picker or disturb sharing.
if pid_file_is_running "$PID_FILE"; then
  echo "Qwen server is already running (PID $(<"$PID_FILE"))."
  if pid_file_is_running "$STATE_DIR/tunnel.pid" && [[ -s "$STATE_DIR/tunnel-url" ]]; then
    echo "Remote API: $(<"$STATE_DIR/tunnel-url")/v1"
  elif (( SHARE )); then
    echo "It was started without an active public link; stop it before enabling --share." >&2
  fi
  open_monitor
  exit 0
fi
rm -f "$PID_FILE"

# A Quick Tunnel can outlive a crashed/stopped llama-server. Keep that process,
# URL, and API key so restarting (with or without --share) restores the exact
# same public endpoint instead of silently replacing it.
PRESERVE_TUNNEL=0
if pid_file_is_running "$STATE_DIR/tunnel.pid" \
    && [[ -s "$STATE_DIR/tunnel-url" && -s "$STATE_DIR/share-api-key" ]]; then
  PRESERVE_TUNNEL=1
  SHARE=1
  echo "Restoring the existing shared endpoint $(<"$STATE_DIR/tunnel-url")/v1."
else
  # These files describe a dead Quick Tunnel and must never be advertised.
  rm -f "$STATE_DIR/tunnel.pid" "$STATE_DIR/tunnel-url" "$STATE_DIR/tunnel.log" \
    "$STATE_DIR/share-api-key"
fi

choose_model() {
  [[ -n "$MODEL_CHOICE" ]] && return
  if [[ -s "$STATE_DIR/model-choice" ]]; then
    MODEL_CHOICE="$(<"$STATE_DIR/model-choice")"
    return
  fi
  MODEL_CHOICE=current
  if [[ -t 0 && -t 1 ]]; then
    printf 'Choose a model:\n  1) Qwen3.8 27B · Unsloth UD-Q4_K_XL\n  2) HauhauCS Aggressive · Q4_K_P\n> '
    read -r answer
    [[ "$answer" == 2 ]] && MODEL_CHOICE=aggressive
  fi
}
choose_model

case "$MODEL_CHOICE" in
  current)
    MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
    MODEL="$MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf"
    MMPROJ="$MODEL_DIR/mmproj-F16.gguf"
    MODEL_ALIAS="qwen3.8-27b"
    MODEL_LABEL="Qwen3.8 27B · UD-Q4_K_XL"
    ;;
  aggressive)
    MODEL_DIR="${QWEN_AGGRESSIVE_MODEL_DIR:-$HOME/models/Qwen3.8-27B-Aggressive}"
    MODEL="$MODEL_DIR/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
    MMPROJ="$MODEL_DIR/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    MODEL_ALIAS="qwen3.8-27b-aggressive"
    MODEL_LABEL="Qwen3.8 27B · Aggressive Q4_K_P"
    ;;
  *) echo "Unknown model '$MODEL_CHOICE' (use current or aggressive)." >&2; exit 2 ;;
esac

# First-run onboarding downloads the model, builds llamAmpere with CUDA, and
# creates a key.
if [[ ! -x "$LLAMA_SERVER" || ! -r "$MODEL" || ! -r "$MMPROJ" || ! -s "$KEY_FILE" ]]; then
  env LLAMA_CPP_DIR="$LLAMA_DIR" LLAMA_SERVER="$LLAMA_SERVER" QWEN_API_KEY_FILE="$KEY_FILE" \
    "$ROOT/setup.sh" --model "$MODEL_CHOICE"
fi

if (( SHARE )); then
  command -v cloudflared >/dev/null || {
    echo "--share requires cloudflared. See the Remote access section in $ROOT/README.md" >&2
    exit 1
  }
  HOST="127.0.0.1"
  ACTIVE_KEY_FILE="$STATE_DIR/share-api-key"
  if (( ! PRESERVE_TUNNEL )); then
    umask 077
    openssl rand -base64 24 | tr '/+' '_-' | tr -d '=\n' >"$ACTIVE_KEY_FILE"
    printf '\n' >>"$ACTIVE_KEY_FILE"
  fi
else
  HOST="${QWEN_HOST:-$(tailscale ip -4 2>/dev/null | head -n1 || true)}"
  HOST="${HOST:-127.0.0.1}"
  ACTIVE_KEY_FILE="$KEY_FILE"
fi

for file in "$LLAMA_SERVER" "$MODEL" "$MMPROJ" "$ACTIVE_KEY_FILE"; do
  [[ -r "$file" ]] || { echo "Required file is missing: $file" >&2; exit 1; }
done

if ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "TCP port $PORT is already in use." >&2
  exit 1
fi

printf '%s\n' "$HOST" >"$STATE_DIR/host"
printf '%s\n' "$PORT" >"$STATE_DIR/port"
printf '%s\n' "$ACTIVE_KEY_FILE" >"$STATE_DIR/key-file"
printf '%s\n' "$MODEL_ALIAS" >"$STATE_DIR/model-alias"
printf '%s\n' "$MODEL_LABEL" >"$STATE_DIR/model-label"
printf '%s\n' "$MODEL_CHOICE" >"$STATE_DIR/model-choice"

# Measured flag set: BENCHMARKS.md. Both GGUFs carry their own MTP head, and
# llamAmpere drafts from it with its built-in 65,536-token vocabulary shortlist.
args=(
  --model "$MODEL"
  --alias "$MODEL_ALIAS"
  --mmproj "$MMPROJ"
  --image-min-tokens 1024
  --n-gpu-layers all
  --fit off
  --ctx-size 131072
  --parallel 1
  --flash-attn on
  # SJ-KVaRN 4-bit KV cache with a full-precision recent window.
  --cache-type-k sjkvarn4
  --cache-type-v sjkvarn4
  --batch-size 4096
  --ubatch-size 1024
  # Adaptive MTP depth 3-5 with exact (lossless) drafting; dropping
  # --no-spec-mb-mtp keeps more drafts at a small cost in sampling fidelity.
  --spec-type draft-mtp-adaptive
  --spec-draft-n-min-adaptive 3
  --spec-draft-n-max 5
  --spec-draft-p-min 0
  --no-spec-mb-mtp
  # The prompt cache parks a conversation evicted by a side request in RAM
  # (2 GiB on this host) and then on disk (~/.cache/llamampere/prompt-cache),
  # so returning to it restores in seconds instead of re-reading the prompt.
  --cache-prompt
  --threads 4
  --threads-batch 8
  --temp 1.0
  --top-k 20
  --top-p 0.95
  --min-p 0.0
  --presence-penalty 0.0
  --repeat-penalty 1.0
  --jinja
  --reasoning on
  --reasoning-effort xhigh
  --reasoning-format deepseek
  --reasoning-preserve
  # Sampling and draft verification run on the GPU when the request allows it
  # (no tool grammar); the 248,320-row vocabulary is too wide for this CPU.
  --backend-sampling
  --host "$HOST"
  --port "$PORT"
  --api-key-file "$ACTIVE_KEY_FILE"
  --metrics
  --no-webui
)

: >"$LOG_FILE"
nohup env LLAMA_MTP_GPU_VERIFY=1 "$LLAMA_SERVER" "${args[@]}" >>"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"

for _ in $(seq 1 180); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Qwen server failed to start. Last log lines:" >&2
    tail -80 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  if curl -fsS --max-time 2 -H "Authorization: Bearer $(<"$ACTIVE_KEY_FILE")" \
      "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "$MODEL_LABEL is ready at http://$HOST:$PORT/v1 (PID $pid)."
    echo "Model ID: $MODEL_ALIAS"
    echo "Log: $LOG_FILE"

    if (( SHARE )); then
      if (( PRESERVE_TUNNEL )); then
        tunnel_url="$(<"$STATE_DIR/tunnel-url")"
      else
        TUNNEL_LOG="$STATE_DIR/tunnel.log"
        : >"$TUNNEL_LOG"
        nohup cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$TUNNEL_LOG" 2>&1 &
        tunnel_pid=$!
        printf '%s\n' "$tunnel_pid" >"$STATE_DIR/tunnel.pid"
        tunnel_url=""
        for _ in $(seq 1 30); do
          tunnel_url=$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -n1 || true)
          [[ -n "$tunnel_url" ]] && break
          kill -0 "$tunnel_pid" 2>/dev/null || break
          sleep 1
        done
        if [[ -z "$tunnel_url" ]]; then
          echo "Cloudflare tunnel failed to start. See $TUNNEL_LOG" >&2
          "$ROOT/stop.sh" >/dev/null
          exit 1
        fi
        printf '%s\n' "$tunnel_url" >"$STATE_DIR/tunnel-url"
      fi
      echo
      echo "══════════════════ TEMPORARY REMOTE ACCESS ══════════════════"
      echo "Base URL: $tunnel_url/v1"
      echo "Model:    $MODEL_ALIAS"
      echo "Password/API key: $(<"$ACTIVE_KEY_FILE")"
      echo "This public URL and password expire when stop.sh is run."
      echo "══════════════════════════════════════════════════════════════"
    fi

    open_monitor
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for Qwen server. See $LOG_FILE" >&2
exit 1
