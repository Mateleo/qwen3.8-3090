#!/usr/bin/env bash
set -euo pipefail

MONITOR=1
SHARE=0
MODEL_CHOICE=""
PROFILE_CHOICE=""
LIST_PROFILES=0
while (($#)); do
  case "$1" in
    --detach|--no-monitor) MONITOR=0 ;;
    --share) SHARE=1 ;;
    --profile)
      shift
      [[ $# -gt 0 ]] || { echo "--profile requires a name; see --list-profiles" >&2; exit 2; }
      PROFILE_CHOICE="$1"
      ;;
    --model)
      shift
      [[ $# -gt 0 ]] || { echo "--model requires current or aggressive" >&2; exit 2; }
      MODEL_CHOICE="$1"
      ;;
    --list-profiles) LIST_PROFILES=1 ;;
    --help|-h)
      cat <<'EOF'
Usage: start.sh [--detach] [--profile NAME] [--model current|aggressive] [--share]

Starts Qwen and opens the live dashboard. With no --profile, an interactive menu
offers every measured runtime/model combination. --share creates a public
Cloudflare Quick Tunnel with a fresh temporary API password (cloudflared must be
installed).

Profiles (also listed by --list-profiles):
  fast               UD-Q4_K_XL on the Ampere runtime   (+23% decode, 22.4 GiB)
  compat             UD-Q4_K_XL on the stock runtime    (unchanged outputs)
  aggressive         Aggressive Q4_K_P on the Ampere runtime
  aggressive-compat  Aggressive Q4_K_P on the stock runtime
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${QWEN_ROOT:-$SCRIPT_DIR}"

# Two runtimes serve the same weights. Each profile below pairs a model with a
# runtime. Evidence and exact numbers: thesis.md, benchmarks/thesis2-*.
STOCK_SERVER="${QWEN_STOCK_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
AMPERE_SERVER="${QWEN_AMPERE_SERVER:-$HOME/llamAmpere-src/repo/build-sm86/bin/llama-server}"
AMPERE_AVAILABLE=0
[[ -x "$AMPERE_SERVER" ]] && AMPERE_AVAILABLE=1

ampere_build_hint() {
  cat >&2 <<EOF
The Ampere runtime is not built yet: $AMPERE_SERVER
Build it (about 25 minutes on this host, keep the parallelism low):
  cd ~/llamAmpere-src/repo
  cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \\
        -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
  cmake --build build-sm86 -j2 --target llama-server
Until then use one of the "stock" profiles (compat / aggressive-compat).
EOF
}
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
  if [[ -z "$PROFILE_CHOICE" ]]; then
    if [[ -s "$STATE_DIR/profile-choice" ]]; then
      PROFILE_CHOICE="$(<"$STATE_DIR/profile-choice")"
    elif [[ -s "$STATE_DIR/model-choice" ]]; then
      # State written by an older version: keep serving the same model.
      case "$(<"$STATE_DIR/model-choice")" in
        aggressive) PROFILE_CHOICE=aggressive-compat ;;
        *)          PROFILE_CHOICE=compat ;;
      esac
    fi
  fi
  echo "Restoring the existing shared endpoint $(<"$STATE_DIR/tunnel-url")/v1."
else
  # These files describe a dead Quick Tunnel and must never be advertised.
  rm -f "$STATE_DIR/tunnel.pid" "$STATE_DIR/tunnel-url" "$STATE_DIR/tunnel.log" \
    "$STATE_DIR/share-api-key"
fi

# ---------------------------------------------------------------------------
# Profiles: a name pairs a model with a runtime. The interactive menu is the
# normal way to pick one; --profile selects it directly for scripts.
# ---------------------------------------------------------------------------
PROFILE_ORDER=(fast compat aggressive aggressive-compat)

profile_model() {
  case "$1" in
    fast|compat)                  echo current ;;
    aggressive|aggressive-compat) echo aggressive ;;
    *) return 1 ;;
  esac
}

profile_runtime() {
  case "$1" in
    fast|aggressive)          echo ampere ;;
    compat|aggressive-compat) echo stock ;;
    *) return 1 ;;
  esac
}

profile_title() {
  case "$1" in
    fast)              echo "Qwen3.8 27B - rapide (+23%, 22.4 GiB)" ;;
    compat)            echo "Qwen3.8 27B - sortie identique au stock" ;;
    aggressive)        echo "Aggressive uncensored - rapide (+23%)" ;;
    aggressive-compat) echo "Aggressive uncensored - sortie identique" ;;
  esac
}

profile_detail() {
  case "$1" in
    fast)              echo "UD-Q4_K_XL, runtime Ampere, verification GPU des drafts" ;;
    compat)            echo "UD-Q4_K_XL, runtime stock, sorties inchangees" ;;
    aggressive)        echo "Aggressive Q4_K_P, runtime Ampere" ;;
    aggressive-compat) echo "Aggressive Q4_K_P, runtime stock" ;;
  esac
}

list_profiles() {
  local p
  for p in "${PROFILE_ORDER[@]}"; do
    printf '  %-18s %-42s %s\n' "$p" "$(profile_title "$p")" "$(profile_detail "$p")"
  done
}

profile_available() {
  [[ "$(profile_runtime "$1")" == ampere && $AMPERE_AVAILABLE -eq 0 ]] && return 1
  return 0
}

choose_profile() {
  if [[ -n "$PROFILE_CHOICE" ]]; then
    return
  fi
  # --model stays supported: it means "this model on the stock runtime".
  if [[ -n "$MODEL_CHOICE" ]]; then
    case "$MODEL_CHOICE" in
      current)    PROFILE_CHOICE=compat ;;
      aggressive) PROFILE_CHOICE=aggressive-compat ;;
      *) echo "Unknown model '$MODEL_CHOICE' (use current or aggressive)." >&2; exit 2 ;;
    esac
    return
  fi
  local default="compat" saved=""
  if [[ -s "$STATE_DIR/profile-choice" ]]; then
    saved="$(<"$STATE_DIR/profile-choice")"
    if profile_model "$saved" >/dev/null 2>&1 && profile_available "$saved"; then
      default="$saved"
    fi
  fi
  if [[ -t 0 && -t 1 ]] && command -v whiptail >/dev/null; then
    local items=() tag note
    for tag in "${PROFILE_ORDER[@]}"; do
      note=""
      profile_available "$tag" || note="  [not built]"
      items+=("$tag" "$(profile_title "$tag")$note")
    done
    PROFILE_CHOICE=$(whiptail --title "Qwen 3.8 - what to serve" --default-item "$default" \
      --menu "Pick a profile. Both runtimes serve the same weights.
Measured on this machine: fast = +23% tokens/s, uses less VRAM.
Last used: $default" 20 96 4 "${items[@]}" 3>&1 1>&2 2>&3) || exit 0
    return
  fi
  if [[ -t 0 && -t 1 ]]; then
    echo "Pick a profile:"
    local i=1 p
    for p in "${PROFILE_ORDER[@]}"; do
      printf '  %d) %-18s %s\n     %s\n' "$i" "$p" "$(profile_title "$p")" "$(profile_detail "$p")"
      i=$((i+1))
    done
    printf 'Choice [%s]: ' "$default"
    read -r answer
    case "$answer" in
      "")            PROFILE_CHOICE="$default" ;;
      1)             PROFILE_CHOICE=fast ;;
      2)             PROFILE_CHOICE=compat ;;
      3)             PROFILE_CHOICE=aggressive ;;
      4)             PROFILE_CHOICE=aggressive-compat ;;
      fast|compat|aggressive|aggressive-compat) PROFILE_CHOICE="$answer" ;;
      *) echo "Invalid choice '$answer'." >&2; exit 2 ;;
    esac
    return
  fi
  PROFILE_CHOICE="$default"
}

if (( LIST_PROFILES )); then
  list_profiles
  exit 0
fi
choose_profile
profile_model "$PROFILE_CHOICE" >/dev/null 2>&1 || {
  echo "Unknown profile '$PROFILE_CHOICE'. Available:" >&2
  list_profiles >&2
  exit 2
}
if ! profile_available "$PROFILE_CHOICE"; then
  ampere_build_hint
  exit 1
fi
MODEL_CHOICE="$(profile_model "$PROFILE_CHOICE")"
SERVER_BUILD="$(profile_runtime "$PROFILE_CHOICE")"
# An explicit QWEN_SERVER_BUILD still overrides the runtime of the chosen model.
if [[ -n "${QWEN_SERVER_BUILD:-}" ]]; then
  case "$QWEN_SERVER_BUILD" in
    stock)  SERVER_BUILD=stock ;;
    ampere) SERVER_BUILD=ampere ;;
    *) echo "QWEN_SERVER_BUILD must be 'stock' or 'ampere' (got '$QWEN_SERVER_BUILD')." >&2; exit 2 ;;
  esac
fi
if [[ "$SERVER_BUILD" == ampere && $AMPERE_AVAILABLE -eq 0 ]]; then
  ampere_build_hint
  exit 1
fi
case "$SERVER_BUILD" in
  stock)  DEFAULT_SERVER="$STOCK_SERVER" ;;
  ampere) DEFAULT_SERVER="$AMPERE_SERVER" ;;
esac
LLAMA_SERVER="${LLAMA_SERVER:-$DEFAULT_SERVER}"

case "$MODEL_CHOICE" in
  current)
    MODEL_DIR="${QWEN_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
    MODEL="$MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf"
    MMPROJ="$MODEL_DIR/mmproj-F16.gguf"
    MTP="$MODEL_DIR/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
    MODEL_ALIAS="qwen3.8-27b"
    MODEL_LABEL="Qwen3.8 27B · UD-Q4_K_XL"
    ;;
  aggressive)
    MODEL_DIR="${QWEN_AGGRESSIVE_MODEL_DIR:-$HOME/models/Qwen3.8-27B-Aggressive}"
    MODEL="$MODEL_DIR/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
    MMPROJ="$MODEL_DIR/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    # This GGUF contains native embedded MTP. The optional FastMTP sidecar
    # requires HauhauCS's patched llama.cpp, so upstream-safe embedded MTP is used.
    MTP=""
    MODEL_ALIAS="qwen3.8-27b-aggressive"
    MODEL_LABEL="Qwen3.8 27B · Aggressive Q4_K_P"
    ;;
  *) echo "Unknown model '$MODEL_CHOICE' (use current or aggressive)." >&2; exit 2 ;;
esac
# The dashboard shows which profile is serving.
MODEL_LABEL="$MODEL_LABEL · $PROFILE_CHOICE"

# The llamAmpere runtime adds three things the stock build does not have: GPU
# verification of MTP drafts, the draft vocabulary shortlist and "--fit off".
# Detect them instead of assuming, so the same script drives both runtimes.
SERVER_FORK=0
if "$LLAMA_SERVER" --help 2>/dev/null | grep -q -- "--spec-draft-vocab-map"; then
  SERVER_FORK=1
fi

# Draft vocabulary shortlist: the drafter scores 65,536 of the 248,320 output
# rows. The loader rejects a map whose vocabulary size differs from the model,
# and a greedy run with the map must reproduce the full-head greedy output.
VOCAB_MAP="${QWEN_VOCAB_MAP:-$MODEL_DIR/MTP/mtp-vocab-65536.txt}"
[[ -r "$VOCAB_MAP" ]] || VOCAB_MAP="$HOME/llamAmpere-src/repo/docs/mtp-vocab/atx_65536.txt"

# MTP drafting depth. The fork's measured best at production sampling is 2; the
# stock build's default (3) is left alone.
SPEC_DRAFT_N_MAX=2

# First-run onboarding installs the downloader only when required, shows the
# Hugging Face download progress, builds llama.cpp with CUDA, and creates a key.
if [[ ! -x "$LLAMA_SERVER" || ! -r "$MODEL" || ! -r "$MMPROJ" || ! -s "$KEY_FILE" ]] \
    || [[ -n "$MTP" && ! -r "$MTP" ]]; then
  env LLAMA_SERVER="$LLAMA_SERVER" QWEN_API_KEY_FILE="$KEY_FILE" \
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
if [[ -n "$MTP" && ! -r "$MTP" ]]; then
  echo "Required file is missing: $MTP" >&2
  exit 1
fi

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
printf '%s\n' "$PROFILE_CHOICE" >"$STATE_DIR/profile-choice"

args=(
  --model "$MODEL"
  --alias "$MODEL_ALIAS"
  --mmproj "$MMPROJ"
  --image-min-tokens 1024
  --spec-type draft-mtp
  --n-gpu-layers all
  --ctx-size 131072
  --parallel 1
  --flash-attn on
  --cache-type-k q4_0
  --cache-type-v q4_0
  --batch-size 2048
  --ubatch-size 512
  --cache-prompt
  --cache-ram 0
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
  # Sampling and draft verification run on the GPU. On this 2400G host the host
  # side of a 248,320-row vocabulary was the decode bottleneck: moving it to the
  # card is worth +37% at greedy sampling and never measurably loses.
  --backend-sampling
  --host "$HOST"
  --port "$PORT"
  --api-key-file "$ACTIVE_KEY_FILE"
  --metrics
  --no-webui
)
if (( SERVER_FORK )); then
  args+=(
    --fit off
    --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
  )
  [[ -r "$VOCAB_MAP" ]] && args+=(--spec-draft-vocab-map "$VOCAB_MAP")
fi
DRAFT_DESC="embedded MTP head"
if [[ -n "$MTP" ]]; then
  if (( SERVER_FORK )) && [[ "${QWEN_MTP_SIDECAR:-0}" != "1" ]]; then
    # This GGUF carries its own MTP layer, and on the fork the embedded head is
    # as fast as the sidecar while using ~1.1 GiB less VRAM.
    MTP=""
  else
    args+=(
      --model-draft "$MTP"
      --n-gpu-layers-draft all
      --cache-type-k-draft q4_0
      --cache-type-v-draft q4_0
    )
    DRAFT_DESC="$MTP"
  fi
fi
{
  printf 'server=%s\n' "$LLAMA_SERVER"
  printf 'fork_features=%s\n' "$SERVER_FORK"
  printf 'backend_sampling=1\n'
  printf 'draft=%s\n' "$DRAFT_DESC"
  printf 'vocab_map=%s\n' "$([[ -r "$VOCAB_MAP" ]] && printf '%s' "$VOCAB_MAP" || printf 'none')"
  if (( SERVER_FORK )); then
    printf 'spec_draft_n_max=%s\n' "$SPEC_DRAFT_N_MAX"
  else
    printf 'spec_draft_n_max=default\n'
  fi
} >"$STATE_DIR/server-profile" 2>/dev/null || true

: >"$LOG_FILE"
if (( SERVER_FORK )); then
  # Verify the drafter's tokens on the GPU as well (LLAMA_MTP_GPU_VERIFY=1).
  nohup env LLAMA_MTP_GPU_VERIFY=1 "$LLAMA_SERVER" "${args[@]}" >>"$LOG_FILE" 2>&1 &
else
  nohup "$LLAMA_SERVER" "${args[@]}" >>"$LOG_FILE" 2>&1 &
fi
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
    echo "Profile: $PROFILE_CHOICE ($SERVER_BUILD runtime)"
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
