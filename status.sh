#!/usr/bin/env bash
set -euo pipefail
STATE_DIR="${QWEN_STATE_DIR:-$HOME/.local/state/qwen3.8}"
PID_FILE="$STATE_DIR/server.pid"
read_state() {
  if [[ -r "$STATE_DIR/$1" ]]; then
    printf '%s' "$(<"$STATE_DIR/$1")"
  else
    printf '%s' "$2"
  fi
}
KEY_FILE="${QWEN_API_KEY_FILE:-$(read_state key-file "$HOME/.config/qwen3.8/api-key")}"
PORT="${QWEN_PORT:-$(read_state port 2222)}"
HOST="${QWEN_HOST:-$(read_state host "$(tailscale ip -4 2>/dev/null | head -n1 || true)")}"
HOST="${HOST:-127.0.0.1}"
MODEL_LABEL="$(read_state model-label 'Qwen3.8 27B')"

if [[ ! -s "$PID_FILE" ]] || ! kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "Qwen server is stopped."
  exit 1
fi
curl -fsS --max-time 3 -H "Authorization: Bearer $(<"$KEY_FILE")" "http://$HOST:$PORT/health"
echo
echo "$MODEL_LABEL"
echo "PID $(<"$PID_FILE"), API http://$HOST:$PORT/v1"
if [[ -s "$STATE_DIR/tunnel.pid" ]] \
    && kill -0 "$(<"$STATE_DIR/tunnel.pid")" 2>/dev/null \
    && [[ -r "$STATE_DIR/tunnel-url" ]]; then
  echo "Remote API $(<"$STATE_DIR/tunnel-url")/v1"
fi
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
