#!/usr/bin/env bash
set -euo pipefail
STATE_DIR="${QWEN_STATE_DIR:-$HOME/.local/state/qwen3.8}"
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
MODEL_ALIAS="$(read_state model-alias qwen3.8-27b)"

curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
  -H "Authorization: Bearer $(<"$KEY_FILE")" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: API works\"}],\"max_tokens\":64,\"temperature\":0}" \
  | python3 -m json.tool
