#!/usr/bin/env bash
set -euo pipefail
STATE_DIR="${QWEN_STATE_DIR:-$HOME/.local/state/qwen3.8}"
PID_FILE="$STATE_DIR/server.pid"
TUNNEL_PID_FILE="$STATE_DIR/tunnel.pid"

stop_pid_file() {
  local file="$1" label="$2" pid
  [[ -s "$file" ]] || return 0
  pid="$(<"$file")"
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$file"
    return 0
  fi
  kill "$pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$file"
      echo "$label stopped."
      return 0
    fi
    sleep 1
  done
  echo "$label did not stop gracefully; sending SIGKILL." >&2
  kill -KILL "$pid"
  rm -f "$file"
}

if [[ ! -s "$PID_FILE" && ! -s "$TUNNEL_PID_FILE" ]]; then
  echo "Qwen server is not running."
else
  stop_pid_file "$TUNNEL_PID_FILE" "Cloudflare tunnel"
  stop_pid_file "$PID_FILE" "Qwen server"
fi
rm -f \
  "$STATE_DIR/share-api-key" "$STATE_DIR/tunnel-url" "$STATE_DIR/tunnel.pid" "$STATE_DIR/tunnel.log" \
  "$STATE_DIR/host" "$STATE_DIR/port" "$STATE_DIR/key-file" \
  "$STATE_DIR/model-alias" "$STATE_DIR/model-label" \
  "$STATE_DIR/profile-choice" "$STATE_DIR/server-profile"
