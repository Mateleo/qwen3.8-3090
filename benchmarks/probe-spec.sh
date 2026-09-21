#!/usr/bin/env bash
# Diagnose whether a llama-server binary actually engages speculative decoding.
# Usage: probe-spec.sh <binary> <logfile> [extra args...]
# Starts on port 2234 with a small context, one greedy 64-token request, then stops.
set -u
BIN="$1"; LOG="$2"; shift 2
PORT="${PROBE_PORT:-2234}"
M=~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
D=~/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf

"$BIN" --model "$M" --model-draft "$D" --spec-type draft-mtp --spec-draft-n-max 3 \
  --n-gpu-layers all --n-gpu-layers-draft all \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
  --ctx-size 4096 --parallel 1 --flash-attn on \
  --cache-prompt --cache-ram 0 --threads 4 --threads-batch 4 \
  --host 127.0.0.1 --port "$PORT" --no-webui -lv 5 "$@" >"$LOG" 2>&1 &
PID=$!
for _ in $(seq 1 120); do
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "server died, see $LOG"; tail -20 "$LOG"; exit 1; }
  sleep 2
done
curl -sS -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
  -d '{"prompt":"Write one sentence about the sea.","n_predict":64,"temperature":0,"ignore_eos":true,"cache_prompt":false,"stream":false}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print("timings:", json.dumps(r.get("timings"), indent=1))'
kill -TERM "$PID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$PID" 2>/dev/null
echo "--- spec/draft log lines ---"
grep -iE "spec|draft|mtp|unused tensor blk.64" "$LOG" | head -40
