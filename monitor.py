#!/usr/bin/env python3
"""Tiny live terminal dashboard for the local llama.cpp Qwen server."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
import urllib.request
from pathlib import Path

STATE_DIR = Path(os.environ.get("QWEN_STATE_DIR", Path.home() / ".local/state/qwen3.8"))


def state_text(name: str, default: str) -> str:
    try:
        return (STATE_DIR / name).read_text().strip() or default
    except OSError:
        return default


KEY_FILE = Path(
    os.environ.get(
        "QWEN_API_KEY_FILE",
        state_text("key-file", str(Path.home() / ".config/qwen3.8/api-key")),
    )
)
PORT = os.environ.get("QWEN_PORT", state_text("port", "2222"))
MODEL_LABEL = state_text("model-label", "Qwen3.8 27B")

C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"


def tailscale_ip() -> str:
    explicit = os.environ.get("QWEN_HOST") or state_text("host", "")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(["tailscale", "ip", "-4"], text=True, timeout=2).splitlines()[0]
    except Exception:
        return "127.0.0.1"


def request(path: str, key: str, host: str):
    req = urllib.request.Request(
        f"http://{host}:{PORT}{path}",
        headers={"Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=1.5) as response:
        body = response.read().decode()
    return json.loads(body) if path == "/slots" else body


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        try:
            name, value = line.rsplit(None, 1)
            values[name] = float(value)
        except ValueError:
            pass
    return values


def gpu_stats() -> dict[str, float]:
    fields = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        ).strip()
        vals = [float(part.strip()) for part in raw.split(",")]
        return dict(zip(("util", "used", "total", "temp", "power", "limit"), vals))
    except Exception:
        return {key: 0.0 for key in ("util", "used", "total", "temp", "power", "limit")}


def server_uptime(pid: int) -> int:
    try:
        return int(subprocess.check_output(["ps", "-o", "etimes=", "-p", str(pid)], text=True).strip())
    except Exception:
        return 0


def fmt_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def fmt_time(seconds: int) -> str:
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}d " if days else "") + f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(value: float, maximum: float, width: int = 22) -> str:
    ratio = min(1.0, max(0.0, value / maximum if maximum else 0.0))
    filled = round(ratio * width)
    color = C_GREEN if ratio < 0.75 else C_YELLOW if ratio < 0.92 else C_RED
    return f"{color}{'█' * filled}{C_DIM}{'░' * (width - filled)}{C_RESET}"


def metric(m: dict[str, float], name: str) -> float:
    return m.get(f"llamacpp:{name}", 0.0)


def render(
    host: str,
    pid: int,
    metrics: dict[str, float],
    slots: list[dict],
    gpu: dict[str, float],
    online: bool,
    live_prefill: float,
    live_decode: float,
) -> str:
    width = max(76, min(shutil.get_terminal_size((96, 24)).columns, 120))
    slot = slots[0] if slots else {}
    ctx_max = float(slot.get("n_ctx", 131072))
    prompt_now = float(slot.get("n_prompt_tokens", 0))
    processing = bool(slot.get("is_processing", False))
    processed_now = float(slot.get("n_prompt_tokens_processed", 0))
    decoded_now = sum(float(x.get("n_decoded", 0)) for x in slot.get("next_token", []))
    context_now = min(ctx_max, prompt_now + decoded_now)

    prompt = metric(metrics, "prompt_tokens_total")
    cached = metric(metrics, "prompt_tokens_cached_total")
    output = metric(metrics, "tokens_predicted_total")
    prompt_secs = metric(metrics, "prompt_seconds_total")
    output_secs = metric(metrics, "tokens_predicted_seconds_total")
    prefill_speed = metric(metrics, "prompt_tokens_seconds") or (prompt / prompt_secs if prompt_secs else 0)
    decode_speed = metric(metrics, "predicted_tokens_seconds") or (output / output_secs if output_secs else 0)
    drafts = metric(metrics, "spec_decode_num_draft_tokens_total")
    accepted = metric(metrics, "spec_decode_num_accepted_tokens_total")
    cache_hit = 100 * cached / (prompt + cached) if prompt + cached else 0
    mtp_hit = 100 * accepted / drafts if drafts else 0
    active = int(metric(metrics, "requests_processing"))
    queued = int(metric(metrics, "requests_deferred"))

    if not online:
        state, state_color = "OFFLINE", C_RED
    elif processing and processed_now < prompt_now:
        state, state_color = "PREFILL", C_YELLOW
    elif processing:
        state, state_color = "DECODE", C_GREEN
    else:
        state, state_color = "IDLE", C_CYAN

    line = "─" * width
    title = f" {MODEL_LABEL}  ·  RTX 3090  ·  {host}:{PORT} "
    lines = [
        f"{C_CYAN}{C_BOLD}{title}{C_RESET}{C_DIM}{'─' * max(0, width - len(title))}{C_RESET}",
        f" {state_color}{C_BOLD}{state:<8}{C_RESET}  uptime {fmt_time(server_uptime(pid)):<12}  requests {active} active / {queued} queued",
        f"{C_DIM}{line}{C_RESET}",
        f" GPU      {bar(gpu['util'], 100)} {gpu['util']:5.1f}%   {gpu['temp']:4.0f}°C   {gpu['power']:5.0f}/{gpu['limit']:.0f} W",
        f" VRAM     {bar(gpu['used'], gpu['total'])} {gpu['used']/1024:5.1f}/{gpu['total']/1024:.1f} GiB",
        f" CONTEXT  {bar(context_now, ctx_max)} {fmt_count(context_now):>7}/{fmt_count(ctx_max):<7} {100*context_now/ctx_max if ctx_max else 0:5.1f}%",
        f"{C_DIM}{line}{C_RESET}",
        f" TOKENS   in {C_BOLD}{fmt_count(prompt + cached):>9}{C_RESET}   out {C_GREEN}{C_BOLD}{fmt_count(output):>9}{C_RESET}   max sequence {fmt_count(metric(metrics, 'n_tokens_max'))}",
        f" INPUT    evaluated {fmt_count(prompt):>8}   cached {fmt_count(cached):>8}   cache hit {C_BOLD}{cache_hit:6.2f}%{C_RESET}",
        f" REASON   {C_DIM}included in tokens out · llama.cpp metrics do not expose a separate count{C_RESET}",
        f" MTP      accepted {fmt_count(accepted):>7}/{fmt_count(drafts):<7}  acceptance {C_BOLD}{mtp_hit:6.2f}%{C_RESET}",
        f"{C_DIM}{line}{C_RESET}",
        f" PREFILL  live {C_YELLOW}{C_BOLD}{live_prefill:8.1f}{C_RESET} tok/s   session avg {prefill_speed:8.1f} tok/s",
        f" DECODE   live {C_GREEN}{C_BOLD}{live_decode:8.1f}{C_RESET} tok/s   session avg {decode_speed:8.1f} tok/s",
    ]
    if processing:
        lines.append(f" CURRENT  prompt {fmt_count(prompt_now):>8}   processed {fmt_count(processed_now):>8}   generated {fmt_count(decoded_now):>7}")
    else:
        lines.append(f" CURRENT  slot idle · retained prefix/context {fmt_count(context_now)} tokens")
    lines += [
        f"{C_DIM}{line}{C_RESET}",
        f" {C_DIM}q / Ctrl-C: close dashboard (server keeps running) · stop: ./stop.sh{C_RESET}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    host = tailscale_ip()
    try:
        key = KEY_FILE.read_text().strip()
        pid = int((STATE_DIR / "server.pid").read_text().strip())
    except (OSError, ValueError) as exc:
        print(f"Qwen monitor: {exc}", file=sys.stderr)
        return 1

    interactive = sys.stdout.isatty() and sys.stdin.isatty() and not args.once
    old_term = None
    if interactive:
        old_term = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write("\033[?1049h\033[?25l")

    previous_task = None
    samples: deque[tuple[float, float, float]] = deque()
    live_prefill = 0.0
    live_decode = 0.0
    try:
        while True:
            online = True
            try:
                metrics = parse_metrics(request("/metrics", key, host))
                slots = request("/slots", key, host)
            except Exception:
                metrics, slots, online = {}, [], False

            now = time.monotonic()
            slot = slots[0] if slots else {}
            task = slot.get("id_task")
            processed = float(slot.get("n_prompt_tokens_processed", 0))
            decoded = sum(float(x.get("n_decoded", 0)) for x in slot.get("next_token", []))
            if task != previous_task:
                samples.clear()
            if slot.get("is_processing"):
                samples.append((now, processed, decoded))
                # A short rolling window smooths llama.cpp's 2048-token progress updates.
                while len(samples) > 2 and samples[1][0] < now - 4.0:
                    samples.popleft()
                if len(samples) > 1:
                    elapsed = max(0.001, samples[-1][0] - samples[0][0])
                    live_prefill = max(0.0, samples[-1][1] - samples[0][1]) / elapsed
                    live_decode = max(0.0, samples[-1][2] - samples[0][2]) / elapsed
                else:
                    live_prefill = live_decode = 0.0
            else:
                samples.clear()
                live_prefill = live_decode = 0.0
            previous_task = task

            dashboard = render(host, pid, metrics, slots, gpu_stats(), online, live_prefill, live_decode)
            if interactive:
                sys.stdout.write("\033[H\033[2J" + dashboard + "\n")
                sys.stdout.flush()
            else:
                # Strip the small ANSI palette for snapshots and redirected output.
                for code in (C_RESET, C_DIM, C_BOLD, C_CYAN, C_GREEN, C_YELLOW, C_RED):
                    dashboard = dashboard.replace(code, "")
                print(dashboard)
                return 0
            ready, _, _ = select.select([sys.stdin], [], [], max(0.1, args.interval))
            if ready and sys.stdin.read(1).lower() == "q":
                return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if interactive:
            if old_term is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
