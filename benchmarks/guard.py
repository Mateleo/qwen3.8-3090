#!/usr/bin/env python3
"""Run a command under hard VRAM and host-RAM limits (WSL2 safety wrapper).

This host has already rebooted once when the card sat about 1.3 GiB from full
(nvlddmkm fault, Kernel-Power 41), and WSL2 has only ~10 GiB of RAM. The screen
harness only checks VRAM between requests; this wrapper checks continuously.

  * host RAM: the command runs in a transient systemd user scope with
    MemoryMax / MemorySwapMax, so the kernel kills *it* (not WSL) on overrun;
  * VRAM: nvidia-smi streams memory.used every 250 ms; above the limit every
    process in the scope gets SIGKILL immediately (no graceful shutdown on purpose);
  * free RAM: if MemAvailable for the whole VM drops below the floor, same kill.

Usage:
    benchmarks/guard.py [--vram-mib N] [--ram-max 5G] [--swap-max 512M] \
        [--min-avail-mib N] [--log FILE] -- cmd args...

Exit code is the command's, or 97 when the guard killed it (reason on stderr and
in the log). Defaults are conservative on purpose; raise them only deliberately.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

GUARD_KILLED = 97


def mem_available_mib():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-mib", type=int,
                    default=int(os.environ.get("GUARD_VRAM_MIB", "23000")),
                    help="kill when total card memory.used exceeds this (MiB)")
    ap.add_argument("--ram-max", default=os.environ.get("GUARD_RAM_MAX", "5G"))
    ap.add_argument("--swap-max", default=os.environ.get("GUARD_SWAP_MAX", "512M"))
    ap.add_argument("--min-avail-mib", type=int,
                    default=int(os.environ.get("GUARD_MIN_AVAIL_MIB", "1200")))
    ap.add_argument("--log", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("missing command")

    logfh = open(a.log, "a") if a.log else None

    def note(msg):
        line = f"[guard {time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stderr, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    # A named scope lets the kill reach every descendant, including servers that a
    # harness started in their own session (a process-group kill would miss those).
    unit = f"qwen-guard-{os.getpid()}.scope"
    scope = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
             "-p", f"MemoryMax={a.ram_max}", "-p", f"MemorySwapMax={a.swap_max}", "--"]
    proc = subprocess.Popen(scope + cmd, start_new_session=True)
    note(f"pid {proc.pid}: vram<={a.vram_mib} MiB, ram<={a.ram_max} (+swap {a.swap_max}), "
         f"MemAvailable>={a.min_avail_mib} MiB")

    killed = {"why": None}
    peak = {"vram": 0, "avail_min": 1 << 30}

    def kill(why):
        if killed["why"]:
            return
        killed["why"] = why
        note(f"KILL: {why}")
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", unit],
                       capture_output=True)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def watch_vram():
        smi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "-lms", "250"], stdout=subprocess.PIPE, text=True)
        try:
            for line in smi.stdout:
                if proc.poll() is not None:
                    break
                try:
                    used = int(line.strip())
                except ValueError:
                    continue
                peak["vram"] = max(peak["vram"], used)
                if used > a.vram_mib:
                    kill(f"VRAM {used} MiB > {a.vram_mib} MiB")
                    break
        finally:
            smi.kill()

    def watch_ram():
        while proc.poll() is None:
            avail = mem_available_mib()
            peak["avail_min"] = min(peak["avail_min"], avail)
            if 0 <= avail < a.min_avail_mib:
                kill(f"MemAvailable {avail} MiB < {a.min_avail_mib} MiB")
                return
            time.sleep(0.5)

    for fn in (watch_vram, watch_ram):
        threading.Thread(target=fn, daemon=True).start()

    def forward(sig, _frame):
        subprocess.run(["systemctl", "--user", "kill", f"--signal={sig}", unit],
                       capture_output=True)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    rc = proc.wait()
    note(f"exit rc={rc}, peak VRAM {peak['vram']} MiB, min MemAvailable {peak['avail_min']} MiB")
    return GUARD_KILLED if killed["why"] else rc


if __name__ == "__main__":
    sys.exit(main())
