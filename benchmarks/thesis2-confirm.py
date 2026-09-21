#!/usr/bin/env python3
"""Production-shaped confirmation stage for the thesis screen.

The screen in thesis2-screen.py runs raw completions at temperature 0 so that
profiles are hash-comparable. This script instead drives the chat endpoint with
the launcher's production sampling (temperature 1, top-k 20, top-p 0.95,
reasoning on), repeats every fixture once per seed, and pairs the profiles so a
noise-driven winner is visible as such.

Usage:
    THESIS2_PROFILES=stock-mtp3,amp-map THESIS2_SEEDS=3 \
        python3 benchmarks/thesis2-confirm.py
Results land in benchmarks/thesis2-confirm-<tag>/.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location("screen", HERE / "thesis2-screen.py")
screen = importlib.util.module_from_spec(spec)
sys.modules["screen"] = screen
spec.loader.exec_module(screen)

FIXTURES = ["novel-code", "prose", "patch", "copy", "prefill8k"]
MAX_TOKENS = int(os.environ.get("THESIS2_MAX_TOKENS", "512"))

if os.environ.get("THESIS2_REASONING", "1") == "1":
    # Match the launcher: jinja chat template, reasoning on, effort xhigh.
    screen.COMMON = screen.COMMON + [
        "--jinja", "--reasoning", "on", "--reasoning-effort", "xhigh",
        "--reasoning-format", "deepseek", "--reasoning-preserve",
    ]


def main():
    names = [n.strip() for n in os.environ.get("THESIS2_PROFILES", "").split(",") if n.strip()]
    if not names:
        print("set THESIS2_PROFILES")
        return 1
    seeds = [int(s) for s in os.environ.get("THESIS2_SEEDS", "3,7,11").split(",")]
    tag = os.environ.get("THESIS2_TAG", time.strftime("confirm-%Y%m%d-%H%M%S"))
    outdir = screen.ROOT / "benchmarks" / f"thesis2-{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / "rows.jsonl"
    if not screen.wait_free():
        return 2

    for name in names:
        cfg = screen.PROFILES.get(name)
        if cfg is None:
            print(f"unknown profile {name}")
            continue
        binary, extra, env, note = cfg
        if not Path(binary).exists():
            continue
        log_path = outdir / f"{name}.log"
        print(f"[{time.strftime('%H:%M:%S')}] confirm {name}: {note}", flush=True)
        proc, load_s, cmd, full_env = screen.start_server(binary, extra, env, log_path)
        rows = []
        try:
            for fixture, seed in ((f, s) for s in seeds for f in FIXTURES):
                body = {
                    "messages": [{"role": "user", "content": screen.FIXTURES[fixture]}],
                    "max_tokens": 128 if fixture == "prefill8k" else MAX_TOKENS,
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                    "min_p": 0.0,
                    "seed": seed,
                    "cache_prompt": False,
                    "stream": False,
                }
                t0 = time.time()
                try:
                    raw = screen.http_json("/v1/chat/completions", body)
                except Exception as exc:
                    rows.append({"profile": name, "fixture": fixture, "seed": seed,
                                 "error": str(exc)[:300]})
                    continue
                wall = time.time() - t0
                timings = raw.get("timings", {}) or {}
                usage = raw.get("usage", {}) or {}
                row = {
                    "tag": tag, "profile": name, "note": note, "fixture": fixture,
                    "seed": seed, "temp": 1.0, "load_s": round(load_s, 2),
                    "wall_s": round(wall, 3),
                    "prompt_n": timings.get("prompt_n"),
                    "prompt_per_second": timings.get("prompt_per_second"),
                    "predicted_n": timings.get("predicted_n"),
                    "predicted_per_second": timings.get("predicted_per_second"),
                    "draft_n": timings.get("draft_n"),
                    "draft_n_accepted": timings.get("draft_n_accepted"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "gpu": screen.gpu_mib(),
                }
                if (row.get("gpu") or {}).get("vram_mib", 0) > screen.VRAM_LIMIT_MIB:
                    row["error"] = (f"vram {(row['gpu'])['vram_mib']} MiB exceeds the "
                                    f"{screen.VRAM_LIMIT_MIB} MiB guard")
                    rows.append(row)
                    print(f"  ABORT: {row['error']}")
                    break
                rows.append(row)
                print("  %-12s seed %-3s decode %6.2f tok/s prefill %8.2f draft %s/%s wall %.1fs"
                      % (fixture, seed, row["predicted_per_second"] or 0.0,
                         row["prompt_per_second"] or 0.0, row["draft_n_accepted"],
                         row["draft_n"], wall), flush=True)
                (outdir / "raw").mkdir(exist_ok=True)
                (outdir / "raw" / f"{name}__{fixture}__s{seed}.json").write_text(
                    json.dumps({"row": row, "usage": usage, "timings": timings}, indent=1))
        finally:
            screen.stop_server(proc)
        with rows_path.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(f"done, results in {rows_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
