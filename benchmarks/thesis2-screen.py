#!/usr/bin/env python3
"""Sequential A/B screen for the Qwen3.8-27B llama-server on this RTX 3090.

One profile = one freshly launched server process. Profiles run one at a time
(a second llama-server would compete for the same card). For every profile the
harness runs the standard fixtures, records server-reported timings, wall times,
VRAM, draft acceptance and output hashes, and appends one JSON row per request
to rows.jsonl in the campaign directory.

Usage:
    THESIS2_PROFILES=stock-mtp3,amp-mtp3 python3 benchmarks/thesis2-screen.py
    THESIS2_PROFILES=... THESIS2_TAG=screen1 python3 benchmarks/thesis2-screen.py

Constraints this harness deliberately respects:
  * one server at a time, no lockfile races (it refuses to start if the port or
    another llama-server is busy);
  * host RAM stays low: --cache-ram 0 is kept and no profile parks state in RAM;
  * every profile is loaded the same way, so load time is comparable.
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis2_fixtures import FIXTURES, PREFILL_ONLY  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
MODEL_DIR = HOME / "models/Qwen3.8-27B"
TARGET = MODEL_DIR / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
MMPROJ = MODEL_DIR / "mmproj-F16.gguf"
MTP = MODEL_DIR / "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"

STOCK = HOME / "llama.cpp/build/bin/llama-server"
AMPERE = HOME / "llamAmpere-src/repo/build-sm86/bin/llama-server"
VOCAB_MAP = HOME / "llamAmpere-src/repo/docs/mtp-vocab/atx_65536.txt"
DFLASH = HOME / "models/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q8_0.gguf"
# Patched libggml-base.so (patches/llamampere-compute-slack.patch), picked up ahead
# of the fork's RUNPATH. With GGML_COMPUTE_SLACK_MIB unset it behaves as upstream.
SLACK_LIB = HOME / "llamAmpere-src/libslack"
SLACK_ENV = {"LD_LIBRARY_PATH": str(SLACK_LIB), "GGML_COMPUTE_SLACK_MIB": "8"}

PORT = int(os.environ.get("THESIS2_PORT", "2233"))
BASE = f"http://127.0.0.1:{PORT}"
N_PREDICT = int(os.environ.get("THESIS2_N_PREDICT", "256"))
# A run must never sit on the card's edge: the Windows display driver owns part
# of the 24576 MiB card, and a GPU fault there reboots the whole PC. Abort the
# profile if allocation ever exceeds this.
VRAM_LIMIT_MIB = int(os.environ.get("THESIS2_VRAM_LIMIT_MIB", "24000"))
HOST = os.environ.get("THESIS2_HOST", "127.0.0.1")

# Flags shared by every profile. These mirror the launcher so that a profile is
# a change of one dimension, not a change of the whole command line.
COMMON = [
    "--model", str(TARGET),
    "--mmproj", str(MMPROJ),
    "--image-min-tokens", "1024",
    "--n-gpu-layers", "all",
    "--ctx-size", "131072",
    "--parallel", "1",
    "--flash-attn", "on",
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--cache-prompt",
    "--cache-ram", "0",
    "--threads", "4",
    "--threads-batch", "8",
    "--temp", "1.0",
    "--top-k", "20",
    "--top-p", "0.95",
    "--min-p", "0.0",
    "--host", HOST,
    "--port", str(PORT),
    "--metrics",
    "--no-webui",
]

MTP3 = [
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3",
    "--model-draft", str(MTP),
    "--n-gpu-layers-draft", "all",
    "--cache-type-k-draft", "q4_0",
    "--cache-type-v-draft", "q4_0",
]


def mtp(n, extra=(), model="sidecar"):
    """MTP draft arguments. model="embedded" omits --model-draft (target MTP head)."""
    args = ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]
    if model == "embedded":
        return args + list(extra)
    args += [
        "--model-draft", str(MTP),
        "--n-gpu-layers-draft", "all",
        "--cache-type-k-draft", "q4_0",
        "--cache-type-v-draft", "q4_0",
    ]
    args.extend(extra)
    return args


# ---------------------------------------------------------------------------
# Profile table. Each entry: binary, extra args, env, one-line note.
# ---------------------------------------------------------------------------
PROFILES = {
    # ---- stock llama.cpp (baseline) ----
    "stock-mtp3": (STOCK, mtp(3), {}, "shipped launcher configuration (MTP3 sidecar)"),
    "stock-mtp1": (STOCK, mtp(1), {}, "shorter MTP draft"),
    "stock-mtp2": (STOCK, mtp(2), {}, "shorter MTP draft"),
    "stock-mtp4": (STOCK, mtp(4), {}, "longer MTP draft"),
    "stock-nomtp": (STOCK, [], {}, "no speculation at all"),
    "stock-mtp3-embed": (
        STOCK, mtp(3, model="embedded"), {},
        "embedded MTP from the target GGUF, sidecar dropped",
    ),
    "stock-mtp2-ngram": (
        STOCK,
        ["--spec-type", "draft-mtp,ngram-mod",
         "--spec-ngram-mod-n-match", "32",
         "--spec-draft-n-max", "2",
         "--model-draft", str(MTP),
         "--n-gpu-layers-draft", "all",
         "--cache-type-k-draft", "q4_0",
         "--cache-type-v-draft", "q4_0"],
        {},
        "MTP2 plus n-gram lookup",
    ),
    "stock-backend": (
        STOCK, mtp(3) + ["--backend-sampling"], {},
        "target sampling on the GPU (CPU is a 2400G)",
    ),
    "stock-be-mtp2": (
        STOCK, mtp(2) + ["--backend-sampling"], {},
        "GPU target sampling with MTP2",
    ),
    "stock-be-mtp2-ngram": (
        STOCK,
        ["--spec-type", "draft-mtp,ngram-mod", "--spec-ngram-mod-n-match", "32",
         "--spec-draft-n-max", "2", "--backend-sampling",
         "--model-draft", str(MTP), "--n-gpu-layers-draft", "all",
         "--cache-type-k-draft", "q4_0", "--cache-type-v-draft", "q4_0"],
        {},
        "GPU target sampling, MTP2 and n-gram lookup",
    ),
    "stock-be-ub1024": (
        STOCK, mtp(2) + ["--backend-sampling", "--batch-size", "4096",
                         "--ubatch-size", "1024", "--threads", "8", "--threads-batch", "8"],
        {},
        "GPU target sampling, MTP2, UB1024 prefill",
    ),
    "stock-be-embed": (
        STOCK, mtp(2, model="embedded") + ["--backend-sampling"], {},
        "GPU target sampling on the embedded MTP head",
    ),
    "stock-be-mtp1": (
        STOCK, mtp(1) + ["--backend-sampling"], {}, "GPU target sampling with MTP1",
    ),
    "stock-be-mtp4": (
        STOCK, mtp(4) + ["--backend-sampling"], {}, "GPU target sampling with MTP4",
    ),
    "stock-be-only": (
        STOCK, ["--backend-sampling"], {},
        "GPU target sampling without speculation (control)",
    ),
    "stock-be-q8draft": (
        STOCK, mtp(2) + ["--backend-sampling", "--cache-type-k-draft", "q8_0",
                         "--cache-type-v-draft", "q8_0"],
        {},
        "GPU target sampling, MTP2, q8_0 drafter cache",
    ),
    "stock-t8": (
        STOCK, mtp(3) + ["--threads", "8", "--threads-batch", "8"], {},
        "shipped MTP3 with 8 host threads instead of 4",
    ),
    "stock-be-t8": (
        STOCK, mtp(2) + ["--backend-sampling", "--threads", "8", "--threads-batch", "8"],
        {}, "GPU target sampling, MTP2, 8 host threads",
    ),
    "stock-ub1024": (
        STOCK,
        mtp(3) + ["--batch-size", "4096", "--ubatch-size", "1024",
                  "--threads", "8", "--threads-batch", "8"],
        {},
        "bigger prefill ubatch and 8 threads",
    ),
    # ---- llamAmpere fork, identical weights and flags ----
    "amp-be-gv-map": (
        AMPERE,
        ["--fit", "off"] + mtp(2, ["--spec-draft-vocab-map", str(VOCAB_MAP)])
        + ["--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "fork, MTP2, GPU sampling, shortlist and GPU draft verification",
    ),
    "amp-be-gv-mtp3": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "fork, MTP3, GPU sampling and GPU draft verification",
    ),
    "amp-be-gv-embed": (
        AMPERE,
        ["--fit", "off"] + mtp(2, ["--spec-draft-vocab-map", str(VOCAB_MAP)],
                               model="embedded") + ["--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "fork, embedded MTP head, shortlist, GPU sampling and GPU verification",
    ),
    "amp-chain-gv": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--spec-chain", "3", "--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "fork, chained MTP3, GPU sampling and GPU verification",
    ),
    "amp-be-mtp2": (
        AMPERE, ["--fit", "off"] + mtp(2) + ["--backend-sampling"], {},
        "fork, MTP2 and GPU target sampling (direct kernel A/B vs stock-be-mtp2)",
    ),
    "amp-be-map": (
        AMPERE, ["--fit", "off"] + mtp(2) + ["--backend-sampling",
                                             "--spec-draft-vocab-map", str(VOCAB_MAP)],
        {},
        "fork, MTP2, GPU sampling, 65K draft shortlist",
    ),
    "amp-be-map-ub1024": (
        AMPERE,
        ["--fit", "off"] + mtp(2) + ["--backend-sampling",
                                     "--spec-draft-vocab-map", str(VOCAB_MAP),
                                     "--batch-size", "4096", "--ubatch-size", "1024",
                                     "--threads", "8", "--threads-batch", "8"],
        {},
        "fork, shortlist, GPU sampling, UB1024 prefill",
    ),
    "amp-be-embed-map": (
        AMPERE,
        ["--fit", "off"] + mtp(2, ["--spec-draft-vocab-map", str(VOCAB_MAP)],
                               model="embedded") + ["--backend-sampling"],
        {},
        "fork, embedded MTP head, shortlist, GPU sampling",
    ),
    "amp-be-gv": (
        AMPERE, ["--fit", "off"] + mtp(2) + ["--backend-sampling"], 
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "fork, MTP2, GPU sampling and GPU draft verification",
    ),
    "amp-be-turbo3": (
        AMPERE,
        ["--fit", "off"] + mtp(2) + ["--backend-sampling",
                                     "--cache-type-k", "q8_0", "--cache-type-v", "turbo3"],
        {"GGML_Q8_TURBO3_MMA_FUSED": "1"},
        "fork, MTP2, GPU sampling, fused q8_0/turbo3 attention",
    ),
    "amp-be-mtp3": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--backend-sampling"], {},
        "fork, MTP3 and GPU sampling",
    ),
    "amp-mtp3": (AMPERE, ["--fit", "off"] + mtp(3), {}, "fork, same flags as stock"),
    "amp-mtp3-embed": (
        AMPERE, ["--fit", "off"] + mtp(3, model="embedded"), {},
        "fork, embedded MTP head, no sidecar",
    ),
    "amp-map": (
        AMPERE,
        ["--fit", "off"] + mtp(3, ["--spec-draft-vocab-map", str(VOCAB_MAP)]),
        {},
        "fork, sidecar + 65K draft vocabulary shortlist",
    ),
    "amp-embed-map": (
        AMPERE,
        ["--fit", "off"] + mtp(3, ["--spec-draft-vocab-map", str(VOCAB_MAP)], model="embedded"),
        {},
        "fork, embedded MTP head + shortlist",
    ),
    "amp-chain3": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--spec-chain", "3", "--backend-sampling"], {},
        "fork, chained MTP: three drafts in one GPU decode",
    ),
    "amp-chain-map": (
        AMPERE,
        ["--fit", "off"] + mtp(3) + ["--spec-chain", "3", "--backend-sampling",
                                     "--spec-draft-vocab-map", str(VOCAB_MAP)],
        {},
        "chained MTP with the shortlist",
    ),
    "amp-dflash7": (
        AMPERE,
        ["--fit", "off", "--spec-type", "draft-dflash", "--model-draft", str(DFLASH),
         "--n-gpu-layers-draft", "all", "--spec-draft-n-max", "7",
         "--cache-type-k-draft", "f16", "--cache-type-v-draft", "f16",
         "--backend-sampling"],
        {},
        "fork with the DFlash2 block drafter (Q8_0), 7 draft tokens",
    ),
    "amp-dflash3": (
        AMPERE,
        ["--fit", "off", "--spec-type", "draft-dflash", "--model-draft", str(DFLASH),
         "--n-gpu-layers-draft", "all", "--spec-draft-n-max", "3",
         "--cache-type-k-draft", "f16", "--cache-type-v-draft", "f16",
         "--backend-sampling"],
        {},
        "fork with the DFlash2 block drafter (Q8_0), 3 draft tokens",
    ),
    "amp-dflash7-map": (
        AMPERE,
        ["--fit", "off", "--spec-type", "draft-dflash", "--model-draft", str(DFLASH),
         "--n-gpu-layers-draft", "all", "--spec-draft-n-max", "7",
         "--cache-type-k-draft", "f16", "--cache-type-v-draft", "f16",
         "--backend-sampling", "--spec-draft-vocab-map", str(VOCAB_MAP)],
        {},
        "DFlash2 drafter plus the 65K draft vocabulary shortlist",
    ),
    "amp-adaptive": (
        AMPERE,
        ["--fit", "off"] + mtp(8) + ["--spec-type", "draft-mtp-adaptive",
                                     "--spec-draft-n-min-adaptive", "2",
                                     "--backend-sampling"],
        {},
        "fork, adaptive MTP depth 2..8",
    ),
    "amp-map-q8draft": (
        AMPERE,
        ["--fit", "off"] + mtp(3) + ["--spec-draft-vocab-map", str(VOCAB_MAP),
                                     "--cache-type-k-draft", "q8_0",
                                     "--cache-type-v-draft", "q8_0"],
        {},
        "shortlist plus the fused q8_0 drafter cache",
    ),
    "amp-ub1024": (
        AMPERE,
        ["--fit", "off"] + mtp(3) + ["--batch-size", "4096", "--ubatch-size", "1024",
                                     "--threads", "8", "--threads-batch", "8"],
        {},
        "fork with the upstream-accepted UB1024 prefill",
    ),
    "amp-turbo3": (
        AMPERE,
        ["--fit", "off"] + mtp(3) + ["--cache-type-k", "q8_0", "--cache-type-v", "turbo3"],
        {"GGML_Q8_TURBO3_MMA_FUSED": "1"},
        "fork with fused q8_0/turbo3 attention KV",
    ),
    "amp-gpuverify": (
        AMPERE,
        ["--fit", "off"] + mtp(3) + ["--spec-draft-vocab-map", str(VOCAB_MAP)],
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "shortlist plus GPU verification of drafts",
    ),
    "amp-pmin45": (
        AMPERE,
        ["--fit", "off"] + mtp(3, ["--spec-draft-p-min", "0.45"]),
        {},
        "fork with the draft emission gate",
    ),
    "amp-backend": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--backend-sampling"], {},
        "fork with GPU target sampling",
    ),
    "amp-nofuse": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--backend-sampling"], {"GGML_CUDA_FUSE_CHAIN": "0"},
        "fork with elementwise chain fusion disabled (control for the fusion)",
    ),
    "amp-nographs": (
        AMPERE, ["--fit", "off"] + mtp(3) + ["--backend-sampling"], {"GGML_CUDA_DISABLE_GRAPHS": "1"},
        "fork with CUDA graphs disabled (control)",
    ),
    # ---- third pass: shared compute buffer (patches/llamampere-compute-slack.patch) ----
    "amp-be-gv-embed-slack": (
        AMPERE,
        ["--fit", "off"] + mtp(2, ["--spec-draft-vocab-map", str(VOCAB_MAP)],
                               model="embedded") + ["--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1", **SLACK_ENV},
        "promoted fork profile + compute slack (MTP shares the target compute buffer)",
    ),
    "amp-chain-gv-embed": (
        AMPERE,
        ["--fit", "off"] + mtp(3, model="embedded") + ["--spec-chain", "3", "--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1"},
        "chained MTP3 on the embedded head, GPU verification",
    ),
    "amp-chain-gv-embed-slack": (
        AMPERE,
        ["--fit", "off"] + mtp(3, model="embedded") + ["--spec-chain", "3", "--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1", **SLACK_ENV},
        "chained MTP3 on the embedded head, GPU verification, compute slack",
    ),
    "amp-chain-gv-embed-map-slack": (
        AMPERE,
        ["--fit", "off"] + mtp(3, ["--spec-draft-vocab-map", str(VOCAB_MAP)], model="embedded")
        + ["--spec-chain", "3", "--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1", **SLACK_ENV},
        "chained MTP3, embedded head, shortlist, GPU verification, compute slack",
    ),
    "amp-be-gv-embed-pmin-slack": (
        AMPERE,
        ["--fit", "off"] + mtp(4, ["--spec-draft-vocab-map", str(VOCAB_MAP),
                                   "--spec-draft-p-min", "0.75"], model="embedded")
        + ["--backend-sampling"],
        {"LLAMA_MTP_GPU_VERIFY": "1", **SLACK_ENV},
        "promoted + MTP depth 4 gated at p-min 0.75 (registry 4090 recipe), compute slack",
    ),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_free(timeout=900):
    """Refuse to run while another server holds the port or the card.

    Matched on the executable, not on any command line that merely mentions
    llama-server (a build target does, which used to stall the harness).
    """
    for _ in range(timeout):
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
        busy = []
        for line in out.stdout.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) < 2:
                continue
            if os.path.basename(fields[1].split()[0]).startswith("llama-server"):
                busy.append(f"{fields[0]} {fields[1][:100]}")
        port = subprocess.run(["ss", "-ltnH", f"sport = :{PORT}"], capture_output=True, text=True)
        if not busy and not port.stdout.strip():
            return True
        log(f"card or port busy ({busy[:1] or 'port ' + str(PORT)}), waiting")
        time.sleep(5)
    return False


def http_json(path, body=None, timeout=1800):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def gpu_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,power.draw,clocks.sm,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    try:
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        return {"vram_mib": int(parts[0]), "watts": float(parts[1]),
                "sm_mhz": int(parts[2]), "temp_c": int(parts[3])}
    except Exception:
        return {}


LOAD_RE = re.compile(
    r"(model buffer|KV self|recurrent|compute buffer|CUDA0|offloaded|n_ctx|CUDA_Host|"
    r"draft|spec|shortlist|checkpoint|WRN|ERR)", re.I)


def start_server(binary, extra, env, log_path):
    cmd = [str(binary)] + COMMON + extra
    # Hermetic environment: knobs set by an earlier profile must not leak forward.
    full_env = dict(os.environ)
    for knob in ("LLAMA_MTP_GPU_VERIFY", "LLAMA_SPEC_PQ", "GGML_Q8_TURBO3_MMA_FUSED",
                 "GGML_Q8_TURBO3_MMA_MIN_Q", "GGML_Q8_TURBO3_MMA_NCOLS1_MIN",
                 "GGML_CUDA_FUSE_CHAIN", "GGML_CUDA_Q8CACHE", "LLAMA_SPEC_CHAIN",
                 "LLAMA_SPEC_DRAFT_VOCAB", "CUDA_SCALE_LAUNCH_QUEUES",
                 "GGML_CUDA_DISABLE_GRAPHS", "LLAMA_MTP_VERBOSE",
                 "GGML_COMPUTE_SLACK_MIB", "LD_LIBRARY_PATH"):
        full_env.pop(knob, None)
    full_env.update(env)
    logfile = open(log_path, "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT,
                            env=full_env, start_new_session=True)
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=2) as r:
                if r.status == 200:
                    return proc, time.time() - t0, cmd, full_env
        except Exception:
            time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not become healthy in 600 s")


def stop_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    time.sleep(float(os.environ.get("THESIS2_COOLDOWN", "8")))


def prefill_rows(log_path):
    """Keep the memory-relevant load lines so the row is self-describing."""
    keep = []
    try:
        for line in Path(log_path).read_text(errors="replace").splitlines():
            if LOAD_RE.search(line) and len(line) < 400:
                keep.append(line.strip())
            if "server is listening" in line:
                break
    except Exception:
        pass
    return keep[-40:]


def run_profile(name, cfg, outdir, fixtures, temperature, tag):
    binary, extra, env, note = cfg
    if not Path(binary).exists():
        log(f"SKIP {name}: binary missing ({binary})")
        return []
    log_path = outdir / f"{name}.log"
    log(f"start {name} ({note})")
    proc, load_s, cmd, full_env = start_server(binary, extra, env, log_path)
    try:
        head = prefill_rows(log_path)
        gpu = gpu_mib()
        log(f"  ready in {load_s:.1f}s, {gpu}")
        vram_peak = gpu.get("vram_mib", 0)
        if vram_peak > VRAM_LIMIT_MIB:
            log(f"  ABORT: {vram_peak} MiB after load exceeds the {VRAM_LIMIT_MIB} MiB guard")
            return [{"tag": tag, "profile": name, "note": note, "error":
                     f"vram {vram_peak} MiB > guard {VRAM_LIMIT_MIB} MiB after load",
                     "load_s": round(load_s, 2), "gpu": gpu}]
        if os.environ.get("THESIS2_WARMUP") == "1":
            # Measure the effect of a startup warm-up request: the first real
            # request otherwise pays CUDA-graph capture for the prefill shapes.
            t0 = time.time()
            warm = http_json("/completion", {"prompt": "warm up", "n_predict": 4,
                                             "temperature": 0, "ignore_eos": True,
                                             "cache_prompt": False, "stream": False})
            log("  warmup done in %.2fs (prefill %.1f tok/s)"
                % (time.time() - t0,
                   (warm.get("timings") or {}).get("prompt_per_second") or 0.0))
        rows = []
        measured = [f for f in fixtures if f not in PREFILL_ONLY]
        prefill_fx = [f for f in fixtures if f in PREFILL_ONLY]
        for fixture in measured + prefill_fx:
            prompt = FIXTURES[fixture]
            body = {
                "prompt": prompt,
                # Long-prompt fixtures exist to time cold prefill; do not spend
                # decode time on them.
                "n_predict": 8 if fixture in PREFILL_ONLY else N_PREDICT,
                "temperature": temperature,
                "top_k": 0 if temperature == 0 else 20,
                "top_p": 1.0 if temperature == 0 else 0.95,
                "min_p": 0.0,
                "ignore_eos": True,
                "cache_prompt": False,
                "stream": False,
            }
            t0 = time.time()
            try:
                raw = http_json("/completion", body)
            except urllib.error.HTTPError as exc:
                rows.append({"profile": name, "fixture": fixture, "error": str(exc),
                             "body": exc.read().decode(errors="replace")[:500]})
                continue
            wall = time.time() - t0
            timings = raw.get("timings", {}) or {}
            text = raw.get("content", "") or ""
            gpu = gpu_mib()
            vram_peak = max(vram_peak, gpu.get("vram_mib", 0))
            row = {
                "tag": tag,
                "profile": name,
                "note": note,
                "fixture": fixture,
                "temperature": temperature,
                "load_s": round(load_s, 2),
                "wall_s": round(wall, 3),
                "prompt_n": timings.get("prompt_n"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "predicted_n": timings.get("predicted_n"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
                "tokens_evaluated": raw.get("tokens_evaluated"),
                "tokens_predicted": raw.get("tokens_predicted"),
                "tokens_cached": raw.get("tokens_cached"),
                "stop_type": raw.get("stop_type"),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chars": len(text),
                "gpu": gpu,
                "vram_peak_mib": vram_peak,
            }
            rows.append(row)
            (outdir / "raw").mkdir(exist_ok=True)
            (outdir / "raw" / f"{name}__{fixture}.json").write_text(
                json.dumps({"row": row, "raw": raw}, indent=1)[:200000])
            if vram_peak > VRAM_LIMIT_MIB:
                log(f"  ABORT after {fixture}: peak {vram_peak} MiB exceeds the "
                    f"{VRAM_LIMIT_MIB} MiB guard; stopping this profile")
                break
            log("  %-12s decode %s tok/s | prefill %s tok/s | draft %s/%s | %ss"
                % (fixture,
                   _fmt(timings.get("predicted_per_second")),
                   _fmt(timings.get("prompt_per_second")),
                   timings.get("draft_n_accepted"), timings.get("draft_n"),
                   round(wall, 1)))
        return rows
    finally:
        stop_server(proc)


def _fmt(v):
    return "n/a" if v is None else f"{float(v):.2f}"


def run_cache_pair(name, cfg, outdir, temperature, tag):
    """Measure prompt-cache reuse: same long prefix, then a small appended turn."""
    binary, extra, env, note = cfg
    if not Path(binary).exists():
        return []
    log_path = outdir / f"{name}.log"
    proc, load_s, cmd, full_env = start_server(binary, extra, env, log_path)
    rows = []
    try:
        base_prompt = FIXTURES["cache-a"]
        follow = base_prompt + "\nNow answer again with one integer: how many rows are listed above?\n"
        for label, prompt, cache in (("cold", base_prompt, True), ("append", follow, True)):
            body = {"prompt": prompt, "n_predict": 16, "temperature": 0, "ignore_eos": True,
                    "cache_prompt": cache, "stream": False}
            t0 = time.time()
            raw = http_json("/completion", body)
            wall = time.time() - t0
            t = raw.get("timings", {}) or {}
            row = {
                "tag": tag, "profile": name, "note": note, "fixture": f"cache-{label}",
                "temperature": temperature, "load_s": round(load_s, 2),
                "wall_s": round(wall, 3),
                "prompt_n": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second"),
                "predicted_n": t.get("predicted_n"),
                "predicted_per_second": t.get("predicted_per_second"),
                "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                "tokens_evaluated": raw.get("tokens_evaluated"),
                "tokens_cached": raw.get("tokens_cached"),
                "sha256": hashlib.sha256((raw.get("content") or "").encode()).hexdigest(),
                "gpu": gpu_mib(),
            }
            rows.append(row)
            log("  cache-%-7s prompt_n=%s cached=%s prefill %s tok/s wall %ss"
                % (label, row["prompt_n"], row["tokens_cached"],
                   _fmt(t.get("prompt_per_second")), round(wall, 2)))
        return rows
    finally:
        stop_server(proc)


def main():
    names = os.environ.get("THESIS2_PROFILES", "").split(",")
    names = [n.strip() for n in names if n.strip()]
    if not names:
        print("set THESIS2_PROFILES (comma separated). Available:")
        for k, v in PROFILES.items():
            print(f"  {k:24s} {v[3]}")
        return 1
    tag = os.environ.get("THESIS2_TAG", time.strftime("run-%Y%m%d-%H%M%S"))
    fixtures = os.environ.get("THESIS2_FIXTURES",
                              "novel-code,prose,patch,copy,prefill8k").split(",")
    temperature = float(os.environ.get("THESIS2_TEMP", "0"))
    outdir = ROOT / "benchmarks" / f"thesis2-{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / "rows.jsonl"

    if not wait_free():
        log("aborting: another server holds the port")
        return 2

    for name in names:
        cfg = PROFILES.get(name)
        if cfg is None:
            log(f"unknown profile {name}")
            continue
        try:
            rows = run_profile(name, cfg, outdir, fixtures, temperature, tag)
        except Exception as exc:  # keep the campaign going
            log(f"FAILED {name}: {exc}")
            rows = [{"tag": tag, "profile": name, "error": str(exc)}]
        with rows_path.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        log(f"wrote {len(rows)} rows for {name}")
    log(f"done, results in {rows_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
