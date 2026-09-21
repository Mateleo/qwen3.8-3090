# Faster classic Qwen3.8-27B on this RTX 3090

Research and handoff. First pass 2026-09-21; this revision is the second pass of
the same day and **corrects the first pass's numbers**, which were not
reproducible (section 2). Target: faster single-user decode and/or prefill of the
**classic model**, preserving its quality as far as practical. Existing 131,072
context and vision support are the comparison baseline.

Constraints in force: nothing beyond 24 GiB of VRAM, no host-RAM fallback (this
machine has 9.7 GiB of RAM), and low parallelism in builds and benchmarks because
the host has 4 cores and WSL is fragile.

Bottom line: a **verified single change worth about +23% decode on the real
production path**, at 1.1 GiB *less* VRAM than the shipped configuration, plus a
harness and a protocol that can keep measuring. Section 1 states the promoted
configuration; section 6 lists what is still open.

## 1. Promoted configuration (measured, verified)

`start.sh` selects a runtime with `QWEN_SERVER_BUILD` and always adds
`--backend-sampling`:

```bash
QWEN_SERVER_BUILD=ampere ./start.sh --detach   # +23%, 22,438 MiB peak
QWEN_SERVER_BUILD=stock  ./start.sh            # shipped runtime, unchanged
```

The `ampere` build is `~/llamAmpere-src/repo/build-sm86/bin/llama-server`
(llamAmpere v0.3.1, built here from `2cb1693`, CUDA 12.0, sm_86). The launcher
detects its features from `--help` instead of assuming them, so one script drives
both runtimes. Its measured flag set:

```
--spec-type draft-mtp --spec-draft-n-max 2
--spec-draft-vocab-map ~/models/Qwen3.8-27B/MTP/mtp-vocab-65536.txt
--backend-sampling
--fit off
(no --model-draft: this GGUF's own MTP layer is used)
LLAMA_MTP_GPU_VERIFY=1
```

Production-shaped measurement (`benchmarks/thesis2-confirm2`): chat endpoint,
temperature 1, top-k 20, top-p 0.95, reasoning on at xhigh effort, 256 output
tokens, three seeds, five workloads, otherwise idle GPU.

| Configuration | novel-code | prose | patch | copy | mean | acceptance | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| shipped: stock binary, MTP3 sidecar | 36.7 +-3.6 | 33.5 +-1.5 | 39.1 +-3.4 | 37.7 +-4.1 | 36.7 | 54.3% | 23,533 MiB |
| stock + `--backend-sampling` + embedded MTP | 36.2 +-0.9 | 33.7 +-0.4 | 38.4 +-2.8 | 37.3 +-3.3 | 36.4 (-1.3%) | 64.2% | 22,690 MiB |
| **ampere, promoted flag set** | 45.3 +-4.9 | 43.2 +-1.7 | 48.2 +-1.2 | 45.4 +-2.8 | **45.5 (+22.7%)** | 58.4% | 22,438 MiB |

Per-fixture deltas against the shipped configuration: +23.4% novel code, +29.2%
prose, +23.3% code editing, +20.5% copying, +17.1% decode after an 8.5K prompt.
The promoted configuration is the only change in this document that wins on every
workload, on the sampled production path, with stable repeats.

Correctness evidence for the promotion:

- **Within the fork, speculation changes nothing.** Every fork configuration
  tested (MTP depth 2/3, the 65K shortlist, GPU verification, chained MTP, the
  embedded MTP head, DFlash2 at depth 3 and 7) reproduced the same greedy
  continuation, byte for byte, on all five fixtures. The only configurations that
  changed output were `-ctv turbo3` (lossy 3.25-bit V) and the adaptive drafter.
- **Across runtimes, output does change.** The fork's greedy text differs from
  the stock binary on 4 of 5 fixtures. The cause is documented and structural:
  the fork carries upstream PR 28068, which makes the Gated DeltaNet q/k
  normalization match the reference flash-linear-attention definition
  (`x * rsqrt(sum(x*x) + eps)`) instead of `ggml_l2_norm`. It is a model-math
  change, not a speculation artifact, and it is why `stock` stays the default:
  pick `ampere` for speed, keep `stock` if byte-identical continuity matters.

VRAM: every configuration in this document fits the card, and the promoted one
has the *most* headroom (speculative sidecar dropped, about 2.1 GiB free for the
Windows desktop). `--cache-ram 0` is kept everywhere, so no run parks state in
host RAM, and no run used `mlock` or CPU offload.

## 2. Corrections to the first pass

The first pass reported 52.03 tok/s for MTP3 on a 256-token greedy screen and
42.99-55.09 tok/s across profiles, from `benchmarks/thesis-screen.py`, which no
longer exists. Re-running the same configuration gives 32.67 tok/s at the same
256-token greedy settings and 36.7 tok/s at production sampling. The original
numbers are **not reproducible on this machine and are withdrawn**; no decision
rests on them. Possible causes, none verified: a different binary, prompt-cache
effects, or an unrecorded launch difference. Everything in this revision comes
from harness runs whose raw rows, logs and output hashes are kept under
`benchmarks/thesis2-*`.

Two first-pass claims survive. The model metadata does advertise
`qwen35.nextn_predict_layers=1`, and the MTP sidecar is optional: the GGUF carries
its own MTP layer (`blk.64.nextn.*`, Q6_K projection), the stock binary uses it
when `--model-draft` is omitted, with the same acceptance and the same greedy
output, and it frees about 1.1 GiB.

## 3. Verified local state

| Item | Observed value |
|---|---|
| Project | `/home/mateleo/qwen3.8-server` |
| Classic launch | `./start.sh --model current` |
| Target | `~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,559,178,144 bytes |
| Draft sidecar | `~/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, 1,369,590,656 bytes, carries its own 248,320-row output head |
| Draft shortlist | `~/models/Qwen3.8-27B/MTP/mtp-vocab-65536.txt` (65,536 ids, header `llama-mtp-vocab-v1 248320 65536`) |
| DFlash2 drafter | `~/models/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q8_0.gguf`, 2,056,414,208 bytes, quantized here from `dflash2-BF16.partial.gguf` |
| Vision | `~/models/Qwen3.8-27B/mmproj-F16.gguf` |
| Stock runtime | `~/llama.cpp/build/bin/llama-server`, 0.4.0-dev, commit `6a1a922d`, CUDA 12.0, `86-real` |
| Fork runtime | `~/llamAmpere-src/repo/build-sm86/bin/llama-server`, ggml 0.24.0, commit `2cb1693`, built here at `-j2` |
| GPU | RTX 3090, 24,576 MiB, 245 W limit, one Windows display attached |
| Host | WSL2 on Windows 10, 9.7 GiB RAM, 24 GiB swap, Ryzen 5 2400G (4c/8t) |
| Serving | 131,072 context, one slot, all GPU layers, FA on, Q4_0 K/V, `-b 2048 -ub 512 -t 4 -tb 8`, `--cache-prompt --cache-ram 0` |
| Production generation | temperature 1, top-k 20, top-p 0.95, min-p 0; thinking on, effort xhigh |

The loader identifies architecture `qwen35`: 64 main layers plus one MTP layer, 16
full-attention layers, 48 recurrent layers, 24 Q heads / 4 KV heads, head
dimension 256, 27.32B parameters. Q4_0 K/V at 131,072 costs 2,304 MiB.

`benchmarks/prefill-rtx3090-128k.md` (older campaign) records about 993 tok/s at
8K, 737 at 64K and 540 at 128K for this quant. Long prefill remains a large part
of interactive latency; nothing promoted here changes it.

## 4. What was measured

Harness (new, kept): `benchmarks/thesis2-screen.py` (raw completions, fixed
fixtures, one server at a time, VRAM guard), `benchmarks/thesis2-confirm.py`
(production sampling), `benchmarks/thesis2-report.py` and
`benchmarks/thesis2-stats.py` (tables), `benchmarks/thesis2_fixtures.py`
(workloads), `benchmarks/probe-spec.sh` (does this binary really speculate),
`benchmarks/gguf-inspect.py` (metadata and tensor list without a VRAM load).
The guard aborts a profile if allocation passes 24,000 MiB.

Fixtures: novel code, prose, a code edit that restates a supplied file, verbatim
copying, and 8.5K/34K prefill. Each row carries server timings, wall time, draft
counts, VRAM, and the SHA256 of the generated text.

### 4a. Stock runtime, greedy screen (256 tokens, `thesis2-stock-screen1..4`)

Some of these rows ran while the fork was compiling at `-j2`, so absolute values
are conservative; the ranking is what matters.

| Profile | novel | prose | patch | copy | geometric mean | acceptance |
|---|---:|---:|---:|---:|---:|---:|
| shipped, MTP3 sidecar | 34.17 | 31.75 | 31.48 | 33.36 | 32.67 | 62.9% |
| MTP3 + `--backend-sampling` | 38.62 | 44.33 | 45.03 | 52.79 | 44.91 (+37.5%) | 62.9% |
| MTP2 + n-gram + backend sampling | 40.34 | 43.01 | 43.24 | 98.84 | 52.18 (+59.7%) | 70.8% |
| MTP2 + backend sampling | 41.34 | 43.17 | 44.42 | 48.74 | 44.33 (+35.7%) | 70.7% |
| embedded MTP + backend sampling | 39.27 | 42.92 | 45.26 | 49.38 | 44.06 (+34.9%) | 69.2% |
| MTP2 + n-gram, no backend sampling | 34.34 | 36.61 | 33.36 | 77.76 | 42.49 (+30.1%) | 70.8% |
| MTP3, `-t 8 -tb 8` | 36.59 | 39.44 | 40.43 | 47.08 | 40.71 (+24.6%) | 62.9% |
| MTP3, `-b 4096 -ub 1024 -t 8 -tb 8` | 35.71 | 39.72 | 39.32 | 46.73 | 40.18 (+23.0%) | 64.9% |
| MTP2 | 36.94 | 36.57 | 38.38 | 40.24 | 38.01 (+16.3%) | 70.7% |
| MTP1 | 35.12 | 35.66 | 35.17 | 37.24 | 35.79 (+9.5%) | 76.9% |
| no speculation | 27.50 | 26.64 | 27.07 | 25.79 | 26.74 (-18.1%) | - |

`--backend-sampling` was the largest greedy win and is now in the launcher. It
moves the sampler onto the card, which matters because this host is a 2400G and
the vocabulary is 248,320 rows wide. Three cautions, all measured:

- At **production sampling (temperature 1) the stock build gains nothing** from
  it (36.4 vs 36.7 tok/s). The +37% is a greedy-path effect (GPU argmax
  verification). A request that asks for `temperature 0` while the server's
  `top-k`/`top-p` remain in force is not eligible and falls back to CPU
  verification, which the server logs. It is kept because it costs nothing
  measurable and helps greedy and tool traffic; it is not the reason to switch
  runtimes.
- n-gram lookup doubles copy-heavy decode (+196% on the copy fixture) and costs
  about 2% on novel code and edits. Left off by default; `--spec-type
  draft-mtp,ngram-mod --spec-ngram-mod-n-match 32` turns it on for sessions that
  rewrite large files verbatim.
- More host threads help only while sampling is on the CPU. With
  `--backend-sampling`, `-t 4` and `-t 8` measure the same.

### 4b. Fork runtime, greedy screen (`thesis2-amp-screen1`, `-screen2`)

Same weights, same context, same KV types, quiet GPU.

| Profile (all with `--backend-sampling`) | novel | prose | patch | copy | geometric mean | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| chained MTP3 + GPU verification | 54.57 | 57.56 | 57.39 | 75.58 | 60.76 | 23,175 MiB |
| MTP2 + shortlist + GPU verification | 55.57 | 54.95 | 56.20 | 65.35 | 57.87 | 23,084 MiB |
| MTP3 + GPU verification | 54.83 | 52.34 | 55.62 | 69.88 | 57.79 | 23,234 MiB |
| **MTP2 + shortlist + GPU verification, embedded MTP (promoted)** | 54.29 | 52.86 | 54.78 | 64.09 | 56.34 | 21,907 MiB |
| MTP2 + GPU verification | 51.09 | 51.74 | 51.48 | 60.95 | 53.66 | 23,084 MiB |
| MTP2 + shortlist | 48.28 | 48.17 | 49.82 | 56.15 | 50.51 | 23,232 MiB |
| chained MTP3 | 49.46 | 49.24 | 51.90 | 62.20 | 52.95 | 23,145 MiB |
| MTP2, elementwise chain fusion off (control) | 46.13 | 45.62 | 46.67 | 55.10 | 48.23 | 23,175 MiB |
| MTP2, adaptive depth 2..8 | 39.14 | 37.40 | 47.40 | 77.53 | 48.16 | 23,923 MiB |
| MTP2 + fused q8_0/turbo3 attention KV | 47.13 | 43.65 | 47.32 | 54.09 | 47.90 | 23,756 MiB |
| MTP2, embedded MTP head | 44.77 | 46.31 | 47.45 | 51.33 | 47.40 | 23,414 MiB |

Two mechanisms carry the fork: GPU verification of drafts (the verification
sampling leaves the 2400G) and the 65,536-row draft vocabulary shortlist (the
drafter reads 0.19 GiB of head rows per draft step instead of 0.72 GiB; the
sidecar carries a full-vocabulary head). Chained MTP is the fastest measured
configuration but was **not** promoted: section 5.

### 4c. DFlash2 (finished and measured, not adopted)

The local `dflash2-BF16.partial.gguf` was complete after all (81 tensors, 16.00
BPW, 3,671 MiB). `llama-quantize` from the fork build converted it to Q8_0
(2,056,414,208 bytes) in 9 seconds. With `--spec-type draft-dflash` it loads and
drafts (acceptance 45%, mean accepted length 4.11 at depth 7). Throughput at
greedy sampling: 53.46 geometric mean at depth 3 and 43.83 at depth 7, against
60.76 for chained MTP3 on the same runtime, and with the highest VRAM of the set.
Conclusion: DFlash2 is real and usable on this machine, and it is not the fastest
option here. Keep the Q8_0 file; retest it if the fork's DFlash path gains a
cheaper block drafter.

### 4d. Prefill

Prefill is unaffected by everything promoted, as expected: no prefill kernel or
batch setting was changed. Single cold observations: 34K prefill 689 tok/s
(shipped), 669 (stock + backend sampling), 706 (promoted, +2.3%); 8.5K prefill
831-920 tok/s across profiles. The first request after load is 2-3x slower on its
prefill portion because the CUDA graphs for those shapes are captured then.

## 5. Host crash during this investigation

At 21:44:45 the PC rebooted while the confirm stage was running the *chained MTP3
+ GPU verification* profile (23.2 GiB VRAM). Windows recorded `nvlddmkm` errors
153 and 14 at 21:44:45, then Kernel-Power 41 and BugCheck 1001 at 21:48. Two
consequences:

1. That campaign (`benchmarks/thesis2-confirm1`) is marked invalid and was not
   used. Its rows remain on disk with `INVALID-crash.md` beside them.
2. The promoted configuration is deliberately the **lower-VRAM** one (22,438 MiB,
   no MTP sidecar). The fastest measured configuration sits at 23.2 GiB, i.e.
   within about 1.3 GiB of the card's limit with a Windows desktop attached.
   Kernel-Power 41 events also exist on 2026-07-04 and 2026-07-12, so this
   machine has a history of unclean shutdowns. Treat the VRAM ceiling as unsafe
   territory and re-measure chained MTP on a quieter card or at a lower context
   before adopting it.

Rules adopted from this: never benchmark above 24,000 MiB (the harness guard
enforces it), never run a profile whose peak is within about 1 GiB of the card,
and treat any host reboot as invalidating the run it interrupted.

## 6. Ranked experiments

| Order | Experiment | Target | Why / deciding measurement |
|---|---|---|---|
| 1 | Chained MTP3 + GPU verification at lower VRAM | Decode | Fastest measured (60.76 vs 45.5 promoted). Needs headroom: `-c 65536`, embedded MTP, or no mmproj, then a production confirm and a soak test |
| 2 | Prefix reuse, recurrent checkpoints, disk cache tier | Effective prefill / TTFT | Untouched. `--cache-ram 0` is forced by the 9.7 GiB RAM cap, so `--cache-disk-path` is the interesting variant |
| 3 | Confirm the promoted configuration on long context | Quality / stability | 131,072 allocation is untested at production sampling with GPU verification: 8K/32K/64K/128K populated decodes, vision and a tool call |
| 4 | `--spec-draft-vocab-hot` adaptive tail | Decode | The shortlist is static at 65,536 ids; 6,144 hot slots can follow recent traffic |
| 5 | Gated DeltaNet prefill kernels | Prefill | Still the only structural prefill lever (34K prefill 706 tok/s; no fused chunked CUDA path upstream) |
| 6 | Quantization shortlist (ATX-IQ4_XS-M, UD-IQ4_XS, UD-Q3_K_XL) | Both | Only if more context or speed per GiB is needed; requires downloads the first pass excluded |
| 7 | vLLM/HyperQwen stack | Both | Larger migration; compare complete workflows and host RAM, not single-stream tok/s |
| 8 | n-gram as an opt-in session flag | Copy-heavy work | Measured: +196% on copying, -2% elsewhere. Wire it as a documented flag, not a default |
| 9 | DFlash2 retest after a cheaper block drafter | Decode | Measured 53.5 at depth 3; revisit only if drafting cost drops |
| 10 | `--backend-sampling` with tools and grammars | Correctness | Sampler fallbacks are documented but untested here for tool schemas and grammar |

Deprioritized with evidence: MTP depth sweeps (2 and 3 tie at production
sampling), `-b/-ub` and thread sweeps (no prefill effect; the thread effect
vanishes with GPU sampling), turbo3 KV (no speed gain, lossy V),
adaptive MTP depth (worst acceptance of the set), elementwise chain fusion
(no measurable difference), context-checkpoint tuning on this host (host RAM),
Bonsai/EXL3 checkpoints (different models, not controlled comparisons).

## 7. Protocol that produced these numbers

1. One server at a time, otherwise idle GPU, 245 W limit, no concurrent compile.
   Every profile loads the same way (mmap, same flags) so load time is comparable.
2. Screen at temperature 0 with EOS suppressed and 256 output tokens on five
   fixtures; compare the SHA256 of the generated text before looking at speed.
3. Confirm what survives at the real settings: chat endpoint, temperature 1,
   top-k 20, top-p 0.95, reasoning xhigh, three seeds, 256 output tokens. Quote
   means and standard deviations per fixture, never a single cell.
4. Record peak VRAM per profile and never cross 24,000 MiB. The card is shared
   with a Windows desktop and a driver fault there reboots the host.
5. Keep raw rows, per-profile logs and raw responses under
   `benchmarks/thesis2-<tag>/`; mark an interrupted campaign invalid instead of
   salvaging its rows.

Reproduce:

```bash
cd /home/mateleo/qwen3.8-server
THESIS2_PROFILES=stock-mtp3,amp-be-gv-embed THESIS2_TAG=again \
  python3 benchmarks/thesis2-screen.py
python3 benchmarks/thesis2-report.py benchmarks/thesis2-again
THESIS2_PROFILES=stock-mtp3,amp-be-gv-embed THESIS2_TAG=again-confirm \
  python3 benchmarks/thesis2-confirm.py
python3 benchmarks/thesis2-stats.py benchmarks/thesis2-again-confirm
```

Rebuild the fork runtime (about 25 minutes at `-j2`; do not raise the job count,
the host has 4 cores and 9.7 GiB of RAM):

```bash
cd ~/llamAmpere-src/repo
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
cmake --build build-sm86 -j2 --target llama-server llama-quantize
```
