# Faster classic Qwen3.8-27B on this RTX 3090

Research and handoff, 2026-09-21. Target: faster single-user decode and/or prefill of the **classic model**, preserving its quality as far as practical. Existing 131,072 context and vision support are the comparison baseline. No changes to the normal launcher, GPU power limit, installed runtime, or model weights were made during this investigation.

## Main thesis

Do not start by downloading a pile of smaller quants. First measure CPU sampling, speculative verification, and prefix reuse on this exact build. Then compare an Ampere-specific runtime with the **same weights**. Qwen's hybrid recurrent architecture and small speculative verification batches make kernel selection unusually important. A smaller GGUF is not automatically faster, and DFlash is not automatically better than MTP.

The clearest measured lead is **n-gram lookup for copying source: 145.6 versus 55.1 tok/s**, with identical greedy output. It did not improve novel code or prose in this screen. Enabling target GPU sampling also did not win. The most promising substantial work remains **SM86 attention/verification kernels, a smaller MTP output vocabulary, and the Gated DeltaNet prefill path**. DFlash2 is worth testing, but a compatible checkpoint and much of the required runtime support are already local; investigate those before installing a new stack.

## 1. Verified local state

| Item | Observed value |
|---|---|
| Project | `/home/mateleo/qwen3.8-server` |
| Classic launch | `./start.sh --model current` |
| Target | `~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,559,178,144 bytes |
| Draft | `~/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, 1,369,590,656 bytes |
| Vision | `~/models/Qwen3.8-27B/mmproj-F16.gguf` |
| Runtime | `~/llama.cpp/build/bin/llama-server`, version `0.4.0-dev`, commit `6a1a922d269908a29cbd4b49c27e6a8e7fd10fae` |
| Build | Release, CUDA, `86-real`, CUDA graphs ON; FORCE_MMQ/FORCE_CUBLAS OFF |
| GPU | RTX 3090, 24,576 MiB; display uses some VRAM |
| Power | **245 W limit**, reported default 350 W; unchanged |
| Driver | Windows driver 595.97; reported CUDA compatibility 13.2, not the installed toolkit version |
| Host | Windows 10 / WSL2, Linux `6.18.33.2-microsoft-standard-WSL2` |
| CPU / RAM | Ryzen 5 2400G, 4 cores / 8 threads; WSL sees 9.7 GiB RAM, 24 GiB swap |
| Serving | 128K allocation, one slot, all GPU layers, FA on, Q4_0 K/V for target and draft |
| Batching | `-b 2048 -ub 512 -t 4 -tb 8` |
| Reuse | `--cache-prompt --cache-ram 0`; default 32 context checkpoints, 8192 minimum spacing |
| Speculation | `draft-mtp`, unspecified draft length means **3** on this binary; draft GPU sampling enabled |
| Target sampling | `--backend-sampling` supported but **disabled by default** |
| Production generation | temperature 1, top-k 20, top-p .95, min-p 0; thinking on, effort xhigh |

The loader identifies architecture `qwen35`; this is expected architecture naming, not evidence that the file is the wrong Qwen release. Observed: 64 main layers, 16 full-attention layers, 48 recurrent layers, 24 Q heads / 4 KV heads, head dimension 256, 27.32B parameters. Main-model GPU buffer: 16,053 MiB; target Q4 KV: 2,304 MiB; recurrent buffer: 598.5 MiB. All 66/66 offloadable layers report GPU offload. Host embeddings do not imply that transformer layers fell back to CPU.

The project already had extensive uncommitted changes, including Bonsai/DFlash work. Preserve them. The normal server was stopped at inspection.

### Existing evidence, not fresh measurements

`benchmarks/prefill-rtx3090-128k.md` records this classic quant with vision and MTP, uncached: about **993 tok/s at 8K**, **737 at 64K**, **540 at 128K** (237 seconds). This makes long prefill a significant part of interactive latency.

`~/ping-llm/PROGRESS.md` is an **older Qwen3.6** experiment, not a Qwen3.8 result. It found only modest batch-tuning gains and a severe Q4-K/Q8-V regression on that older build. Use it to choose experiments, not to declare Q8 KV universally bad.

### Screening performed in this session

See `benchmarks/thesis-screen.py` and `benchmarks/thesis-2026-09-21/` for exact commands, raw responses, hashes, timings, and logs. The screen uses temperature 0, non-thinking text, 256 generated tokens with EOS suppressed, three distinct workloads, and 8,192 uncached synthetic prompt tokens. It retains 128K allocation and the vision projector. It is **not** a quality evaluation or a measurement of production temperature-1/xhigh performance.

| Profile | Novel code decode | Prose decode | Source copying decode | 8K prefill |
|---|---:|---:|---:|---:|
| MTP n=3 | 52.03 | 42.99 | 53.13 | 878.59 |
| MTP n=2 | 50.51 | 47.20 | 55.09 | 923.60 |
| MTP n=2 + ngram-mod, match=32 | 49.66 | 45.22 | **145.57** | 925.73 |
| MTP n=2 + target backend sampling | 49.29 | 43.22 | 49.26 | 888.54 |

All values are server-reported tokens/s; one observation per cell. For each workload, **output SHA256 matched across all four profiles**, and prompt-cache hits were zero. The copy prompt was encountered once per fresh server, so this gain is prompt copying, not memorization of an earlier identical completion. N-grams made copy decode **2.64x** faster than MTP n=2; full request wall time fell from **5.82 to 2.98 seconds**. The copy fixture is deliberately favorable and artificial.

MTP n=2 has mixed results versus n=3; its apparent prose advantage needs repeats. Prefill variation of roughly 879-926 tok/s is not evidence that n-grams accelerate prefill. Target backend sampling was a negative result under these settings. The no-MTP control, repeat baseline, long populated context and production sampling were **not run**. The helper includes those first two profiles for continuation. No launcher change is justified by this screen alone.

The default mmap load did not become ready within a 150-second limit. A second attempt with `--load-mode none` became ready in about 138 seconds; subsequent profiles loaded in 26-29 seconds. Different cache state and timeout censoring prevent a controlled loading-mode speedup claim. This is a startup issue, not measured decode/prefill acceleration. Keep the loading mode identical across subsequent A/B runs. Do not use `mlock` for a 17.6 GB file under this 9.7 GiB WSL RAM cap.

## 2. Ranked experiments

| Order | Experiment | Target | Why / deciding measurement |
|---|---|---|---|
| 1 | Confirm n-grams on real code editing, temperature 1 | Decode | Large copying gain measured; test useful edits and quality, not only exact reproduction |
| 2 | MTP off, n=1,2,3,4 | Decode | Current n=3 is an implicit default; acceptance alone does not price verification cost |
| 3 | Embedded MTP versus separate sidecar | Decode / memory | Target metadata advertises an MTP layer; verify actual weights/support, acceptance and VRAM |
| 4 | Prefix reuse and recurrent checkpoints | Effective prefill / TTFT | Avoid replaying tens of thousands of tokens after edits or alternating conversations |
| 5 | Same GGUF on a pinned Ampere fork | Decode, long-context attention | Directly tests kernel hypothesis without conflating quantization |
| 6 | `b=2048/4096`, `ub=256/512/1024` | Prefill | Cheap sweep; cap VRAM and retain populated 128K capability |
| 7 | Same-base IQ4_XS / Q4_K_M / Q3_K_XL | Primarily decode, memory | Test both target-only and best speculation; quality and acceptance may change |
| 8 | Existing DFlash2 checkpoint, Q8 draft, n=3/7 | Decode | Parallel drafting may win; verify acceptance, VRAM, long-context prefill and correctness |
| 9 | Profile GDN; fused chunked prefill implementation | Prefill | Concrete source-level opportunity; requires kernel work and state correctness checks |
| 10 | Threads 2 vs 4; revisit target GPU sampling at temperature 1 | Decode | GPU sampling lost the greedy screen; do not enable by default |
| 11 | Pinned HyperQwen/vLLM Marlin stack | Both / concurrency | Larger migration; compare complete workflows, startup and WSL RAM as well as tok/s |

### A. CPU sampling and launch overhead

`llama-server --help` confirms `--backend-sampling` is experimental and off. Draft backend sampling is already on, so enabling it again is not a new optimization. The local vocabulary is 248,320 entries: copying/scanning target logits and CPU scheduling are plausible overhead on this 2400G/WSL host. However, enabling **target** backend sampling did not win the greedy A/B above. Deprioritize it unless production sampling behaves differently; confirm actual sampler placement/fallback when profiling.

Use greedy output hashes for initial comparison; repeat promising settings at the actual temperature/top-k/top-p and with tool schemas. CUDA graphs are already compiled in. Check that `GGML_CUDA_DISABLE_GRAPHS` is absent; setting it to `0` still disables graphs because this checkout tests whether the variable exists. Try `CUDA_SCALE_LAUNCH_QUEUES=4x` as a separate measured experiment, not an assumed universal win.

### B. MTP and n-grams

Smaller drafts can win because accepted tokens per verification step must repay **draft time + verification time + rollback/sampling overhead**. A useful cost model is `tokens/s = E[accepted draft tokens + target token] / E[round wall time]`. Measure draft counts, accepted counts, mean emitted tokens per round, and elapsed time. Acceptance percentage alone is insufficient.

Another no-download clue: the target loader reports `nextn_predict_layers=1` and `n_layer_all=65`, while the launcher explicitly loads a separate Q4 sidecar. Verify whether the GGUF actually contains usable embedded MTP tensors, then test `--spec-type draft-mtp` with the external draft argument omitted. Metadata alone does not prove this will work or save memory; compare the loaded buffers and head quantization. Do not remove the sidecar from production without that check.

Test `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-ngram-mod-n-match 32`, then tune n-gram min/max lengths if it wins on code edits. Local defaults are min 48 / max 64, independent of the MTP n-max. Draftless lookup gets precedence when combined with a model drafter. These details are in local `docs/speculative.md` and binary help.

N-grams predict continuations already present in token history. Expect gains on rewrites, patch generation and copied source; do not expect the same on novel prose or hard new reasoning. `cache_prompt:false` does **not** clear n-gram history. The published engine trial explicitly retracts 300-400 tok/s repeated-prompt figures as fresh-prompt speed: subsequent requests had memorized prior outputs. Fresh and deliberately warm workloads must be reported separately. [Original experiment and correction](https://github.com/0x7067/rtx3090-llm-lab/tree/main/benchmarks/engine-trial-2026-09-02).

### C. Avoid prefill work: caching is especially important for this model

Current `--cache-prompt` can reuse the active slot; `--cache-ram 0` disables the extra host cache used for evicted states. These are different features. Test:

1. Cold 32K prompt; append a small turn.
2. Edit the last user turn and retry.
3. Conversation A, unrelated B, then A again.
4. Real client chat templating, including preserved reasoning and tool messages.

Record actual evaluated tokens, cached tokens, checkpoint restoration and wall TTFT. A hybrid recurrent state cannot simply be truncated like a transformer KV cache; checkpoint placement matters. Try 32 checkpoints with spacing 2048/4096/8192 and `--cache-ram 1024` or `2048`, watching RSS and swap. Local logs show a checkpoint around **149.6 MiB**, so allocating an 8 GiB RAM cache under a 9.7 GiB cap would be a poor blind default. More checkpoints also cost copies and memory.

Keep stable system/tool prefixes and append history consistently. A changed template or reordered tools can destroy reuse. These changes reduce the amount of prefill; report them as latency wins rather than faster cold-prefill kernels.

### D. Concrete RTX 3090 kernel candidates

**First candidate: `JakeATX/llamAmpere`.** This is an actively documented Ampere fork, with tuned attention, speculative verification, recurrent rollback, and a draft vocabulary shortlist. Its documentation reports v0.3 results on a **3090 Ti at 350 W**, using an IQ4_XS-based quant: 99.4 tok/s on its clean fixtures versus 67.9 for stock, plus long-context measurements. These are external results, not a prediction for this 245 W 3090. Pin a release/commit, run the existing UD-Q4_K_XL first, then test its quant and KV modes separately. Its current recommendations include altered Swift checkpoints: those change the model and are outside the initial classic-model comparison. [Runtime and pinned-release guidance](https://github.com/JakeATX/llamAmpere/blob/main/QWEN_AMPERE.md).

**Second candidate: `0x7067/rtx3090-llm-lab`.** Its kernel investigation targets two costs relevant here: repeated KV reads/dequantization for grouped-query attention, and MMVQ versus MMQ dispatch for tiny verification batches. Inspect `patches-v14/` against our newer checkout before porting. Its `GGML_CUDA_MMVQ_NE11_MAX` and `GGML_CUDA_MMQ_SMALLN` variables were **not found in local CUDA sources**. Setting them alone has no effect. Do not combine percentage gains from different patch generations. [Kernel measurements and mechanisms](https://github.com/0x7067/rtx3090-llm-lab/blob/main/docs/PERFORMANCE.md).

Local investigation points:

- `ggml/src/ggml-cuda/mmvq.cu`: `ggml_cuda_should_use_mmvq`, currently falls back to `MMVQ_MAX_BATCH_SIZE`.
- `ggml/src/ggml-cuda/mmq.cu`: tensor-core quantized matmul selection.
- `ggml/src/ggml-cuda/fattn.cu` and `fattn-common.cuh`: SM86 selection and quantized-cache conversion.
- `ggml/src/ggml-cuda/gated_delta_net.cu`: recurrent compute and state snapshots.
- `common/speculative.cpp`: draft GPU sampling, DFlash2 handling and verification.

FORCE_MMQ is a **compile-time** switch and does not automatically override earlier MMVQ selection. Benchmark `MUL_MAT` at verification widths 1,2,3,4,8,16 and attention at 8K/32K/128K depth before editing thresholds. Check outputs and recurrent rollback, not just timing. Build isolated runtimes at `-j2` or `-j3`; this WSL RAM limit makes unrestricted nvcc parallelism inappropriate.

### E. Quantization: shortlist, not ideology

Same base weights first: Unsloth UD-IQ4_XS, UD-Q4_K_M, then UD-Q3_K_XL if a quality tradeoff is acceptable. Current model-card download sizes are approximately 14.3, 16.5 and 13.1 GB versus 17.6 GB for UD-Q4_K_XL. Pin file revision and exact filename; dynamic tensor maps can change between releases. [Publisher's available quants](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).

A particularly relevant candidate is **ATX-IQ4_XS-M**, a quant of the original base with a per-tensor mix selected for Ampere. Its reported wins use a modified runtime, MTP and different KV types on a 3090 Ti: do not attribute the whole gain to the GGUF. Inspect quality on coding, instruction following, tools and long-context retrieval. [Quant recipe and measurement conditions](https://huggingface.co/jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF).

Fewer bits reduce weight traffic but may increase dequantization cost, change kernel dispatch, reduce draft acceptance, or harm quality. Prefill is often more compute-sensitive than single-stream decode, so the decode winner need not win prefill. Quantizing an already Q4 file is not the right way to generate a clean alternative; download a publisher quant or quantize original higher-precision weights with appropriate calibration.

Q8/BF16 target weights exceed this GPU's capacity before runtime caches; CPU offload would likely sacrifice the desired decode speed. Bonsai and Aggressive are different checkpoints, not controlled quant-only comparisons. The existing Bonsai 128K prefill result is ~527 tok/s, already similar to the classic model's ~540; its tiny weights alone did not solve long prefill.

### F. DFlash2: finish the right local experiment

Local `~/models/Qwen3.8-27B-DFlash2/` already contains `model.safetensors`, config, target tokenizer, and `dflash2-BF16.partial.gguf`. The finished `Qwen3.8-27B-DFlash2-Q8_0.gguf` is **absent**. A `.partial` file is not proof of completed conversion. Config says five draft layers, block size 8, selector top-k 16, target feature layers `[5,19,33,47,61]`.

The classic `~/llama.cpp/common/speculative.cpp` **already contains DFlash2 selector handling** (`is_dflash2`, `selector_top_k`); generic `draft-dflash` help alone would not establish that. `~/llama.cpp-bonsai-dflash2/build/bin/llama-server` is absent. Existing `setup-dflash2.sh` is wired to that unfinished Bonsai runtime, and `benchmarks/speed-dflash2.py` benchmarks **Bonsai**, not the requested classic model. Do not run them and mislabel the result.

Build `llama-quantize` in the classic checkout if required; it and `llama-bench` are currently missing there. Validate or repeat conversion with the classic converter and the matching target-tokenizer metadata, then quantize the draft to Q8_0. Preserve partial/original files until validation. Test against classic UD-Q4_K_XL with `--spec-type draft-dflash -md <finished-draft> --spec-draft-n-max 7 -ngld all -ctkd f16 -ctvd f16`. Compare n=3 and n=7, MTP, then the winning drafter plus n-grams. Keep target settings fixed; measure actual full-depth VRAM and prefill overhead.

The publisher evaluates DFlash2 on **H200**, not RTX 3090. Its seven draft tokens must not be confused with the eight-position block. Hardware speedups will not transfer directly. Demand target-correct verification, greedy comparison and sampled evaluation of the actual implementation. [DFlash2 model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

### G. Prefill beyond batch sizes

At local `ggml/src/ggml-cuda/gated_delta_net.cu:180` there is still a TODO for a chunked prefill kernel. `src/models/delta-net-base.cpp` contains a chunked graph reference and chooses a fused path when enabled. This is a concrete profiling lead, **not proof that GDN dominates total prefill**. The context calls a supported fused op “chunked” even though that CUDA implementation still carries the TODO; inspect the actual kernel rather than relying on the log label.

Use CUDA profiling, where supported on this WSL/driver combination, to separate matmul, attention, GDN and synchronization time for 8K/32K/128K input. A chunked/tensor-core GDN path may pay off more than weight quantization because 48 layers are recurrent. Compare outputs and final recurrent state with the reference, including partial chunks and speculation rollback. The upstream RFC identifies this same gap but is **closed as not planned**, so it is not an available merged fix. [RFC #22967](https://github.com/ggml-org/llama.cpp/issues/22967).

### H. Other engine, GPU power and WSL

**HyperQwen/vLLM** is a serious second engine experiment: pinned patches, prepared W4A16 weights and Ampere kernels. Its own guidance switches between MTP and DFlash2 by context/workload and includes a WSL pinned-memory workaround. It does not establish that vLLM always beats llama.cpp. The stack requires significant extra host RAM/storage and initialization work; verify feasibility under this VM cap first. [Project and workload profiles](https://github.com/syv-ai/HyperQwen). Marlin explicitly targets FP16 x INT4 on compute capability 8.0+, making it relevant to a 3090; it is not a GGUF runtime flag. [Marlin authors](https://github.com/IST-DASLab/marlin).

Do not pursue native FP8/FP4 tensor-core headlines as though this card were Hopper/Blackwell. Compressed storage can still be useful through compatible kernels; arithmetic capability and cache format are separate issues. ExLlama/EXL3 or SGLang require verified support for this exact hybrid model, vision and serving features before investing in conversion.

Keep the current 245 W power cap as baseline. If higher power is acceptable, a separate 245/285/320/350 W experiment can quantify the tradeoff; the runtime must sustain the clocks without thermal throttling. Expect prefill and decode to respond differently. Neither 350 W nor published 3090 Ti performance is a free software optimization.

Models already live on WSL's Linux filesystem, not `/mnt/c`, so “move off NTFS” is not the missing fix. Use the Windows-provided CUDA driver path; pinned host-memory availability and some profiling/management features are restricted under WSL. [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Check swap and Windows display VRAM during full-depth runs. Avoid assuming bare Linux would yield a particular percentage improvement without measurement.

## 3. Efficient continuation protocol

1. Read this file and the screening results. Keep the original launcher and runtime as rollback. Pin every new build and record model hashes/revisions.
2. Complete the cheap flag screen. Run candidates sequentially on an otherwise idle GPU at 245 W. Do not run simultaneous GPU benchmarks.
3. Confirm any winner with interleaved A/B/A runs, three seeds and actual temperature-1/xhigh workloads. Include novel code, prose, tools, intentional copying and long-context retrieval. A 256-token synthetic screen is insufficient for promotion.
4. Measure cold prefill at 8K/32K/64K/128K, decode after populated 8K/32K/128K prompts, and append/edit/A-B-A cache behavior. For a 131,072-token allocation, leave space for output: e.g. 130,560 prompt + 256 output. Report exact token counts.
5. Log server prefill/decode timings **and** wall TTFT/total time, generated thinking+answer tokens, draft acceptance, cache hits, peak VRAM/RSS/swap, clocks and power. Keep cold startup separate. A warm `llama-bench` tg test at depth zero cannot answer long-context serving speed.
6. Test the Ampere runtime on current weights before altering quant/KV. Then evaluate one promising quant and finish classic-model DFlash2. Stop broad sweeps when a convincing workload-level winner emerges.
7. Promote only improvements larger than repeat noise with acceptable quality, correct tool/vision behavior and full required context. Record exact revert command/config. Lower reasoning effort is an optional task-latency tradeoff, not a raw decode/prefill speedup; do not silently change it.

Suggested starting commands:

```bash
cd /home/mateleo/qwen3.8-server
# The harness owns its temporary localhost server and cleans it up.
# It refuses to run if another llama-server process is active.
THESIS_PROFILES=mtp3,none,mtp2,mtp2-ngram,mtp2-gpu-sampling,mtp3-repeat \
  python3 benchmarks/thesis-screen.py

# Inspect actual supported arguments before following an online recipe.
~/llama.cpp/build/bin/llama-server --help
```

The harness appends raw rows and rewrites per-profile logs/metadata on rerun; archive the output directory first when preserving an earlier campaign. For kernel experiments, create a separate build/checkout and cap compile parallelism. No model download, kernel patch or server-default change is needed just to test the first three hypotheses.
