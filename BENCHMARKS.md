# Benchmarks

Measured on this host: RTX 3090 (24 GB, 245 W limit, one Windows display
attached), WSL2, Ryzen 5 2400G, 9.7 GiB RAM, CUDA 12.0.

## Current configuration: llamAmpere v0.6 (2026-10-29)

`benchmarks/agentbench.py` reproduces the traffic of a coding agent (Hermes): a
~49K-token context with three tools defined, temperature 1, top-k 20, top-p
0.95, reasoning on at xhigh effort, 1,200 output tokens. It runs four requests:

1. **cold**: the whole context, first read;
2. **turn**: the same conversation plus a new user turn;
3. **side**: an unrelated 3.6K-token request without tools, which takes the
   single slot (agents send these for titles and compression);
4. **back**: the original conversation plus another turn.

Same model (HauhauCS Aggressive Q4_K_P), same 131,072-token context, single runs:

| Step | v0.3.1 `fast` profile | v0.6 (this launcher) |
|---|---:|---:|
| cold prefill | 787 tok/s | **929 tok/s** (+18%) |
| cold decode | 41.9 tok/s | **55.7 tok/s** (+33%) |
| turn decode | 41.1 tok/s | **51.2 tok/s** (+25%) |
| back: prompt work | 49,474 tokens re-read, 64.9 s | **322 tokens + 5.8 s disk restore** |
| back: wall time | 94.6 s | **31.3 s** |
| back decode | 32.5 tok/s | **60.6 tok/s** |
| peak VRAM (whole card, 0.5 s sampling) | 22,315 MiB | **22,248 MiB** |

Raw rows: `benchmarks/results/2026-10-29-*.jsonl` (context text came from
`~/llama.cpp` at `6a1a922`).

What changed between the two columns:

- **Adaptive MTP depth 3-5** instead of a fixed depth 2. Tool requests carry a
  grammar, which keeps draft verification on the CPU in both versions; deeper
  drafts more than make up for it.
- **SJ-KVaRN 4-bit KV cache** (`sjkvarn4`) instead of `q4_0`: a full-precision
  recent window and lower KL than plain 4-bit, at the same VRAM.
- **Prompt cache tiers.** v0.3.1 ran with `--cache-ram 0`, so any side request
  discarded the main conversation and the next turn re-read it (60-90 s at
  50-90K tokens). v0.6 parks it in a 2 GiB RAM tier, then in a 16 GiB disk tier
  at `~/.cache/llamampere/prompt-cache`.
- `-b 4096 -ub 1024` and the upstream GDN prefill work for faster prompt reading.

Drafting is exact (`--no-spec-mb-mtp`): the output distribution is the model's
own. Dropping that flag enables mb-MTP, which keeps more drafts at a measured
0.009 KL/token toward the most likely tokens (upstream figure, not re-measured
here).

Tool calls were checked to come back well-formed (`finish_reason: tool_calls`,
valid JSON arguments).

## Reproduce

Run a server, then point the benchmark at it:

```bash
./start.sh --detach --model aggressive
python3 benchmarks/agentbench.py "http://$(cat ~/.local/state/qwen3.8/host):2222/v1" \
  ~/.config/qwen3.8/api-key qwen3.8-27b-aggressive mytag 60000
```

`benchmarks/guard.py` runs any command in a systemd scope with hard VRAM and
host-RAM limits and kills it before the card or WSL runs out. Use it for any
configuration whose memory use is not yet known:

```bash
python3 benchmarks/guard.py --vram-mib 23300 --ram-max 6G -- <server command>
```

## Host notes

- The card is shared with the Windows desktop (and Parsec when connected). A
  driver fault near full VRAM has rebooted this host before, so keep peak card
  usage under about 23 GiB.
- In the NVIDIA Control Panel, set *CUDA - Sysmem Fallback Policy* to **Prefer
  No Sysmem Fallback**. Otherwise an over-allocation silently spills to system
  RAM and decode drops to about 1 tok/s instead of failing.
- Building llamAmpere: `ptxas` needs about 8.5 GB of RAM on `mmvq.cu`, and
  the full build takes 1-3 hours on this CPU. Keep the job count at 2-4.

## History

Earlier campaigns (stock llama.cpp vs llamAmpere v0.3.1, backend sampling, MTP
depth, DFlash2, compute-buffer slack) are in git history: `thesis.md` and
`benchmarks/thesis2-*` up to commit `067d739`.
