# Qwen3.8 27B on one RTX 3090

A first-run-friendly, OpenAI-compatible `llama.cpp` server tuned for a single
24 GB RTX 3090. It keeps a 131,072-token context, vision, MTP speculative
decoding, Q4 KV caches, prompt caching, optional remote access, and a live
terminal dashboard.

> This configuration is specifically benchmarked on an RTX 3090. Other NVIDIA
> GPUs may work if they have enough VRAM, but may require tuning.

## Quick start

Requirements: Linux, an NVIDIA driver, the CUDA toolkit (`nvcc`), Git, CMake,
Python 3, OpenSSL, curl, and roughly 20 GB of free model storage. Build tools
must already be available through your operating system's package manager.

```bash
git clone https://github.com/Mateleo/qwen3.8-3090.git
cd qwen3.8-3090
./start.sh
```

On first launch, the interactive onboarding:

1. asks which of the two supported model variants to use;
2. installs the Hugging Face `hf` CLI in an isolated local environment if it is
   not already installed;
3. downloads only that variant's required GGUF files with resumable progress
   bars;
4. clones and compiles `llama.cpp` with CUDA using the detected CPU job count;
5. generates a random 256-bit API key in `~/.config/qwen3.8/api-key`; and
6. launches the server and dashboard.

Both model repositories are public, so a Hugging Face account or token is not
normally required. Run onboarding separately with:

```bash
./setup.sh --model current
./setup.sh --model aggressive
```

Existing model files, builds, and keys are preserved. For unattended setup, add
`--yes`.

## Commands

```bash
./start.sh                              # interactive profile menu, then dashboard
./start.sh --profile fast               # UD-Q4_K_XL on the Ampere runtime (+23%)
./start.sh --profile compat             # UD-Q4_K_XL on the stock runtime
./start.sh --profile aggressive         # HauhauCS Aggressive on the Ampere runtime
./start.sh --list-profiles              # print the profiles and their numbers
./start.sh --detach                     # start without dashboard
./start.sh --model current              # kept for compatibility (stock runtime)
./monitor.py                            # attach dashboard later
./status.sh
./test-api.sh
./stop.sh
```

The profile menu uses `whiptail` when available and has a plain terminal
fallback. It preselects the profile used last, so pressing Enter repeats it.
Running `start.sh` while the server is alive only reattaches the dashboard.
Pressing `q` or `Ctrl-C` closes the dashboard without stopping the server. Logs
are written to `~/.local/state/qwen3.8/server.log`.

## Models

| Profile | API model ID | Model | Runtime |
|---|---|---|---|
| `fast` | `qwen3.8-27b` | Unsloth UD-Q4_K_XL | Ampere (+23% decode, 22.4 GiB) |
| `compat` | `qwen3.8-27b` | Unsloth UD-Q4_K_XL | stock (23.5 GiB) |
| `aggressive` | `qwen3.8-27b-aggressive` | HauhauCS Q4_K_P | Ampere |
| `aggressive-compat` | `qwen3.8-27b-aggressive` | HauhauCS Q4_K_P | stock |

The Aggressive repository also offers a FastMTP sidecar that requires a patched
`llama.cpp`. This project deliberately uses the GGUF's native embedded MTP, and
the Ampere runtime reuses the same embedded MTP head.

## API client

By default, the server binds to the machine's Tailscale IPv4 address when one
is available, otherwise to `127.0.0.1`. Set `QWEN_HOST` to override it.
`status.sh` prints the active address without exposing the key.

```bash
export OPENAI_BASE_URL="http://127.0.0.1:2222/v1" # replace with status.sh address
export OPENAI_API_KEY="$(cat ~/.config/qwen3.8/api-key)"
```

The generated key is required as an OpenAI API key/Bearer token. It is stored
with owner-only permissions and must never be committed.

## Temporary public access

For a device that cannot use your private network:

```bash
./start.sh --share
```

This binds the server to localhost, starts a Cloudflare Quick Tunnel, and prints
a temporary HTTPS base URL and a separate per-run API key. Install
`cloudflared` first:

```bash
install -d ~/.local/bin
curl -fL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o ~/.local/bin/cloudflared
chmod 755 ~/.local/bin/cloudflared
```

The tunnel URL is public; keep its API key secret. `stop.sh` stops both
processes and removes the temporary key. Quick Tunnels have no uptime guarantee;
prefer Tailscale or a named authenticated tunnel for stable access.

## RTX 3090 inference configuration

- full CUDA GPU offload and Flash Attention;
- 131,072-token single slot;
- Q4-class model and Q4_0 K/V caches;
- batch / microbatch sizes of 2048 / 512;
- native F16/BF16 vision projector;
- MTP speculative decoding and prompt-prefix caching;
- backend (GPU) sampling, so host-side sampling of the 248,320-row vocabulary
  does not gate decode on this CPU;
- Qwen thinking defaults: temperature 1.0, top-k 20, top-p 0.95, xhigh effort;
- reasoning returned separately as `message.reasoning_content`.

The dashboard reports generated reasoning as part of **tokens out** because
`llama.cpp` does not expose a separate reasoning-token Prometheus counter.

### Runtime profiles

A profile pairs a model with a runtime. `./start.sh` shows them in a menu;
`--profile NAME` picks one directly and `--list-profiles` prints them.

| Profile | Runtime binary | Notes |
|---|---|---|
| `fast` | `~/llamAmpere-src/repo/build-sm86/bin/llama-server` | SM86 kernels, GPU verification of MTP drafts, draft vocabulary shortlist |
| `compat` | `~/llama.cpp/build/bin/llama-server` | upstream checkout |
| `aggressive` | Ampere binary | HauhauCS Aggressive weights |
| `aggressive-compat` | stock binary | HauhauCS Aggressive weights |

Measured on this machine at the production sampling settings (temperature 1,
top-k 20, top-p 0.95, reasoning xhigh, 256 output tokens, three seeds, five
workloads, quiet GPU, `benchmarks/thesis2-confirm2`):

| Configuration | mean decode | peak VRAM |
|---|---|---|
| `compat` (stock, MTP sidecar, MTP3) | 36.7 tok/s | 23,533 MiB |
| stock + `--backend-sampling` + embedded MTP | 36.4 tok/s | 22,690 MiB |
| `fast` (`--backend-sampling` + shortlist + GPU draft verification) | **45.5 tok/s (+23%)** | 22,438 MiB |
| `fast` + `--spec-chain 3` (measured, not promoted: 23.2 GiB, see thesis) | 51-58 tok/s | 23,200 MiB |

The `fast` profile merges a later upstream that deliberately changes model
output on this architecture: the Gated DeltaNet q/k normalization now matches the
reference definition (upstream PR 28068). Greedy text therefore differs from
`compat` on free-form prompts, and is identical within the fork across every
speculation setting. Pick `fast` for the speed, keep `compat` if byte-identical
continuity of past outputs matters.

The `fast` profile also drops the MTP sidecar: that GGUF already carries its own
MTP layer, which measures the same speed and leaves about 1.1 GiB more VRAM for
the Windows desktop.

Both profiles are also selectable through the environment, which the menu writes
back to `~/.local/state/qwen3.8/profile-choice`:

```bash
./start.sh --profile fast          # fastest, measured
./start.sh --profile compat        # unchanged outputs
QWEN_SERVER_BUILD=stock ./start.sh --profile fast   # runtime override
QWEN_MTP_SIDECAR=1 ./start.sh --profile fast        # keep the Q4 sidecar
```

`~/.local/state/qwen3.8/server-profile` records what a running server actually
uses (runtime, draft source, vocabulary map, draft depth).

Environment overrides include `LLAMA_SERVER`, `LLAMA_CPP_DIR`, `QWEN_HOST`,
`QWEN_PORT`, `QWEN_MODEL_DIR`, `QWEN_AGGRESSIVE_MODEL_DIR`,
`QWEN_API_KEY_FILE`, `QWEN_STATE_DIR`, `QWEN_BUILD_JOBS`, `QWEN_SERVER_BUILD`,
`QWEN_STOCK_SERVER`, `QWEN_AMPERE_SERVER`, `QWEN_VOCAB_MAP`, and
`QWEN_MTP_SIDECAR`.

## License

The scripts are available under the [MIT License](LICENSE). Downloaded models
are not included and retain their respective Hugging Face repository licenses.
