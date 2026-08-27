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
./start.sh                              # choose model, start, open dashboard
./start.sh --model current              # Unsloth UD-Q4_K_XL
./start.sh --model aggressive           # HauhauCS Aggressive Q4_K_P
./start.sh --detach                     # start without dashboard
./monitor.py                            # attach dashboard later
./status.sh
./test-api.sh
./stop.sh
```

The picker uses `whiptail` when available and has a plain terminal fallback.
Running `start.sh` while the server is alive only reattaches the dashboard.
Pressing `q` or `Ctrl-C` closes the dashboard without stopping the server. Logs
are written to `~/.local/state/qwen3.8/server.log`.

## Models

| Picker ID | API model ID | Quantization | MTP |
|---|---|---|---|
| `current` | `qwen3.8-27b` | Unsloth UD-Q4_K_XL | Q4 sidecar |
| `aggressive` | `qwen3.8-27b-aggressive` | HauhauCS Q4_K_P | embedded native MTP |

The Aggressive repository also offers a FastMTP sidecar that requires a patched
`llama.cpp`. This project deliberately uses the GGUF's native embedded MTP so
it remains compatible with upstream `llama.cpp`.

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
- Qwen thinking defaults: temperature 1.0, top-k 20, top-p 0.95, xhigh effort;
- reasoning returned separately as `message.reasoning_content`.

The dashboard reports generated reasoning as part of **tokens out** because
`llama.cpp` does not expose a separate reasoning-token Prometheus counter.

Environment overrides include `LLAMA_SERVER`, `LLAMA_CPP_DIR`, `QWEN_HOST`,
`QWEN_PORT`, `QWEN_MODEL_DIR`, `QWEN_AGGRESSIVE_MODEL_DIR`,
`QWEN_API_KEY_FILE`, `QWEN_STATE_DIR`, and `QWEN_BUILD_JOBS`.

## License

The scripts are available under the [MIT License](LICENSE). Downloaded models
are not included and retain their respective Hugging Face repository licenses.
