# Qwen3.8 27B on one RTX 3090

A first-run-friendly, OpenAI-compatible server for Qwen3.8 27B on a single
24 GB RTX 3090. It runs [llamAmpere](https://github.com/JakeATX/llamAmpere), a
llama.cpp fork tuned for Ampere cards, with a 131,072-token context, vision,
MTP speculative decoding, a 4-bit SJ-KVaRN KV cache, a RAM and disk prompt
cache, optional remote access, and a live terminal dashboard.

> This configuration is benchmarked on an RTX 3090 (see
> [BENCHMARKS.md](BENCHMARKS.md)). Other NVIDIA GPUs may work if they have
> enough VRAM, but may require tuning.

## Quick start

Requirements: Linux, an NVIDIA driver, the CUDA toolkit (`nvcc`), Git, CMake,
Python 3, OpenSSL, curl, about 20 GB of free model storage, and about 9 GB of
RAM for the build. Build tools must already be available through your
operating system's package manager.

```bash
git clone https://github.com/Mateleo/qwen3.8-3090.git
cd qwen3.8-3090
./start.sh
```

On first launch, the interactive onboarding:

1. asks which of the two supported models to use;
2. installs the Hugging Face `hf` CLI in an isolated local environment if it is
   not already installed;
3. downloads only that model's GGUF files with resumable progress bars;
4. clones llamAmpere v0.6 into `~/llamAmpere-src/v0.6`, applies
   `patches/llamampere-v0.6-cuda12.0.patch`, and builds it with CUDA (1-3 hours,
   2 parallel jobs by default; set `QWEN_BUILD_JOBS` to change it);
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
./start.sh                        # serve the model used last, then dashboard
./start.sh --model aggressive     # switch model (remembered for next time)
./start.sh --detach               # start without dashboard
./monitor.py                      # attach dashboard later
./status.sh
./test-api.sh
./stop.sh
```

Running `start.sh` while the server is alive only reattaches the dashboard.
Pressing `q` or `Ctrl-C` closes the dashboard without stopping the server. Logs
are written to `~/.local/state/qwen3.8/server.log`.

## Models

| `--model` | API model ID | Weights |
|---|---|---|
| `current` | `qwen3.8-27b` | Unsloth UD-Q4_K_XL |
| `aggressive` | `qwen3.8-27b-aggressive` | HauhauCS Aggressive Q4_K_P (uncensored) |

Both GGUFs carry their own MTP layer, which drafts the speculative tokens. No
separate draft model is downloaded.

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

## Inference configuration

- full CUDA GPU offload and Flash Attention;
- 131,072-token single slot;
- Q4-class weights and an SJ-KVaRN 4-bit K/V cache;
- batch / microbatch sizes of 4096 / 1024;
- native F16/BF16 vision projector;
- adaptive MTP speculative decoding (depth 3-5, exact drafting);
- prompt cache: a conversation pushed out of the slot by another request is
  kept in RAM (2 GiB) and then on disk (`~/.cache/llamampere/prompt-cache`,
  16 GiB), so returning to it takes seconds instead of re-reading the prompt;
- backend (GPU) sampling and draft verification when the request allows it;
- Qwen thinking defaults: temperature 1.0, top-k 20, top-p 0.95, xhigh effort;
- reasoning returned separately as `message.reasoning_content`.

On a ~49K-token agent workload with tools this decodes at 51-61 tok/s and reads
prompts at about 930 tok/s, against 41 and 787 tok/s for the previous
v0.3.1-based launcher. Details in [BENCHMARKS.md](BENCHMARKS.md).

The dashboard reports generated reasoning as part of **tokens out** because
`llama.cpp` does not expose a separate reasoning-token Prometheus counter.

Environment overrides: `LLAMA_SERVER`, `LLAMA_CPP_DIR`, `QWEN_HOST`,
`QWEN_PORT`, `QWEN_MODEL_DIR`, `QWEN_AGGRESSIVE_MODEL_DIR`,
`QWEN_API_KEY_FILE`, `QWEN_STATE_DIR`, and `QWEN_BUILD_JOBS`.

## License

The scripts are available under the [MIT License](LICENSE). Downloaded models
are not included and retain their respective Hugging Face repository licenses.
