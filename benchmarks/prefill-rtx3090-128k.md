# Qwen3.8-27B RTX 3090 prefill benchmark

- Model: Unsloth Dynamic 3.0 UD-Q4_K_XL
- Context allocation: 131,072 tokens
- Flash Attention: enabled
- KV cache: Q4_0
- Batch / microbatch: 2048 / 512
- Vision projector and Q4 MTP loaded
- Prompt cache disabled per request

| Prompt tokens | Prefill tok/s | Prefill time | Wall time | Cached |
|---:|---:|---:|---:|---:|
| 1,024 | 673.3 | 1.52s | 1.58s | 0 |
| 2,024 | 893.0 | 2.27s | 2.28s | 0 |
| 4,024 | 969.3 | 4.15s | 4.16s | 0 |
| 8,024 | 992.9 | 8.08s | 8.09s | 0 |
| 16,025 | 968.7 | 16.54s | 16.57s | 0 |
| 32,025 | 889.1 | 36.02s | 36.05s | 0 |
| 64,025 | 737.4 | 86.83s | 86.87s | 0 |
| 96,025 | 625.2 | 153.60s | 153.69s | 0 |
| 120,026 | 558.1 | 215.05s | 215.13s | 0 |
| 128,026 | 539.7 | 237.23s | 237.33s | 0 |
