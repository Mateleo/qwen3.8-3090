#!/usr/bin/env python3
"""Two production-sampling chat requests against the running launcher.

Reports decode tok/s, draft acceptance and whether GPU verification engaged.
Reads the key file from the launcher state; never prints it.
"""

import json
import urllib.request
from pathlib import Path

STATE = Path.home() / ".local/state/qwen3.8"
key = (STATE / "key-file").read_text().strip()
host = (STATE / "host").read_text().strip()
port = (STATE / "port").read_text().strip()
key_value = Path(key).read_text().strip()

prompts = [
    "Write a Python function that reverses a linked list iteratively. Output code only.",
    "Write the same function for a doubly linked list. Output code only.",
]
for prompt in prompts:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key_value})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    t = data.get("timings", {})
    print(f"decode {t.get('predicted_per_second', 0):6.2f} tok/s | prefill "
          f"{t.get('prompt_per_second', 0):7.1f} tok/s | draft {t.get('draft_n_accepted')}"
          f"/{t.get('draft_n')} | completion tokens {data['usage']['completion_tokens']}")
