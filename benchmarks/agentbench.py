#!/usr/bin/env python3
"""Agent-shaped benchmark: long tool-enabled context, like Hermes traffic.

usage: benchmarks/agentbench.py BASE_URL KEY_FILE MODEL TAG [TARGET_TOKENS]
Steps:
  1. cold: ~TARGET_TOKENS context with tools -> prefill + decode (grammar path)
  2. turn: same context + a short new user turn -> cached-prefix decode
  3. side: unrelated 5K request without tools (evicts the slot on 1 slot)
  4. back: main context again + another turn -> measures eviction cost
Prints one JSON line per step. The context is C++ source from BENCH_SRC_DIR
(default: the llamAmpere tree start.sh runs).
"""
import glob, json, os, sys, time, urllib.request

base, keyf, model, tag = sys.argv[1:5]
target = int(sys.argv[5]) if len(sys.argv) > 5 else 60000
key = open(keyf).read().strip()
src_dir = os.environ.get("BENCH_SRC_DIR", os.path.expanduser("~/llamAmpere-src/v0.6"))

src = ""
for f in sorted(glob.glob(f"{src_dir}/tools/server/*.cpp")) + \
         sorted(glob.glob(f"{src_dir}/common/*.cpp")):
    src += f"\n\n===== {f} =====\n" + open(f, errors="ignore").read()
    if len(src) > target * 3.3:
        break
src = src[: int(target * 3.3)]

tools = [{
    "type": "function",
    "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(props)},
    }} for name, desc, props in [
        ("read_file", "Read a file", {"path": {"type": "string"}}),
        ("write_file", "Write a file", {"path": {"type": "string"}, "content": {"type": "string"}}),
        ("run_shell", "Run a shell command", {"command": {"type": "string"}}),
    ]]

def chat(messages, use_tools=True, max_tokens=1200):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7}
    if use_tools:
        body["tools"] = tools
    req = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.time() - t0
    t = r.get("timings", {})
    return r, {"wall_s": round(wall, 1),
               "prompt_n": t.get("prompt_n"), "prefill_tps": round(t.get("prompt_per_second", 0), 1),
               "prefill_s": round(t.get("prompt_ms", 0) / 1e3, 1),
               "gen_n": t.get("predicted_n"), "decode_tps": round(t.get("predicted_per_second", 0), 2),
               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")}

sys_msg = {"role": "system", "content": "You are a coding agent. Use the tools when useful."}
history = [sys_msg,
           {"role": "user", "content": "Here is the source I'm working on:\n" + src},
           {"role": "assistant", "content": "I've read the source. What should I do?"},
           {"role": "user", "content": "Explain how the server picks a slot for a new request, then propose a patch "
                                       "to prefer idle slots. Write the patched function with write_file."}]

def step(name, msgs, **kw):
    r, m = chat(msgs, **kw)
    print(json.dumps({"tag": tag, "step": name, **m}), flush=True)
    return r

r = step("cold", history)
history.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or "ok"})
history.append({"role": "user", "content": "Now summarize the risks of that patch in detail, then list tests to add."})
r = step("turn", history)
step("side", [{"role": "user", "content": "Summarize this file:\n" + src[:16000]}], use_tools=False, max_tokens=200)
history.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or "ok"})
history.append({"role": "user", "content": "Write the first of those tests."})
step("back", history)
