#!/usr/bin/env python3
"""Per-fixture mean/SD, acceptance and VRAM for a thesis2 campaign directory."""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

campaign = Path(sys.argv[1])
rows = [json.loads(line) for line in (campaign / "rows.jsonl").read_text().splitlines() if line.strip()]
d = defaultdict(lambda: defaultdict(list))
acc = defaultdict(list)
vram = defaultdict(list)
for row in rows:
    if row.get("error"):
        print(f"error row: {row['profile']}: {row['error']}")
        continue
    d[row["profile"]][row["fixture"]].append(row["predicted_per_second"])
    if row.get("draft_n"):
        acc[row["profile"]].append(row["draft_n_accepted"] / row["draft_n"])
    gpu = row.get("gpu") or {}
    if gpu.get("vram_mib"):
        vram[row["profile"]].append(gpu["vram_mib"])

fixtures = ["novel-code", "prose", "patch", "copy", "prefill8k"]
print(f"{'profile':22s} " + " ".join(f"{f:>16s}" for f in fixtures)
      + f" {'real mean':>10s} {'acc':>6s} {'vram':>6s}")
for profile in d:
    cells = []
    for fixture in fixtures:
        vals = d[profile].get(fixture)
        cells.append(f"{st.mean(vals):8.1f}+-{st.pstdev(vals):4.1f}" if vals else f"{'-':>16s}")
    real = [st.mean(d[profile][f]) for f in fixtures[:4] if d[profile].get(f)]
    real_txt = f"{st.mean(real):10.1f}" if real else f"{'-':>10s}"
    acc_txt = f"{100*st.mean(acc[profile]):5.1f}%" if acc[profile] else f"{'-':>6s}"
    print(f"{profile:22s} " + " ".join(cells)
          + f" {real_txt} {acc_txt} {max(vram[profile]) if vram[profile] else 0:6d}")

base = {f: st.mean(d["stock-mtp3"][f]) for f in fixtures if d["stock-mtp3"].get(f)}
print("\nrelative to stock-mtp3 (production sampling, 3 seeds, per fixture):")
for profile in d:
    if profile == "stock-mtp3":
        continue
    rel = [100 * (st.mean(d[profile][f]) / base[f] - 1)
           for f in fixtures if d[profile].get(f) and base.get(f)]
    if rel:
        print(f"  {profile:22s} " + ", ".join(f"{r:+6.1f}%" for r in rel)
              + f"   mean {st.mean(rel):+.1f}%")
