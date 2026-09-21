#!/usr/bin/env python3
"""Summarise a thesis2 campaign directory into markdown plus a deltas table.

Usage:
    python3 benchmarks/thesis2-report.py benchmarks/thesis2-screen1 [--baseline stock-mtp3]
"""

import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

DECODE_FIXTURES = ["novel-code", "prose", "patch", "copy"]


def load_rows(rows_path):
    rows = []
    for line in rows_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        rows.append(row)
    return rows


def geo(values):
    values = [v for v in values if v and v > 0]
    if not values:
        return 0.0
    return stats.geometric_mean(values)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    campaigns = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    baseline = "stock-mtp3"
    if "--baseline" in sys.argv:
        baseline = sys.argv[sys.argv.index("--baseline") + 1]
    rows = []
    for campaign in campaigns:
        rows.extend(load_rows(campaign / "rows.jsonl"))
    campaign = campaigns[0]

    by_profile = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row)

    baseline_rows = {r["fixture"]: r for r in by_profile.get(baseline, [])}
    order = [p for p in by_profile if p != baseline]
    order.sort(key=lambda p: -geo([r.get("predicted_per_second") for r in by_profile[p]
                                   if r["fixture"] in DECODE_FIXTURES
                                   and r.get("predicted_per_second")]))

    print(f"# Campaign {' + '.join(c.name for c in campaigns)}\n")
    if not baseline_rows:
        print(f"note: baseline profile '{baseline}' has no rows in this campaign\n")

    fixtures = [f for f in DECODE_FIXTURES + ["prefill8k"]]
    head = "| profile | " + " | ".join(fixtures) + " | decode geo | acceptance |"
    print(head)
    print("|" + "---|" * (len(fixtures) + 3))
    for name in [baseline] + order if baseline_rows else order:
        rows_p = by_profile[name]
        cells = []
        for fixture in fixtures:
            match = [r for r in rows_p if r["fixture"] == fixture]
            if not match:
                cells.append("-")
                continue
            r = match[0]
            v = r.get("predicted_per_second") if fixture != "prefill8k" else r.get("prompt_per_second")
            cells.append("n/a" if v is None else f"{v:.2f}")
        dec = geo([r.get("predicted_per_second") for r in rows_p if r["fixture"] in DECODE_FIXTURES])
        acc = [r["draft_n_accepted"] / r["draft_n"] for r in rows_p
               if r.get("draft_n") and r.get("draft_n_accepted") is not None]
        print(f"| {name} | " + " | ".join(cells) + f" | {dec:.2f} | "
              + (f"{100*sum(acc)/len(acc):.1f}%" if acc else "-") + " |")

    if baseline_rows:
        print("\n## Change versus baseline (decode tok/s, per fixture)\n")
        print("| profile | " + " | ".join(DECODE_FIXTURES) + " | geo |")
        print("|" + "---|" * (len(DECODE_FIXTURES) + 2))
        for name in order:
            rows_p = by_profile[name]
            cells, ratios = [], []
            for fixture in DECODE_FIXTURES:
                base = baseline_rows.get(fixture, {}).get("predicted_per_second")
                match = [r for r in rows_p if r["fixture"] == fixture]
                cur = match[0].get("predicted_per_second") if match else None
                if base and cur:
                    cells.append(f"{100*(cur/base-1):+.1f}%")
                    ratios.append(cur / base)
                else:
                    cells.append("-")
            tail = f"{100*(geo(ratios)-1):+.1f}%" if ratios else "-"
            print(f"| {name} | " + " | ".join(cells) + f" | {tail} |")

    print("\n## Greedy output agreement with baseline\n")
    for name in [baseline] + order:
        rows_p = {r["fixture"]: r for r in by_profile[name]}
        same, diff, missing = [], [], []
        for fixture in DECODE_FIXTURES + ["prefill8k"]:
            b = baseline_rows.get(fixture)
            c = rows_p.get(fixture)
            if not b or not c:
                missing.append(fixture)
            elif b.get("sha256") == c.get("sha256"):
                same.append(fixture)
            else:
                diff.append(fixture)
        print(f"- {name}: same {len(same)}/{len(same)+len(diff)}"
              + (f", DIFFERENT {diff}" if diff else "")
              + (f", missing {missing}" if missing else ""))

    print("\n## VRAM and load\n")
    print("| profile | load s | peak VRAM MiB | note |")
    print("|---|---|---|---|")
    for name in [baseline] + order:
        rows_p = by_profile[name]
        vram = [r.get("gpu", {}).get("vram_mib") for r in rows_p if r.get("gpu")]
        load = [r.get("load_s") for r in rows_p if r.get("load_s")]
        note = rows_p[0].get("note", "")
        print(f"| {name} | {load[0] if load else '-'} | {max(vram) if vram else '-'} | {note} |")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print("\n## Errors\n")
        for row in errors:
            print(f"- {row['profile']}/{row['fixture']}: {row['error'][:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
