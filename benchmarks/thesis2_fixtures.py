"""Prompt fixtures for the thesis screen.

Plain, deterministic prompt text. No chat template is applied: the screen posts
raw completions so that server-side timings are directly comparable between
runtimes. Each fixture targets one workload property:

  novel-code  : code the model cannot have seen (novel reasoning/composition)
  prose       : descriptive prose, low lexical overlap with the prompt
  patch       : a concrete edit of a supplied file (copy-heavy, realistic)
  copy        : verbatim reproduction (n-gram friendly, deliberately favourable)
  prefill8k   : long context, one-token answer (cold prefill throughput)
  cache-a/b   : long shared prefix for prompt-cache reuse measurement
"""

NUMBERED_TABLE = "".join(
    "row %04d | station=%s | reading=%d.%02d | status=%s\n"
    % (i, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i % 26] + str(i % 97), 100 + (i * 7) % 900,
       (i * 13) % 100, ("ok" if i % 3 else "drift"))
    for i in range(1, 641)
)

NUMBERED_TABLE_32K = "".join(
    "row %05d | station=%s | reading=%d.%02d | status=%s\n"
    % (i, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i % 26] + str(i % 97), 100 + (i * 7) % 900,
       (i * 13) % 100, ("ok" if i % 3 else "drift"))
    for i in range(1, 2561)
)

SOURCE_FILE = '''\
import json
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Station:
    name: str
    readings: list[float]

    def mean(self) -> float:
        return sum(self.readings) / max(len(self.readings), 1)

    def peak(self) -> float:
        return max(self.readings, default=0.0)


def load_station(path: str) -> Station:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Station(name=payload["name"], readings=[float(v) for v in payload["values"]])


def summarise(stations: Iterable[Station]) -> dict:
    out = {}
    for station in stations:
        out[station.name] = {"mean": station.mean(), "peak": station.peak()}
    return out


def main() -> None:
    paths = sorted(p for p in os.listdir("data") if p.endswith(".json"))
    stations = [load_station(os.path.join("data", p)) for p in paths]
    print(json.dumps(summarise(stations), indent=2))


if __name__ == "__main__":
    main()
'''

FIXTURES = {
    "novel-code": (
        "Write a Python 3.12 module that solves the following problem from scratch.\n"
        "Problem: given a directed graph of tasks with integer durations and a set of\n"
        "resource types with fixed capacities, compute a schedule minimising the makespan\n"
        "under the constraint that a task may only start when every predecessor finished\n"
        "and the resources it needs are free. Return the schedule and the makespan.\n"
        "Use only the standard library (heapq, collections, dataclasses). Include input\n"
        "parsing for a compact text format, deterministic ordering of tie-breaks, and a\n"
        "self-test in a __main__ block. Explain none of it; output the module only.\n"
    ),
    "prose": (
        "Explain, in continuous prose, why hysteresis losses appear in ferromagnetic\n"
        "materials in a switching magnetic field, how domain wall motion and pinning\n"
        "contribute to them, and how a designer of a small transformer would keep them\n"
        "low without changing the operating frequency. Do not use lists or headings.\n"
    ),
    "patch": (
        "Apply exactly one change to this file: every station that ever reported a\n"
        "reading below zero must be dropped from the summary, and the summary must also\n"
        "report the number of dropped readings per station. Output the complete revised\n"
        "file, with the same imports and the same public function names.\n\n"
        "```python\n" + SOURCE_FILE + "```\n"
    ),
    "copy": (
        "Reproduce the following source file byte for byte, without any commentary,\n"
        "without a code fence, and without changing a single character:\n\n"
        "```python\n" + SOURCE_FILE + "```\n"
    ),
    "prefill8k": (
        "The table below is a day of sensor telemetry. Answer with a single integer: the\n"
        "number of rows whose status is 'drift'.\n\n" + NUMBERED_TABLE + "\nAnswer:"
    ),
    "prefill32k": (
        "The table below is sensor telemetry in five blocks. Answer with a single\n"
        "integer: the number of rows whose status is 'drift'.\n\n" + NUMBERED_TABLE_32K + "\nAnswer:"
    ),
    "cache-a": (
        "Context block for a long conversation. Read it, then answer the question.\n\n"
        + NUMBERED_TABLE + "\nQuestion: how many rows are listed above? Answer with one integer.\n"
    ),
    "cache-b": (
        "Context block for a long conversation. Read it, then answer the question.\n\n"
        + NUMBERED_TABLE + "\nQuestion: what is the reading of row 0007? Answer with one number.\n"
    ),
}

# Fixtures that measure cold prefill rather than decode.
PREFILL_ONLY = {"prefill8k", "prefill32k"}
