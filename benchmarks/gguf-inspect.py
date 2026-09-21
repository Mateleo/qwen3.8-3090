#!/usr/bin/env python3
"""Minimal GGUF reader: dump metadata and tensor names (no gguf package needed).

Used to answer two questions about a local GGUF without loading it into VRAM:
  * what architecture / vocabulary size it declares,
  * whether it actually carries embedded MTP (nextn) tensors, which decides if
    the separate MTP drafter sidecar is required.

Usage: python3 gguf-inspect.py FILE.gguf [name-substring ...]
"""

import struct
import sys
from collections import Counter

_FMTS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _str(fh):
    (n,) = struct.unpack("<Q", fh.read(8))
    return fh.read(n).decode("utf-8", "replace")


def _value(fh, vtype):
    if vtype == 8:
        return _str(fh)
    if vtype == 9:  # array
        (atype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        vals = [_value(fh, atype) for _ in range(count)]
        return vals if count <= 32 else f"<array of {count} {atype}>"
    fmt = _FMTS.get(vtype)
    if fmt is None:
        raise ValueError(f"unknown value type {vtype}")
    return struct.unpack("<" + fmt, fh.read(_SCALARS[vtype]))[0]


def read(path, want_names=True):
    meta = {}
    tensors = []
    with open(path, "rb") as fh:
        magic, version = struct.unpack("<4sI", fh.read(8))
        if magic != b"GGUF":
            raise ValueError("not a GGUF file")
        (n_tensors,), (n_kv,) = struct.unpack("<Q", fh.read(8)), struct.unpack("<Q", fh.read(8))
        for _ in range(n_kv):
            key = _str(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            meta[key] = _value(fh, vtype)
        for _ in range(n_tensors):
            name = _str(fh)
            (nd,) = struct.unpack("<I", fh.read(4))
            dims = struct.unpack("<" + "Q" * nd, fh.read(8 * nd))
            (ttype,) = struct.unpack("<I", fh.read(4))
            (offset,) = struct.unpack("<Q", fh.read(8))
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
    return {"version": version, "meta": meta, "tensors": tensors}


def main():
    path = sys.argv[1]
    wanted = sys.argv[2:] or ["nextn", "mtp", "draft", "output", "token_embd"]
    info = read(path)
    meta, tensors = info["meta"], info["tensors"]
    print(f"file: {path}")
    print(f"gguf version {info['version']}, {len(tensors)} tensors, {len(meta)} metadata keys")
    for key in sorted(meta):
        if key.startswith("tokenizer.ggml.tokens"):
            print(f"  {key} = <omitted>")
            continue
        val = meta[key]
        if isinstance(val, str) and len(val) > 120:
            val = val[:117] + "..."
        print(f"  {key} = {val}")
    print("tensor name histogram (top 15 prefixes):")
    pref = Counter(".".join(t["name"].split(".")[:2]) for t in tensors)
    for name, count in pref.most_common(15):
        print(f"  {count:5d}  {name}")
    for token in wanted:
        hits = [t for t in tensors if token in t["name"]]
        print(f"tensors matching {token!r}: {len(hits)}")
        for t in hits[:12]:
            print(f"  {t['name']} dims={t['dims']} type={t['type']}")


if __name__ == "__main__":
    main()
