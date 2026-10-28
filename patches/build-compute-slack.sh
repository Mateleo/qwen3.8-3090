#!/usr/bin/env bash
# Build a patched libggml-base.so for the llamAmpere runtime without touching
# its source tree or build directory: compile the patched ggml-alloc.c, relink it
# with the existing objects, and install the result in its own directory. The
# server loads it ahead of its RUNPATH through LD_LIBRARY_PATH (start.sh does this).
# Takes seconds and one core; no CUDA code is recompiled.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QWEN_AMPERE_REPO:-$HOME/llamAmpere-src/repo}"
BUILD="${QWEN_AMPERE_BUILD:-$REPO/build-sm86}"
OUT="${QWEN_SLACK_LIB:-$HOME/llamAmpere-src/libslack}"
PATCH="$HERE/llamampere-compute-slack.patch"
OBJ="$BUILD/ggml/src/CMakeFiles/ggml-base.dir"

[[ -d "$OBJ" ]] || { echo "No ggml-base objects in $OBJ; build the runtime first." >&2; exit 1; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/ggml/src"
cp "$REPO/ggml/src/ggml-alloc.c" "$work/ggml/src/"
patch -d "$work" -p1 --quiet <"$PATCH"

# Same defines and flags as the runtime's own ggml-base target (flags.make).
flags="$(sed -n 's/^C_DEFINES = //p; s/^C_FLAGS = //p' "$OBJ/flags.make" | tr '\n' ' ')"
mkdir -p "$OUT"
# shellcheck disable=SC2086
cc $flags -I"$REPO/ggml/src" -I"$BUILD/ggml/src" -I"$REPO/ggml/include" \
  -c "$work/ggml/src/ggml-alloc.c" -o "$work/ggml-alloc.c.o"

objs=()
for o in "$OBJ"/*.o; do
  [[ "$(basename "$o")" == ggml-alloc.c.o ]] && objs+=("$work/ggml-alloc.c.o") || objs+=("$o")
done
soname="$(readlink "$BUILD/bin/libggml-base.so.0")"
# Link line of the original target, minus its objects and output.
libs="$(grep -oE ' -l[^ ]+| /[^ ]+\.(so|a)' "$OBJ/link.txt" | tr '\n' ' ')"
# shellcheck disable=SC2086
c++ -fPIC -O3 -DNDEBUG -shared -Wl,-soname,libggml-base.so.0 -o "$OUT/$soname" "${objs[@]}" $libs
ln -sf "$soname" "$OUT/libggml-base.so.0"
ln -sf libggml-base.so.0 "$OUT/libggml-base.so"
echo "Installed $OUT/$soname"
