#!/usr/bin/env bash
# Regenerate the golden parity data (tests/golden/cases.txt) from the C
# reference implementation. Requires a C toolchain + cmake; run once when
# the case table changes. CI only replays the committed file.
set -euo pipefail

DACE_C_SRC="${DACE_C_SRC:-/home/ouyangjiahong/codes/dace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$DACE_C_SRC/build-c"

cmake -S "$DACE_C_SRC" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

gcc -O2 -I"$BUILD_DIR/core/include" -I"$DACE_C_SRC/core/include" \
    "$SCRIPT_DIR/main.c" \
    "$BUILD_DIR/libdace_s.a" -lm -lstdc++ -o "$BUILD_DIR/golden_gen"

"$BUILD_DIR/golden_gen" > "$REPO_ROOT/tests/golden/cases.txt"
echo "wrote $REPO_ROOT/tests/golden/cases.txt ($(grep -c '^CASE' "$REPO_ROOT/tests/golden/cases.txt") cases)"
