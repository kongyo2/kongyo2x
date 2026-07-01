#!/usr/bin/env bash
# Build the Rust numeric kernels to WebAssembly and stage the optimized artifact
# at wasm/kongyo2x.wasm. The result is committed so the published package and CI
# never need a Rust toolchain; run this only when the Rust sources change.
set -euo pipefail

cd "$(dirname "$0")/.."

CRATE=kongyo2x-wasm
TARGET=wasm32-unknown-unknown
RAW="target/${TARGET}/release/${CRATE//-/_}.wasm"
OUT="wasm/kongyo2x.wasm"

echo "· cargo build --release --target ${TARGET}"
cargo build -p "$CRATE" --target "$TARGET" --release

mkdir -p wasm

if command -v wasm-opt >/dev/null 2>&1; then
  echo "· wasm-opt -O3"
  wasm-opt -O3 --enable-simd --enable-bulk-memory "$RAW" -o "$OUT"
else
  echo "· wasm-opt not found; copying unoptimized module"
  cp "$RAW" "$OUT"
fi

printf '· wrote %s (%s bytes)\n' "$OUT" "$(wc -c < "$OUT")"
