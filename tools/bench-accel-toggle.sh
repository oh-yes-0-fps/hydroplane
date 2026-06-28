#!/usr/bin/env bash
# Run the `gemm_accel_toggle` criterion bench twice — the *identical* kernel, once per backend — and
# diff them. The backend is a compile-time choice, so each pass is a separate compilation:
#
#   1. default build              → Apple Accelerate (cblas → AMX/SME), saved as baseline "accelerate"
#   2. `--cfg no_apple_accelerate` → spmd's hand-rolled SME2 grid kernel, compared vs that baseline
#
# Criterion prints the per-size % change of the SME path against Accelerate (with p-values). Any extra
# args are forwarded to criterion, e.g. a quick pass:
#   tools/bench-accel-toggle.sh --warm-up-time 0.5 --measurement-time 2 --sample-size 10
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BENCH=(cargo bench --bench gemm_accel_toggle --)

echo "==> [1/2] Apple Accelerate (default cfg) — saving baseline 'accelerate'"
"${BENCH[@]}" "$@" --save-baseline accelerate

echo
echo "==> [2/2] hand-rolled SME (--cfg no_apple_accelerate) — comparing vs 'accelerate'"
RUSTFLAGS="--cfg no_apple_accelerate" "${BENCH[@]}" "$@" --baseline accelerate
