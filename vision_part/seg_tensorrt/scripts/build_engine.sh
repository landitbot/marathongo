#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_PATH="${1:-}"
ENGINE_PATH="${2:-$ROOT_DIR/engines/weights/fusion_branch10_960x544_fp16.engine}"
PRECISION="${PRECISION:-fp16}"
WORKSPACE_MIB="${WORKSPACE_MIB:-4096}"
LOG_DIR="$ROOT_DIR/outputs/logs"
LOG_PATH="$LOG_DIR/trtexec_${PRECISION}.log"

if [ -z "$ONNX_PATH" ]; then
  echo "Usage: bash scripts/build_engine.sh <onnx_path> [engine_output_path]" >&2
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")" "$LOG_DIR"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Please install TensorRT first." >&2
  exit 1
fi

CMD=(
  trtexec
  "--onnx=$ONNX_PATH"
  "--saveEngine=$ENGINE_PATH"
  "--workspace=$WORKSPACE_MIB"
  "--verbose"
)

case "$PRECISION" in
  fp16) CMD+=(--fp16) ;;
  int8) CMD+=(--int8) ;;
  fp32) ;;
  *)
    echo "Unsupported PRECISION: $PRECISION" >&2
    exit 2
    ;;
esac

printf 'Running:'
for arg in "${CMD[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${CMD[@]}" 2>&1 | tee "$LOG_PATH"
