#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs/benchmarks/fp16_baseline"
mkdir -p "$OUTPUT_DIR"
SOURCE_VIDEO="${1:-$ROOT_DIR/data/videos/video8.mp4}"
SOURCE_STEM="$(basename "${SOURCE_VIDEO%.*}")"

if [ ! -f "$SOURCE_VIDEO" ]; then
  echo "Benchmark source video not found: $SOURCE_VIDEO" >&2
  echo "Usage: bash scripts/benchmark_video.sh /path/to/test_video.mp4" >&2
  exit 1
fi

python3 "$ROOT_DIR/engines/python/robot_seg_trt_bench.py" \
  --engine "$ROOT_DIR/engines/weights/fusion_fp16.engine" \
  --source "$SOURCE_VIDEO" \
  --output-video "$OUTPUT_DIR/${SOURCE_STEM}_fp16_result.mp4" \
  --output-dir "$OUTPUT_DIR" \
  --label fp16_baseline \
  --max-frames 300
