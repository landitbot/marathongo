#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs/visualizations/images"
mkdir -p "$OUTPUT_DIR"

if ! find "$ROOT_DIR/data/images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | grep -q .; then
  echo "No images found in $ROOT_DIR/data/images" >&2
  echo "Put test images there first, then rerun bash scripts/infer_image.sh" >&2
  exit 1
fi

python3 "$ROOT_DIR/engines/python/robot_seg_trt_image.py" \
  --engine "$ROOT_DIR/engines/weights/fusion_branch10_960x544_b1_fp16.engine" \
  --input-dir "$ROOT_DIR/data/images" \
  --output-dir "$OUTPUT_DIR" \
  --conf 0.25 \
  --iou 0.70
