#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs/visualizations/videos"
mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
VIDEOS=("$ROOT_DIR"/data/videos/*)

if [ "${#VIDEOS[@]}" -eq 0 ]; then
  echo "No videos found in $ROOT_DIR/data/videos" >&2
  echo "Put test videos there first, then rerun bash scripts/infer_video.sh" >&2
  exit 1
fi

python3 "$ROOT_DIR/engines/python/robot_seg_fusion_video.py" \
  --engine "$ROOT_DIR/engines/weights/fusion_fp16.engine" \
  --videos "${VIDEOS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --max-frames "${MAX_FRAMES:-300}"
