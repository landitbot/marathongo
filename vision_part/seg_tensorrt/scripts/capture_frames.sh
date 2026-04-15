#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/data/images/camera_capture"
mkdir -p "$OUTPUT_DIR"

python3 "$ROOT_DIR/engines/python/camera_frame_capture.py" \
  --method opencv \
  --device /dev/video0 \
  --count 20 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --pixel-format MJPG \
  --output-dir "$OUTPUT_DIR" \
  --prefix robot_seg
