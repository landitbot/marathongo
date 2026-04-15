#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-fusion-v4l2-deepstream:7.1}"

docker run --rm \
  --runtime=nvidia \
  -v "${repo_root}:/workspace" \
  "${image_name}" \
  bash -lc '
    set -euo pipefail
    cd /workspace/fusion_v4l2_deepstream_app
    rm -rf build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel "$(nproc)"
  '
