#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-fusion-v4l2-deepstream:7.1}"
base_image="${BASE_IMAGE:-ds-segformer:7.1-ready}"

docker build \
  --build-arg BASE_IMAGE="${base_image}" \
  -f "${repo_root}/fusion_v4l2_deepstream_app/Dockerfile" \
  -t "${image_name}" \
  "${repo_root}"
