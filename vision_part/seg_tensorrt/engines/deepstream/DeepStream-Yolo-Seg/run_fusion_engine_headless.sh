#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${script_dir}/.." && pwd)"

image="${IMAGE:-ds-segformer:7.1-ready}"
camera_device="${CAMERA_DEVICE:-/dev/video0}"
camera_caps="${CAMERA_CAPS:-image/jpeg,width=1280,height=720,framerate=30/1}"
config_path="${CONFIG_PATH:-/workspace/DeepStream-Yolo-Seg/config_infer_primary_fusion_960x544_engine_tensormeta.txt}"

docker run --rm -it \
  --runtime=nvidia \
  --network=host \
  --ipc=host \
  --device "${camera_device}:${camera_device}" \
  -v "${workspace_root}:/workspace" \
  "${image}" \
  bash -lc "
    gst-launch-1.0 -e -v \
      nvstreammux name=mux batch-size=1 width=960 height=544 live-source=1 batched-push-timeout=33000 nvbuf-memory-type=0 ! \
      nvinfer config-file-path=${config_path} ! \
      fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
      nvcompositor name=comp sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 ! \
      nvvideoconvert compute-hw=2 interpolation-method=4 ! \
      'video/x-raw(memory:NVMM),format=NV12,width=960,height=544' ! \
      mux.sink_0 \
      v4l2src device=${camera_device} ! ${camera_caps} ! \
      nvv4l2decoder mjpeg=1 ! \
      nvvideoconvert compute-hw=2 interpolation-method=4 ! \
      'video/x-raw(memory:NVMM),format=NV12,width=960,height=540' ! \
      comp.sink_0
  "
