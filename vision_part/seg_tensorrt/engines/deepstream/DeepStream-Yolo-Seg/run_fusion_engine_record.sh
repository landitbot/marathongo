#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${script_dir}/.." && pwd)"
repo_in_container="/workspace/DeepStream-Yolo-Seg"

image="${IMAGE:-ds-segformer:7.1-ready}"
camera_device="${CAMERA_DEVICE:-/dev/video0}"
camera_caps="${CAMERA_CAPS:-image/jpeg,width=1280,height=720,framerate=30/1}"
config_path="${CONFIG_PATH:-${repo_in_container}/config_infer_primary_fusion_960x544_runtime_tensormeta.txt}"
bitrate="${BITRATE:-4000000}"
duration_sec="${DURATION_SEC:-}"
host_uid="$(id -u)"
host_gid="$(id -g)"
camera_gid="$(stat -c '%g' "${camera_device}")"

host_output_dir="${OUTPUT_DIR:-${script_dir}/output}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_name="${OUTPUT_NAME:-usb_detect_${timestamp}.mp4}"
host_output_path="${host_output_dir%/}/${output_name}"
container_output_path="${OUTPUT_PATH_IN_CONTAINER:-${repo_in_container}/output/${output_name}}"

mkdir -p "${host_output_dir}"

echo "image=${image}"
echo "camera_device=${camera_device}"
echo "config_path=${config_path}"
echo "output=${host_output_path}"
if [[ -n "${duration_sec}" ]]; then
  echo "duration_sec=${duration_sec}"
fi

gst_cmd="
cd ${repo_in_container}
$(if [[ -n "${duration_sec}" ]]; then printf 'timeout --signal=INT %ss ' "${duration_sec}"; fi)gst-launch-1.0 -e -v \
  nvstreammux name=mux batch-size=1 width=960 height=544 live-source=1 batched-push-timeout=33000 nvbuf-memory-type=0 ! \
  nvinfer config-file-path=${config_path} ! \
  tee name=t \
    t. ! queue ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
    t. ! queue ! nvvideoconvert compute-hw=2 interpolation-method=4 ! \
      'video/x-raw(memory:NVMM),format=NV12' ! \
      nvv4l2h264enc bitrate=${bitrate} iframeinterval=30 insert-sps-pps=1 idrinterval=30 ! \
      h264parse ! qtmux ! filesink location=${container_output_path} sync=false async=false \
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

docker_exit=0

docker run --rm \
  --runtime=nvidia \
  --network=host \
  --ipc=host \
  --user "${host_uid}:${host_gid}" \
  --group-add "${camera_gid}" \
  --device "${camera_device}:${camera_device}" \
  -v "${workspace_root}:/workspace" \
  "${image}" \
  bash -lc "${gst_cmd}" || docker_exit=$?

if [[ "${docker_exit}" -ne 0 ]]; then
  if [[ -n "${duration_sec}" && "${docker_exit}" -eq 124 ]]; then
    docker_exit=0
  else
    exit "${docker_exit}"
  fi
fi

echo "saved=${host_output_path}"
