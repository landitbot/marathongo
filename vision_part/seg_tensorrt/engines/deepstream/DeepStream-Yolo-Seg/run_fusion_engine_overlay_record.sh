#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${script_dir}/.." && pwd)"
repo_in_container="/workspace/DeepStream-Yolo-Seg"

image="${IMAGE:-ds-segformer:7.1-ready}"
camera_device="${CAMERA_DEVICE:-/dev/video0}"
camera_caps="${CAMERA_CAPS:-image/jpeg,width=1280,height=720,framerate=30/1}"
config_path="${CONFIG_PATH:-${repo_in_container}/config_infer_primary_fusion_640x640_seg.txt}"
bitrate="${BITRATE:-4000000}"
duration_sec="${DURATION_SEC:-}"
save_segment_points="${SAVE_SEGMENT_POINTS:-0}"
segment_point_stride="${SEGMENT_POINT_STRIDE:-4}"

frame_width="${FRAME_WIDTH:-960}"
frame_height="${FRAME_HEIGHT:-544}"
content_height="${CONTENT_HEIGHT:-540}"

host_uid="$(id -u)"
host_gid="$(id -g)"
camera_gid="$(stat -c '%g' "${camera_device}")"

host_output_dir="${OUTPUT_DIR:-${script_dir}/output}"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_name="${OUTPUT_NAME:-usb_detect_overlay_${timestamp}.mp4}"
host_output_path="${host_output_dir%/}/${output_name}"
container_output_path="${OUTPUT_PATH_IN_CONTAINER:-${repo_in_container}/output/${output_name}}"
coord_name="${COORD_NAME:-${output_name%.mp4}.coords.jsonl}"
host_coord_path="${host_output_dir%/}/${coord_name}"
container_coord_path="${COORD_PATH_IN_CONTAINER:-${repo_in_container}/output/${coord_name}}"

mkdir -p "${host_output_dir}"

echo "image=${image}"
echo "camera_device=${camera_device}"
echo "config_path=${config_path}"
echo "output=${host_output_path}"
echo "coord_log=${host_coord_path}"
echo "save_segment_points=${save_segment_points}"
echo "segment_point_stride=${segment_point_stride}"
if [[ -n "${duration_sec}" ]]; then
  echo "duration_sec=${duration_sec}"
fi

gst_cmd="
cd ${repo_in_container}
$(if [[ -n "${duration_sec}" ]]; then printf 'timeout --signal=INT %ss ' "${duration_sec}"; fi)gst-launch-1.0 -e -v \
  nvstreammux name=mux batch-size=1 width=${frame_width} height=${frame_height} live-source=1 batched-push-timeout=33000 nvbuf-memory-type=0 ! \
  nvinfer config-file-path=${config_path} ! \
  nvvideoconvert compute-hw=2 interpolation-method=4 ! \
  'video/x-raw(memory:NVMM),format=RGBA,width=${frame_width},height=${frame_height}' ! \
  nvdsosd display-mask=true display-bbox=true display-text=true process-mode=1 ! \
  tee name=t \
    t. ! queue ! nvvideoconvert compute-hw=2 interpolation-method=4 ! \
      'video/x-raw(memory:NVMM),format=NV12' ! \
      nvv4l2h264enc bitrate=${bitrate} iframeinterval=30 insert-sps-pps=1 idrinterval=30 ! \
      h264parse ! qtmux ! filesink location=${container_output_path} sync=false async=false \
    t. ! queue ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
  nvcompositor name=comp sink_0::xpos=0 sink_0::ypos=0 sink_0::width=${frame_width} sink_0::height=${content_height} ! \
  nvvideoconvert compute-hw=2 interpolation-method=4 ! \
  'video/x-raw(memory:NVMM),format=NV12,width=${frame_width},height=${frame_height}' ! \
  mux.sink_0 \
  v4l2src device=${camera_device} ! ${camera_caps} ! \
  nvv4l2decoder mjpeg=1 ! \
  nvvideoconvert compute-hw=2 interpolation-method=4 ! \
  'video/x-raw(memory:NVMM),format=NV12,width=${frame_width},height=${content_height}' ! \
  comp.sink_0
"

docker_exit=0

docker run --rm \
  --runtime=nvidia \
  --network=host \
  --ipc=host \
  --user "${host_uid}:${host_gid}" \
  --group-add "${camera_gid}" \
  --env FUSION_COORD_LOG_PATH="${container_coord_path}" \
  --env FUSION_COORD_STDOUT="${COORD_STDOUT:-1}" \
  --env FUSION_SEGMENT_POINTS="${save_segment_points}" \
  --env FUSION_SEGMENT_POINT_STRIDE="${segment_point_stride}" \
  --env FUSION_SOURCE_WIDTH="${frame_width}" \
  --env FUSION_SOURCE_HEIGHT="${frame_height}" \
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
echo "coords=${host_coord_path}"
