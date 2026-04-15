#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${script_dir}/.." && pwd)"

image="${IMAGE:-ds-segformer:7.1-ready}"
config_path="${CONFIG_PATH:-/workspace/DeepStream-Yolo-Seg/config_infer_primary_fusion_960x544_engine_tensormeta.txt}"
output_dir="${OUTPUT_DIR:-${script_dir}/deepstream_fps_outputs}"

mkdir -p "${output_dir}"

if [ "$#" -gt 0 ]; then
  host_videos=("$@")
else
  shopt -s nullglob
  host_videos=("${workspace_root}"/video*.mp4)
  shopt -u nullglob
fi

if [ "${#host_videos[@]}" -eq 0 ]; then
  echo "No input videos found. Pass video paths or place video*.mp4 in ${workspace_root}." >&2
  exit 1
fi

summary_csv="${output_dir}/fps_summary.csv"
printf 'video,average_fps,last_current_fps,rendered_frames,log_file\n' > "${summary_csv}"

for host_video in "${host_videos[@]}"; do
  if [ ! -f "${host_video}" ]; then
    echo "Skip missing video: ${host_video}" >&2
    continue
  fi

  video_name="$(basename "${host_video}")"
  video_stem="${video_name%.*}"
  container_video="/workspace/${video_name}"
  log_file="${output_dir}/${video_stem}_deepstream_fps.log"

  echo "Running ${video_name} ..."

  docker run --rm \
    --runtime=nvidia \
    --network=host \
    --ipc=host \
    -v "${workspace_root}:/workspace" \
    -e DS_VIDEO="${container_video}" \
    -e DS_CONFIG="${config_path}" \
    "${image}" \
    bash -lc '
      set -euo pipefail
      gst-launch-1.0 -e -v \
        nvstreammux name=mux batch-size=1 width=960 height=544 live-source=0 batched-push-timeout=40000 nvbuf-memory-type=0 ! \
        nvinfer config-file-path="${DS_CONFIG}" ! \
        fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
        nvcompositor name=comp sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 ! \
        nvvideoconvert compute-hw=2 interpolation-method=4 ! \
        "video/x-raw(memory:NVMM),format=NV12,width=960,height=544" ! \
        mux.sink_0 \
        uridecodebin uri="file://${DS_VIDEO}" ! \
        queue ! \
        nvvideoconvert compute-hw=2 interpolation-method=4 ! \
        "video/x-raw(memory:NVMM),format=NV12,width=960,height=540" ! \
        comp.sink_0
    ' > "${log_file}" 2>&1

  last_fps_line="$(grep 'last-message = rendered:' "${log_file}" | tail -n 1 || true)"
  average_fps="$(printf '%s\n' "${last_fps_line}" | sed -n 's/.*average: \([0-9.]*\).*/\1/p')"
  current_fps="$(printf '%s\n' "${last_fps_line}" | sed -n 's/.*current: \([0-9.]*\).*/\1/p')"
  rendered_frames="$(printf '%s\n' "${last_fps_line}" | sed -n 's/.*rendered: \([0-9]*\).*/\1/p')"

  average_fps="${average_fps:-NA}"
  current_fps="${current_fps:-NA}"
  rendered_frames="${rendered_frames:-NA}"

  printf '%s,%s,%s,%s,%s\n' "${video_name}" "${average_fps}" "${current_fps}" "${rendered_frames}" "${log_file}" >> "${summary_csv}"
  echo "  average_fps=${average_fps} current_fps=${current_fps} rendered=${rendered_frames}"
done

echo "Summary written to ${summary_csv}"
