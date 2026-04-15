#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_root="${repo_root}/fusion_v4l2_deepstream_app"
image_name="${IMAGE_NAME:-fusion-v4l2-deepstream:7.1}"
camera_device="${CAMERA_DEVICE:-/dev/video0}"
display_value="${DISPLAY:-}"
xauth_value="${XAUTHORITY:-}"
output_dir="${app_root}/output"
host_build_binary="${app_root}/build/fusion_v4l2_deepstream_app"
mkdir -p "${output_dir}"

timestamp="$(date +%Y%m%d_%H%M%S)"
default_output="/workspace/fusion_v4l2_deepstream_app/output/fusion_seg_${timestamp}.mp4"
stats_log="${output_dir}/tegrastats_${timestamp}.log"

enable_preview=0
has_output_arg=0
forward_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --preview)
      enable_preview=1
      forward_args+=("$1")
      shift
      ;;
    --output)
      has_output_arg=1
      forward_args+=("$1")
      shift
      if [ "$#" -gt 0 ]; then
        forward_args+=("$1")
        shift
      fi
      ;;
    --device)
      forward_args+=("$1")
      shift
      if [ "$#" -gt 0 ]; then
        camera_device="$1"
        forward_args+=("$1")
        shift
      fi
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

if [ ! -e "${camera_device}" ]; then
  echo "Camera device not found: ${camera_device}" >&2
  ls -l /dev/video* 2>/dev/null || true
  exit 1
fi

if [ "${has_output_arg}" = "0" ]; then
  forward_args+=(--output "${default_output}")
fi

docker_args=(--rm)
if [ -t 0 ] && [ -t 1 ]; then
  docker_args+=(-it)
else
  docker_args+=(-i)
fi

extra_args=()
if [ "${enable_preview}" = "1" ] && [ -n "${display_value}" ] && [ -d /tmp/.X11-unix ]; then
  extra_args+=(-e "DISPLAY=${display_value}")
  extra_args+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

if [ "${enable_preview}" = "1" ] && [ -n "${xauth_value}" ] && [ -f "${xauth_value}" ]; then
  extra_args+=(-e "XAUTHORITY=${xauth_value}")
  extra_args+=(-v "${xauth_value}:${xauth_value}:ro")
fi

tegrastats_pid=""

cleanup() {
  if [ -n "${tegrastats_pid}" ] && kill -0 "${tegrastats_pid}" 2>/dev/null; then
    kill "${tegrastats_pid}" 2>/dev/null || true
    wait "${tegrastats_pid}" 2>/dev/null || true
  fi

  if [ -f "${stats_log}" ]; then
    python3 - "$stats_log" <<'PY'
import re
import sys

path = sys.argv[1]
peak_ram = 0
total_ram = 0
peak_gr3d = 0
peak_vic = 0
sum_gr3d = 0
sum_vic = 0
count_gr3d = 0
count_vic = 0

for line in open(path, encoding="utf-8", errors="ignore"):
    m = re.search(r"RAM (\d+)/(\d+)MB", line)
    if m:
        used = int(m.group(1))
        total = int(m.group(2))
        peak_ram = max(peak_ram, used)
        total_ram = total

    m = re.search(r"GR3D_FREQ (\d+)%", line)
    if m:
        util = int(m.group(1))
        peak_gr3d = max(peak_gr3d, util)
        sum_gr3d += util
        count_gr3d += 1

    m = re.search(r"VIC_FREQ (\d+)%", line)
    if m:
        util = int(m.group(1))
        peak_vic = max(peak_vic, util)
        sum_vic += util
        count_vic += 1

print("[tegrastats-summary]")
print(f"  shared_ram_peak_mb={peak_ram}")
if total_ram:
    print(f"  shared_ram_total_mb={total_ram}")
if count_gr3d:
    print(f"  gr3d_mean_pct={sum_gr3d / count_gr3d:.2f}")
    print(f"  gr3d_peak_pct={peak_gr3d}")
if count_vic:
    print(f"  vic_mean_pct={sum_vic / count_vic:.2f}")
    print(f"  vic_peak_pct={peak_vic}")
PY
  fi
}

trap cleanup EXIT

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 1000 > "${stats_log}" &
  tegrastats_pid="$!"
fi

set +e
container_exec="/opt/fusion_v4l2_deepstream_app/build/fusion_v4l2_deepstream_app"
if [ -x "${host_build_binary}" ]; then
  container_exec="/workspace/fusion_v4l2_deepstream_app/build/fusion_v4l2_deepstream_app"
fi

docker run "${docker_args[@]}" \
  --runtime=nvidia \
  --network=host \
  --ipc=host \
  --device "${camera_device}:${camera_device}" \
  -v "${repo_root}:/workspace" \
  "${extra_args[@]}" \
  "${image_name}" \
  "${container_exec}" "${forward_args[@]}"
status=$?
set -e

if [ "${status}" = "130" ] || [ "${status}" = "143" ]; then
  exit 0
fi

exit "${status}"
