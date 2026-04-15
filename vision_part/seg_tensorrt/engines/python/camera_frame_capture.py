#!/usr/bin/env python3
"""Capture N frames from a camera with OpenCV, v4l2-ctl, or gst-launch-1.0.

Examples
--------
OpenCV:
  python3 camera_frame_capture.py --method opencv --device /dev/video0 \
      --count 30 --width 1280 --height 720 --fps 30 --pixel-format MJPG

V4L2 (MJPG cameras recommended):
  python3 camera_frame_capture.py --method v4l2 --device /dev/video0 \
      --count 30 --width 1280 --height 720 --fps 30 --pixel-format MJPG

GStreamer:
  python3 camera_frame_capture.py --method gst --device /dev/video0 \
      --count 30 --width 1280 --height 720 --fps 30 --pixel-format MJPG
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


def require_command(cmd: str) -> None:
    if shutil.which(cmd) is None:
        raise FileNotFoundError(f"未找到命令: {cmd}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def camera_index_from_device(device: str) -> int:
    """Map /dev/videoN -> N for OpenCV. Fallback to 0."""
    base = os.path.basename(device)
    if base.startswith("video"):
        suffix = base[5:]
        if suffix.isdigit():
            return int(suffix)
    return 0


def save_report(paths: Iterable[Path]) -> None:
    files = [str(p) for p in paths]
    print(f"\n共输出 {len(files)} 帧:")
    for p in files[:10]:
        print(f"  {p}")
    if len(files) > 10:
        print("  ...")


def opencv_capture(
    device: str,
    count: int,
    out_dir: Path,
    prefix: str,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
    warmup: int,
) -> List[Path]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV 未安装，请先安装 python3-opencv 或 pip install opencv-python") from exc

    ensure_dir(out_dir)
    cam_index = camera_index_from_device(device)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开摄像头: {device} (索引 {cam_index})")

    # Best effort property setting. Availability depends on backend and driver.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if hasattr(cv2, "VideoWriter_fourcc"):
        fourcc = cv2.VideoWriter_fourcc(*pixel_format[:4])
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    for _ in range(max(0, warmup)):
        ok, _ = cap.read()
        if not ok:
            break

    saved: List[Path] = []
    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            print(f"[OpenCV] 第 {i+1} 帧读取失败，提前结束。", file=sys.stderr)
            break
        path = out_dir / f"{prefix}_{i:04d}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"OpenCV 保存失败: {path}")
        saved.append(path)

    cap.release()
    return saved


def split_mjpeg_stream(stream_path: Path, out_dir: Path, prefix: str, limit: int) -> List[Path]:
    """Split concatenated JPEG bitstream into individual JPEG files."""
    data = stream_path.read_bytes()
    ensure_dir(out_dir)

    saved: List[Path] = []
    pos = 0
    while len(saved) < limit:
        soi = data.find(b"\xff\xd8", pos)
        if soi < 0:
            break
        eoi = data.find(b"\xff\xd9", soi + 2)
        if eoi < 0:
            break
        frame = data[soi : eoi + 2]
        path = out_dir / f"{prefix}_{len(saved):04d}.jpg"
        path.write_bytes(frame)
        saved.append(path)
        pos = eoi + 2
    return saved


def v4l2_capture(
    device: str,
    count: int,
    out_dir: Path,
    prefix: str,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
) -> List[Path]:
    require_command("v4l2-ctl")
    ensure_dir(out_dir)

    pixel_format = pixel_format.upper()
    if pixel_format != "MJPG":
        raise RuntimeError("当前 v4l2 模式仅实现了 MJPG 摄像头切帧。请改用 --pixel-format MJPG，或使用 opencv/gst 模式。")

    stream_path = out_dir / "_capture_stream.mjpg"
    fmt = f"width={width},height={height},pixelformat={pixel_format}"
    cmd = [
        "v4l2-ctl",
        "-d",
        device,
        f"--set-fmt-video={fmt}",
        f"--set-parm={fps}",
        "--stream-mmap=3",
        f"--stream-count={count}",
        f"--stream-to={stream_path}",
    ]
    print("[V4L2] 运行:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    saved = split_mjpeg_stream(stream_path, out_dir, prefix, count)
    try:
        stream_path.unlink(missing_ok=True)
    except Exception:
        pass
    return saved


def gst_capture(
    device: str,
    count: int,
    out_dir: Path,
    prefix: str,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
) -> List[Path]:
    require_command("gst-launch-1.0")
    ensure_dir(out_dir)

    pixel_format = pixel_format.upper()
    location = out_dir / f"{prefix}_%04d.jpg"

    if pixel_format == "MJPG":
        caps = f"image/jpeg,width={width},height={height},framerate={fps}/1"
        pipeline = (
            f"gst-launch-1.0 -e "
            f"v4l2src device={device} num-buffers={count} ! "
            f"'{caps}' ! multifilesink location='{location}'"
        )
    else:
        # Common raw formats: YUYV -> YUY2 caps for GStreamer
        gst_fmt = "YUY2" if pixel_format == "YUYV" else pixel_format
        caps = f"video/x-raw,format={gst_fmt},width={width},height={height},framerate={fps}/1"
        pipeline = (
            f"gst-launch-1.0 -e "
            f"v4l2src device={device} num-buffers={count} ! "
            f"'{caps}' ! videoconvert ! jpegenc ! "
            f"multifilesink location='{location}'"
        )

    print("[GStreamer] 运行:")
    print(pipeline)
    subprocess.run(["bash", "-lc", pipeline], check=True)
    return sorted(out_dir.glob(f"{prefix}_*.jpg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自定义帧数量读取摄像头并输出图像")
    parser.add_argument("--method", choices=["opencv", "v4l2", "gst"], required=True, help="切帧方法")
    parser.add_argument("--device", default="/dev/video0", help="摄像头设备，例如 /dev/video0")
    parser.add_argument("--count", type=int, default=10, help="输出帧数量")
    parser.add_argument("--width", type=int, default=1280, help="宽度")
    parser.add_argument("--height", type=int, default=720, help="高度")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    parser.add_argument("--pixel-format", default="MJPG", help="像素格式，例如 MJPG / YUYV")
    parser.add_argument("--output-dir", default="frames_out", help="输出目录")
    parser.add_argument("--prefix", default="frame", help="输出文件名前缀")
    parser.add_argument("--warmup", type=int, default=5, help="OpenCV 模式预热帧数")
    return parser.parse_args()


METHODS = {
    "opencv": opencv_capture,
    "v4l2": v4l2_capture,
    "gst": gst_capture,
}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    start = time.time()

    func = METHODS[args.method]
    try:
        saved = func(
            device=args.device,
            count=args.count,
            out_dir=out_dir,
            prefix=args.prefix,
            width=args.width,
            height=args.height,
            fps=args.fps,
            pixel_format=args.pixel_format,
            warmup=args.warmup,
        ) if args.method == "opencv" else func(
            device=args.device,
            count=args.count,
            out_dir=out_dir,
            prefix=args.prefix,
            width=args.width,
            height=args.height,
            fps=args.fps,
            pixel_format=args.pixel_format,
        )
    except TypeError:
        # Fallback if function signature changes.
        saved = func(
            device=args.device,
            count=args.count,
            out_dir=out_dir,
            prefix=args.prefix,
            width=args.width,
            height=args.height,
            fps=args.fps,
            pixel_format=args.pixel_format,
        )
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    save_report(saved)
    print(f"耗时: {elapsed:.2f}s")
    if len(saved) != args.count:
        print(f"提示: 期望 {args.count} 帧，实际输出 {len(saved)} 帧。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
