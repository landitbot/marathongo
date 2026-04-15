#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from robot_seg_trt_bench import run_source


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fusion TensorRT video inference with correct person/car/robot labels."
    )
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--videos", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    return parser.parse_args()


def extract_mid_frame(video_path: Path, screenshot_dir: Path) -> str | None:
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame = min(30, max(frame_count // 2, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    screenshot_path = screenshot_dir / f"{video_path.stem}_frame{target_frame:04d}.jpg"
    if not cv2.imwrite(str(screenshot_path), frame):
        return None
    return str(screenshot_path)


def collect_videos(candidates: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for path in candidates:
        if path.is_dir():
            videos.extend(
                sorted(child for child in path.iterdir() if child.is_file() and child.suffix.lower() in VIDEO_EXTS)
            )
        elif path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            videos.append(path)
    return videos


def run_batch(
    engine: Path,
    videos: list[Path],
    output_dir: Path,
    max_frames: int | None,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output_dir / "screenshots"
    summary: list[dict[str, object]] = []

    for video in videos:
        label = f"{video.stem}_{engine.stem}"
        output_video = output_dir / f"{label}.mp4"
        metric = run_source(
            engine_path=engine,
            source=str(video),
            output_video=output_video,
            max_frames=max_frames,
            camera_width=camera_width,
            camera_height=camera_height,
            camera_fps=camera_fps,
        )
        screenshot_path = extract_mid_frame(output_video, screenshot_dir)
        result = asdict(metric)
        result["mid_frame"] = screenshot_path
        summary.append({"label": label, "result": result})

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    videos = collect_videos(args.videos)
    if not videos:
        raise FileNotFoundError("No valid videos were provided")
    run_batch(
        engine=args.engine,
        videos=videos,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
    )


if __name__ == "__main__":
    main()
