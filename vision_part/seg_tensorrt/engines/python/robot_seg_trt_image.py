#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from robot_seg_trt_video import YoloSegTrt, draw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TensorRT robot segmentation on images.")
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    return parser.parse_args()


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def run(engine: Path, input_dir: Path, output_dir: Path, conf: float, iou: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    runner = YoloSegTrt(engine)
    summary: list[dict[str, object]] = []
    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            batch = runner.preprocess(frame)
            t0 = time.perf_counter()
            outputs = runner.infer(batch)
            detections = runner.postprocess(outputs, frame.shape[:2], conf, iou)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            rendered = draw(frame, detections)
            out_path = output_dir / f"{image_path.stem}_{engine.stem}.jpg"
            cv2.imwrite(str(out_path), rendered)

            summary.append(
                {
                    "image": image_path.name,
                    "output": out_path.name,
                    "infer_ms": round(infer_ms, 4),
                    "num_detections": len(detections),
                    "classes": [det["name"] for det in detections],
                }
            )
    finally:
        runner.close()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    run(args.engine, args.input_dir, args.output_dir, args.conf, args.iou)


if __name__ == "__main__":
    main()
