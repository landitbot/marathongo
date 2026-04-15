from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from onnx_inference_test import YOLOv8Seg


def run_video_inference(
    model_path: str,
    source: str,
    width: int = 1280,
    height: int = 720,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    writer = None
    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_file), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create VideoWriter: {save_file}")

    model = YOLOv8Seg(model_path, imgsz=(height, width))
    model.debug = False

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        results = model(frame)
        vis = results[0].plot()

        cv2.putText(
            vis,
            f"frame: {frame_idx}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if writer is not None:
            writer.write(vis)

        if show:
            cv2.imshow("ONNX Video Inference", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Processed frames: {frame_idx}")
    if save_path:
        print(f"Saved visualization video to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ONNX video inference with OpenCV (1280x720)")
    parser.add_argument("--model", type=str, default="/path/to/fusion_robotbase.onnx")
    parser.add_argument("--source", type=str, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save", type=str, default="/path/to/vis/onnx_video/test1_vis.mp4")
    parser.add_argument("--noshow", action="store_true", help="Disable preview window")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_video_inference(
        model_path=args.model,
        source=args.source,
        width=args.width,
        height=args.height,
        save_path=args.save,
        show=not args.noshow,
    )
