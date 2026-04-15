#!/usr/bin/env python3
from __future__ import annotations

"""Generic YOLO11 segmentation video runner.

This script decodes standard YOLO segmentation outputs such as ``output0`` and
``output1``. It is not the correct entrypoint for the fusion dual-branch engine
that renders ``person / car / robot`` labels. For fusion engines, use
``robot_seg_fusion_video.py`` instead.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

FILTER_CLASSES = {"person", "car", "bus", "truck", "bicycle", "motorcycle"}
COLORS = {
    "person": np.array([40, 220, 40], dtype=np.uint8),
    "car": np.array([0, 180, 255], dtype=np.uint8),
    "bus": np.array([255, 180, 0], dtype=np.uint8),
    "truck": np.array([255, 120, 0], dtype=np.uint8),
    "bicycle": np.array([200, 100, 255], dtype=np.uint8),
    "motorcycle": np.array([120, 100, 255], dtype=np.uint8),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=np.int32)


def letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def scale_boxes(img1_shape: tuple[int, int], boxes: np.ndarray, img0_shape: tuple[int, int]) -> np.ndarray:
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2.0
    pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2.0
    out = boxes.copy()
    out[:, [0, 2]] -= pad_w
    out[:, [1, 3]] -= pad_h
    out[:, :4] /= gain
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, img0_shape[1] - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, img0_shape[0] - 1)
    return out


def scale_masks(masks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if masks.size == 0:
        return np.empty((0, shape[0], shape[1]), dtype=np.float32)
    mh, mw = masks.shape[1:]
    gain = min(mh / shape[0], mw / shape[1])
    pad_w = (mw - shape[1] * gain) / 2.0
    pad_h = (mh - shape[0] * gain) / 2.0
    top = max(int(round(pad_h - 0.1)), 0)
    left = max(int(round(pad_w - 0.1)), 0)
    bottom = min(int(round(mh - pad_h + 0.1)), mh)
    right = min(int(round(mw - pad_w + 0.1)), mw)
    cropped = masks[:, top:bottom, left:right]
    resized = [cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) for mask in cropped]
    return np.stack(resized, axis=0)


def crop_masks(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if masks.size == 0:
        return np.empty_like(masks, dtype=bool)
    out = np.zeros_like(masks, dtype=bool)
    h, w = masks.shape[1:]
    for i, box in enumerate(boxes.astype(np.int32)):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2] > 0.5
    return out


class YoloSegTrt:
    def __init__(self, engine_path: Path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: dict[str, dict[str, object]] = {}
        self.input_name = ""

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(trt.volume(shape))
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            meta = {"shape": shape, "dtype": dtype, "host": host, "device": device, "mode": self.engine.get_tensor_mode(name)}
            self.bindings[name] = meta
            if meta["mode"] == trt.TensorIOMode.INPUT:
                self.input_name = name

        self.input_shape = tuple(int(v) for v in self.bindings[self.input_name]["shape"])
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

    def close(self) -> None:
        self.cuda_ctx.pop()

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = letterbox(frame, (self.input_h, self.input_w))
        img = img[..., ::-1].transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32) / 255.0

    def infer(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        inp = self.bindings[self.input_name]
        np.copyto(inp["host"], batch.ravel())
        cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for name, meta in self.bindings.items():
            if meta["mode"] == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(meta["host"], meta["device"], self.stream)
        self.stream.synchronize()
        return {name: np.array(meta["host"]).reshape(meta["shape"]) for name, meta in self.bindings.items() if meta["mode"] == trt.TensorIOMode.OUTPUT}

    def postprocess(self, outputs: dict[str, np.ndarray], frame_shape: tuple[int, int], conf: float, iou: float) -> list[dict[str, object]]:
        pred = outputs["output0"][0].T  # (8400, 116)
        protos = outputs["output1"][0]  # (32, 160, 160)
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:84]
        mask_coeff = pred[:, 84:]
        cls_idx = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
        names = np.array([COCO_NAMES[int(i)] for i in cls_idx], dtype=object)
        keep = (scores > conf) & np.isin(names, list(FILTER_CLASSES))
        if not np.any(keep):
            return []

        boxes = boxes_xywh[keep].copy()
        boxes[:, 0] -= boxes[:, 2] / 2.0
        boxes[:, 1] -= boxes[:, 3] / 2.0
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        scores = scores[keep]
        cls_idx = cls_idx[keep]
        names = names[keep]
        coeffs = mask_coeff[keep]

        detections: list[dict[str, object]] = []
        for cls_name in sorted(set(names.tolist())):
            idx = np.where(names == cls_name)[0]
            kept = nms(boxes[idx], scores[idx], iou)
            for local_i in kept.tolist():
                gi = idx[local_i]
                detections.append({
                    "box": boxes[gi].copy(),
                    "score": float(scores[gi]),
                    "cls_id": int(cls_idx[gi]),
                    "name": cls_name,
                    "coeff": coeffs[gi].copy(),
                })

        if not detections:
            return []

        det_boxes = np.stack([d["box"] for d in detections], axis=0).astype(np.float32)
        det_boxes = scale_boxes((self.input_h, self.input_w), det_boxes, frame_shape)
        protos_flat = protos.reshape(protos.shape[0], -1)
        mask_coeffs = np.stack([d["coeff"] for d in detections], axis=0).astype(np.float32)
        masks = sigmoid(mask_coeffs @ protos_flat).reshape(-1, protos.shape[1], protos.shape[2])
        masks = scale_masks(masks, frame_shape)
        masks = crop_masks(masks, det_boxes)

        out = []
        for det, box, mask in zip(detections, det_boxes, masks):
            out.append({"box": box, "score": det["score"], "name": det["name"], "mask": mask})
        return out


def draw(frame: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        color = COLORS.get(det["name"], np.array([255, 255, 0], dtype=np.uint8))
        mask = det["mask"]
        if mask.any():
            out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
    for det in detections:
        color = tuple(int(v) for v in COLORS.get(det["name"], np.array([255, 255, 0], dtype=np.uint8)).tolist())
        x1, y1, x2, y2 = det["box"].astype(np.int32).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{det['name']} {det['score']:.2f}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def run(engine: Path, videos: list[Path], out_dir: Path, max_frames: int | None, conf: float, iou: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = YoloSegTrt(engine)
    summary = []
    try:
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(out_dir / f"{video.stem}_{engine.stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            frame_count = 0
            infer_total = 0.0
            det_counter = {}
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_count >= max_frames:
                    break
                batch = runner.preprocess(frame)
                t0 = time.perf_counter()
                outputs = runner.infer(batch)
                dets = runner.postprocess(outputs, frame.shape[:2], conf, iou)
                infer_total += (time.perf_counter() - t0) * 1000.0
                for det in dets:
                    det_counter[det["name"]] = det_counter.get(det["name"], 0) + 1
                writer.write(draw(frame, dets))
                frame_count += 1
            cap.release()
            writer.release()
            summary.append({"video": video.name, "frames": frame_count, "infer_ms_avg": round(infer_total / max(frame_count,1), 4), "detections": det_counter})
    finally:
        runner.close()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--videos", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()
    run(args.engine, args.videos, args.output_dir, args.max_frames, args.conf, args.iou)


if __name__ == "__main__":
    main()
