#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


PERSON_MAP = {0: 0, 2: 1, 5: 1, 7: 1}
ROBOT_MAP = {0: 2}
CLASS_NAMES = {0: "person", 1: "car", 2: "robot"}
CLASS_COLORS = {
    0: np.array([40, 220, 40], dtype=np.uint8),
    1: np.array([0, 180, 255], dtype=np.uint8),
    2: np.array([50, 80, 255], dtype=np.uint8),
}


def _load_cudart() -> ctypes.CDLL:
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("Failed to load libcudart.so")


class CudaRt:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self) -> None:
        self.lib = _load_cudart()
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int

        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaMemGetInfo.restype = ctypes.c_int

        self.check(self.lib.cudaSetDevice(0))

    def check(self, code: int) -> None:
        if code != 0:
            msg = self.lib.cudaGetErrorString(code)
            raise RuntimeError(f"CUDA runtime error {code}: {msg.decode() if msg else 'unknown'}")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes))
        return int(ptr.value)

    def free(self, ptr: int) -> None:
        if ptr:
            self.check(self.lib.cudaFree(ctypes.c_void_p(ptr)))

    def memcpy_htod_async(self, dst: int, src: np.ndarray, stream: int) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(src.ctypes.data),
                ctypes.c_size_t(src.nbytes),
                ctypes.c_int(self.cudaMemcpyHostToDevice),
                ctypes.c_void_p(stream),
            )
        )

    def memcpy_dtoh_async(self, dst: np.ndarray, src: int, stream: int) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst.ctypes.data),
                ctypes.c_void_p(src),
                ctypes.c_size_t(dst.nbytes),
                ctypes.c_int(self.cudaMemcpyDeviceToHost),
                ctypes.c_void_p(stream),
            )
        )

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)))
        return int(stream.value)

    def stream_destroy(self, stream: int) -> None:
        if stream:
            self.check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)))

    def stream_synchronize(self, stream: int) -> None:
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)))

    def mem_get_info_mb(self) -> tuple[float, float]:
        free_b = ctypes.c_size_t()
        total_b = ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b)))
        used_mb = (total_b.value - free_b.value) / (1024.0 * 1024.0)
        total_mb = total_b.value / (1024.0 * 1024.0)
        return used_mb, total_mb


class TegrastatsLogger:
    def __init__(self, log_path: Path, interval_ms: int = 500):
        self.log_path = log_path
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        subprocess.run(["tegrastats", "--stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms), "--logfile", str(self.log_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop(self) -> None:
        subprocess.run(["tegrastats", "--stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if self.proc is not None:
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def parse_tegrastats(log_path: Path) -> dict[str, float]:
    metrics = {
        "shared_ram_peak_mb": 0.0,
        "gr3d_mean_pct": 0.0,
        "gr3d_peak_pct": 0.0,
        "vic_mean_pct": 0.0,
        "vic_peak_pct": 0.0,
    }
    if not log_path.exists():
        return metrics

    gr3d_values: list[float] = []
    vic_values: list[float] = []

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if m:
            metrics["shared_ram_peak_mb"] = max(metrics["shared_ram_peak_mb"], float(m.group(1)))
        m = re.search(r"GR3D_FREQ (\d+)%", line)
        if m:
            gr3d_values.append(float(m.group(1)))
        m = re.search(r"VIC_FREQ (\d+)%", line)
        if m:
            vic_values.append(float(m.group(1)))

    if gr3d_values:
        metrics["gr3d_mean_pct"] = float(np.mean(gr3d_values))
        metrics["gr3d_peak_pct"] = float(np.max(gr3d_values))
    if vic_values:
        metrics["vic_mean_pct"] = float(np.mean(vic_values))
        metrics["vic_peak_pct"] = float(np.max(vic_values))
    return metrics


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter + 1e-6
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.asarray(keep, dtype=np.int32)


def merged_class_nms(detections: list[dict[str, object]], iou_thres: float) -> list[dict[str, object]]:
    if not detections:
        return detections

    kept: list[dict[str, object]] = []
    classes = sorted({int(det["cls"]) for det in detections})
    for cls_id in classes:
        cls_dets = [det for det in detections if int(det["cls"]) == cls_id]
        boxes = np.stack([det["box"] for det in cls_dets], axis=0).astype(np.float32)
        scores = np.asarray([float(det["score"]) for det in cls_dets], dtype=np.float32)
        keep_indices = nms(boxes, scores, iou_thres)
        kept.extend(cls_dets[int(idx)] for idx in keep_indices.tolist())

    kept.sort(key=lambda det: float(det["score"]), reverse=True)
    return kept


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
    resized = [
        cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        for mask in cropped
    ]
    return np.stack(resized, axis=0) if resized else np.empty((0, shape[0], shape[1]), dtype=np.float32)


def crop_masks(masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if masks.size == 0:
        return masks.astype(bool)
    out = np.zeros_like(masks, dtype=bool)
    h, w = masks.shape[1:]
    for i, box in enumerate(boxes.astype(np.int32)):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 > x1 and y2 > y1:
            out[i, y1:y2, x1:x2] = masks[i, y1:y2, x1:x2] > 0.0
    return out


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


class TrtSegRunner:
    def __init__(self, engine_path: Path) -> None:
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.cuda = CudaRt()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.stream = self.cuda.stream_create()
        self.bindings: dict[str, dict[str, object]] = {}
        self.input_name = ""

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT and any(dim < 0 for dim in shape):
                raise RuntimeError(f"Dynamic input shape not supported: {name} {shape}")
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host = np.empty(shape, dtype=dtype)
            device = self.cuda.malloc(host.nbytes)
            self.context.set_tensor_address(name, int(device))
            self.bindings[name] = {
                "shape": shape,
                "dtype": dtype,
                "host": host,
                "device": device,
                "mode": mode,
            }
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name

        if not self.input_name:
            raise RuntimeError("No input tensor found")

        self.input_shape = tuple(int(x) for x in self.bindings[self.input_name]["shape"])
        self.input_h = int(self.input_shape[2])
        self.input_w = int(self.input_shape[3])
        self.model_h = int(math.ceil(self.input_h / 32.0) * 32)
        self.model_w = int(math.ceil(self.input_w / 32.0) * 32)
        self.anchor_points, self.stride_tensor = self.make_anchor_points((self.model_h, self.model_w))

    def close(self) -> None:
        for meta in self.bindings.values():
            self.cuda.free(int(meta["device"]))
        self.bindings.clear()
        self.cuda.stream_destroy(self.stream)

    @staticmethod
    def make_anchor_points(model_imgsz: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = model_imgsz
        shapes = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
        strides = [8, 16, 32]
        anchor_points = []
        stride_tensor = []
        for (fh, fw), stride in zip(shapes, strides):
            sy = np.arange(fh, dtype=np.float32) + 0.5
            sx = np.arange(fw, dtype=np.float32) + 0.5
            yy, xx = np.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
            stride_tensor.append(np.full((fh * fw, 1), stride, dtype=np.float32))
        return np.concatenate(anchor_points, axis=0), np.concatenate(stride_tensor, axis=0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = letterbox(frame, (self.input_h, self.input_w))
        img = img[..., ::-1].transpose(2, 0, 1)[None]
        return np.ascontiguousarray(img, dtype=np.float32) / 255.0

    def infer(self, batch: np.ndarray) -> dict[str, np.ndarray]:
        binding = self.bindings[self.input_name]
        host_input = binding["host"]
        np.copyto(host_input, batch)
        self.cuda.memcpy_htod_async(int(binding["device"]), host_input, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream)

        for name, meta in self.bindings.items():
            if meta["mode"] == trt.TensorIOMode.OUTPUT:
                self.cuda.memcpy_dtoh_async(meta["host"], int(meta["device"]), self.stream)
        self.cuda.stream_synchronize(self.stream)

        outputs: dict[str, np.ndarray] = {}
        for name, meta in self.bindings.items():
            if meta["mode"] == trt.TensorIOMode.OUTPUT:
                outputs[name] = np.array(meta["host"], copy=True)
        return outputs

    def decode_dfl_boxes(self, box_dist: np.ndarray) -> np.ndarray:
        reg_max = 16
        n = box_dist.shape[1]
        dist = box_dist.reshape(4, reg_max, n)
        dist = softmax(dist, axis=1)
        proj = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
        dist = (dist * proj).sum(axis=1)
        lt = dist[:2].T
        rb = dist[2:].T
        xy1 = self.anchor_points - lt
        xy2 = self.anchor_points + rb
        c_xy = (xy1 + xy2) / 2.0
        wh = xy2 - xy1
        return np.concatenate((c_xy, wh), axis=1) * self.stride_tensor

    def postprocess_branch(
        self,
        frame_shape: tuple[int, int],
        box_dist: np.ndarray,
        cls_logits: np.ndarray,
        mask_coeff: np.ndarray,
        protos: np.ndarray,
        valid_map: dict[int, int],
        conf_thres: float,
        iou_thres: float,
    ) -> list[dict[str, object]]:
        boxes_xywh = self.decode_dfl_boxes(box_dist)
        cls_scores = sigmoid(cls_logits)
        protos_flat = protos.reshape(protos.shape[0], -1)
        detections: list[dict[str, object]] = []

        for src_cls, dst_cls in valid_map.items():
            scores = cls_scores[src_cls]
            keep = scores > conf_thres
            if not np.any(keep):
                continue
            boxes_xyxy = xywh_to_xyxy(boxes_xywh[keep])
            scores_kept = scores[keep]
            coeff_kept = mask_coeff[:, keep].T
            keep_indices = nms(boxes_xyxy, scores_kept, iou_thres)
            if keep_indices.size == 0:
                continue

            boxes_scaled = scale_boxes((self.model_h, self.model_w), boxes_xyxy[keep_indices], frame_shape)
            coeffs = coeff_kept[keep_indices]
            masks = (coeffs @ protos_flat).reshape(-1, protos.shape[1], protos.shape[2])
            masks = scale_masks(masks, frame_shape)
            masks = crop_masks(masks, boxes_scaled)

            for box, score, mask in zip(boxes_scaled, scores_kept[keep_indices], masks):
                detections.append(
                    {
                        "box": box.astype(np.float32),
                        "score": float(score),
                        "cls": int(dst_cls),
                        "mask": mask,
                    }
                )
        return detections

    def postprocess(self, outputs: dict[str, np.ndarray], frame_shape: tuple[int, int]) -> list[dict[str, object]]:
        detections = (
            self.postprocess_branch(
                frame_shape,
                outputs["boxes1"][0],
                outputs["scores1"][0],
                outputs["mask1"][0],
                outputs["proto1"][0],
                PERSON_MAP,
                conf_thres=0.01,
                iou_thres=0.45,
            )
            + self.postprocess_branch(
                frame_shape,
                outputs["boxes2"][0],
                outputs["scores2"][0],
                outputs["mask2"][0],
                outputs["proto2"][0],
                ROBOT_MAP,
                conf_thres=0.01,
                iou_thres=0.45,
            )
        )
        return merged_class_nms(detections, iou_thres=0.45)


def draw_detections(frame: np.ndarray, detections: list[dict[str, object]]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        cls_id = int(det["cls"])
        color = CLASS_COLORS[cls_id]
        mask = det["mask"]
        if mask.any():
            out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
    for det in detections:
        cls_id = int(det["cls"])
        color = tuple(int(v) for v in CLASS_COLORS[cls_id].tolist())
        x1, y1, x2, y2 = det["box"].astype(np.int32).tolist()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {det['score']:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


@dataclass
class RunMetric:
    source: str
    engine: str
    frames: int
    source_fps: float
    fps_total: float
    fps_infer: float
    preprocess_ms_avg: float
    infer_ms_avg: float
    postprocess_ms_avg: float
    render_ms_avg: float
    total_ms_avg: float
    cuda_mem_used_mb_avg: float
    cuda_mem_used_mb_peak: float
    shared_ram_peak_mb: float
    gr3d_mean_pct: float
    gr3d_peak_pct: float
    vic_mean_pct: float
    vic_peak_pct: float
    output_video: str


def open_source(source: str, width: int, height: int, fps: int) -> tuple[cv2.VideoCapture, float]:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
    elif source.startswith("/dev/video"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    if source.isdigit() or source.startswith("/dev/video"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

    source_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    return cap, float(source_fps)


def run_source(
    engine_path: Path,
    source: str,
    output_video: Path,
    max_frames: int | None,
    camera_width: int,
    camera_height: int,
    camera_fps: int,
) -> RunMetric:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tegrastats_log = output_video.with_suffix(".tegrastats.log")
    cap, source_fps = open_source(source, camera_width, camera_height, camera_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps if source_fps > 0 else float(camera_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video}")

    runner = TrtSegRunner(engine_path)
    tegrastats = TegrastatsLogger(tegrastats_log)

    preprocess_ms = 0.0
    infer_ms = 0.0
    post_ms = 0.0
    render_ms = 0.0
    frame_count = 0
    cuda_mem_samples: list[float] = []
    stop = False

    def on_sigint(signum, frame):
        nonlocal stop
        stop = True

    old_sigint = signal.signal(signal.SIGINT, on_sigint)
    wall_start = time.perf_counter()
    tegrastats.start()
    try:
        while not stop:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_count >= max_frames:
                break

            t0 = time.perf_counter()
            batch = runner.preprocess(frame)
            t1 = time.perf_counter()
            outputs = runner.infer(batch)
            t2 = time.perf_counter()
            detections = runner.postprocess(outputs, frame.shape[:2])
            t3 = time.perf_counter()
            rendered = draw_detections(frame, detections)
            writer.write(rendered)
            t4 = time.perf_counter()

            preprocess_ms += (t1 - t0) * 1000.0
            infer_ms += (t2 - t1) * 1000.0
            post_ms += (t3 - t2) * 1000.0
            render_ms += (t4 - t3) * 1000.0
            cuda_used_mb, _ = runner.cuda.mem_get_info_mb()
            cuda_mem_samples.append(cuda_used_mb)
            frame_count += 1
    finally:
        tegrastats.stop()
        signal.signal(signal.SIGINT, old_sigint)
        cap.release()
        writer.release()
        runner.close()

    wall_total = max(time.perf_counter() - wall_start, 1e-6)
    if frame_count == 0:
        raise RuntimeError("No frames were processed")

    total_ms = preprocess_ms + infer_ms + post_ms + render_ms
    tegra = parse_tegrastats(tegrastats_log)
    metric = RunMetric(
        source=source,
        engine=engine_path.name,
        frames=frame_count,
        source_fps=round(source_fps, 4),
        fps_total=round(frame_count / wall_total, 4),
        fps_infer=round(frame_count / max(infer_ms / 1000.0, 1e-6), 4),
        preprocess_ms_avg=round(preprocess_ms / frame_count, 4),
        infer_ms_avg=round(infer_ms / frame_count, 4),
        postprocess_ms_avg=round(post_ms / frame_count, 4),
        render_ms_avg=round(render_ms / frame_count, 4),
        total_ms_avg=round(total_ms / frame_count, 4),
        cuda_mem_used_mb_avg=round(float(np.mean(cuda_mem_samples)), 4),
        cuda_mem_used_mb_peak=round(float(np.max(cuda_mem_samples)), 4),
        shared_ram_peak_mb=round(tegra["shared_ram_peak_mb"], 4),
        gr3d_mean_pct=round(tegra["gr3d_mean_pct"], 4),
        gr3d_peak_pct=round(tegra["gr3d_peak_pct"], 4),
        vic_mean_pct=round(tegra["vic_mean_pct"], 4),
        vic_peak_pct=round(tegra["vic_peak_pct"], 4),
        output_video=str(output_video),
    )
    return metric


def write_summary(metric: RunMetric, output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": label, "result": asdict(metric)}
    (output_dir / f"{label}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / f"{label}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metric).keys()))
        writer.writeheader()
        writer.writerow(asdict(metric))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV + TensorRT benchmark for fusion_branch10_960x544_b1_fp16.engine"
    )
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Video file path, /dev/videoX, or camera index")
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    args = parser.parse_args()

    metric = run_source(
        engine_path=args.engine,
        source=str(args.source),
        output_video=args.output_video,
        max_frames=args.max_frames,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
    )
    write_summary(metric, args.output_dir, args.label)
    print(json.dumps({"label": args.label, "result": asdict(metric)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
