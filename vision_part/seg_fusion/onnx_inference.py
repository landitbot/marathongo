# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path

from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML, nms, ops
from ultralytics.utils.checks import check_yaml


class YOLOv8Seg:
    """YOLOv8 segmentation model for performing instance segmentation using ONNX Runtime.

    This class implements a YOLOv8 instance segmentation model using ONNX Runtime for inference. It handles
    preprocessing of input images, running inference with the ONNX model, and postprocessing the results to generate
    bounding boxes and segmentation masks.

    Attributes:
        session (ort.InferenceSession): ONNX Runtime inference session for model execution.
        imgsz (tuple[int, int]): Input image size as (height, width) for the model.
        classes (dict): Dictionary mapping class indices to class names from the dataset.
        conf (float): Confidence threshold for filtering detections.
        iou (float): IoU threshold used by non-maximum suppression.

    Methods:
        letterbox: Resize and pad image while maintaining aspect ratio.
        preprocess: Preprocess the input image before feeding it into the model.
        postprocess: Post-process model predictions to extract meaningful results.
        process_mask: Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

    Examples:
        >>> model = YOLOv8Seg("yolov8n-seg.onnx", conf=0.25, iou=0.7)
        >>> img = cv2.imread("image.jpg")
        >>> results = model(img)
        >>> cv2.imshow("Segmentation", results[0].plot())
    """

    def __init__(
        self,
        onnx_model: str,
        imgsz: int | tuple[int, int] = 640,
    ):
        """Initialize the instance segmentation model using an ONNX model.

        Args:
            onnx_model (str): Path to the ONNX model file.
            conf (float, optional): Confidence threshold for filtering detections.
            iou (float, optional): IoU threshold for non-maximum suppression.
            imgsz (int | tuple[int, int], optional): Input image size of the model. Can be an integer for square input
                or a tuple for rectangular input.
        """
        available = ort.get_available_providers()
        providers = [p for p in ("CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(
            onnx_model, providers=providers or available
        )

        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        # Keep external test input at imgsz, but align decode/scaling with model stride-padding.
        self.model_imgsz = (
            ((self.imgsz[0] + 31) // 32) * 32,
            ((self.imgsz[1] + 31) // 32) * 32,
        )
        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]
        self.debug = True
        self.debug_dir = Path("/path/to/debug_vis")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, img: np.ndarray) -> list[Results]:
        """Run inference on the input image using the ONNX model.

        Args:
            img (np.ndarray): The original input image in BGR format.

        Returns:
            (list[Results]): Processed detection results after post-processing, containing bounding boxes and
                segmentation masks.
        """
        prep_img = self.preprocess(img, self.imgsz)
        outs = self.session.run(None, {self.session.get_inputs()[0].name: prep_img})
        return self.postprocess(img, prep_img, outs)

    def letterbox(
        self, img: np.ndarray, new_shape: tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """Resize and pad image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image in BGR format.
            new_shape (tuple[int, int], optional): Target shape as (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (
            new_shape[0] - new_unpad[1]
        ) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img

    def preprocess(self, img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
        """Preprocess the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (tuple[int, int]): The target shape for resizing as (height, width).

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized
                to [0, 1].
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255  # Normalize to [0, 1]
        return img

    def postprocess(self, img: np.ndarray, prep_img: np.ndarray, outs: list):
        valid_dict_person = {0: 0, 2: 1, 5: 1, 7: 1}
        boxes = outs[0][0]
        scores = outs[1][0]
        masks = outs[2][0]
        protos = outs[3][0]
        boxes_list, masks_list = self.postprocess_(
            img,
            prep_img,
            (boxes, scores, masks, protos),
            valid_dict_person,
            num_classes=80,
        )
        boxes_list = boxes_list or []
        masks_list = masks_list or []

        valid_dict_robot = {0: 2}
        boxes = outs[4][0]
        scores = outs[5][0]
        masks = outs[6][0]
        protos = outs[7][0]
        boxes2, masks2 = self.postprocess_(
            img,
            prep_img,
            (boxes, scores, masks, protos),
            valid_dict_robot,
            num_classes=3,
        )
        if boxes2 is not None:
            boxes_list.extend(boxes2)
            masks_list.extend(masks2)

        names = {0: "person", 1: "car", 2: "robot"}
        if boxes_list:
            merged_boxes = torch.cat(boxes_list, dim=0)
            merged_masks = torch.cat(masks_list, dim=0)
            merged_boxes, merged_masks = self.suppress_person_overlapping_robot(
                merged_boxes, merged_masks
            )
            return [Results(img, path="", names=names, boxes=merged_boxes, masks=merged_masks)]
        return [Results(img, path="", names=names)]

    def postprocess_(
        self,
        img: np.ndarray,
        prep_img: np.ndarray,
        outs: list,
        valid_dict: dict[int, str],
        num_classes: int,
    ):
        """Post-process model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (list): Model outputs containing predictions and prototype masks.

        Returns:
            (list[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        box_dist = torch.from_numpy(outs[0][None]).float()
        cls_logits = torch.from_numpy(outs[1][None]).float()
        mask_coeff = torch.from_numpy(outs[2][None]).float()
        protos = torch.from_numpy(outs[3]).float()

        decoded_boxes = self.decode_dfl_boxes(box_dist)
        cls_scores = cls_logits.sigmoid()
        preds = torch.cat((decoded_boxes, cls_scores, mask_coeff), dim=1)

        preds = nms.non_max_suppression(
            preds,
            conf_thres=0.4,
            iou_thres=0.7,
            nc=num_classes,
            classes=list(valid_dict.keys()),
        )[0]

        if preds.shape[0] > 0:
            if self.debug:
                print("\n[DEBUG] box_dist shape:", tuple(outs[0].shape))
                print("[DEBUG] cls_logits shape:", tuple(outs[1].shape))
                print("[DEBUG] mask_coeff shape:", tuple(outs[2].shape))
                print("[DEBUG] proto shape:", tuple(outs[3].shape))
                print("[DEBUG] detections after NMS:", preds.shape[0])
                print("[DEBUG] first 5 boxes before scale:")
                print(preds[:5, :6].cpu())

            cls = preds[:, 5].to(torch.int64)
            mapped_cls = torch.tensor(
                [valid_dict[int(c.item())] for c in cls], dtype=preds.dtype
            ).unsqueeze(1)

            raw_boxes = preds[:, :4].clone()
            preds[:, :4] = ops.scale_boxes(self.model_imgsz, preds[:, :4], img.shape)
            if self.debug:
                print("[DEBUG] first 5 boxes after scale:")
                print(preds[:5, :6].cpu())

            masks = self.process_mask(protos, preds[:, 6:], preds[:, :4], img.shape[:2])
            if self.debug:
                unique_cls, counts = torch.unique(mapped_cls.squeeze(1).to(torch.int64), return_counts=True)
                print("[DEBUG] mapped classes and counts:", list(zip(unique_cls.tolist(), counts.tolist())))
                self.save_debug_overlay(img, raw_boxes, preds[:, :4], mapped_cls.squeeze(1), preds[:, 4])

            merged_boxes = torch.cat((preds[:, :4], preds[:, 4:5], mapped_cls), dim=1)
            return [merged_boxes], [masks]
        return None, None

    def decode_dfl_boxes(self, box_dist: torch.Tensor) -> torch.Tensor:
        reg_max = 16
        box_dist = box_dist.view(1, 4, reg_max, -1).softmax(2)
        proj = torch.arange(reg_max, dtype=box_dist.dtype, device=box_dist.device).view(
            1, 1, reg_max, 1
        )
        dist = (box_dist * proj).sum(2)  # (1, 4, 8400)

        anchor_points, stride_tensor = self.make_anchor_points(
            self.model_imgsz, device=dist.device, dtype=dist.dtype
        )
        anchor_points = anchor_points.t().unsqueeze(0)  # (1, 2, 8400)
        stride_tensor = stride_tensor.t().unsqueeze(0)  # (1, 1, 8400)

        lt, rb = dist[:, :2], dist[:, 2:]
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim=1) * stride_tensor

    def make_anchor_points(self, imgsz, device, dtype):
        h, w = imgsz
        shapes = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
        strides = [8, 16, 32]
        anchor_points = []
        stride_tensor = []

        for (fh, fw), stride in zip(shapes, strides):
            sy = torch.arange(fh, device=device, dtype=dtype) + 0.5
            sx = torch.arange(fw, device=device, dtype=dtype) + 0.5
            yy, xx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((xx, yy), -1).view(-1, 2))
            stride_tensor.append(torch.full((fh * fw, 1), stride, device=device, dtype=dtype))

        return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)

    def box_iou_xyxy(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        if boxes1.numel() == 0 or boxes2.numel() == 0:
            return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype)

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:4], boxes2[None, :, 2:4])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
            boxes1[:, 3] - boxes1[:, 1]
        ).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
            boxes2[:, 3] - boxes2[:, 1]
        ).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-6)

    def suppress_person_overlapping_robot(
        self,
        boxes: torch.Tensor,
        masks: torch.Tensor,
        iou_thres: float = 0.7,
        person_cls: int = 0,
        robot_cls: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_ids = boxes[:, 5].to(torch.int64)
        person_idx = torch.where(cls_ids == person_cls)[0]
        robot_idx = torch.where(cls_ids == robot_cls)[0]
        if person_idx.numel() == 0 or robot_idx.numel() == 0:
            return boxes, masks

        ious = self.box_iou_xyxy(boxes[person_idx, :4], boxes[robot_idx, :4])
        suppress_person = (ious >= iou_thres).any(dim=1)
        if not suppress_person.any():
            return boxes, masks

        keep = torch.ones(boxes.shape[0], dtype=torch.bool)
        keep[person_idx[suppress_person]] = False
        return boxes[keep], masks[keep]

    def save_debug_overlay(self, img, raw_boxes, scaled_boxes, cls_ids, confs):
        raw_canvas = img.copy()
        scaled_canvas = img.copy()
        names = {0: "person", 1: "car", 2: "robot"}

        limit = min(10, scaled_boxes.shape[0])
        h, w = img.shape[:2]
        for i in range(limit):
            rx1, ry1, rx2, ry2 = raw_boxes[i].tolist()
            sx1, sy1, sx2, sy2 = scaled_boxes[i].tolist()
            label = f"{names[int(cls_ids[i].item())]}:{float(confs[i].item()):.2f}"

            # Draw raw boxes directly on original image to see if they collapse to the corner.
            p1 = (int(max(0, min(w - 1, rx1))), int(max(0, min(h - 1, ry1))))
            p2 = (int(max(0, min(w - 1, rx2))), int(max(0, min(h - 1, ry2))))
            cv2.rectangle(raw_canvas, p1, p2, (0, 255, 255), 2)
            cv2.putText(raw_canvas, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            p1 = (int(max(0, min(w - 1, sx1))), int(max(0, min(h - 1, sy1))))
            p2 = (int(max(0, min(w - 1, sx2))), int(max(0, min(h - 1, sy2))))
            cv2.rectangle(scaled_canvas, p1, p2, (0, 255, 0), 2)
            cv2.putText(scaled_canvas, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(str(self.debug_dir / "raw_boxes.jpg"), raw_canvas)
        cv2.imwrite(str(self.debug_dir / "scaled_boxes.jpg"), scaled_canvas)

    def process_mask(
        self,
        protos: torch.Tensor,
        masks_in: torch.Tensor,
        bboxes: torch.Tensor,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        """Process prototype masks with predicted mask coefficients to generate instance segmentation masks.

        Args:
            protos (torch.Tensor): Prototype masks with shape (mask_dim, mask_h, mask_w).
            masks_in (torch.Tensor): Predicted mask coefficients with shape (N, mask_dim), where N is number of
                detections.
            bboxes (torch.Tensor): Bounding boxes with shape (N, 4), where N is number of detections.
            shape (tuple[int, int]): The size of the input image as (height, width).

        Returns:
            (torch.Tensor): Binary segmentation masks with shape (N, height, width).
        """
        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).view(
            -1, mh, mw
        )  # Matrix multiplication
        masks = ops.scale_masks(masks[None], shape)[
            0
        ]  # Scale masks to original image size
        masks = ops.crop_mask(masks, bboxes)  # Crop masks to bounding boxes
        return masks.gt_(0.0)  # Convert to binary masks


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 ONNX segmentation on images or video.")
    parser.add_argument(
        "--model",
        default="/path/to/fusion_robotbase.onnx",
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--source",
        default="data_4/images/val",
        help="Image directory, image file, or video file.",
    )
    parser.add_argument(
        "--output",
        default="/path/to/vis/onnx_inference4",
        help="Output image directory or output video path.",
    )
    parser.add_argument("--imgsz", type=int, nargs=2, default=(720, 1280), metavar=("H", "W"))
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to process when source is a directory.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug prints and overlays.",
    )
    return parser.parse_args()


def infer_images(model: YOLOv8Seg, source_path: Path, output_path: Path, max_images: int):
    output_path.mkdir(parents=True, exist_ok=True)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if source_path.is_dir():
        sample_images = sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in image_suffixes
        )[:max_images]
    elif source_path.suffix.lower() in image_suffixes:
        sample_images = [source_path]
    else:
        raise ValueError(f"Unsupported image source: {source_path}")

    if not sample_images:
        raise FileNotFoundError(f"No image files found in: {source_path}")

    for source_img in sample_images:
        print(f"\n===== inference: {source_img.name} =====")
        img = cv2.imread(str(source_img))
        if img is None:
            print(f"skip unreadable image: {source_img}")
            continue

        results = model(img)
        result_img = results[0].plot()
        output_file = output_path / f"{source_img.stem}_segmented.jpg"
        cv2.imwrite(str(output_file), result_img)
        print(f"saved: {output_file}")


def infer_video(model: YOLOv8Seg, source_path: Path, output_path: Path):
    if output_path.exists() and output_path.is_dir():
        output_path = output_path / f"{source_path.stem}_segmented.mp4"
    elif output_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        output_path = output_path / f"{source_path.stem}_segmented.mp4" if not output_path.suffix else output_path.with_suffix(".mp4")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model(frame)
            writer.write(results[0].plot())
            frame_idx += 1

            if frame_idx % 10 == 0:
                total = frame_count if frame_count > 0 else "?"
                print(f"processed {frame_idx}/{total} frames")
    finally:
        cap.release()
        writer.release()

    print(f"saved video: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    model = YOLOv8Seg(args.model, imgsz=tuple(args.imgsz))
    model.debug = not args.no_debug

    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if source_path.is_file() and source_path.suffix.lower() in video_suffixes:
        infer_video(model, source_path, output_path)
    else:
        infer_images(model, source_path, output_path, args.max_images)
