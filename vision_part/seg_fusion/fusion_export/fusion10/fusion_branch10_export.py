from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


ROOT = Path(__file__).resolve().parent.parent
MODEL_YAML = Path(__file__).resolve().with_name("fusion_branch10.yaml")
IMAGE_DIR = ROOT / "yolo/data_4/images/val"
OUTPUT_DIR = ROOT / "vis/fusion10"
SUMMARY_PATH = OUTPUT_DIR / "fusion_branch10_summary.json"
ONNX_PATH = ROOT / "yolo/fusion_branch10_960x544.onnx"
IMG_SIZE = (640, 640)
ONNX_INPUT_SIZE = (544, 960)
NAMES = {0: "person", 1: "car", 2: "robot"}
PERSON_MAP = {0: 0, 2: 1, 5: 1, 7: 1}
ROBOT_MAP = {0: 2}


def build_model():
    model = YOLO(str(MODEL_YAML))
    model.model.cpu()

    person_model = torch.load(ROOT / "yolo/yolo11s-seg.pt", map_location="cpu")["model"].state_dict()
    robot_model = torch.load(ROOT / "yolo/runs/segment/train12/weights/best.pt", map_location="cpu")["model"].state_dict()

    person_dict = {"model." + k: v for k, v in person_model.items()}
    missing_keys, unexpected_keys = model.load_state_dict(person_dict, strict=False)
    print("person missing:", len(missing_keys), "unexpected:", len(unexpected_keys))

    robot_dict = {}
    for k, v in robot_model.items():
        parts = k.split(".")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        layer = int(parts[1])
        if layer < 10:
            continue
        parts[1] = str(layer + 14)  # 10->24, ..., 23->37
        robot_dict["model." + ".".join(parts)] = v

    missing_keys, unexpected_keys = model.load_state_dict(robot_dict, strict=False)
    print("robot missing:", len(missing_keys), "unexpected:", len(unexpected_keys))
    return model


def letterbox(img, new_shape=IMG_SIZE):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def preprocess(img_bgr):
    prep = letterbox(img_bgr, IMG_SIZE)
    rgb = prep[..., ::-1].transpose(2, 0, 1)
    x = torch.from_numpy(rgb.copy()).float().unsqueeze(0) / 255.0
    return x, prep


def make_anchor_points(imgsz, device, dtype):
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


def decode_dfl_boxes(box_dist: torch.Tensor, imgsz=IMG_SIZE):
    reg_max = 16
    box_dist = box_dist.view(1, 4, reg_max, -1).softmax(2)
    proj = torch.arange(reg_max, dtype=box_dist.dtype, device=box_dist.device).view(1, 1, reg_max, 1)
    dist = (box_dist * proj).sum(2)

    anchor_points, stride_tensor = make_anchor_points(imgsz, device=dist.device, dtype=dist.dtype)
    anchor_points = anchor_points.t().unsqueeze(0)
    stride_tensor = stride_tensor.t().unsqueeze(0)

    lt, rb = dist[:, :2], dist[:, 2:]
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat((c_xy, wh), dim=1) * stride_tensor


def process_mask(protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, shape):
    c, mh, mw = protos.shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = ops.scale_masks(masks[None], shape)[0]
    masks = ops.crop_mask(masks, bboxes)
    return masks.gt_(0.0)


def extract_details(head_output):
    if isinstance(head_output, tuple) and len(head_output) > 1 and isinstance(head_output[1], dict):
        return head_output[1]
    if isinstance(head_output, dict):
        return head_output
    raise TypeError(f"Unexpected head output type: {type(head_output)}")


def postprocess_head(details, img_bgr, prep_img, valid_dict, num_classes, conf_thres=0.6, iou_thres=0.7):
    box_dist = details["boxes"].float()
    cls_logits = details["scores"].float()
    mask_coeff = details["mask_coefficient"].float()
    protos = details["proto"][0].float()

    decoded_boxes = decode_dfl_boxes(box_dist)
    cls_scores = cls_logits.sigmoid()
    preds = torch.cat((decoded_boxes, cls_scores, mask_coeff), dim=1)

    preds = nms.non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=num_classes,
        classes=list(valid_dict.keys()),
    )[0]

    if preds.shape[0] == 0:
        return None, None

    cls = preds[:, 5].to(torch.int64)
    mapped_cls = torch.tensor([valid_dict[int(c.item())] for c in cls], dtype=preds.dtype).unsqueeze(1)
    preds[:, :4] = ops.scale_boxes(prep_img.shape[:2], preds[:, :4], img_bgr.shape[:2])
    masks = process_mask(protos, preds[:, 6:], preds[:, :4], img_bgr.shape[:2])
    boxes = torch.cat((preds[:, :4], preds[:, 4:5], mapped_cls), dim=1)
    return boxes, masks


class HookedFusionModel:
    def __init__(self, model, head_specs):
        self.model = model
        self.outputs = {}
        self.handles = []
        for name, idx in head_specs.items():
            handle = self.model.model[idx].register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.outputs[name] = output

        return hook

    def __call__(self, x):
        self.outputs.clear()
        _ = self.model(x)
        return {name: extract_details(output) for name, output in self.outputs.items()}


class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.head_outputs = {}
        self.head1_idx = 23
        self.head2_idx = 37
        self._hook1 = self.model.model[self.head1_idx].register_forward_hook(self._make_hook("head1"))
        self._hook2 = self.model.model[self.head2_idx].register_forward_hook(self._make_hook("head2"))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.head_outputs[name] = output

        return hook

    def _print_head_debug(self, name, head):
        details = extract_details(head)
        boxes = details["boxes"]
        scores = details["scores"]
        mask_coefficient = details["mask_coefficient"]
        proto_detail = details["proto"]

        print(f"\n===== {name} =====")
        print("boxes shape:", tuple(boxes.shape))
        print("scores shape:", tuple(scores.shape))
        print("mask_coefficient shape:", tuple(mask_coefficient.shape))
        print("proto(detail) shape:", tuple(proto_detail.shape))

        print("boxes first 3 anchors:")
        print(boxes[0, :, :3])
        print("scores first 3 anchors:")
        print(scores[0, :, :3])
        print("mask_coefficient first 3 anchors:")
        print(mask_coefficient[0, :, :3])

    def forward(self, x):
        self.head_outputs.clear()
        # Keep exported ONNX input at 960x540 and pad internally to stride-32 alignment.
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        _ = self.model(x)

        head1 = self.head_outputs["head1"]
        head2 = self.head_outputs["head2"]

        self._print_head_debug("head1", head1)
        self._print_head_debug("head2", head2)

        details1 = extract_details(head1)
        details2 = extract_details(head2)

        return (
            details1["boxes"],
            details1["scores"],
            details1["mask_coefficient"],
            details1["proto"],
            details2["boxes"],
            details2["scores"],
            details2["mask_coefficient"],
            details2["proto"],
        )


def summarize_detections(tag, boxes):
    print(f"\n===== {tag} =====")
    if boxes is None or boxes.shape[0] == 0:
        print("no detections")
        return

    order = torch.argsort(boxes[:, 4], descending=True)
    boxes = boxes[order]
    print("num detections:", int(boxes.shape[0]))
    for i, det in enumerate(boxes[:10]):
        x1, y1, x2, y2, conf, cls = det.tolist()
        print(
            {
                "rank": i + 1,
                "class": NAMES[int(cls)],
                "conf": round(conf, 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )


def detection_records(boxes):
    if boxes is None or boxes.shape[0] == 0:
        return []

    order = torch.argsort(boxes[:, 4], descending=True)
    boxes = boxes[order]
    records = []
    for det in boxes:
        x1, y1, x2, y2, conf, cls = det.tolist()
        records.append(
            {
                "class_id": int(cls),
                "class_name": NAMES[int(cls)],
                "conf": round(conf, 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return records


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:4], boxes2[None, :, 2:4])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def suppress_person_overlapping_robot(
    boxes_list,
    masks_list,
    robot_cls: int = 2,
    person_cls: int = 0,
    mask_cover_thres: float = 0.5,
):
    if not boxes_list:
        return boxes_list, masks_list

    merged_boxes = torch.cat(boxes_list, dim=0)
    merged_masks = torch.cat(masks_list, dim=0)
    cls_ids = merged_boxes[:, 5].to(torch.int64)

    robot_idx = torch.where(cls_ids == robot_cls)[0]
    person_idx = torch.where(cls_ids == person_cls)[0]
    if robot_idx.numel() == 0 or person_idx.numel() == 0:
        return boxes_list, masks_list

    person_boxes = merged_boxes[person_idx, :4]
    robot_boxes = merged_boxes[robot_idx, :4]
    person_centers = (person_boxes[:, :2] + person_boxes[:, 2:4]) / 2
    person_masks = merged_masks[person_idx].to(torch.bool)
    robot_masks = merged_masks[robot_idx].to(torch.bool)

    inside_x = (person_centers[:, None, 0] >= robot_boxes[None, :, 0]) & (
        person_centers[:, None, 0] <= robot_boxes[None, :, 2]
    )
    inside_y = (person_centers[:, None, 1] >= robot_boxes[None, :, 1]) & (
        person_centers[:, None, 1] <= robot_boxes[None, :, 3]
    )
    center_hit = (inside_x & inside_y).any(dim=1)

    person_area = person_masks.flatten(1).sum(dim=1).clamp(min=1)
    inter = (person_masks[:, None] & robot_masks[None, :]).flatten(2).sum(dim=2)
    cover_ratio = inter / person_area[:, None]
    mask_hit = (cover_ratio >= mask_cover_thres).any(dim=1)

    suppress_person = center_hit | mask_hit

    if not suppress_person.any():
        return boxes_list, masks_list

    keep = torch.ones(merged_boxes.shape[0], dtype=torch.bool)
    keep[person_idx[suppress_person]] = False
    return [merged_boxes[keep]], [merged_masks[keep]]


def build_results(img_bgr, boxes_list, masks_list):
    if not boxes_list:
        return Results(img_bgr, path="", names=NAMES)
    merged_boxes = torch.cat(boxes_list, dim=0)
    merged_masks = torch.cat(masks_list, dim=0)
    return Results(img_bgr, path="", names=NAMES, boxes=merged_boxes, masks=merged_masks)


def export_onnx(model, sample_image_path):
    img_bgr = cv2.imread(str(sample_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {sample_image_path}")

    resized = cv2.resize(img_bgr, (ONNX_INPUT_SIZE[1], ONNX_INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    rgb = resized[..., ::-1].transpose(2, 0, 1)
    x = torch.from_numpy(rgb.copy()).float().unsqueeze(0) / 255.0
    print("onnx sample image:", sample_image_path)
    print("input tensor shape:", tuple(x.shape))

    m = ExportModel(model.model).eval()
    with torch.no_grad():
        y = m(x)
        for i, item in enumerate(y):
            print(f"output[{i}] shape:", tuple(item.shape))

    torch.onnx.export(
        m,
        x,
        str(ONNX_PATH),
        input_names=["images"],
        output_names=[
            "boxes1",
            "scores1",
            "mask1",
            "proto1",
            "boxes2",
            "scores2",
            "mask2",
            "proto2",
        ],
        opset_version=12,
        do_constant_folding=True,
    )
    print("Exported ONNX to", ONNX_PATH)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))[:10]
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in: {IMAGE_DIR}")

    model = build_model()
    export_onnx(model, image_paths[0])
    hooked = HookedFusionModel(model.model, {"person80_head": 23, "robot3_head": 37})

    all_summaries = []
    for image_path in image_paths:
        print(f"\n\n######## {image_path.name} ########")
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            print("skip unreadable image:", image_path)
            continue

        prep_tensor, prep_img = preprocess(img_bgr)
        with torch.no_grad():
            heads = hooked(prep_tensor)

        boxes_list = []
        masks_list = []
        image_summary = {}
        for head_name, cfg in (
            ("person80_head", {"valid_dict": PERSON_MAP, "num_classes": 80}),
            ("robot3_head", {"valid_dict": ROBOT_MAP, "num_classes": 3}),
        ):
            boxes, masks = postprocess_head(
                heads[head_name],
                img_bgr,
                prep_img,
                valid_dict=cfg["valid_dict"],
                num_classes=cfg["num_classes"],
            )
            summarize_detections(f"{image_path.name} / {head_name}", boxes)
            image_summary[head_name] = detection_records(boxes)
            if boxes is not None:
                boxes_list.append(boxes)
                masks_list.append(masks)

        boxes_list, masks_list = suppress_person_overlapping_robot(boxes_list, masks_list)
        result = build_results(img_bgr, boxes_list, masks_list)
        output_path = OUTPUT_DIR / f"{image_path.stem}_fusion10.jpg"
        cv2.imwrite(str(output_path), result.plot())
        print("saved:", output_path)

        all_summaries.append({"image": image_path.name, "fusion10": image_summary})

    SUMMARY_PATH.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print("saved summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
