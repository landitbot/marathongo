import os
import sys
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder
import ultralytics.utils
import ultralytics.models.yolo
import ultralytics.utils.tal as _m

sys.modules["ultralytics.yolo"] = ultralytics.models.yolo
sys.modules["ultralytics.yolo.utils"] = ultralytics.utils


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat([x1y1, x2y2], dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class RoiAlign(torch.autograd.Function):
    @staticmethod
    def forward(
        self,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode,
        mode,
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale
    ):
        C = X.shape[1]
        num_rois = rois.shape[0]
        return torch.randn([num_rois, C, output_height, output_width], device=rois.device, dtype=rois.dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode,
        mode,
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale
    ):
        return g.op(
            "TRT::ROIAlignX_TRT",
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_i=coordinate_transformation_mode,
            mode_i=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale
        )


class NMS(torch.autograd.Function):
    @staticmethod
    def forward(self, boxes, scores, score_threshold, iou_threshold, max_output_boxes):
        batch_size = scores.shape[0]
        num_classes = scores.shape[-1]
        num_detections = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        detection_boxes = torch.randn(batch_size, max_output_boxes, 4)
        detection_scores = torch.randn(batch_size, max_output_boxes)
        detection_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        detections_indices = torch.randint(0, max_output_boxes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_detections, detection_boxes, detection_scores, detection_classes, detections_indices

    @staticmethod
    def symbolic(g, boxes, scores, score_threshold, iou_threshold, max_output_boxes):
        return g.op(
            "TRT::EfficientNMSX_TRT",
            boxes,
            scores,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=-1,
            score_activation_i=0,
            class_agnostic_i=0,
            box_coding_i=0,
            outputs=5
        )


class DeepStreamOutput(nn.Module):
    def __init__(self, nc, conf_threshold, iou_threshold, max_detections):
        super().__init__()
        self.nc = nc
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, x):
        preds = x[0].transpose(1, 2)
        boxes = preds[:, :, :4]
        scores = preds[:, :, 4:self.nc+4]
        masks = preds[:, :, self.nc+4:]
        protos = x[1]

        num_detections, detection_boxes, detection_scores, detection_classes, detections_indices = NMS.apply(
            boxes, scores, self.conf_threshold, self.iou_threshold, self.max_detections
        )

        batch_size, num_protos, h_protos, w_protos = protos.shape

        total_detections = batch_size * self.max_detections

        batch_index = torch.ones_like(detections_indices) * torch.arange(
            batch_size, device=boxes.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_index = batch_index.view(total_detections).to(torch.int32)
        box_index = detections_indices.view(total_detections).to(torch.int32)

        selected_boxes = boxes[batch_index, box_index]
        selected_masks = masks[batch_index, box_index]

        pooled_proto = RoiAlign.apply(protos, selected_boxes, batch_index, 1, 1, int(h_protos), int(w_protos), 0, 0.25)

        masks_protos = torch.matmul(
            selected_masks.unsqueeze(1), pooled_proto.view(total_detections, num_protos, h_protos * w_protos)
        )
        masks_protos = masks_protos.sigmoid().view(batch_size, self.max_detections, h_protos * w_protos)

        return torch.cat(
            [detection_boxes, detection_scores.unsqueeze(-1), detection_classes.unsqueeze(-1), masks_protos], dim=-1
        )


def yolo11_seg_export(weights, device, fuse=True):
    model = YOLO(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    if fuse:
        model = model.fuse()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model


def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f"\nStarting: {args.weights}")

    print("Opening YOLO11-Seg model")

    device = torch.device("cpu")
    model = yolo11_seg_export(args.weights, device)

    if len(model.names.keys()) > 0:
        print("Creating labels.txt file")
        with open("labels.txt", "w", encoding="utf-8") as f:
            for name in model.names.values():
                f.write(f"{name}\n")

    model = nn.Sequential(
        model, DeepStreamOutput(len(model.names), args.conf_threshold, args.iou_threshold, args.max_detections)
    )

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = args.weights.rsplit(".", 1)[0] + ".onnx"

    dynamic_axes = {
        "input": {
            0: "batch"
        },
        "output": {
            0: "batch"
        }
    }

    print("Exporting the model to ONNX")
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print("Simplifying the ONNX model")
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f"Done: {onnx_output_file}\n")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="DeepStream YOLO11-Seg conversion")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    parser.add_argument(
        "--conf-threshold", type=float, default=0.25, help="Minimum detection confidence threshold (default 0.25)"
    )
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    parser.add_argument(
        "--max-detections", type=int, default=100, help="Maximum number of output detections (default 100)"
    )
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if args.dynamic and args.batch > 1:
        raise SystemExit("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
