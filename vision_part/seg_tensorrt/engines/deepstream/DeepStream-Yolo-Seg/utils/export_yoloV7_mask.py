import os
import yaml
import onnx
import torch
import torch.nn as nn

from utils.general import merge_bases


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
    def __init__(self, nc, conf_threshold, iou_threshold, max_detections, attn_resolution, num_base):
        super().__init__()
        self.nc = nc
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.attn_resolution = attn_resolution
        self.num_base = num_base

    def forward(self, x):
        preds = x["test"]
        boxes = preds[:, :, :4]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        objectness = preds[:, :, 4:5]
        scores = preds[:, :, 5:self.nc+5]
        scores *= objectness
        attn = x["attn"]
        bases = torch.cat([x["bases"], x["sem"]], dim=1)

        num_detections, detection_boxes, detection_scores, detection_classes, detections_indices = NMS.apply(
            boxes, scores, self.conf_threshold, self.iou_threshold, self.max_detections
        )

        batch_size, num_protos, h_protos, w_protos = bases.shape

        total_detections = batch_size * self.max_detections

        batch_index = torch.ones_like(detections_indices) * torch.arange(
            batch_size, device=boxes.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_index = batch_index.view(total_detections).to(torch.int32)
        box_index = detections_indices.view(total_detections).to(torch.int32)

        selected_boxes = boxes[batch_index, box_index]
        selected_masks = attn[batch_index, box_index]

        pooled_proto = RoiAlign.apply(bases, selected_boxes, batch_index, 1, 1, int(h_protos), int(w_protos), 0, 0.25)

        masks_protos = merge_bases(
            pooled_proto, selected_masks, self.attn_resolution, self.num_base
        ).view(batch_size, self.max_detections, h_protos * w_protos).sigmoid()

        return torch.cat(
            [detection_boxes, detection_scores.unsqueeze(-1), detection_classes.unsqueeze(-1), masks_protos], dim=-1
        )


def yolov7_mask_export(weights, device):
    ckpt = torch.load(weights)
    model = ckpt["model"]
    model = model.float().to(device)
    model.eval()
    with open("data/hyp.scratch.mask.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    return model, hyp["attn_resolution"], hyp["num_base"]


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

    print("Opening YOLOv7-Mask model")

    device = torch.device("cpu")
    model, attn_resolution, num_base = yolov7_mask_export(args.weights, device)

    if hasattr(model, "names") and len(model.names) > 0:
        print("Creating labels.txt file")
        with open("labels.txt", "w", encoding="utf-8") as f:
            for name in model.names:
                f.write(f"{name}\n")

    model = nn.Sequential(
        model, DeepStreamOutput(
            len(model.names), args.conf_threshold, args.iou_threshold, args.max_detections, attn_resolution, num_base
        )
    )

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    if img_size == [640, 640] and args.p6:
        img_size = [1280] * 2

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
    parser = argparse.ArgumentParser(description="DeepStream YOLOv7-Mask conversion")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[640], help="Inference size [H,W] (default [640])")
    parser.add_argument("--p6", action="store_true", help="P6 model")
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
