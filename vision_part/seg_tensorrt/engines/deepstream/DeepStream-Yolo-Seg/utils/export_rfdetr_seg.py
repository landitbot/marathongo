import os
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from rfdetr import RFDETRSegPreview
import rfdetr.models.backbone.projector as _m1
import rfdetr.models.segmentation_head as _m2
import rfdetr.models.ops.modules.ms_deform_attn as _m3


def LayerNorm_forward(self, x):
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, (int(x.size(3)),), self.weight, self.bias, self.eps)
    x = x.permute(0, 3, 1, 2)
    return x

_m1.LayerNorm.forward.__code__ = LayerNorm_forward.__code__


def SegmentationHead_forward_export(self, spatial_features, query_features, image_size, skip_blocks=False):
    assert len(query_features) == 1, "at export time, segmentation head expects exactly one query feature"

    target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
    spatial_features = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)

    if not skip_blocks:
        for block in self.blocks:
            spatial_features = block(spatial_features)

    spatial_features_proj = self.spatial_features_proj(spatial_features)

    qf = self.query_features_proj(self.query_features_block(query_features[0]))

    return [[spatial_features_proj, qf, self.bias]]


_m2.SegmentationHead.forward_export.__code__ = SegmentationHead_forward_export.__code__


def MSDeformAttn_forward(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None
):
    class MultiscaleDeformableAttnPlugin(torch.autograd.Function):
        @staticmethod
        def forward(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
            value = value.permute(0, 2, 3, 1)
            N, Lq, M, L, P, n = sampling_locations.shape
            attention_weights = attention_weights.view(N, Lq, M, L * P)
            return ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        @staticmethod
        def symbolic(g, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
            return g.op(
                "TRT::MultiscaleDeformableAttnPlugin_TRT",
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights
            )

    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
    attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2] \
                                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
    else:
        raise ValueError(f"Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.")

    attention_weights = F.softmax(attention_weights, -1)

    value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)

    value = value.permute(0, 3, 1, 2)

    L, P = sampling_locations.shape[3:5]

    attention_weights = attention_weights.view(N, Len_q, self.n_heads, L, P)

    output = MultiscaleDeformableAttnPlugin.apply(
        value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights
    )

    output = output.view(N, Len_q, self.d_model)

    output = self.output_proj(output)
    return output


_m3.MSDeformAttn.forward.__code__ = MSDeformAttn_forward.__code__


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


class DeepStreamOutput(nn.Module):
    def __init__(self, img_size, max_detections):
        super().__init__()
        self.img_size = img_size
        self.max_detections = max_detections

    def forward(self, x):
        boxes = x[0]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        boxes *= torch.as_tensor([[*self.img_size]]).flip(1).tile([1, 2]).unsqueeze(1)
        scores = x[1].sigmoid()
        protos, masks, mask_bias = x[2]

        num_classes = scores.shape[2]
        batch_size, num_protos, h_protos, w_protos = protos.shape

        topk_values, topk_indexes = torch.topk(scores.view(batch_size, -1), self.max_detections, dim=1, sorted=False)

        scores = topk_values.unsqueeze(-1)

        topk_boxes = topk_indexes // num_classes
        labels = topk_indexes % num_classes

        topk_boxes = topk_boxes.unsqueeze(-1)
        labels = labels.unsqueeze(-1)

        boxes = torch.gather(boxes, 1, topk_boxes.repeat(1, 1, 4))
        masks = torch.gather(masks, 1, topk_boxes.repeat(1, 1, num_protos))

        total_detections = batch_size * self.max_detections

        batch_index = torch.ones(
            [batch_size, self.max_detections], device=boxes.device, dtype=torch.int32
        ) * torch.arange(batch_size, device=boxes.device, dtype=torch.int32).unsqueeze(1)
        batch_index = batch_index.view(total_detections)

        selected_boxes = boxes.view(total_detections, 4)
        selected_masks = masks.view(total_detections, -1)

        pooled_proto = RoiAlign.apply(protos, selected_boxes, batch_index, 1, 1, int(h_protos), int(w_protos), 0, 0.25)

        masks_protos = torch.matmul(
            selected_masks.unsqueeze(1), pooled_proto.view(total_detections, num_protos, h_protos * w_protos)
        )
        masks_protos = masks_protos.view(batch_size, self.max_detections, h_protos * w_protos) + mask_bias

        return torch.cat([boxes, scores, labels.to(boxes.dtype), masks_protos], dim=-1)


def rfdetr_seg_export(model_name, weights, nc, img_size, max_detections, device):
    if model_name == "rfdetr-seg-preview":
        model = RFDETRSegPreview(pretrain_weights=weights, resolution=img_size[0], num_classes=nc, device=device.type)
    else:
        raise NotImplementedError("Model not supported")
    class_names = model.class_names
    model = deepcopy(model.model.model)
    model.to(device)
    model.eval()
    if hasattr(model, "export"):
        model.export()
    if max_detections > model.num_queries:
        raise ValueError(
            f"The `max_detections={max_detections}` is higher than the model `num_queries={model.num_queries}`")
    return model, class_names


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

    print("Opening RF-DETR-Seg model")

    device = torch.device("cpu")
    model, class_names = rfdetr_seg_export(
        args.model, args.weights, args.classes, args.size, args.max_detections, device
    )

    if len(class_names.keys()) > 0:
        print("Creating labels.txt file")
        with open("labels.txt", "w", encoding="utf-8") as f:
            f.write("background\n")
            for i in range(1, args.classes + 1):
                if i in class_names:
                    f.write(f"{class_names[i]}\n")
                else:
                    f.write("empty\n")


    img_size = args.size * 2 if len(args.size) == 1 else args.size

    model = nn.Sequential(
        model, DeepStreamOutput(img_size, args.max_detections)
    )

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
    parser = argparse.ArgumentParser(description="DeepStream RF-DETR-Seg conversion")
    parser.add_argument("-m", "--model", required=True, help="Model name (required)")
    parser.add_argument("-w", "--weights", required=True, type=str, help="Input weights (.pt) file path (required)")
    parser.add_argument("-n", "--classes", type=int, default=91, help="Number of trained classes (default 91)")
    parser.add_argument("-s", "--size", nargs="+", type=int, default=[432], help="Inference size [H,W] (default [432])")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", help="ONNX simplify model")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch-size")
    parser.add_argument("--batch", type=int, default=1, help="Static batch-size")
    parser.add_argument(
        "--max-detections", type=int, default=100, help="Maximum number of output detections (default 100)"
    )
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("Invalid weights file")
    if len(args.size) > 1 and args.size[0] != args.size[1]:
        raise SystemExit("RF-DETR model requires square resolution (width = height)")
    if args.dynamic and args.batch > 1:
        raise SystemExit("Cannot set dynamic batch-size and static batch-size at same time")
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
