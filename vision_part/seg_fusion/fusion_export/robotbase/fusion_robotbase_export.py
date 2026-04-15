from ultralytics import YOLO
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


model = YOLO("fusion_robotbase.yaml")
model.model.cpu()

train4_model = torch.load("runs/segment/train10/weights/best.pt", map_location="cpu")["model"].state_dict()
person_model = torch.load("yolo11s-seg.pt", map_location="cpu")["model"].state_dict()

# Keep train4 intact at model.0~23
robot_base_dict = {"model." + k: v for k, v in train4_model.items()}
missing_keys, unexpected_keys = model.load_state_dict(robot_base_dict, strict=False)
print("train4 missing:", len(missing_keys), "unexpected:", len(unexpected_keys))

# Add original yolo11m-seg head at model.24~36
person_head_dict = {}
for k, v in person_model.items():
    parts = k.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        continue
    layer = int(parts[1])
    if layer < 11:
        continue
    parts[1] = str(layer + 13)  # 11->24, ..., 23->36
    person_head_dict["model." + ".".join(parts)] = v

missing_keys, unexpected_keys = model.load_state_dict(person_head_dict, strict=False)
print("person head missing:", len(missing_keys), "unexpected:", len(unexpected_keys))


def find_sample_image():
    candidates = sorted(Path("data_4/images/val").glob("*"))
    if not candidates:
        candidates = sorted(Path("data_4/images/train").glob("*"))
    if not candidates:
        raise FileNotFoundError("No images found under data_4/images/val or data_4/images/train")
    return candidates[0]


def load_image_tensor(image_path, width=1280, height=720):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


class ExportModel(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model
        self.head_outputs = {}
        self.robot_head_idx = 23
        self.person_head_idx = 36
        self._hook1 = self.model.model[self.robot_head_idx].register_forward_hook(self._make_hook("robot_head"))
        self._hook2 = self.model.model[self.person_head_idx].register_forward_hook(self._make_hook("person_head"))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self.head_outputs[name] = output
        return hook

    def _print_head_debug(self, name, head):
        details = head[1]
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

        scores_sigmoid = scores.sigmoid()
        print("max sigmoid score:", float(scores_sigmoid.max()))
        vals, inds = torch.topk(scores_sigmoid.view(-1), k=10)
        print("top10 sigmoid scores:")
        for v, idx in zip(vals.tolist(), inds.tolist()):
            cls = idx // scores.shape[-1]
            anc = idx % scores.shape[-1]
            print(
                {
                    "score": round(v, 4),
                    "cls": int(cls),
                    "anchor": int(anc),
                }
            )

    def forward(self, x):
        self.head_outputs.clear()
        # YOLO heads require stride alignment (typically 32). Keep external input at 1280x720,
        # and pad internally to avoid concat shape mismatch (e.g. 45 vs 46).
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        _ = self.model(x)

        robot_head = self.head_outputs["robot_head"]
        person_head = self.head_outputs["person_head"]

        self._print_head_debug("robot_head", robot_head)
        self._print_head_debug("person_head", person_head)

        details1 = robot_head[1]
        details2 = person_head[1]

        # Keep ONNX output order aligned with new_test.py and onnx_inference_test.py:
        # first 4 tensors are the 80-class person/car head, last 4 are the 3-class robot head.
        return (
            details2["boxes"],
            details2["scores"],
            details2["mask_coefficient"],
            details2["proto"],
            details1["boxes"],
            details1["scores"],
            details1["mask_coefficient"],
            details1["proto"],
        )


m = ExportModel(model).eval()
image_path = find_sample_image()
x = load_image_tensor(image_path, width=1280, height=720)
print("input image:", image_path)
print("input tensor shape:", tuple(x.shape))

with torch.no_grad():
    y = m(x)
    for i, item in enumerate(y):
        print(f"output[{i}] shape:", tuple(item.shape))

torch.onnx.export(
    m,
    x,
    "fusion_robotbase.onnx",
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

print("Exported ONNX to fusion_robotbase.onnx")
