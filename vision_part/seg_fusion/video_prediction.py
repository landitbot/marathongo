from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect, Segment


# Compatibility shim for checkpoints saved with a custom __main__.SegmentP2 class.
class SegmentP2(Segment):
    def forward(self, x):
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        # P2 head uses x[1] (P3 feature) for proto generation, not x[0].
        proto = self.proto(x[1])
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

# Load a model
model = YOLO("/path/to/runs/segment/train12/weights/best.pt")

results = model.predict(
    source = "/path/to/videos/video8_11_2x2.mp4",
    stream=True,
    save=True,
    conf=0.25,
    verbose=True,
    )

for i, result in enumerate(results, 1):
    n = 0 if result.boxes is None else len(result.boxes)
    print(f"frame {i}: {n} detections")

print("done")
