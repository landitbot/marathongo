from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s-seg.pt")
model.load()

results = model.train(
    data="data_3/dataset.yaml",
    epochs=100,
    imgsz=(640, 640),
    device=[1],
    freeze=10,
)
