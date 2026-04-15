# Engines Layout

- `python/`：Python 推理、benchmark、取帧工具。
- `cpp/`：C++ / DeepStream / V4L2 入口代码。
- `deepstream/`：独立的 DeepStream 系列工程。
- `weights/`：已经转换完成的 TensorRT `.engine` 权重。

后续新增文件时，保持下面约定：

- Python 入口只放 `engines/python/`
- C++ 入口只放 `engines/cpp/`
- DeepStream 工程只放 `engines/deepstream/`
- 所有 TensorRT Engine 只放 `engines/weights/`
