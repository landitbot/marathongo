# 机器人分割 TensorRT 项目整理模板

## 1. 项目定位

- 任务类型：机器人分割 / 实例分割
- 部署方式：TensorRT Engine 推理
- 代码组织：`engines/python`、`engines/cpp` 和 `engines/deepstream` 明确分开
- 结果管理：输入数据、Engine 权重、可视化结果、benchmark 日志各自独立

## 2. 当前目录结构

```text
robot_seg_tensorrt_template/
├── README.md
├── .gitignore
├── configs/
│   ├── deploy/
│   │   └── trtexec_fp16.yaml
│   ├── model/
│   │   └── robot_seg.yaml
│   └── runtime/
│       ├── image.yaml
│       └── video.yaml
├── data/
│   ├── images/
│   └── videos/
├── engines/
│   ├── cpp/
│   │   └── fusion_v4l2_deepstream_app/
│   ├── deepstream/
│   │   └── DeepStream-Yolo-Seg/
│   ├── python/
│   │   ├── camera_frame_capture.py
│   │   ├── robot_seg_trt_bench.py
│   │   ├── robot_seg_trt_image.py
│   │   └── robot_seg_trt_video.py
│   └── weights/
│       ├── fusion_branch10_960x544_b1_fp16.engine
│       ├── fusion_fp16.engine
│       ├── fusion_int8.engine
│       ├── fusion_int8_protofp16_opt0.engine
│       ├── fusion_robotbase_fp16.engine
│       └── yolo11s-seg_fp16.engine
├── outputs/
│   ├── benchmarks/
│   ├── logs/
│   └── visualizations/
└── scripts/
    ├── benchmark_video.sh
    ├── build_engine.sh
    ├── capture_frames.sh
    ├── infer_image.sh
    └── infer_video.sh
```

## 3. 目录职责

- `configs/`：模型尺寸、类别、运行参数、`trtexec` 构建参数。
- `data/images`：图片推理输入。
- `data/videos`：视频推理输入。
- `engines/python`：Python 推理、benchmark、取帧脚本。
- `engines/cpp`：C++ 入口代码，方便后续补 DeepStream / V4L2 / 纯 TensorRT 应用。
- `engines/deepstream`：DeepStream 系列工程、推理配置、插件源码和导出脚本。
- `engines/weights`：已经转换好的 TensorRT `.engine` 权重。
- `outputs/visualizations`：图片叠加结果、输出视频、关键帧。
- `outputs/benchmarks`：FPS、延迟、显存/共享内存统计。
- `outputs/logs`：`trtexec` 和运行日志。

## 4. 依赖环境

最少需要：

- Python 3.8+
- OpenCV
- NumPy
- TensorRT Python
- PyCUDA

示例安装：

```bash
pip install numpy opencv-python PyYAML
```

`tensorrt` 和 `pycuda` 建议按目标设备环境安装，尤其是 Jetson 上要和 JetPack 版本对应。

## 5. TensorRT Engine 构建

如果你已经有 ONNX，可以直接用脚本转换成 Engine：

```bash
cd robot_seg_tensorrt_template

bash scripts/build_engine.sh \
  /path/to/fusion_branch10_960x544.onnx \
  engines/weights/fusion_branch10_960x544_fp16.engine
```

切到 INT8：

```bash
PRECISION=int8 bash scripts/build_engine.sh \
  /path/to/fusion_branch10_960x544.onnx \
  engines/weights/fusion_branch10_960x544_int8.engine
```

构建日志会写到 `outputs/logs/`。

## 6. 图片推理

先把待测图片放到 `data/images/`，然后执行：

```bash
bash scripts/infer_image.sh
```

默认入口：

- 脚本：`engines/python/robot_seg_trt_image.py`
- 默认 Engine：`engines/weights/fusion_branch10_960x544_b1_fp16.engine`
- 输出目录：`outputs/visualizations/images/`

输出内容：

- 叠加分割结果图片
- `summary.json`

## 7. 视频推理

先把待测视频放到 `data/videos/`，然后执行：

```bash
bash scripts/infer_video.sh
```

默认入口：

- 脚本：`engines/python/robot_seg_fusion_video.py`
- 默认 Engine：`engines/weights/fusion_fp16.engine`
- 输出目录：`outputs/visualizations/videos/`

输出内容：

- 推理后视频
- 中间截图：`outputs/visualizations/videos/screenshots/`
- `summary.json`

## 8. Benchmark

如果你要导出论文附录或简历里的性能数据，直接跑：

```bash
bash scripts/benchmark_video.sh
```

默认入口：

- 脚本：`engines/python/robot_seg_trt_bench.py`
- 输出目录：`outputs/benchmarks/fp16_baseline/`

输出内容：

- benchmark 视频
- `fp16_baseline_summary.json`
- `fp16_baseline_summary.csv`

## 9. 摄像头样例取帧

如果你要先从摄像头抓图，再喂给分割模型：

```bash
bash scripts/capture_frames.sh
```

抓到的图片默认保存到 `data/images/camera_capture/`。

## 10. 当前已整理进 `engines/` 的文件

Python：

- `robot_seg_trt_video.py`：TensorRT 视频分割推理
- `robot_seg_fusion_video.py`：按 `trt_video_bench.py` 的解析方式运行 `fusion` 系列视频推理，标签为 `person / car / robot`
- `robot_seg_trt_image.py`：TensorRT 图片分割推理
- `robot_seg_trt_bench.py`：TensorRT 视频 benchmark
- `camera_frame_capture.py`：摄像头取帧工具

C++：

- `fusion_v4l2_deepstream_app/`：现有 V4L2 + DeepStream 入口代码

DeepStream：

- `DeepStream-Yolo-Seg/`：完整同步的 DeepStream 分割工程，保留了配置、插件源码、导出脚本、说明文档和模型相关文件

TensorRT 权重：

- `fusion_branch10_960x544_b1_fp16.engine`
- `fusion_fp16.engine`
- `fusion_int8.engine`
- `fusion_int8_protofp16_opt0.engine`
- `fusion_robotbase_fp16.engine`
- `yolo11s-seg_fp16.engine`


