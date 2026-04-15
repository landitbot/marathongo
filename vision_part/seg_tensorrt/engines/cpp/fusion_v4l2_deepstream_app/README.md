# Fusion V4L2 DeepStream App

这个项目把两部分能力合在一起：

- `nvidiajetsonnanVv4L2example` 提供的 Jetson 相机硬件采集 / 缩放实现思路
- `DeepStream-Yolo-Seg` 提供的 DeepStream `nvinfer` 配置、自定义库和 fusion 模型文件

目标是生成一个**可以在 Docker 内直接运行的可执行程序**，而不是继续手工拼长 `gst-launch-1.0` 命令。

## 当前实现

程序名：

```text
fusion_v4l2_deepstream_app
```

默认链路：

```text
/dev/video0 MJPEG
-> nvv4l2decoder
-> nvvideoconvert (VIC) -> 960x540
-> nvcompositor -> nvvideoconvert -> 960x544
-> nvstreammux
-> nvinfer
-> bbox / mask postprocess
-> nvdsosd
-> mp4 recording
```

当前默认阈值：

- `head1_threshold=0.01`
- `head2_threshold=0.01`
- `nms_iou=0.45`
- `mask_threshold=0.5`

这是按当前 `fusion_branch10_960x544.onnx` 的实际输出尺度调出来的。此前 `0.25` 阈值下没有有效检测输出。

它已经把之前验证过的链路固化进代码里，并在 `nvinfer` 的 `src` pad 上挂了 probe，会定期打印：

- `NvDsInferTensorMeta`
- 输出层名字
- 每个输出层的维度

这对当前 `fusion_branch10_960x544.onnx` 非常重要，因为它是双头输出：

```text
boxes1 / scores1 / mask1 / proto1
boxes2 / scores2 / mask2 / proto2
```

## 默认依赖路径

程序默认使用：

- 推理配置：
  `/workspace/DeepStream-Yolo-Seg/config_infer_primary_fusion_960x544_engine_tensormeta.txt`
- engine：
  `/workspace/fusion_branch10_960x544_b1_fp16.engine`

也就是说，容器运行时默认假设你把整个 `/home/user/repo` 挂载到 `/workspace`。

## 目录结构

```text
/home/user/repo
├── DeepStream-Yolo-Seg
├── nvidiajetsonnanVv4L2example
├── fusion_branch10_960x544.onnx
├── fusion_branch10_960x544_b1_fp16.engine
└── fusion_v4l2_deepstream_app
```

## 编译 Docker 镜像

在 `/home/user/repo` 下执行：

```bash
cd /home/user/repo/fusion_v4l2_deepstream_app
./build_image.sh
```

如果你希望把可执行文件直接编到挂载目录里的 `build/`，而不是只用镜像内预编译版本：

```bash
./build.sh
```

生成产物：

```text
/home/user/repo/fusion_v4l2_deepstream_app/build/fusion_v4l2_deepstream_app
```

说明：

- `run_container.sh` 会优先使用挂载目录里的 `build/fusion_v4l2_deepstream_app`
- 所以你改完源码后，先执行一次 `./build.sh`，就不必每次都重建整个镜像

默认镜像名：

```text
fusion-v4l2-deepstream:7.1
```

它会基于：

```text
ds-segformer:7.1-ready
```

安装 GStreamer 开发包，然后编译 `fusion_v4l2_deepstream_app`。

## 运行

默认 headless 推理模式：

```bash
cd /home/user/repo/fusion_v4l2_deepstream_app
./run_container.sh
```

说明：

- 默认模式不会把宿主机 `DISPLAY` / `XAUTHORITY` 传进容器
- 这样可以避免在桌面环境里误触发 EGL/X11 路径，导致 headless 推理启动失败
- 默认会输出一段带分割叠加的 mp4 视频到 `output/`
- 默认会在宿主机启动 `tegrastats`，程序结束后打印共享内存 / GR3D / VIC 统计

直接打印每帧检测结果：

```bash
./run_container.sh --tensor-log-interval 1
```

指定配置文件：

```bash
./run_container.sh --config /workspace/DeepStream-Yolo-Seg/config_infer_primary_fusion_960x544_onnx_tensormeta.txt
```

指定输出视频路径：

```bash
./run_container.sh --output /workspace/fusion_v4l2_deepstream_app/output/custom_run.mp4
```

打印完整 pipeline：

```bash
./run_container.sh --verbose
```

如果要打开预览分支：

```bash
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority ./run_container.sh --preview
```

注意：

- `--preview` 会创建 `960x560` 预览画布，并尝试走 `nveglglessink`
- 预览分支已经包含 `nvdsosd`，会尝试显示程序内附加的 bbox / text metadata
- 当前回合主要完成的是 headless 闭环，X11 预览链路尚未做同等强度的实机验证
- `run_container.sh` 只会在显式传入 `--preview` 时转发 `DISPLAY` / `XAUTHORITY`

## 命令行参数

```text
--device PATH
--camera-width N
--camera-height N
--fps N
--exposure-auto N
--exposure-absolute N
--gain N
--extra-controls STR
--config PATH
--output PATH
--labels PATH
--head1-threshold X
--head2-threshold X
--mask-threshold X
--nms-iou X
--max-detections N
--bitrate N
--preview
--tensor-log-interval N
--verbose
```

## 这个程序已经解决了什么

- 不再手工输入长 `gst-launch-1.0`
- 固化 `1280x720 -> 960x540 -> 960x544` 的推理输入链路
- 自动接入 `nvstreammux`，避免 `NvDsBatchMeta not found`
- 自动读取 `output-tensor-meta=1` 的输出层信息
- 双头 bbox 解码
- mask/proto 解码并附加到 `mask_params`
- `nvdsosd` 叠加分割结果
- MP4 录像输出
- 退出时打印 FPS
- 宿主机 `tegrastats` 统计共享内存与 GR3D/VIC 占用
- Docker 内可编译、可运行

## 还没完成的部分

下面两件事还没在这个程序里完成：

1. `NvDsInferParseFusionSeg`
   还没有把同样的 fusion 逻辑移植进 `DeepStream-Yolo-Seg` 自定义 parser

2. 底部 4 行 footer-aware preprocess
   如果你要把 timestamp 编进 `960x544` 输入底部 4 行，需要单独的 preprocess 逻辑

当前版本已经完成的闭环是：

- Docker 内相机采集
- 硬解 / VIC 缩放
- `960x544` 推理输入
- `nvinfer` 加载 engine
- raw tensor 解析
- bbox 结果生成并附加为 `NvDsObjectMeta`
- mask 结果生成并附加为 `mask_params`
- 分割视频编码输出
- 程序内 FPS 汇总
- 宿主机显存/共享内存统计

## 推荐的下一步

当前最合理的继续方向是：

1. 在 `DeepStream-Yolo-Seg` 里新增 `nvdsparseseg_fusion.cpp`
2. 把 `config_infer_primary_fusion_960x544_engine_tensormeta.txt` 切换到真正的 parser 模式
3. 再决定是否把底部 4 行 footer preprocess 整合进程序
