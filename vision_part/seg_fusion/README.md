# split_yolo_seg

## 项目概述

`split_yolo_seg` 是一个面向 `robot`、`person`、`car` 混合场景的实例分割实验方案整理目录。

该项目所要解决的核心问题是自定义数据集存在部分标注：

- 当前自定义数据中仅标注了 `robot` 类别；
- 同一图像中仍可能同时出现 `person` 和 `car`；
- 若直接将该数据作为完整监督数据进行训练，未标注的 `person` 和 `car` 会被错误视为背景，从而影响分类与分割质量。

为降低部分标注带来的干扰，当前方案采用分阶段训练与权重融合的技术路线：

1. 使用自定义 `robot` 数据训练面向目标域的模型；
2. 在第一阶段冻结 backbone，以降低对通用视觉特征的破坏；
3. 将自定义数据训练得到的权重与 COCO 数据训练得到的权重进行融合；
4. 使用共享 backbone 与双 head 结构，分别保留通用类别能力与 `robot` 类别能力。

本 README 主要说明训练方案、目录结构、实验流程与中间导出产物，不包含部署、运行时集成及生产化推理服务内容。

## 背景问题

在当前自定义数据集中，仅有 `robot` 被显式标注，而 `person`、`car` 可能真实存在于同一批图像中。

在常规监督训练设置下，未标注目标会被隐式当作背景处理，由此会带来以下风险：

- 原本应由通用预训练模型保留的 `person`、`car` 识别能力被抑制；
- `robot` 周边区域的特征学习受到干扰，进而影响分割边界质量与类别判别稳定性。

因此，本项目不直接采用单阶段常规微调方式，而是将“目标域适配”与“通用类别能力保留”拆分处理。

## 方法说明

### 第一阶段：面向 `robot` 的目标域适配

第一阶段使用自定义 `robot` 数据对模型进行目标域适配。

关键设计如下：

- 训练过程中冻结 backbone；
- 仅由 head 或主要由 head 完成面向 `robot` 的适配。

该设计的目的在于：

- 尽量保留预训练模型中的通用视觉表示；
- 在不显著重写基础特征提取器的前提下，提高模型对 `robot` 类别的分割能力。

### 第二阶段：权重融合

完成 `robot` 目标域适配后，将所得权重与基于 COCO 数据训练的权重进行融合。

当前方案的结构性设想为：

- 使用同一个共享 backbone；
- 使用两个相互独立的 head；
- 其中一个 head 偏向保留 `person`、`car` 等通用类别能力；
- 另一个 head 偏向强化 `robot` 类别能力。

该阶段的目标是在保持通用类别识别能力的同时，增强模型对 `robot` 类别的适配效果。

## 目录结构

当前目录中已整理的文件与子目录如下：

- [train.py](train.py)：当前第一阶段训练入口
- [video_prediction.py](video_prediction.py)：基于 `.pt` 权重的视频验证脚本
- [onnx_inference.py](onnx_inference.py)：ONNX 离线推理脚本
- [onnx_camera_inference.py](onnx_camera_inference.py)：ONNX 摄像头推理脚本
- [demo_video.py](demo_video.py)：视频演示脚本
- [demo/](demo)：演示相关资源目录
- [data_boost/](data_boost)：数据增强脚本目录
- [fusion_export/](fusion_export)：模型融合、权重拼接与导出相关脚本目录
- [ckpt/](ckpt)：阶段性权重与导出产物目录

其中：

- [data_boost/data_copypaste.py](data_boost/data_copypaste.py) 用于 Copy-Paste 增强；
- [data_boost/data_zoomout.py](data_boost/data_zoomout.py) 用于 Zoom-out / 小目标场景增强；
- [fusion_export/robotbase](fusion_export/robotbase) 用于保留完整 `robot` head 的融合实验；
- [fusion_export/fusion10](fusion_export/fusion10) 用于保留完整 YOLO 主干分支并融合 `robot` head 的实验。

## 环境依赖

建议使用 Python 3.10 及以上版本，并安装以下核心依赖：

```bash
pip install ultralytics torch opencv-python onnx onnxruntime numpy
```

如使用 GPU 环境，请根据本机 CUDA 版本安装对应的 PyTorch。

## 当前训练配置

当前训练入口位于 [train.py](train.py)，默认使用如下配置：

- 预训练模型：`yolo11s-seg.pt`
- 数据集配置：`data_3/dataset.yaml`
- 训练轮数：`100`
- 输入尺寸：`640 x 640`
- 训练设备：`device=[1]`
- 冻结层数：`freeze=10`

在当前阶段，该脚本应理解为面向 `robot` 的目标域适配训练入口，而非最终完整训练系统。

## 已有权重与中间产物

[ckpt](ckpt) 目录当前包含以下相关文件：

- [robot_best.pt](ckpt/robot_best.pt)
- [person_car_best.pt](ckpt/person_car_best.pt)
- [fusion_model.onnx](ckpt/fusion_model.onnx)

上述文件属于当前研究过程中的阶段性结果，应视为实验产物，而非正式发布版本。

其中，`fusion_model.onnx` 表示当前融合流程产生的中间导出结果，可用于离线验证，但不应等同于最终部署版本。

## 快速开始

### 训练

在当前目录下运行第一阶段训练流程：

```bash
python train.py
```

如需调整训练参数，可直接修改 [train.py](train.py)。

### 权重融合与导出

融合与导出相关逻辑位于 [fusion_export/](fusion_export) 目录。当前可用脚本以具体实验实现为准，例如：

```bash
python fusion_export/robotbase/fusion_robotbase_export.py
python fusion_export/fusion10/fusion_branch10_export.py
```

说明：

- 不同脚本对应不同的权重融合策略；
- 导出的 ONNX 结果属于研究过程中的中间产物；
- 输出顺序、类别定义及 mask 解析方式需要与后处理逻辑保持一致。

### 离线验证

如需对中间导出模型进行离线验证，可使用：

```bash
python onnx_inference.py --model ckpt/fusion_model.onnx --source /path/to/input
```

如需直接验证 `.pt` 权重效果，可使用：

```bash
python video_prediction.py
```

## 推荐实验流程

建议按照以下顺序开展实验：

1. 使用自定义数据完成 `robot` 定向训练；
2. 对 `robot` 阶段性权重进行验证与分析；
3. 将该权重与保留 `person`、`car` 能力的权重进行融合；
4. 导出中间 ONNX 模型；
5. 通过离线推理脚本对融合结果进行一致性验证。

## 注意事项

1. 当前自定义数据集属于部分标注数据，所有训练结果均需结合这一前提进行解读。
2. 第一阶段冻结 backbone 是有意设计，用于尽量保留预训练模型的通用视觉表示。
3. 当前 [train.py](train.py) 仅对应第一阶段训练，不代表最终完整训练框架。
4. 权重融合仍属于实验性流程，需要对类别一致性、输出行为与效果稳定性进行单独验证。
5. 若任一阶段调整了类别顺序，必须显式检查类别索引。在当前设定中，自定义数据中的 `robot` 索引为 `2`。
6. ONNX 导出结果的输出顺序必须与后处理脚本保持一致，否则会导致类别解析或 mask 解析错误。
7. 若从历史脚本迁移到当前目录结构，需要特别注意原有绝对路径配置已发生变化，当前文档中的路径均以本目录为根目录。

## 文档范围

本 README 仅覆盖以下内容：

- 训练方案说明；
- 方法设计背景；
- 代码目录组织；
- 融合与导出中间流程；
- 当前实验流程。

以下内容不属于本 README 范围，应在独立项目或独立文档中维护：

- 部署流程；
- 运行时集成；
- TensorRT 相关转换与执行链路；
- 生产环境推理服务。
