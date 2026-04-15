# YOLO11s-seg Lightning 实现

基于 PyTorch Lightning 和 ultralytics 实现的 YOLO11s-seg 实例分割模型，支持多数据集混合训练，并实现了数据集感知的 loss 屏蔽策略。

## 项目结构

```
yolo_lightning/
├── configs/
│   └── yolo11s_seg.yaml          # 网络结构配置（类似 ultralytics 格式）
├── model/
│   ├── __init__.py
│   └── yolo_seg.py                # 模型构建（从 YAML 动态解析网络）
├── data/
│   ├── __init__.py
│   ├── dataset.py                 # YOLO 格式数据集（支持 source 标记）
│   └── datamodule.py              # Lightning DataModule（多数据集混合）
├── loss/
│   ├── __init__.py
│   └── seg_loss.py                # 数据集感知的分割 Loss
├── trainer/
│   ├── __init__.py
│   └── yolo_lightning.py          # Lightning 训练模块
└── train.py                       # 训练入口脚本
```

## 核心特性

### 1. 模块化设计
- **模型模块**: 从 YAML 配置动态构建网络，复用 ultralytics 的基础组件（Conv, C3k2, SPPF, C2PSA, Segment）
- **数据模块**: 支持多数据集混合，每张图片携带 `source` 标记
- **Loss 模块**: 在 ultralytics v8SegmentationLoss 基础上增加数据集感知屏蔽
- **训练模块**: 基于 Lightning 实现，支持分布式训练、混合精度、梯度裁剪等

### 2. 数据集感知的 Loss 屏蔽

**问题场景**:
- COCO 数据集：包含 person(0) 和 car(1)，提取后用于训练
- 机器人数据集：只标注了 robot(2)，但图片中也有 person 和 car（未标注）
- 直接混合训练会导致：机器人数据集中的 person/car 被当作负样本，抑制这些类别的检测能力

**解决方案**:
在 `loss/seg_loss.py` 中实现 `SourceAwareSegLoss`，核心策略：
1. 每张图片携带 `source` 标记（'coco' / 'robot'）
2. 定义各数据集实际包含的类别：`{'coco': [0, 2], 'robot': [1]}`
3. 在 cls loss 计算时，对每张图片屏蔽不属于该数据集的类别预测：
   - robot 图片：person/car 的预测不参与 cls loss（既不作为正样本也不作为负样本）
   - coco 图片：robot 的预测不参与 cls loss
4. bbox/seg loss 天然不受影响（因为 GT 中就不包含被屏蔽的类别）

**实现细节**:
```python
# loss/seg_loss.py:_apply_cls_ignore()
# cls_loss: (B, num_anchors, nc)
for i, ignore_cls in enumerate(ignore_cls_per_img):
    if ignore_cls:
        cls_ids = list(ignore_cls)
        cls_loss[i, :, cls_ids] = 0.0  # 屏蔽对应类别的 loss
```

### 3. 网络结构配置

`configs/yolo11s_seg.yaml` 采用与 ultralytics 相同的格式：
```yaml
nc: 3  # person, robot, car

scales:
  s: [0.50, 0.50, 1024]  # depth_mult, width_mult, max_channels

backbone:
  - [-1, 1, Conv,  [64, 3, 2]]   # [from, repeats, module, args]
  - [-1, 1, Conv,  [128, 3, 2]]
  - [-1, 2, C3k2,  [256, False, 0.25]]
  ...

head:
  - [-1, 1, Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  ...
  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]
```

## 使用方法

### 1. 环境依赖

```bash
pip install torch torchvision
pip install lightning
pip install ultralytics  # 复用基础模块
pip install opencv-python numpy
```

### 2. 数据准备

数据格式采用 YOLO seg 格式：
```
<dataset_root>/
├── images/
│   └── train/
│       ├── img1.jpg
│       └── img2.jpg
└── labels/
    └── train/
        ├── img1.txt  # 每行: cls x1 y1 x2 y2 ... (归一化多边形坐标)
        └── img2.txt
```

### 3. 训练

```bash
python train.py \
  --cfg configs/yolo11s_seg.yaml \
  --coco_img  /path/to/coco/images/train \
  --coco_lbl  /path/to/coco/labels/train \
  --robot_img /path/to/robot/images/train \
  --robot_lbl /path/to/robot/labels/train \
  --epochs 100 \
  --batch_size 16 \
  --img_size 640 \
  --lr 0.01 \
  --devices 1 \
  --precision 16-mixed
```

训练日志和 checkpoint 保存在 `checkpoints/` 目录。

### 4. 自定义数据集

修改 `train.py` 中的 `SOURCE_CLASS_MAP`：
```python
SOURCE_CLASS_MAP = {
    "coco":  [0, 2],  # COCO 包含 person(0) + car(2)
    "robot": [1],     # 机器人数据集只包含 robot(1)
    "custom": [0, 1, 2],  # 自定义数据集包含所有类别
}
```

然后在 `MultiDatasetModule` 中添加对应数据集配置。

## 方案可行性分析

### 优点
1. **简洁高效**: 直接在 cls loss 层面屏蔽，无需修改数据标注或模型结构
2. **理论正确**: 
   - 被屏蔽的类别在 GT 中本就不存在，bbox/seg loss 天然为 0
   - cls loss 屏蔽后，这些类别的预测既不被鼓励也不被惩罚，保持中性
3. **易于扩展**: 支持任意多个数据集，只需配置 `source_class_map`

### 注意事项
1. **类别不平衡**: 如果某个类别只在少数数据集中出现，可能导致该类别训练不充分
   - 解决方案：调整各数据集的采样比例，或使用 focal loss
2. **伪标签噪声**: 机器人数据集中未标注的 person/car 可能被模型误检为其他物体
   - 解决方案：在训练后期引入伪标签精炼（用高置信度预测作为伪 GT）
3. **验证集评估**: 需要分别在各数据集的验证集上评估，避免混淆

## 代码说明

### 模型构建 (model/yolo_seg.py)
- `YOLOSegModel`: 从 YAML 配置动态构建网络
- `_LazySegment`: 延迟初始化的 Segment head（在第一次 forward 时根据实际通道数初始化）
- 通道推断逻辑：`ch[i+1] = layer_i 输出通道`，`from_=-1` 表示上一层，`from_=k` 表示第 k 层

### 数据加载 (data/dataset.py)
- `YOLOSegDataset`: 单个数据集，返回 `{img, labels, segments, source}`
- `collate_fn`: 将 batch 合并为 ultralytics loss 兼容格式
- `letterbox`: 保持宽高比的 resize + padding

### Loss 计算 (loss/seg_loss.py)
- `SourceAwareSegLoss`: 继承 `v8SegmentationLoss`，重写 `loss()` 方法
- `_build_ignore_cls()`: 根据 source 计算每张图需要忽略的类别
- `_apply_cls_ignore()`: 将 cls_loss 中对应位置置 0

### Lightning 训练 (trainer/yolo_lightning.py)
- `YOLOSegLightning`: Lightning Module，封装训练逻辑
- `configure_optimizers()`: SGD + cosine LR + warmup
- `training_step()`: 前向 + loss 计算 + 日志记录

## 性能优化建议

1. **混合精度训练**: 使用 `--precision 16-mixed` 可加速 2x 并减少显存
2. **梯度累积**: 如果显存不足，可设置 `accumulate_grad_batches=4`
3. **多 GPU 训练**: 设置 `--devices 4` 使用 DDP 分布式训练
4. **数据增强**: 在 `data/dataset.py` 中可添加 Mosaic、MixUp 等增强

## 参考

- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) - YOLO11 官方实现
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) - 训练框架
