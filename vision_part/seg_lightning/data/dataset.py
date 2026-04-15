"""
data/dataset.py
YOLO 格式的实例分割数据集。
支持多数据集，每张图片携带 dataset_source 标记（用于 loss 屏蔽）。

目录结构（YOLO格式）:
  <root>/images/train/*.jpg
  <root>/labels/train/*.txt   # YOLO seg 格式: cls x1 y1 x2 y2 ... (归一化多边形)
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class YOLOSegDataset(Dataset):
    """
    单个数据集（YOLO seg 格式），每个样本包含:
      - image: (3, H, W) float32 [0,1]
      - labels: (N, 5) float32  [cls, cx, cy, w, h] 归一化
      - segments: list of (K, 2) float32 归一化多边形点
      - source: str, 数据集来源标识 ('coco' / 'robot')
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        source: str,        # 数据集来源标识
        img_size: int = 640,
        augment: bool = True,
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.source = source
        self.img_size = img_size
        self.augment = augment

        # 收集图片路径
        self.img_files = sorted([
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])
        assert len(self.img_files) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')

        # 读取图片
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        # Letterbox resize
        img, ratio, pad = letterbox(img, self.img_size)
        h, w = img.shape[:2]

        # 读取标签
        labels = []    # [cls, cx, cy, bw, bh]
        segments = []  # 每个实例的分割多边形 [(K,2), ...]

        if label_path.exists():
            with open(label_path) as f:
                for line in f.read().strip().splitlines():
                    vals = list(map(float, line.split()))
                    cls = int(vals[0])
                    pts = np.array(vals[1:]).reshape(-1, 2)  # (K, 2) 归一化多边形

                    # 将多边形坐标调整到 letterbox 后尺寸（仍归一化）
                    pts[:, 0] = (pts[:, 0] * w0 * ratio + pad[0]) / w
                    pts[:, 1] = (pts[:, 1] * h0 * ratio + pad[1]) / h
                    pts = pts.clip(0, 1)

                    # 从多边形计算 bbox
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    bw = x2 - x1
                    bh = y2 - y1

                    labels.append([cls, cx, cy, bw, bh])
                    segments.append(pts.astype(np.float32))

        # 简单数据增强（仅翻转）
        if self.augment:
            if np.random.random() > 0.5:  # 水平翻转
                img = np.fliplr(img)
                for i, seg in enumerate(segments):
                    seg[:, 0] = 1 - seg[:, 0]
                    segments[i] = seg
                if labels:
                    labels = [[c, 1 - cx, cy, bw, bh] for c, cx, cy, bw, bh in labels]

        # 转换为 tensor
        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0

        labels_t = torch.tensor(labels, dtype=torch.float32) if labels else \
                   torch.zeros((0, 5), dtype=torch.float32)
        segs_t = [torch.from_numpy(s) for s in segments]

        return {
            "img": img_t,
            "labels": labels_t,       # (N, 5): [cls, cx, cy, w, h]
            "segments": segs_t,       # list of (K,2)
            "source": self.source,    # 数据集来源
            "img_path": str(img_path),
        }


def letterbox(img: np.ndarray, target: int) -> Tuple[np.ndarray, float, Tuple]:
    """
    Letterbox resize: 保持宽高比缩放到 target x target，填充灰色边框。
    返回: (img, ratio, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    ratio = min(target / h, target / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target - new_w) / 2
    pad_h = (target - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right  = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, ratio, (pad_w, pad_h)


def collate_fn(batch: List[dict]) -> dict:
    """
    将一个 batch 的样本合并为字典，格式与 ultralytics loss 兼容。
    
    返回 batch 字典:
      imgs:       (B, 3, H, W)
      batch_idx:  (N_total,) 各 gt 属于哪张图
      cls:        (N_total,)
      bboxes:     (N_total, 4)  xywh 归一化
      masks:      (N_total, H, W)  二值 mask（与 proto 同尺寸前会在 loss 里 resize）
      sources:    (B,) list[str]  每张图的数据集来源
    """
    imgs = torch.stack([b["img"] for b in batch])
    sources = [b["source"] for b in batch]
    B, _, H, W = imgs.shape

    batch_idx_list, cls_list, bbox_list, mask_list = [], [], [], []

    for i, b in enumerate(batch):
        n = len(b["labels"])
        if n == 0:
            continue
        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        cls_list.append(b["labels"][:, 0].long())
        bbox_list.append(b["labels"][:, 1:])

        # 将多边形转换为像素级 mask (H x W)
        for seg in b["segments"]:
            mask = poly_to_mask(seg, H, W)
            mask_list.append(mask)

    if batch_idx_list:
        batch_idx = torch.cat(batch_idx_list)
        cls       = torch.cat(cls_list)
        bboxes    = torch.cat(bbox_list)
        masks     = torch.stack(mask_list) if mask_list else torch.zeros(0, H, W)
    else:
        batch_idx = torch.zeros(0, dtype=torch.long)
        cls       = torch.zeros(0, dtype=torch.long)
        bboxes    = torch.zeros(0, 4)
        masks     = torch.zeros(0, H, W)

    return {
        "imgs": imgs,
        "batch_idx": batch_idx,
        "cls": cls,
        "bboxes": bboxes,
        "masks": masks,
        "sources": sources,
    }


def poly_to_mask(poly: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """将归一化多边形转为 (H, W) 二值 mask"""
    pts = (poly.numpy() * np.array([W, H])).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return torch.from_numpy(mask).float()
