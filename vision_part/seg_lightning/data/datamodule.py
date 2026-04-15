"""
data/datamodule.py
Lightning DataModule，支持多数据集混合训练。
"""
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, ConcatDataset

from .dataset import YOLOSegDataset, collate_fn


class MultiDatasetModule(L.LightningDataModule):
    """
    多数据集混合训练的 DataModule。
    
    Args:
        datasets_config: 数据集配置列表，每项包含:
            - img_dir: 图片目录
            - label_dir: 标签目录
            - source: 数据集来源标识 ('coco' / 'robot')
        img_size: 输入图片尺寸
        batch_size: batch size
        num_workers: DataLoader workers
    
    Example:
        datasets_config = [
            {"img_dir": "coco/images/train", "label_dir": "coco/labels/train", "source": "coco"},
            {"img_dir": "robot/images/train", "label_dir": "robot/labels/train", "source": "robot"},
        ]
    """

    def __init__(
        self,
        datasets_config: list[dict],
        img_size: int = 640,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.datasets_config = datasets_config
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """构建训练/验证数据集"""
        if stage == "fit" or stage is None:
            # 训练集：混合多个数据集
            train_datasets = [
                YOLOSegDataset(
                    img_dir=cfg["img_dir"],
                    label_dir=cfg["label_dir"],
                    source=cfg["source"],
                    img_size=self.img_size,
                    augment=True,
                )
                for cfg in self.datasets_config
            ]
            self.train_dataset = ConcatDataset(train_datasets)

            # 验证集（可选，这里简化为空）
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
