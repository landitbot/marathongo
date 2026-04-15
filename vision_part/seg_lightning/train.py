"""
train.py
训练入口，基于 PyTorch Lightning 实现 YOLO11s-seg 的多数据集训练。

使用示例:
  python train.py \
    --cfg configs/yolo11s_seg.yaml \
    --coco_img  /data/coco/images/train2017 \
    --coco_lbl  /data/coco/labels/train2017 \
    --robot_img /data/robot/images/train \
    --robot_lbl /data/robot/labels/train \
    --epochs 100 --batch_size 16 --img_size 640
"""
import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from trainer.yolo_lightning import YOLOSegLightning
from data.datamodule import MultiDatasetModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",       default="configs/yolo11s_seg.yaml", help="模型配置文件")
    p.add_argument("--coco_img",  required=True, help="COCO 图片目录")
    p.add_argument("--coco_lbl",  required=True, help="COCO 标签目录")
    p.add_argument("--robot_img", required=True, help="机器人图片目录")
    p.add_argument("--robot_lbl", required=True, help="机器人标签目录")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--img_size",   type=int,   default=640)
    p.add_argument("--lr",         type=float, default=1e-2)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--devices",    type=int,   default=1)
    p.add_argument("--precision",  default="16-mixed", help="训练精度 (32/16-mixed/bf16-mixed)")
    p.add_argument("--save_dir",   default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    # --- 数据集感知配置 ---
    # 定义每个数据集实际包含的类别 id
    # 类别映射（见 configs/yolo11s_seg.yaml）: 0=person, 1=robot, 2=car
    SOURCE_CLASS_MAP = {
        "coco":  [0, 2],  # COCO 包含 person + car
        "robot": [1],     # 机器人数据集只包含 robot
    }

    # --- 数据 ---
    datamodule = MultiDatasetModule(
        datasets_config=[
            {"img_dir": args.coco_img,  "label_dir": args.coco_lbl,  "source": "coco"},
            {"img_dir": args.robot_img, "label_dir": args.robot_lbl, "source": "robot"},
        ],
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # --- 模型 ---
    lightning_model = YOLOSegLightning(
        cfg_path=args.cfg,
        source_class_map=SOURCE_CLASS_MAP,
        lr=args.lr,
    )

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="yolo11s-seg-{epoch:03d}-{train/loss:.3f}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # --- Trainer ---
    trainer = L.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="auto",
        precision=args.precision,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=10.0,  # 防止梯度爆炸
        accumulate_grad_batches=1,
    )

    trainer.fit(lightning_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
