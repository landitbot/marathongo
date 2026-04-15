"""
trainer/yolo_lightning.py
Lightning Module，封装 YOLO11s-seg 的训练/验证逻辑。
"""
import torch
import lightning as L

from model.yolo_seg import build_model_from_config
from loss.seg_loss import SourceAwareSegLoss


class YOLOSegLightning(L.LightningModule):
    """
    YOLO11s-seg Lightning 训练模块。
    
    Args:
        cfg_path: 网络配置 YAML 路径
        source_class_map: 各数据集的有效类别，如 {'coco':[0,2], 'robot':[1]}
        lr: 初始学习率
        warmup_epochs: 学习率 warmup epoch 数
        weight_decay: 权重衰减
    """

    def __init__(
        self,
        cfg_path: str,
        source_class_map: dict[str, list[int]],
        lr: float = 1e-2,
        warmup_epochs: int = 3,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 构建模型
        self.model = build_model_from_config(cfg_path)
        self.source_class_map = source_class_map

        # loss 在 setup 时初始化（需要先完成一次前向以初始化 Segment head）
        self._loss_fn: SourceAwareSegLoss | None = None

    def setup(self, stage: str):
        """在 trainer.fit 前调用，此时 Segment head 已初始化"""
        if stage == "fit" and self._loss_fn is None:
            # 做一次虚拟前向，确保 _LazySegment 初始化
            self.model.eval()
            with torch.no_grad():
                dummy = torch.zeros(2, 3, 640, 640, device=self.device)
                self.model(dummy)
            self.model.train()

            # 初始化 loss
            self._loss_fn = SourceAwareSegLoss(
                model=self.model,
                source_class_map=self.source_class_map,
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int):
        imgs = batch["imgs"]
        preds = self.model(imgs)

        # loss 格式化为 ultralytics 兼容格式
        batch_for_loss = {
            "batch_idx": batch["batch_idx"],
            "cls":       batch["cls"],
            "bboxes":    batch["bboxes"],
            "masks":     batch["masks"],
            "sources":   batch["sources"],
        }

        total_loss, loss_items = self._loss_fn.loss(preds, batch_for_loss)

        # 记录各分项 loss
        self.log("train/loss_box", loss_items[0], prog_bar=False, sync_dist=True)
        self.log("train/loss_seg", loss_items[1], prog_bar=False, sync_dist=True)
        self.log("train/loss_cls", loss_items[2], prog_bar=True,  sync_dist=True)
        self.log("train/loss_dfl", loss_items[3], prog_bar=False, sync_dist=True)
        self.log("train/loss",     total_loss / imgs.shape[0], prog_bar=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        """使用 SGD + cosine LR schedule + warmup"""
        hp = self.hparams

        # 分组参数：BN 不加 weight_decay
        bn_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bn' in name or '.bn' in name:
                bn_params.append(p)
            else:
                other_params.append(p)

        optimizer = torch.optim.SGD(
            [
                {"params": other_params, "weight_decay": hp.weight_decay},
                {"params": bn_params,    "weight_decay": 0.0},
            ],
            lr=hp.lr,
            momentum=0.937,
            nesterov=True,
        )

        # cosine 衰减 scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs - hp.warmup_epochs, eta_min=hp.lr * 0.01
        )

        # warmup：前 warmup_epochs 用线性 warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=hp.warmup_epochs
        )

        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[hp.warmup_epochs],
        )

        return [optimizer], [{"scheduler": combined, "interval": "epoch"}]

    def on_train_epoch_start(self):
        """训练前设置 BN momentum"""
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = max(1 - 0.01 * self.current_epoch, 0.01)
