"""
loss/seg_loss.py
YOLO11s-seg 的 Loss，在 ultralytics v8SegmentationLoss 基础上增加数据集感知屏蔽。

屏蔽策略：
  robot 数据集: GT 中没有 person/car，但模型可能预测出它们。
    若将预测出的 person/car 作为负样本参与 cls loss，会抑制这些类别的检测能力。
    解决方案: 在 robot 图片中，cls loss 对 person/car 的预测位置屏蔽。
    
  coco 数据集: GT 中没有 robot，robot cls loss 天然为 0（无正样本），
    但模型在 coco 图片预测出的 robot 会被当作负样本惩罚（一般可接受，因为 coco
    图片里确实没有机器人，但保险起见也屏蔽）。
    
实现方式：
  在 cls loss 计算后，按每张图的 source 生成 class_ignore_mask，
  将对应位置的 cls loss 置 0，再重新归一化。
  
  bbox/seg loss 本身依赖 fg_mask（assigner 分配的正样本），
  由于被屏蔽的类别在 GT 中就不存在，fg_mask 里天然不会包含它们，
  因此 bbox/seg loss 不需要额外处理。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import make_anchors


class SourceAwareSegLoss(v8SegmentationLoss):
    """
    数据集感知的分割 Loss。
    
    在每个 batch 中，根据图片来源动态屏蔽特定类别的 cls loss。
    
    Args:
        model: YOLOSegModel（需要已完成一次 forward 以初始化 Segment head）
        source_class_map: 各数据集来源实际包含的类别 id，格式如:
            {'coco': [0, 2], 'robot': [1]}
        tal_topk: TaskAlignedAssigner 的 topk 参数
    """

    def __init__(self, model: nn.Module, source_class_map: dict[str, list[int]], tal_topk: int = 10):
        # 找到真实的 Segment head（通过 detect 属性或直接搜索）
        super().__init__(self._unwrap_model(model), tal_topk)
        self.source_class_map = source_class_map  # {'coco':[0,2], 'robot':[1]}

    @staticmethod
    def _unwrap_model(model):
        """
        ultralytics loss 需要一个有 .model[-1] / .args 的模型对象。
        这里做适配，包装成 loss 需要的样子。
        """
        return _ModelAdapter(model)

    def loss(self, preds: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        在原始 loss 基础上，对 cls loss 做数据集感知屏蔽。
        
        batch 新增字段:
          sources: list[str], 长度=batch_size, 每张图的数据集来源
        """
        sources = batch.get("sources", None)

        if sources is None:
            # 没有来源信息，走原始 loss
            return super().loss(preds, batch)

        # 使用 hooks 临时替换 cls loss 计算以注入屏蔽
        return self._loss_with_ignore(preds, batch, sources)

    def _loss_with_ignore(self, preds: dict, batch: dict, sources: list[str]):
        """带忽略屏蔽的 loss 计算"""
        pred_masks = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        proto = preds["proto"]
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl

        # 1. 在计算之前，构建 cls ignore mask（每张图哪些类别不参与 cls loss）
        #    ignore_cls_mask[i] = set of class ids to ignore for image i
        ignore_cls_per_img = self._build_ignore_cls(sources)

        # 2. 目标处理（与原版相同）
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            preds["feats"][0].shape[2:], device=self.device, dtype=dtype
        ) * self.stride[0]

        # 3. 按 source 过滤 GT（robot 图片不包含 person/car 的 GT，coco 不含 robot GT）
        #    这里 GT 本身就没有这些类，无需额外过滤；但为安全起见做一次显式过滤
        targets = self._filter_targets(batch, sources, batch_size)
        targets_proc = self.preprocess(targets.to(self.device), batch_size,
                                        scale_tensor=imgsz[[1, 0, 1, 0]])

        gt_labels, gt_bboxes = targets_proc.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 4. Cls loss（核心屏蔽点）
        raw_cls_loss = self.bce(pred_scores, target_scores.to(dtype))
        # 应用 ignore mask：对 robot 图片屏蔽 person/car 预测，对 coco 图片屏蔽 robot 预测
        raw_cls_loss = self._apply_cls_ignore(raw_cls_loss, ignore_cls_per_img)
        loss[2] = raw_cls_loss.sum() / target_scores_sum

        # 5. Bbox + DFL loss
        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum,
                fg_mask, imgsz, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        # 6. Seg loss
        if fg_mask.sum():
            masks = batch["masks"].to(self.device).float()
            _, _, mask_h, mask_w = proto.shape
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes,
                batch["batch_idx"].view(-1, 1), proto, pred_masks, imgsz,
            )
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[1] *= self.hyp.box

        total = loss.sum()
        return total * batch_size, loss.detach()

    def _build_ignore_cls(self, sources: list[str]) -> list[set]:
        """
        为每张图计算需要忽略的类别集合。
        逻辑：该数据集实际包含的类 = source_class_map[source]
              需要忽略的类 = 全部类别 - 该数据集包含的类
        """
        all_cls = set(range(self.nc))
        ignore_per_img = []
        for src in sources:
            valid_cls = set(self.source_class_map.get(src, list(all_cls)))
            ignore_cls = all_cls - valid_cls
            ignore_per_img.append(ignore_cls)
        return ignore_per_img

    def _apply_cls_ignore(
        self,
        cls_loss: torch.Tensor,    # (B, num_anchors, nc)
        ignore_cls_per_img: list[set],
    ) -> torch.Tensor:
        """
        将需要忽略的类别位置的 cls loss 置 0。
        只屏蔽「被忽略类别的预测」对负样本 loss 的贡献。
        正样本（fg_mask=True 的位置）对应的 cls 已由 assigner 处理，
        由于 GT 中没有这些类，它们不会出现在正样本中，无需单独处理。
        """
        cls_loss = cls_loss.clone()
        B = cls_loss.shape[0]
        for i, ignore_cls in enumerate(ignore_cls_per_img):
            if ignore_cls:
                cls_ids = list(ignore_cls)
                # 第 i 张图：所有 anchor 上 ignore 类别的 loss 置 0
                cls_loss[i, :, cls_ids] = 0.0
        return cls_loss

    def _filter_targets(self, batch: dict, sources: list[str], batch_size: int) -> torch.Tensor:
        """
        根据数据集来源过滤 GT targets（本来 GT 就不含被屏蔽的类，这里做安全保障）。
        返回格式: (N, 6) [batch_idx, cls, cx, cy, w, h]
        """
        all_cls = set(range(self.nc))
        batch_idx = batch["batch_idx"]
        cls = batch["cls"]
        bboxes = batch["bboxes"]

        if len(cls) == 0:
            return torch.zeros(0, 6, device=cls.device)

        keep = []
        for j in range(len(cls)):
            img_i = int(batch_idx[j].item())
            src = sources[img_i] if img_i < len(sources) else None
            if src is not None:
                valid_cls = set(self.source_class_map.get(src, list(all_cls)))
                if int(cls[j].item()) not in valid_cls:
                    continue  # 过滤掉不属于该数据集的 GT（理论上不会出现）
            keep.append(j)

        if len(keep) == 0:
            return torch.zeros(0, 6, device=cls.device)

        keep = torch.tensor(keep, dtype=torch.long, device=cls.device)
        return torch.cat([
            batch_idx[keep].float().unsqueeze(1),
            cls[keep].float().unsqueeze(1),
            bboxes[keep],
        ], dim=1)


class _ModelAdapter:
    class _FakeArgs:
        box = 7.5
        cls = 0.5
        dfl = 1.5
        overlap_mask = False

    def __init__(self, model):
        self._model = model
        self._detect = self._find_detect(model)
        self.args = self._FakeArgs()

        # 从 model.cfg 里读 hyp
        if hasattr(model, 'cfg') and 'hyp' in model.cfg:
            hyp = model.cfg['hyp']
            self.args.box = hyp.get('box', 7.5)
            self.args.cls = hyp.get('cls', 0.5)
            self.args.dfl = hyp.get('dfl', 1.5)
            self.args.overlap_mask = hyp.get('overlap_mask', False)

        # 伪装成 model.model 列表（ultralytics loss 会访问 model.model[-1]）
        self.model = _FakeModelList(self._detect)

    def _find_detect(self, model):
        """找到模型最后一个 _LazySegment 里的 Segment"""
        from model.yolo_seg import _LazySegment
        for m in reversed(list(model.model)):
            if isinstance(m, _LazySegment) and m.seg is not None:
                return m.seg

    def parameters(self):
        return self._model.parameters()


class _FakeModelList:
    """伪装成 list，让 model.model[-1] 返回 detect head"""
    def __init__(self, detect):
        self._detect = detect

    def __getitem__(self, idx):
        if idx == -1:
            return self._detect
