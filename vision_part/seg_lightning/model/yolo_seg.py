"""
model/yolo_seg.py
从 YAML config 构建 YOLO11s-seg，复用 ultralytics 基础模块。

网络层格式: [from, repeats, module, args]
  from:    -1=上一层, 整数=指定层索引, 列表=多层concat
  repeats: 模块重复次数（受 depth_mult 缩放）
  module:  模块名称字符串
  args:    模块参数（通道数受 width_mult 缩放）
"""

import math
import yaml
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, C3k2, SPPF, C2PSA, Concat, Segment


def make_div(x, d=8):
    return math.ceil(x / d) * d


def _scale_ch(c, width, max_ch):
    """按 width_mult 缩放通道数"""
    return min(make_div(c * width), max_ch)


class YOLOSegModel(nn.Module):
    """
    YOLO11s-seg 模型，从 YAML 配置动态构建。
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.nc = cfg["nc"]

        scale_name = cfg.get("scale", "s")
        depth, width, max_ch = cfg["scales"][scale_name]

        # 解析所有层
        self.module_list, self.froms, self.saves = self._parse(
            cfg["backbone"] + cfg["head"], depth, width, max_ch
        )
        self.model = nn.ModuleList(self.module_list)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------
    def _parse(self, layers_cfg, depth, width, max_ch):
        """
        解析层配置，返回 (module_list, froms, saves_set)
        ch: ch[0]=3(RGB), ch[i+1] = 第 i 层输出通道
        from_=-1 表示上一层，from_=k (k>=0) 表示第 k 层的输出
        对应关系: ch[i] = layer_{i-1} 输出，所以 from_=k -> ch[k+1], from_=-1 -> ch[i]
        """
        ch = [3]
        modules = []
        froms = []
        saves = set()

        def get_ch(f, i):
            """获取 from 索引对应的通道数"""
            return ch[i] if f == -1 else ch[f + 1]

        for i, (from_, n, name, args) in enumerate(layers_cfg):
            # 输入通道
            if isinstance(from_, int):
                c_in = get_ch(from_, i)
                branch_ch = None
            else:
                branch_ch = [get_ch(f, i) for f in from_]
                c_in = sum(branch_ch)

            # 构建层 + 推断输出通道
            layer, c_out = self._make_layer(
                name, c_in, branch_ch, n, args, depth, width, max_ch
            )
            layer.i = i
            layer.f = from_

            modules.append(layer)
            froms.append(from_)
            ch.append(c_out)

            # 记录哪些层需要缓存（被后续层直接引用）
            refs = from_ if isinstance(from_, list) else [from_]
            for f in refs:
                if f >= 0:
                    saves.add(f)

        return modules, froms, saves

    def _make_layer(self, name, c_in, branch_ch, n, args, depth, width, max_ch):
        """构建单个层，返回 (module, out_channels)"""

        if name == "Conv":
            c_out = _scale_ch(args[0], width, max_ch)
            layer = Conv(c_in, c_out, *args[1:])

        elif name == "C3k2":
            # args: [c_out, c3k_bool, e_float]
            c_out = _scale_ch(args[0], width, max_ch)
            n = max(round(n * depth), 1)
            c3k = args[1] if len(args) > 1 else False
            e = args[2] if len(args) > 2 else 0.5
            layer = C3k2(c_in, c_out, n=n, c3k=c3k, e=e)

        elif name == "SPPF":
            c_out = _scale_ch(args[0], width, max_ch)
            k = args[1] if len(args) > 1 else 5
            layer = SPPF(c_in, c_out, k=k)

        elif name == "C2PSA":
            c_out = _scale_ch(args[0], width, max_ch)
            n = max(round(n * depth), 1)
            layer = C2PSA(c_in, c_out, n=n)

        elif name == "Upsample":
            # yaml 里 None 会被读成字符串 'None'
            size = None if args[0] in (None, "None") else args[0]
            scale = args[1] if size is None else None
            layer = nn.Upsample(size=size, scale_factor=scale, mode=args[2])
            c_out = c_in

        elif name == "Concat":
            layer = Concat(dimension=args[0])
            c_out = c_in  # c_in 已是 sum

        elif name == "Segment":
            # branch_ch 是各 P 层的实际通道，传给 Segment
            seg_ch = tuple(branch_ch)
            layer = _LazySegment(self.nc, nm=args[1], npr=args[2], ch_hint=seg_ch)
            c_out = c_in  # 不影响后续层通道推断

        else:
            raise ValueError(f"Unknown module: {name}")

        return layer, c_out

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, x):
        saved = {}  # {layer_idx: output_tensor}
        y = x

        for i, (m, from_) in enumerate(zip(self.model, self.froms)):
            # 准备输入
            if isinstance(from_, list):
                inp = [y if f == -1 else saved[f] for f in from_]
            else:
                inp = y if from_ == -1 else saved[from_]

            y = m(inp) if isinstance(inp, list) else m(inp)

            if i in self.saves:
                saved[i] = y

        return y

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @property
    def detect(self):
        """返回 Segment 检测头（供 loss 访问 stride/nc 等）"""
        for m in reversed(self.model):
            if isinstance(m, (_LazySegment,)):
                return m.seg  # 训练开始后已初始化
        return None


class _LazySegment(nn.Module):
    """
    延迟初始化的 Segment head。
    Segment 需要知道各尺度输入通道，在第一次 forward 时根据实际 tensor 初始化。
    """

    def __init__(self, nc, nm=32, npr=256, ch_hint=None):
        super().__init__()
        self.nc = nc
        self.nm = nm
        self.npr = npr
        self.ch_hint = ch_hint  # 预先知道的通道数（可选）
        self.seg: Segment | None = None

    def forward(self, x: list):
        if self.seg is None:
            ch = self.ch_hint if self.ch_hint else tuple(f.shape[1] for f in x)
            self.seg = Segment(self.nc, self.nm, self.npr, ch=ch).to(x[0].device)
            # 计算 stride：输入图像尺寸 / 各特征图尺寸
            # 第一个特征图分辨率最高（P3），stride = img_size / feat_size
            # 这里用运行时的实际 tensor 尺寸推断
            img_size = x[0].shape[-1] * 8  # P3的步长通常是8（640/80=8）
            strides = [img_size / feat.shape[-1] for feat in x]
            self.seg.stride = torch.tensor(strides, device=x[0].device)
        return self.seg(x)


def build_model_from_config(cfg_path: str) -> YOLOSegModel:
    """工厂函数：从 YAML 配置文件构建模型"""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return YOLOSegModel(cfg)
