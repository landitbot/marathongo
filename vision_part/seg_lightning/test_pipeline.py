"""
test_pipeline.py
端到端测试脚本，验证模型、数据、loss 的完整流程。
"""
import torch
from model.yolo_seg import build_model_from_config
from loss.seg_loss import SourceAwareSegLoss


def test_model():
    """测试模型构建和前向传播"""
    print("=" * 60)
    print("测试 1: 模型构建和前向传播")
    print("=" * 60)
    
    model = build_model_from_config('configs/yolo11s_seg.yaml')
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(2, 3, 640, 640)
        out = model(x)
    
    print(f"✓ 模型前向传播成功")
    print(f"  输出 keys: {list(out.keys())}")
    print(f"  boxes:  {out['boxes'].shape}")
    print(f"  scores: {out['scores'].shape}")
    print(f"  mask_coefficient: {out['mask_coefficient'].shape}")
    print(f"  proto:  {out['proto'].shape}")
    print()
    return model


def test_loss(model):
    """测试 Loss 计算"""
    print("=" * 60)
    print("测试 2: Loss 计算（数据集感知屏蔽）")
    print("=" * 60)
    
    # 初始化 loss
    model.train()
    source_class_map = {'coco': [0, 2], 'robot': [1]}
    loss_fn = SourceAwareSegLoss(model, source_class_map)
    print(f"✓ Loss 初始化成功, stride: {loss_fn.stride.tolist()}")
    
    # 构造测试 batch
    B = 4
    H, W = 640, 640
    
    # 模拟 4 张图：2 张 coco (person+car), 2 张 robot (robot)
    batch = {
        'batch_idx': torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long),
        'cls':       torch.tensor([0, 2, 0, 2, 1, 1], dtype=torch.long),
        'bboxes':    torch.rand(6, 4) * 0.5 + 0.25,  # 随机 bbox
        'masks':     torch.zeros(6, H, W),
        'sources':   ['coco', 'coco', 'robot', 'robot'],
    }
    
    imgs = torch.randn(B, 3, H, W)
    preds = model(imgs)
    
    total_loss, loss_items = loss_fn.loss(preds, batch)
    
    print(f"✓ Loss 计算成功")
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  box_loss:   {loss_items[0].item():.4f}")
    print(f"  seg_loss:   {loss_items[1].item():.4f}")
    print(f"  cls_loss:   {loss_items[2].item():.4f}")
    print(f"  dfl_loss:   {loss_items[3].item():.4f}")
    print()


def test_ignore_strategy():
    """测试 cls loss 屏蔽策略"""
    print("=" * 60)
    print("测试 3: 数据集感知屏蔽策略验证")
    print("=" * 60)
    
    model = build_model_from_config('configs/yolo11s_seg.yaml')
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 3, 640, 640))
    model.train()
    
    source_class_map = {'coco': [0, 2], 'robot': [1]}
    loss_fn = SourceAwareSegLoss(model, source_class_map)
    
    # 测试 ignore mask 构建
    sources = ['coco', 'robot', 'coco']
    ignore_cls = loss_fn._build_ignore_cls(sources)
    
    print(f"✓ 屏蔽策略验证:")
    print(f"  coco 图片忽略类别:  {ignore_cls[0]} (应为 {{1}} - robot)")
    print(f"  robot 图片忽略类别: {ignore_cls[1]} (应为 {{0, 2}} - person/car)")
    print(f"  coco 图片忽略类别:  {ignore_cls[2]} (应为 {{1}} - robot)")
    
    assert ignore_cls[0] == {1}, "coco 应忽略 robot(1)"
    assert ignore_cls[1] == {0, 2}, "robot 应忽略 person(0)/car(2)"
    print(f"✓ 屏蔽策略正确!")
    print()


def test_channel_inference():
    """测试通道推断逻辑"""
    print("=" * 60)
    print("测试 4: 网络通道推断")
    print("=" * 60)
    
    import yaml
    import math
    
    with open('configs/yolo11s_seg.yaml') as f:
        cfg = yaml.safe_load(f)
    
    depth, width, max_ch = cfg['scales']['s']
    
    def make_div(x, d=8): 
        return math.ceil(x / d) * d
    
    def sc(c): 
        return min(make_div(c * width), max_ch)
    
    # 验证关键层的通道数
    expected = {
        0: 32,    # Conv [64, 3, 2] -> 64*0.5=32
        4: 256,   # C3k2 [512, ...] -> 512*0.5=256
        10: 512,  # C2PSA [1024] -> 1024*0.5=512
        16: 128,  # C3k2 [256, ...] -> 256*0.5=128
    }
    
    ch = [3]
    layers_cfg = cfg['backbone'] + cfg['head']
    
    for i, (from_, n, name, args) in enumerate(layers_cfg):
        def get_ch(f):
            return ch[i] if f == -1 else ch[f + 1]
        
        if isinstance(from_, int):
            c_in = get_ch(from_)
        else:
            c_in = sum(get_ch(f) for f in from_)
        
        if name in ('Conv', 'C3k2', 'SPPF', 'C2PSA'):
            c_out = sc(args[0])
        else:
            c_out = c_in
        
        ch.append(c_out)
        
        if i in expected:
            assert c_out == expected[i], f"Layer {i} 通道数错误: {c_out} != {expected[i]}"
    
    print(f"✓ 通道推断正确!")
    print(f"  Layer 0 (Conv):   {ch[1]} (预期 32)")
    print(f"  Layer 4 (C3k2):   {ch[5]} (预期 256)")
    print(f"  Layer 10 (C2PSA): {ch[11]} (预期 512)")
    print(f"  Layer 16 (C3k2):  {ch[17]} (预期 128)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("YOLO11s-seg Lightning 端到端测试")
    print("=" * 60 + "\n")
    
    try:
        model = test_model()
        test_loss(model)
        test_ignore_strategy()
        test_channel_inference()
        
        print("=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
