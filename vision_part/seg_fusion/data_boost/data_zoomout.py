import os
import cv2
import glob
import random
import shutil
import numpy as np
from pathlib import Path


# =========================================================
# 配置区
# =========================================================
SRC_ROOT = "/path/to/data_5"
DST_ROOT = "/path/to/data_6_mixed_zoomout"

# 要处理哪些子集
SUBSETS = ["train","val"]          # 你后面如果要加 val，可改成 ["train", "val"]

RANDOM_SEED = 42

# -------------------------
# 混合模式
# -------------------------
# "append_aug"
#   保留全部原始数据，再额外追加增强数据
#
# "fixed_ratio"
#   按目标总数做固定比例混合，例如 70% 原始 + 30% 增强
MIX_MODE = "append_aug"

# 当 MIX_MODE = "append_aug" 时使用：
# 生成增强数据数量 = 原始数据数量 * AUG_RATIO
# 例如 0.3 表示生成 30% 数量的增强图
AUG_RATIO = 0.30

# 当 MIX_MODE = "fixed_ratio" 时使用：
# 最终数据集里，原始占比 / 增强占比
ORIG_RATIO = 0.70
AUG_RATIO_FIXED = 0.30

# fixed_ratio 模式下，最终总数是否与原始数据集一样大
# True  -> 最终总数 = 原始数量
# False -> 使用 TARGET_TOTAL
KEEP_TOTAL_SAME_AS_ORIGINAL = True
TARGET_TOTAL = 3000


# -------------------------
# zoom-out 参数
# -------------------------
SCALE_RANGE = (0.35, 0.55)
PLACEMENT_MODE = "random"

# 背景模式可选：
# "reflect"
# "blur_reflect"
# "mean_color"
# "random_crop_fill"
# "blurred_scene_fill"
BG_MODE = "random_crop_fill"

PATCH_SIZE = 96
FEATHER_SIZE = 8

JPG_QUALITY = 95
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# =========================================================
# 通用工具函数
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_images(img_dir):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(img_dir, f"*{ext.upper()}")))
    return sorted(paths)


def read_label_lines(label_path):
    if not os.path.exists(label_path):
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def copy_file(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def is_float_list(tokens):
    try:
        [float(x) for x in tokens]
        return True
    except Exception:
        return False


# =========================================================
# 背景构造
# =========================================================
def build_reflect(img):
    h, w = img.shape[:2]
    bg = cv2.copyMakeBorder(
        img, h, h, w, w,
        borderType=cv2.BORDER_REFLECT_101
    )
    return bg[h:2*h, w:2*w]


def build_blur_reflect(img):
    bg = build_reflect(img)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=15, sigmaY=15)
    return bg


def build_mean_color(img):
    mean_color = img.mean(axis=(0, 1)).astype(np.uint8)
    bg = np.zeros_like(img)
    bg[:] = mean_color
    return bg


def build_random_crop_fill(img, patch_size=96):
    h, w = img.shape[:2]
    bg = np.zeros_like(img)

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            ph = min(patch_size, h - y)
            pw = min(patch_size, w - x)

            sy = random.randint(0, max(0, h - ph))
            sx = random.randint(0, max(0, w - pw))

            bg[y:y+ph, x:x+pw] = img[sy:sy+ph, sx:sx+pw]

    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=6, sigmaY=6)
    return bg


def build_blurred_scene_fill(img):
    h, w = img.shape[:2]
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=18, sigmaY=18).astype(np.float32)
    grad = np.linspace(0.92, 1.05, h, dtype=np.float32).reshape(h, 1, 1)
    bg = bg * grad
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    return bg


def build_background(img, mode="blur_reflect", patch_size=96):
    if mode == "reflect":
        return build_reflect(img)
    elif mode == "blur_reflect":
        return build_blur_reflect(img)
    elif mode == "mean_color":
        return build_mean_color(img)
    elif mode == "random_crop_fill":
        return build_random_crop_fill(img, patch_size=patch_size)
    elif mode == "blurred_scene_fill":
        return build_blurred_scene_fill(img)
    else:
        raise ValueError(f"未知 BG_MODE: {mode}")


# =========================================================
# 融合
# =========================================================
def hard_paste(base, fg, x0, y0):
    out = base.copy()
    h, w = fg.shape[:2]
    out[y0:y0+h, x0:x0+w] = fg
    return out


def feather_blend(base, fg, x0, y0, feather=8):
    out = base.copy().astype(np.float32)
    h, w = fg.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w]
    dist_left = xx
    dist_right = w - 1 - xx
    dist_top = yy
    dist_bottom = h - 1 - yy
    edge_dist = np.minimum.reduce([dist_left, dist_right, dist_top, dist_bottom]).astype(np.float32)

    alpha = np.clip(edge_dist / max(feather, 1), 0, 1)[..., None]

    roi = out[y0:y0+h, x0:x0+w]
    roi[:] = roi * (1 - alpha) + fg.astype(np.float32) * alpha

    inner = feather + 2
    if h > 2 * inner and w > 2 * inner:
        roi[inner:h-inner, inner:w-inner] = fg[inner:h-inner, inner:w-inner]

    out[y0:y0+h, x0:x0+w] = roi
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================
# 标签变换
# =========================================================
def parse_and_transform_label_line(line, scale, x0, y0, canvas_w, canvas_h):
    tokens = line.strip().split()
    if len(tokens) < 2 or not is_float_list(tokens[1:]):
        return None

    cls_id = tokens[0]
    nums = [float(x) for x in tokens[1:]]

    # YOLO bbox
    if len(nums) == 4:
        cx, cy, bw, bh = nums

        cx_new = (cx * canvas_w * scale + x0) / canvas_w
        cy_new = (cy * canvas_h * scale + y0) / canvas_h
        bw_new = bw * scale
        bh_new = bh * scale

        cx_new = clamp(cx_new)
        cy_new = clamp(cy_new)
        bw_new = clamp(bw_new)
        bh_new = clamp(bh_new)

        if bw_new <= 1e-6 or bh_new <= 1e-6:
            return None

        return f"{cls_id} {cx_new:.6f} {cy_new:.6f} {bw_new:.6f} {bh_new:.6f}"

    # YOLO segmentation polygon
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        new_points = []
        for i in range(0, len(nums), 2):
            x = nums[i]
            y = nums[i + 1]

            x_new = (x * canvas_w * scale + x0) / canvas_w
            y_new = (y * canvas_h * scale + y0) / canvas_h

            x_new = clamp(x_new)
            y_new = clamp(y_new)

            new_points.append(f"{x_new:.6f}")
            new_points.append(f"{y_new:.6f}")

        return f"{cls_id} " + " ".join(new_points)

    return None


# =========================================================
# 核心增强
# =========================================================
def make_zoomout_image(img, scale, placement_mode="random", bg_mode="blur_reflect",
                       patch_size=96, feather_size=8):
    h, w = img.shape[:2]

    canvas = build_background(img, mode=bg_mode, patch_size=patch_size)

    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    if placement_mode == "center":
        x0 = (w - small_w) // 2
        y0 = (h - small_h) // 2
    elif placement_mode == "random":
        x0 = random.randint(0, max(0, w - small_w))
        y0 = random.randint(0, max(0, h - small_h))
    else:
        raise ValueError(f"未知 PLACEMENT_MODE: {placement_mode}")

    if bg_mode in ["random_crop_fill", "blurred_scene_fill"]:
        out = feather_blend(canvas, small_img, x0, y0, feather=feather_size)
    else:
        out = hard_paste(canvas, small_img, x0, y0)

    return out, x0, y0, scale


# =========================================================
# 复制原始数据
# =========================================================
def copy_original_subset(src_root, dst_root, subset):
    src_img_dir = os.path.join(src_root, "images", subset)
    src_lbl_dir = os.path.join(src_root, "labels", subset)

    dst_img_dir = os.path.join(dst_root, "images", subset)
    dst_lbl_dir = os.path.join(dst_root, "labels", subset)

    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    img_paths = list_images(src_img_dir)

    copied = 0
    for img_path in img_paths:
        name = Path(img_path).name
        stem = Path(img_path).stem
        label_path = os.path.join(src_lbl_dir, stem + ".txt")

        copy_file(img_path, os.path.join(dst_img_dir, name))
        if os.path.exists(label_path):
            copy_file(label_path, os.path.join(dst_lbl_dir, stem + ".txt"))

        copied += 1

    return copied, img_paths


# =========================================================
# 生成增强数据并写入新数据集
# =========================================================
def generate_augmented_subset(src_root, dst_root, subset, aug_count):
    src_img_dir = os.path.join(src_root, "images", subset)
    src_lbl_dir = os.path.join(src_root, "labels", subset)

    dst_img_dir = os.path.join(dst_root, "images", subset)
    dst_lbl_dir = os.path.join(dst_root, "labels", subset)

    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    img_paths = list_images(src_img_dir)
    if not img_paths:
        print(f"[警告] 未找到图片: {src_img_dir}")
        return 0

    generated = 0

    for i in range(aug_count):
        img_path = random.choice(img_paths)
        name = Path(img_path).stem
        ext = Path(img_path).suffix
        label_path = os.path.join(src_lbl_dir, name + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 读图失败: {img_path}")
            continue

        h, w = img.shape[:2]
        scale = random.uniform(*SCALE_RANGE)

        out_img, x0, y0, scale = make_zoomout_image(
            img=img,
            scale=scale,
            placement_mode=PLACEMENT_MODE,
            bg_mode=BG_MODE,
            patch_size=PATCH_SIZE,
            feather_size=FEATHER_SIZE
        )

        # 保证文件名不重复
        out_stem = f"{name}_zoomout_{BG_MODE}_{i:06d}"
        out_img_path = os.path.join(dst_img_dir, out_stem + ext)
        out_lbl_path = os.path.join(dst_lbl_dir, out_stem + ".txt")

        if ext.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(out_img_path, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
        else:
            cv2.imwrite(out_img_path, out_img)

        label_lines = read_label_lines(label_path)
        out_label_lines = []

        for line in label_lines:
            new_line = parse_and_transform_label_line(
                line=line,
                scale=scale,
                x0=x0,
                y0=y0,
                canvas_w=w,
                canvas_h=h
            )
            if new_line is not None:
                out_label_lines.append(new_line)

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            for line in out_label_lines:
                f.write(line + "\n")

        generated += 1

        if (i + 1) % 50 == 0 or i == aug_count - 1:
            print(f"[{subset}] 已生成增强图: {i + 1}/{aug_count}")

    return generated


# =========================================================
# 可选：复制 data.yaml
# =========================================================
def copy_data_yaml(src_root, dst_root):
    src_yaml = os.path.join(src_root, "dataset.yaml")
    dst_yaml = os.path.join(dst_root, "dataset.yaml")
    if os.path.exists(src_yaml):
        copy_file(src_yaml, dst_yaml)
        print(f"已复制 dataset.yaml -> {dst_yaml}")
    else:
        print("未找到 dataset.yaml，已跳过复制。")


# =========================================================
# 主流程
# =========================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if os.path.exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)

    ensure_dir(DST_ROOT)

    copy_data_yaml(SRC_ROOT, DST_ROOT)

    for subset in SUBSETS:
        print(f"\n================ 处理子集: {subset} ================")

        # 先统计原始数量
        src_img_dir = os.path.join(SRC_ROOT, "images", subset)
        orig_img_paths = list_images(src_img_dir)
        orig_count = len(orig_img_paths)

        if orig_count == 0:
            print(f"[跳过] 子集为空: {subset}")
            continue

        print(f"原始图片数量: {orig_count}")

        # 先复制全部原始数据到新数据集
        copied_count, _ = copy_original_subset(SRC_ROOT, DST_ROOT, subset)
        print(f"已复制原始数据: {copied_count}")

        # 决定增强数量
        if MIX_MODE == "append_aug":
            aug_count = int(round(orig_count * AUG_RATIO))

        elif MIX_MODE == "fixed_ratio":
            if KEEP_TOTAL_SAME_AS_ORIGINAL:
                total_count = orig_count
            else:
                total_count = TARGET_TOTAL

            orig_target = int(round(total_count * ORIG_RATIO))
            aug_target = total_count - orig_target

            # 因为我们已经复制了全部原始图，所以这里只能“追加增强”
            # 若你一定要严格固定总数=orig_target+aug_target，需要再写“原图子采样”
            # 当前这里采用实用策略：
            # 保留全部原图 + 补 aug_target
            # 如果原图已经超过 orig_target，也不删原图
            aug_count = aug_target

            print(f"[提示] fixed_ratio 当前采用“保留全部原图 + 追加增强”的安全策略")
            print(f"目标总数参考: {total_count}, 理论增强数: {aug_target}")

        else:
            raise ValueError(f"未知 MIX_MODE: {MIX_MODE}")

        print(f"本次将生成增强数量: {aug_count}")

        generated_count = generate_augmented_subset(
            src_root=SRC_ROOT,
            dst_root=DST_ROOT,
            subset=subset,
            aug_count=aug_count
        )

        final_count = len(list_images(os.path.join(DST_ROOT, "images", subset)))

        print(f"[{subset}] 原始复制: {copied_count}")
        print(f"[{subset}] 增强生成: {generated_count}")
        print(f"[{subset}] 最终总图数: {final_count}")

    print("\n================ 全部完成 ================")
    print(f"输出新数据集目录: {DST_ROOT}")


if __name__ == "__main__":
    main()