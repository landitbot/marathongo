import os
import cv2
import glob
import math
import shutil
import random
import numpy as np
from pathlib import Path


# =========================
# 1. 路径配置
# =========================
SRC_DATASET = "/path/to/data_5"
BG_DIR = "/path/to/bg"
OUT_DATASET = "/path/to/data_6"

ROBOT_CLASS_ID = 0
SCALE_RANGE = (0.3, 0.5)

# 每张原图生成多少张候选合成图（这里只是候选上限，真正数量由目标比例控制）
NUM_SYNTH_PER_IMAGE = 10

# 每张合成图随机贴几个目标
MIN_INSTANCES_PER_IMAGE = 1
MAX_INSTANCES_PER_IMAGE = 2

# 阴影参数
SHADOW_OFFSET_RANGE = (6, 15)         # 阴影偏移像素
SHADOW_BLUR_RANGE = (11, 17)          # 高斯核，必须是奇数
SHADOW_DARKNESS_RANGE = (0.25, 0.45)  # 阴影强度，越大越暗

# 混合比例：70% 原图 + 30% copypaste
ORIG_RATIO = 0.7
CP_RATIO = 0.3

# 前景颜色调整参数
COLOR_MATCH_STRENGTH = 0.22   # 原来是 0.5，这里降低，避免太像背景
SATURATION_GAIN_RANGE = (1.08, 1.22)  # 提升一点饱和度
SEPARATION_DELTA_RANGE = (10, 28)     # 前景亮度与背景均值拉开一点


# =========================
# 2. 工具函数
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def list_images(img_dir):
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    return sorted(files)


def load_yolo_seg_label(label_path, img_w, img_h):
    """
    读取 YOLO segmentation 格式标注
    每一行: class_id x1 y1 x2 y2 x3 y3 ...
    坐标是归一化的
    """
    results = []
    if not os.path.exists(label_path):
        return results

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        cls_id = int(float(parts[0]))
        coords = list(map(float, parts[1:]))

        if len(coords) % 2 != 0:
            continue

        pts = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_w
            y = coords[i + 1] * img_h
            pts.append([x, y])

        poly_abs = np.array(pts, dtype=np.float32)
        results.append({
            "cls": cls_id,
            "poly_abs": poly_abs
        })

    return results


def polygon_to_mask(poly, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = poly.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_instance_patch(image, poly):
    """
    根据 polygon 从原图中抠出实例前景与mask
    返回:
        fg
        patch_mask
        cropped_poly   # 相对于 patch 左上角的 polygon
    """
    h, w = image.shape[:2]
    mask = polygon_to_mask(poly, h, w)

    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    if bw <= 1 or bh <= 1:
        return None, None, None

    patch = image[y:y+bh, x:x+bw].copy()
    patch_mask = mask[y:y+bh, x:x+bw].copy()

    fg = cv2.bitwise_and(patch, patch, mask=patch_mask)

    cropped_poly = poly.copy()
    cropped_poly[:, 0] -= x
    cropped_poly[:, 1] -= y

    return fg, patch_mask, cropped_poly


def resize_patch_and_poly(patch, mask, poly, scale):
    h, w = patch.shape[:2]
    new_w = max(2, int(w * scale))
    new_h = max(2, int(h * scale))

    patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    poly_resized = poly.copy()
    poly_resized[:, 0] *= (new_w / w)
    poly_resized[:, 1] *= (new_h / h)

    return patch_resized, mask_resized, poly_resized


def match_foreground_to_background(patch, patch_mask, bg_roi, strength=0.22, eps=1e-6):
    """
    轻度颜色匹配：只让前景稍微靠近背景，不要完全贴合
    strength:
        0.0 = 不调整
        1.0 = 完全匹配
    """
    patch_f = patch.astype(np.float32).copy()
    bg_f = bg_roi.astype(np.float32).copy()

    fg_mask = patch_mask > 0
    if fg_mask.sum() < 10:
        return patch.copy()

    matched = patch_f.copy()

    for c in range(3):
        fg_pixels = patch_f[:, :, c][fg_mask]
        bg_pixels = bg_f[:, :, c].reshape(-1)

        fg_mean = fg_pixels.mean()
        fg_std = fg_pixels.std() + eps

        bg_mean = bg_pixels.mean()
        bg_std = bg_pixels.std() + eps

        channel = matched[:, :, c]
        channel_fg = channel[fg_mask]

        fully_matched = (channel_fg - fg_mean) / fg_std * bg_std + bg_mean
        channel_fg = (1 - strength) * channel_fg + strength * fully_matched

        channel[fg_mask] = channel_fg
        matched[:, :, c] = channel

    return np.clip(matched, 0, 255).astype(np.uint8)


def separate_foreground_from_background(patch, patch_mask, bg_roi):
    """
    在轻度颜色匹配后，再把前景和背景稍微“拉开”一点：
    1) 轻微提升前景饱和度
    2) 让前景亮度与背景均值拉开一点
    """
    fg_mask = patch_mask > 0
    if fg_mask.sum() < 10:
        return patch.copy()

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_hsv = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 取前景像素与背景整体均值
    fg_v_mean = hsv[:, :, 2][fg_mask].mean()
    bg_v_mean = bg_hsv[:, :, 2].mean()

    fg_s = hsv[:, :, 1]
    fg_v = hsv[:, :, 2]

    # 1) 稍微提高前景饱和度，让目标更“立住”
    sat_gain = random.uniform(*SATURATION_GAIN_RANGE)
    fg_s[fg_mask] = np.clip(fg_s[fg_mask] * sat_gain, 0, 255)

    # 2) 亮度和背景拉开一点
    delta = random.uniform(*SEPARATION_DELTA_RANGE)

    # 如果前景本来比背景亮，就再亮一点；否则就再暗一点
    if fg_v_mean >= bg_v_mean:
        target_shift = delta
    else:
        target_shift = -delta

    fg_v[fg_mask] = np.clip(fg_v[fg_mask] + target_shift, 0, 255)

    hsv[:, :, 1] = fg_s
    hsv[:, :, 2] = fg_v

    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted


def harmonize_but_keep_separation(patch, patch_mask, bg_roi):
    """
    先做轻度匹配，再拉开一点颜色/亮度差异
    """
    patch_matched = match_foreground_to_background(
        patch, patch_mask, bg_roi, strength=COLOR_MATCH_STRENGTH
    )
    patch_adjusted = separate_foreground_from_background(
        patch_matched, patch_mask, bg_roi
    )
    return patch_adjusted


def add_shadow_to_background(
    bg,
    patch_mask,
    x_offset,
    y_offset,
    shadow_offset=(12, 12),
    blur_ksize=15,
    darkness=0.45
):
    bg_h, bg_w = bg.shape[:2]
    ph, pw = patch_mask.shape[:2]

    shadow_x = x_offset + shadow_offset[0]
    shadow_y = y_offset + shadow_offset[1]

    if shadow_x < 0 or shadow_y < 0 or shadow_x + pw > bg_w or shadow_y + ph > bg_h:
        return bg

    roi = bg[shadow_y:shadow_y + ph, shadow_x:shadow_x + pw].copy()

    kernel = np.ones((3, 3), np.uint8)
    shadow_mask = cv2.erode(patch_mask, kernel, iterations=1)

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    shadow_alpha = shadow_mask.astype(np.float32) / 255.0
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (blur_ksize, blur_ksize), 0)

    shadow_alpha = shadow_alpha * darkness
    shadow_alpha = np.clip(shadow_alpha, 0.0, 1.0)

    shadow_alpha_3 = shadow_alpha[:, :, None]

    dark_roi = roi.astype(np.float32) * (1 - shadow_alpha_3)
    dark_roi = np.clip(dark_roi, 0, 255).astype(np.uint8)

    bg[shadow_y:shadow_y + ph, shadow_x:shadow_x + pw] = dark_roi
    return bg


def sample_shadow_params():
    dx = random.randint(*SHADOW_OFFSET_RANGE)
    dy = random.randint(*SHADOW_OFFSET_RANGE)

    blur_ksize = random.randint(*SHADOW_BLUR_RANGE)
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    darkness = random.uniform(*SHADOW_DARKNESS_RANGE)
    return (dx, dy), blur_ksize, darkness


def masks_overlap_too_much(occupied_mask, patch_mask, x_offset, y_offset, max_iou_like=0.3):
    ph, pw = patch_mask.shape[:2]
    new_region = np.zeros_like(occupied_mask, dtype=np.uint8)
    new_region[y_offset:y_offset + ph, x_offset:x_offset + pw] = (patch_mask > 0).astype(np.uint8)

    inter = ((occupied_mask > 0) & (new_region > 0)).sum()
    area_new = (new_region > 0).sum()

    if area_new == 0:
        return True

    overlap_ratio = inter / area_new
    return overlap_ratio > max_iou_like


def paste_instance(bg, patch, patch_mask, poly, x_offset, y_offset, blur_ksize=9):
    bg_h, bg_w = bg.shape[:2]
    ph, pw = patch.shape[:2]

    if x_offset < 0 or y_offset < 0 or x_offset + pw > bg_w or y_offset + ph > bg_h:
        return None, None

    roi = bg[y_offset:y_offset + ph, x_offset:x_offset + pw].copy()

    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.erode(patch_mask, kernel, iterations=1)

    alpha = clean_mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    alpha_3 = alpha[:, :, None]

    blended = patch.astype(np.float32) * alpha_3 + roi.astype(np.float32) * (1 - alpha_3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    bg[y_offset:y_offset + ph, x_offset:x_offset + pw] = blended

    pasted_poly = poly.copy()
    pasted_poly[:, 0] += x_offset
    pasted_poly[:, 1] += y_offset

    return bg, pasted_poly


def poly_to_yolo_seg_line(cls_id, poly_abs, img_w, img_h):
    coords = []
    for x, y in poly_abs:
        xn = np.clip(x / img_w, 0.0, 1.0)
        yn = np.clip(y / img_h, 0.0, 1.0)
        coords.append(f"{xn:.6f}")
        coords.append(f"{yn:.6f}")

    return str(cls_id) + " " + " ".join(coords)


def random_bg_image(bg_dir):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(bg_dir, ext)))
    if not files:
        return None
    return random.choice(files)


def safe_copy_original_split(split):
    """
    把原始 data_5 的图片和标签完整复制到新数据集
    为避免和合成图重名，原图统一加前缀 orig_
    """
    src_img_dir = os.path.join(SRC_DATASET, "images", split)
    src_lbl_dir = os.path.join(SRC_DATASET, "labels", split)

    out_img_dir = os.path.join(OUT_DATASET, "images", split)
    out_lbl_dir = os.path.join(OUT_DATASET, "labels", split)

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    img_files = list_images(src_img_dir)
    copied_count = 0

    for img_path in img_files:
        stem = Path(img_path).stem
        suffix = Path(img_path).suffix
        src_label_path = os.path.join(src_lbl_dir, stem + ".txt")

        new_name = f"orig_{stem}"
        dst_img_path = os.path.join(out_img_dir, new_name + suffix)
        dst_lbl_path = os.path.join(out_lbl_dir, new_name + ".txt")

        shutil.copy2(img_path, dst_img_path)

        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_lbl_path)
        else:
            # 没有标签也给一个空文件，方便训练时成对存在
            open(dst_lbl_path, "w").close()

        copied_count += 1

    print(f"[{split}] copied original images: {copied_count}")
    return copied_count


def generate_copypaste_split(split, target_cp_count):
    """
    生成指定数量的 copypaste 图
    """
    src_img_dir = os.path.join(SRC_DATASET, "images", split)
    src_lbl_dir = os.path.join(SRC_DATASET, "labels", split)

    out_img_dir = os.path.join(OUT_DATASET, "images", split)
    out_lbl_dir = os.path.join(OUT_DATASET, "labels", split)

    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    img_files = list_images(src_img_dir)
    print(f"[{split}] found {len(img_files)} source images for cp generation")
    print(f"[{split}] target cp count = {target_cp_count}")

    if len(img_files) == 0:
        return 0

    save_idx = 0

    # 为了能生成足够多的 cp 图，循环使用源图
    while save_idx < target_cp_count:
        random.shuffle(img_files)

        for img_path in img_files:
            if save_idx >= target_cp_count:
                break

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            stem = Path(img_path).stem
            label_path = os.path.join(src_lbl_dir, stem + ".txt")

            anns = load_yolo_seg_label(label_path, w, h)
            robot_anns = [ann for ann in anns if ann["cls"] == ROBOT_CLASS_ID]
            if len(robot_anns) == 0:
                continue

            # 每张源图可以尝试生成多张
            for synth_id in range(NUM_SYNTH_PER_IMAGE):
                if save_idx >= target_cp_count:
                    break

                bg_path = random_bg_image(BG_DIR)
                if bg_path is None:
                    raise RuntimeError("No background images found in BG_DIR.")

                composed = cv2.imread(bg_path)
                if composed is None:
                    continue

                bg_h, bg_w = composed.shape[:2]

                num_instances = random.randint(MIN_INSTANCES_PER_IMAGE, MAX_INSTANCES_PER_IMAGE)

                label_lines = []
                occupied_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
                pasted_count = 0

                for _ in range(num_instances):
                    ann = random.choice(robot_anns)
                    poly = ann["poly_abs"]

                    patch, patch_mask, cropped_poly = extract_instance_patch(img, poly)
                    if patch is None:
                        continue

                    scale = random.uniform(*SCALE_RANGE)
                    patch, patch_mask, cropped_poly = resize_patch_and_poly(
                        patch, patch_mask, cropped_poly, scale
                    )

                    ph, pw = patch.shape[:2]
                    if ph >= bg_h or pw >= bg_w:
                        continue

                    placed = False
                    for _try in range(30):
                        x_offset = random.randint(0, bg_w - pw)
                        y_offset = random.randint(0, bg_h - ph)

                        if masks_overlap_too_much(
                            occupied_mask, patch_mask, x_offset, y_offset, max_iou_like=0.3
                        ):
                            continue

                        roi = composed[y_offset:y_offset + ph, x_offset:x_offset + pw].copy()
                        if roi.shape[:2] != patch.shape[:2]:
                            continue

                        # 关键修改：
                        # 不再让前景过度贴近背景，而是“轻度融合 + 稍微拉开”
                        patch_adjusted = harmonize_but_keep_separation(
                            patch, patch_mask, roi
                        )

                        shadow_offset, blur_ksize, darkness = sample_shadow_params()
                        composed = add_shadow_to_background(
                            composed,
                            patch_mask,
                            x_offset,
                            y_offset,
                            shadow_offset=shadow_offset,
                            blur_ksize=blur_ksize,
                            darkness=darkness
                        )

                        composed, pasted_poly = paste_instance(
                            composed,
                            patch_adjusted,
                            patch_mask,
                            cropped_poly,
                            x_offset,
                            y_offset
                        )
                        if composed is None:
                            continue

                        occupied_mask[y_offset:y_offset + ph, x_offset:x_offset + pw] = np.maximum(
                            occupied_mask[y_offset:y_offset + ph, x_offset:x_offset + pw],
                            (patch_mask > 0).astype(np.uint8) * 255
                        )

                        line = poly_to_yolo_seg_line(ROBOT_CLASS_ID, pasted_poly, bg_w, bg_h)
                        label_lines.append(line)

                        pasted_count += 1
                        placed = True
                        break

                    if not placed:
                        continue

                if pasted_count == 0:
                    continue

                out_name = f"cp_{stem}_{save_idx:06d}"
                out_img_path = os.path.join(out_img_dir, out_name + ".jpg")
                out_lbl_path = os.path.join(out_lbl_dir, out_name + ".txt")

                cv2.imwrite(out_img_path, composed)

                with open(out_lbl_path, "w") as f:
                    for line in label_lines:
                        f.write(line + "\n")

                save_idx += 1

                if save_idx % 50 == 0 or save_idx == target_cp_count:
                    print(f"[{split}] generated cp: {save_idx}/{target_cp_count}")

    print(f"[{split}] done, generated cp images = {save_idx}")
    return save_idx


def build_mix_dataset(split):
    """
    构建单个 split:
    - 先复制全部原始数据
    - 再按 70/30 比例推算需要生成多少 cp 图
    """
    orig_count = safe_copy_original_split(split)

    if orig_count == 0:
        print(f"[{split}] no original data found.")
        return

    # 已知：
    # final_total = orig_count / 0.7
    # cp_count = final_total * 0.3 = orig_count * 0.3 / 0.7
    target_cp_count = int(round(orig_count * CP_RATIO / ORIG_RATIO))

    print(f"[{split}] original count = {orig_count}")
    print(f"[{split}] target cp count = {target_cp_count}")
    print(f"[{split}] expected ratio ~ {ORIG_RATIO:.0%}/{CP_RATIO:.0%}")

    cp_count = generate_copypaste_split(split, target_cp_count)

    total = orig_count + cp_count
    real_orig_ratio = orig_count / total if total > 0 else 0
    real_cp_ratio = cp_count / total if total > 0 else 0

    print(f"[{split}] final total = {total}")
    print(f"[{split}] real ratio = original:{real_orig_ratio:.4f}, cp:{real_cp_ratio:.4f}")


def prepare_output_dirs():
    reset_dir(OUT_DATASET)
    ensure_dir(os.path.join(OUT_DATASET, "images", "train"))
    ensure_dir(os.path.join(OUT_DATASET, "images", "val"))
    ensure_dir(os.path.join(OUT_DATASET, "labels", "train"))
    ensure_dir(os.path.join(OUT_DATASET, "labels", "val"))


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    prepare_output_dirs()

    build_mix_dataset("train")
    build_mix_dataset("val")

    print(f"Done. Mixed dataset saved to: {OUT_DATASET}")