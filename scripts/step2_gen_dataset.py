#!/usr/bin/env python3
"""
STEP 2: Tạo Dataset Blur/Sharp (Fixed v2 — Tiered Blur Distribution)
=====================================================================
Phân phối blur theo 3 tầng, mô phỏng thực tế:
  Tầng 1 — Nhẹ (60%): Kernel 3–9px  → Tay rung nhẹ, chụp nhanh
  Tầng 2 — Vừa (30%): Kernel 9–19px → Thiếu sáng, rung nhiều hơn
  Tầng 3 — Nặng(10%): Kernel 19–27px → Edge case: chụp rất tệ

Phân phối lệch về nhẹ vì:
  - Ảnh thực tế chụp thẻ thường ở điều kiện đủ sáng → blur nhẹ
  - Tầng nặng đủ để model không bị "ngợp" khi gặp blur thật,
    nhưng không chiếm đa số để không xóa thông tin
  - Kernel max 27px < chiều cao chữ 10–18px chỉ khi chữ đủ lớn
    (bài tây OK, Yugioh text nhỏ chịu được đến ~19px)

Các fix khác (từ v1):
  ✅ XÓA perspective warp khỏi pipeline
  ✅ Tăng JPEG quality: 75–95%
  ✅ Giảm noise: sigma 1.0–4.0
  ✅ Tăng pairs/image: 20

Chạy:
  python scripts/step2_gen_dataset.py
  python scripts/step2_gen_dataset.py --fold 1
  python scripts/step2_gen_dataset.py --all_folds

Input:  data/raw_gt/
Output: data/train/{sharp,blur}/ và data/test/{sharp,blur}/
"""

import cv2
import numpy as np
import random
import os
import sys
import argparse
from pathlib import Path

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
RAW_GT_DIR      = "data/raw_gt"
TRAIN_SHARP     = "data/train/sharp"
TRAIN_BLUR      = "data/train/blur"
TEST_SHARP      = "data/test/sharp"
TEST_BLUR       = "data/test/blur"

PAIRS_PER_IMAGE = 20
K_FOLDS         = 5
TARGET_SIZE     = (512, 512)
RANDOM_SEED     = 42

# ── Phân phối tầng blur ──────────────────────────────────
# Xác suất rơi vào từng tầng
TIER_PROBS      = [0.60, 0.30, 0.10]   # nhẹ / vừa / nặng

# Range kernel cho từng tầng (odd values only)
MOTION_TIERS    = [
    [3, 5, 7, 9],            # Tầng 1: nhẹ   3–9px
    [9, 11, 13, 15, 17, 19], # Tầng 2: vừa   9–19px
    [19, 21, 23, 25, 27],    # Tầng 3: nặng  19–27px
]
DEFOCUS_TIERS   = [
    [1, 2],       # Tầng 1: radius 1–2
    [2, 3, 4],    # Tầng 2: radius 2–4
    [4, 5, 6],    # Tầng 3: radius 4–6
]
GAUSSIAN_TIERS  = [
    (0.5, 1.5),   # Tầng 1: sigma 0.5–1.5
    (1.5, 3.0),   # Tầng 2: sigma 1.5–3.0
    (3.0, 4.5),   # Tầng 3: sigma 3.0–4.5
]
# ───────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def pick_tier():
    """Chọn tầng blur ngẫu nhiên theo xác suất định sẵn."""
    r = random.random()
    if r < TIER_PROBS[0]:
        return 0
    elif r < TIER_PROBS[0] + TIER_PROBS[1]:
        return 1
    else:
        return 2


# ══════════════════════════════════════════════
#  CÁC HÀM TẠO KERNEL
# ══════════════════════════════════════════════

def motion_blur_kernel(size, angle_deg):
    k = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    rad    = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    for i in range(size):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    total = k.sum()
    return k / total if total > 0 else (k + 1e-8)


def disk_blur_kernel(radius):
    size = 2 * radius + 1
    k = np.zeros((size, size), dtype=np.float32)
    cv2.circle(k, (radius, radius), radius, 1.0, -1)
    return k / k.sum()


# ══════════════════════════════════════════════
#  CÁC HÀM APPLY BLUR (tiered)
# ══════════════════════════════════════════════

def apply_motion_blur(img):
    """
    Motion blur với phân phối 3 tầng.

    Vì sao không dùng uniform(3, 27)?
    → Uniform sẽ cho 50% sample > 15px, quá nhiều blur nặng.
    → Tiered đảm bảo 60% sample blur nhẹ như thực tế.
    """
    tier   = pick_tier()
    length = random.choice(MOTION_TIERS[tier])
    angle  = random.uniform(0, 180)
    kernel = motion_blur_kernel(length, angle)
    return cv2.filter2D(img, -1, kernel), tier


def apply_defocus_blur(img):
    """Defocus blur (mất tiêu cự) với 3 tầng."""
    tier   = pick_tier()
    radius = random.choice(DEFOCUS_TIERS[tier])
    kernel = disk_blur_kernel(radius)
    return cv2.filter2D(img, -1, kernel), tier


def apply_gaussian_blur(img):
    """Gaussian blur (rung tay) với 3 tầng."""
    tier  = pick_tier()
    lo, hi = GAUSSIAN_TIERS[tier]
    sigma  = random.uniform(lo, hi)
    ksize  = int(6 * sigma + 1) | 1   # force odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigma), tier


def apply_combined_blur(img):
    """
    Motion nhẹ + Gaussian rất nhẹ — mô phỏng rung tay thực tế nhất.
    Luôn dùng Tầng 1–2 cho combined (không dùng tầng 3 riêng lẻ).
    """
    tier   = 0 if random.random() < 0.7 else 1
    length = random.choice(MOTION_TIERS[tier])
    angle  = random.uniform(0, 180)
    kernel = motion_blur_kernel(length, angle)
    img    = cv2.filter2D(img, -1, kernel)
    img    = cv2.GaussianBlur(img, (5, 5), random.uniform(0.3, 1.0))
    return img, tier


def apply_mild_blur(img):
    """Blur cực nhẹ — gần như không thấy, giúp model học deblur nhỏ."""
    choice = random.randint(0, 2)
    if choice == 0:
        kernel = motion_blur_kernel(3, random.uniform(0, 180))
        return cv2.filter2D(img, -1, kernel), 0
    elif choice == 1:
        sigma = random.uniform(0.4, 0.9)
        ksize = int(6 * sigma + 1) | 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigma), 0
    else:
        return cv2.filter2D(img, -1, disk_blur_kernel(1)), 0


# ══════════════════════════════════════════════
#  CÁC HÀM SUY BIẾN KHÁC
# ══════════════════════════════════════════════

def add_noise(img, tier=0):
    """
    Gaussian noise — mức độ tương ứng với tầng blur.
    Tầng nặng hơn thường đi kèm điều kiện chụp tệ hơn → nhiễu nhiều hơn.
    """
    sigma_range = {0: (1.0, 3.0), 1: (2.0, 5.0), 2: (3.0, 7.0)}
    lo, hi = sigma_range.get(tier, (1.0, 3.0))
    sigma  = random.uniform(lo, hi)
    noise  = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img, tier=0):
    """
    JPEG compression — quality cao cho tầng nhẹ, thấp hơn cho tầng nặng.
    Thực tế: ảnh blur nặng thường share qua mạng nhiều lần → JPEG nhiều hơn.
    """
    quality_range = {0: (82, 95), 1: (72, 88), 2: (65, 80)}
    lo, hi  = quality_range.get(tier, (75, 95))
    quality = random.randint(lo, hi)
    _, buf  = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ══════════════════════════════════════════════
#  PIPELINE CHÍNH
# ══════════════════════════════════════════════

def generate_one_blur(sharp_img):
    """
    Nhận 1 ảnh sharp → trả ra 1 ảnh blur.

    Phân phối loại blur (thực tế):
      28% → motion blur (tiered)
      22% → defocus blur (tiered)
      20% → gaussian blur (tiered)
      20% → combined motion+gaussian
      10% → mild (gần như không blur)

    KHÔNG có perspective warp trong pipeline.
    (Perspective xử lý ở step5/step6 riêng biệt.)
    """
    img = sharp_img.copy()

    # ── Stage 1: Blur chính ──────────────────────────────────
    r = random.random()
    if r < 0.28:
        img, tier = apply_motion_blur(img)
    elif r < 0.50:
        img, tier = apply_defocus_blur(img)
    elif r < 0.70:
        img, tier = apply_gaussian_blur(img)
    elif r < 0.90:
        img, tier = apply_combined_blur(img)
    else:
        img, tier = apply_mild_blur(img)

    # ── Stage 2: Noise (tương ứng với tầng blur) ─────────────
    img = add_noise(img, tier)

    # ── Stage 3: JPEG compression (50% chance) ───────────────
    if random.random() < 0.50:
        img = apply_jpeg_compression(img, tier)

    return img


# ══════════════════════════════════════════════
#  RESIZE & PADDING
# ══════════════════════════════════════════════

def resize_with_padding(img, target=(512, 512), pad_color=128):
    th, tw  = target
    h, w    = img.shape[:2]
    scale   = min(tw / w, th / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad_top    = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left   = (tw - new_w) // 2
    pad_right  = tw - new_w - pad_left
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[pad_color, pad_color, pad_color]
    )
    return padded


# ══════════════════════════════════════════════
#  K-FOLD SPLIT
# ══════════════════════════════════════════════

def make_kfold_splits(all_imgs, k, seed=RANDOM_SEED):
    imgs = list(all_imgs)
    rng  = random.Random(seed)
    rng.shuffle(imgs)
    folds = [[] for _ in range(k)]
    for idx, img in enumerate(imgs):
        folds[idx % k].append(img)
    splits = []
    for i in range(k):
        test_imgs  = folds[i]
        train_imgs = [img for j in range(k) if j != i for img in folds[j]]
        splits.append((train_imgs, test_imgs))
    return splits


# ══════════════════════════════════════════════
#  HÀM CHÍNH
# ══════════════════════════════════════════════

def generate_dataset(fold_idx=0, k=K_FOLDS):
    try:
        from tqdm import tqdm
    except ImportError:
        os.system("pip install tqdm -q")
        from tqdm import tqdm

    raw_path = Path(RAW_GT_DIR)
    all_imgs = sorted(
        list(raw_path.glob("*.jpg")) +
        list(raw_path.glob("*.png")) +
        list(raw_path.glob("*.jpeg"))
    )

    if not all_imgs:
        print(f"❌ Không tìm thấy ảnh trong {RAW_GT_DIR}")
        print("   Hãy chạy step1_download_cards.py trước!")
        sys.exit(1)

    print(f"Tổng ảnh GT   : {len(all_imgs)}")
    print(f"K-Fold        : k={k},  Fold: {fold_idx}")
    print(f"Pairs/ảnh     : {PAIRS_PER_IMAGE}")
    print(f"Blur tầng     : Nhẹ {TIER_PROBS[0]*100:.0f}% / Vừa {TIER_PROBS[1]*100:.0f}% / Nặng {TIER_PROBS[2]*100:.0f}%")
    print()

    splits = make_kfold_splits(all_imgs, k)
    train_imgs, test_imgs = splits[fold_idx]

    print(f"Train: {len(train_imgs)} ảnh → {len(train_imgs) * PAIRS_PER_IMAGE} cặp")
    print(f"Test:  {len(test_imgs)} ảnh  → {len(test_imgs)  * PAIRS_PER_IMAGE} cặp")

    import shutil
    for d in [TRAIN_SHARP, TRAIN_BLUR, TEST_SHARP, TEST_BLUR]:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # ── Training pairs ────────────────────────────────────────
    print("\n🔄 Đang tạo training pairs...")
    train_count = 0
    failed = 0
    for img_path in tqdm(train_imgs, desc="Train", ncols=70):
        sharp_orig = cv2.imread(str(img_path))
        if sharp_orig is None:
            failed += 1
            continue
        sharp = resize_with_padding(sharp_orig, TARGET_SIZE)
        for i in range(PAIRS_PER_IMAGE):
            pair_id  = f"{img_path.stem}_{i:02d}"
            blur_img = generate_one_blur(sharp)
            cv2.imwrite(str(Path(TRAIN_SHARP) / f"{pair_id}.png"), sharp)
            cv2.imwrite(str(Path(TRAIN_BLUR)  / f"{pair_id}.png"), blur_img)
            train_count += 1

    # ── Test pairs ────────────────────────────────────────────
    print("\n🔄 Đang tạo test pairs...")
    test_count = 0
    for img_path in tqdm(test_imgs, desc="Test ", ncols=70):
        sharp_orig = cv2.imread(str(img_path))
        if sharp_orig is None:
            failed += 1
            continue
        sharp = resize_with_padding(sharp_orig, TARGET_SIZE)
        for i in range(PAIRS_PER_IMAGE):
            pair_id  = f"{img_path.stem}_{i:02d}"
            blur_img = generate_one_blur(sharp)
            cv2.imwrite(str(Path(TEST_SHARP) / f"{pair_id}.png"), sharp)
            cv2.imwrite(str(Path(TEST_BLUR)  / f"{pair_id}.png"), blur_img)
            test_count += 1

    make_preview()

    print("\n" + "=" * 58)
    print(f"  BƯỚC 2 HOÀN THÀNH — Fold {fold_idx}/{k-1}")
    print("=" * 58)
    print(f"  Train pairs : {train_count}")
    print(f"  Test pairs  : {test_count}")
    print(f"  Lỗi bỏ qua  : {failed}")
    print()
    print("  ⚠️  Kiểm tra preview trước khi upload Colab:")
    print("  → data/preview_samples.jpg")
    print()
    print("  Blur nhìn thấy 3 mức: nhẹ / vừa / đôi khi rõ nặng")
    print("  Chữ vẫn NHẬN DẠNG ĐƯỢC ở tầng nhẹ-vừa → ĐÚNG")
    print("=" * 58)
    return train_count, test_count


def make_preview(n_samples=9):
    """
    Preview 3×3 grid: 3 hàng (nhẹ/vừa/nặng), 3 cột (blur|sharp).
    Giúp kiểm tra phân phối bằng mắt.
    """
    sharp_path = Path(TRAIN_SHARP)
    blur_path  = Path(TRAIN_BLUR)
    sharp_files = sorted(list(sharp_path.glob("*.png")))[:n_samples]
    if not sharp_files:
        return

    rows = []
    for i, sf in enumerate(sharp_files):
        s = cv2.imread(str(sf))
        b = cv2.imread(str(blur_path / sf.name))
        if s is None or b is None:
            continue
        s = cv2.resize(s, (256, 256))
        b = cv2.resize(b, (256, 256))

        # Label tầng ước tính (theo số thứ tự sample)
        tier_label = ["TIER1-NHẸ", "TIER1-NHẸ", "TIER1-NHẸ",
                      "TIER1-NHẸ", "TIER1-NHẸ", "TIER2-VỪA",
                      "TIER2-VỪA", "TIER2-VỪA", "TIER3-NẶNG"]
        label = tier_label[i] if i < len(tier_label) else ""

        cv2.putText(s, "SHARP",  (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(b, f"BLUR {label}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 2)
        rows.append(np.hstack([b, s]))

    if rows:
        grid = np.vstack(rows)
        cv2.imwrite("data/preview_samples.jpg", grid)
        print(f"\n👁️  Preview: data/preview_samples.jpg")
        print("   Tầng 1 (nhẹ): chữ rõ dù mờ nhẹ")
        print("   Tầng 2 (vừa): chữ mờ nhưng vẫn đoán được")
        print("   Tầng 3 (nặng): chữ mờ, khó đọc — 10% dataset")


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo dataset blur/sharp tiered")
    parser.add_argument("--fold",      type=int, default=0)
    parser.add_argument("--k",         type=int, default=K_FOLDS)
    parser.add_argument("--all_folds", action="store_true")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 58)
    print("  STEP 2: Tạo Dataset Blur/Sharp (Tiered Distribution)")
    print("=" * 58)
    print(f"  Phân phối tầng:")
    print(f"    Tầng 1 — Nhẹ  (motion 3–9px,  defocus r1–2): 60%")
    print(f"    Tầng 2 — Vừa  (motion 9–19px, defocus r2–4): 30%")
    print(f"    Tầng 3 — Nặng (motion 19–27px,defocus r4–6): 10%")
    print()

    if args.all_folds:
        for fold_i in range(args.k):
            print(f"\n{'─'*55}")
            print(f"  FOLD {fold_i}/{args.k-1}")
            generate_dataset(fold_idx=fold_i, k=args.k)
    else:
        if args.fold < 0 or args.fold >= args.k:
            print(f"❌ --fold phải từ 0 đến {args.k-1}")
            sys.exit(1)
        generate_dataset(fold_idx=args.fold, k=args.k)