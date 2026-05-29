#!/usr/bin/env python3
"""
STEP 2: Tạo Dataset Blur/Sharp (Fixed Version)
===============================================
Các thay đổi so với version cũ:
  ✅ XÓA perspective warp khỏi pipeline (step5 xử lý riêng)
  ✅ Giảm motion blur kernel: 9-43px → 5-15px
  ✅ Giảm defocus radius: 3-14px → 2-5px
  ✅ Giảm Gaussian sigma: 1.5-5.5 → 0.8-2.2
  ✅ Giảm noise: sigma 1.5-9.0 → 1.0-4.0
  ✅ Tăng JPEG quality: 60-88% → 75-95%
  ✅ Tăng pairs/image: 15 → 20

Lý do: Kernel cũ (max 43px) lớn hơn chiều cao chữ (10-18px) ở 512px
→ thông tin bị xóa hoàn toàn, model không thể recover.
Kernel mới (max 15px) < chiều cao chữ → thông tin còn đủ để học.

Chạy:
  python scripts/step2_gen_dataset.py
  python scripts/step2_gen_dataset.py --fold 1   # fold khác
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

PAIRS_PER_IMAGE = 20          # Tăng từ 15 → 20
K_FOLDS         = 5
TARGET_SIZE     = (512, 512)
RANDOM_SEED     = 42
# ───────────────────────────────────────────────

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════
#  CÁC HÀM TẠO BLUR  (đã điều chỉnh)
# ══════════════════════════════════════════════

def motion_blur_kernel(size, angle_deg):
    """
    Tạo kernel motion blur tuyến tính.
    size: kích thước kernel (pixels)
    angle_deg: góc chuyển động (0–180 độ)
    """
    k = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    rad = np.radians(angle_deg)
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
    """Kernel defocus blur hình đĩa tròn."""
    size = 2 * radius + 1
    k = np.zeros((size, size), dtype=np.float32)
    cv2.circle(k, (radius, radius), radius, 1.0, -1)
    return k / k.sum()


def apply_motion_blur(img):
    """
    Motion blur với angle và length ngẫu nhiên.
    ✅ FIX: range 9-43px → 5-15px (odd only)
    Lý do: Kernel cũ max 43px > chiều cao chữ 10-18px → mất thông tin
    """
    angle  = random.uniform(0, 180)
    length = random.choice([5, 7, 9, 11, 13, 15])  # FIX: max 15px
    kernel = motion_blur_kernel(length, angle)
    return cv2.filter2D(img, -1, kernel)


def apply_defocus_blur(img):
    """
    Mờ mất tiêu cự.
    ✅ FIX: radius 3-14px → 2-5px
    """
    radius = random.randint(2, 5)  # FIX: max 5px
    kernel = disk_blur_kernel(radius)
    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_blur(img):
    """
    Mờ Gaussian (rung tay nhẹ, camera shake).
    ✅ FIX: sigma 1.5-5.5 → 0.8-2.2
    """
    sigma = random.uniform(0.8, 2.2)  # FIX: giảm sigma
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_combined_blur(img):
    """
    Motion + Gaussian nhẹ (thực tế nhất).
    ✅ FIX: motion 7-25px → 5-9px
    """
    angle  = random.uniform(0, 180)
    length = random.choice([5, 7, 9])   # FIX: max 9px
    kernel = motion_blur_kernel(length, angle)
    img    = cv2.filter2D(img, -1, kernel)
    img    = cv2.GaussianBlur(img, (5, 5), random.uniform(0.3, 0.8))
    return img


def apply_mild_blur(img):
    """
    Blur nhẹ — thêm loại này để model học cả case blur rất nhỏ.
    Đây là level blur gần với thực tế nhất (chụp bằng tay hơi rung nhẹ).
    """
    choice = random.randint(0, 2)
    if choice == 0:
        # Rất nhẹ: motion 3px
        kernel = motion_blur_kernel(3, random.uniform(0, 180))
        return cv2.filter2D(img, -1, kernel)
    elif choice == 1:
        # Gaussian rất nhẹ
        sigma = random.uniform(0.5, 1.2)
        ksize = int(6 * sigma + 1) | 1  # force odd
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    else:
        # Defocus nhỏ
        kernel = disk_blur_kernel(1)
        return cv2.filter2D(img, -1, kernel)


# ══════════════════════════════════════════════
#  CÁC HÀM SUY BIẾN KHÁC  (đã điều chỉnh)
# ══════════════════════════════════════════════

def add_noise(img, sigma=None):
    """
    Gaussian noise.
    ✅ FIX: sigma 2.0-10.0 → 1.0-4.0
    """
    if sigma is None:
        sigma = random.uniform(1.0, 4.0)  # FIX
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_jpeg_compression(img, quality=None):
    """
    JPEG compression artifact.
    ✅ FIX: quality 60-88% → 75-95%
    """
    if quality is None:
        quality = random.randint(75, 95)   # FIX: ít artifact hơn
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ══════════════════════════════════════════════
#  PIPELINE CHÍNH  (XÓA perspective warp)
# ══════════════════════════════════════════════

def generate_one_blur(sharp_img):
    """
    Nhận 1 ảnh sharp → trả ra 1 ảnh blur.

    ✅ FIX QUAN TRỌNG: XÓA perspective warp khỏi pipeline.
    Lý do: Trong inference (step5/step6), perspective được sửa TRƯỚC khi
    vào model. Nếu train có warp trong blur pipeline → train/inference
    distribution mismatch → model không generalize được.

    Phân phối blur mới (thực tế hơn):
      25% → motion blur (5-15px)
      25% → defocus blur (radius 2-5px)
      20% → gaussian blur (sigma 0.8-2.2)
      20% → combined (motion nhẹ + gaussian)
      10% → mild blur (rất nhẹ, gần thực tế nhất)
    """
    img = sharp_img.copy()

    # ── Stage 1: Blur chính ──────────────────────────────────
    r = random.random()
    if r < 0.25:
        img = apply_motion_blur(img)       # 25%
    elif r < 0.50:
        img = apply_defocus_blur(img)      # 25%
    elif r < 0.70:
        img = apply_gaussian_blur(img)     # 20%
    elif r < 0.90:
        img = apply_combined_blur(img)     # 20%
    else:
        img = apply_mild_blur(img)         # 10%

    # ── Stage 2: Sensor noise (luôn có, nhưng nhẹ hơn) ──────
    img = add_noise(img, sigma=random.uniform(1.0, 4.0))

    # ── Stage 3: JPEG compression (50% chance, quality cao hơn)
    if random.random() < 0.50:
        img = apply_jpeg_compression(img, quality=random.randint(75, 95))

    return img


# ══════════════════════════════════════════════
#  RESIZE & PADDING
# ══════════════════════════════════════════════

def resize_with_padding(img, target=(512, 512), pad_color=128):
    """
    Resize ảnh về target size, giữ aspect ratio, pad bằng màu trung tính.
    """
    th, tw = target
    h, w   = img.shape[:2]
    scale  = min(tw / w, th / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
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
        train_imgs = []
        for j in range(k):
            if j != i:
                train_imgs.extend(folds[j])
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
    print(f"K-Fold        : k={k}")
    print(f"Fold hiện tại : {fold_idx} / {k-1}")
    print(f"Pairs/ảnh     : {PAIRS_PER_IMAGE}")
    print()

    splits = make_kfold_splits(all_imgs, k)
    train_imgs, test_imgs = splits[fold_idx]

    expected_train = len(train_imgs) * PAIRS_PER_IMAGE
    expected_test  = len(test_imgs)  * PAIRS_PER_IMAGE
    print(f"Train: {len(train_imgs)} ảnh → {expected_train} cặp")
    print(f"Test:  {len(test_imgs)} ảnh  → {expected_test} cặp")
    print(f"Overlap train/test: {len(set(train_imgs) & set(test_imgs))} ảnh")  # = 0

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

    print("\n" + "=" * 55)
    print(f"  BƯỚC 2 HOÀN THÀNH — Fold {fold_idx}/{k-1}")
    print("=" * 55)
    print(f"  Train pairs : {train_count}")
    print(f"  Test pairs  : {test_count}")
    print(f"  Lỗi bỏ qua  : {failed}")
    print()
    print("  Kiểm tra bằng mắt ảnh preview:")
    print("  → data/preview_samples.jpg")
    print()
    print("  Lưu ý khi so sánh preview với version cũ:")
    print("  → Blur nhìn nhẹ hơn NHIỀU — đây là ĐÚNG")
    print("  → Chữ vẫn đọc được dù mờ → model có thể học")
    print()
    print("  ➡️  Nén data và upload lên Colab để train!")
    print("=" * 55)

    return train_count, test_count


def make_preview(n_samples=6):
    """Tạo ảnh grid preview để kiểm tra chất lượng blur bằng mắt."""
    sharp_path = Path(TRAIN_SHARP)
    blur_path  = Path(TRAIN_BLUR)
    sharp_files = sorted(list(sharp_path.glob("*.png")))[:n_samples]
    if not sharp_files:
        return
    rows = []
    for sf in sharp_files:
        s = cv2.imread(str(sf))
        b = cv2.imread(str(blur_path / sf.name))
        if s is None or b is None:
            continue
        s = cv2.resize(s, (256, 256))
        b = cv2.resize(b, (256, 256))
        cv2.putText(s, "SHARP", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(b, "BLUR",  (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        rows.append(np.hstack([b, s]))
    if rows:
        grid = np.vstack(rows)
        cv2.imwrite("data/preview_samples.jpg", grid)
        print(f"\n👁️  Preview: data/preview_samples.jpg  (blur trái | sharp phải)")
        print("   Blur nên nhìn nhẹ, vẫn đọc được chữ — đó là ĐÚNG!")


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo dataset blur/sharp với K-Fold CV")
    parser.add_argument("--fold",      type=int, default=0)
    parser.add_argument("--k",         type=int, default=K_FOLDS)
    parser.add_argument("--all_folds", action="store_true")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 55)
    print("  STEP 2: Tạo Dataset Blur/Sharp (FIXED)")
    print("=" * 55)
    print(f"  Ảnh GT dir    : {RAW_GT_DIR}")
    print(f"  Pairs/ảnh     : {PAIRS_PER_IMAGE}")
    print(f"  Target size   : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  Motion blur   : 5-15px  (cũ: 9-43px) ✅")
    print(f"  Defocus radius: 2-5px   (cũ: 3-14px) ✅")
    print(f"  Gaussian sigma: 0.8-2.2 (cũ: 1.5-5.5) ✅")
    print(f"  Perspective   : ĐÃ XÓA khỏi pipeline ✅")
    print()

    if args.all_folds:
        for fold_i in range(args.k):
            print(f"\n{'─'*55}")
            print(f"  FOLD {fold_i}/{args.k-1}")
            print(f"{'─'*55}")
            generate_dataset(fold_idx=fold_i, k=args.k)
    else:
        if args.fold < 0 or args.fold >= args.k:
            print(f"❌ --fold phải từ 0 đến {args.k-1}")
            sys.exit(1)
        generate_dataset(fold_idx=args.fold, k=args.k)