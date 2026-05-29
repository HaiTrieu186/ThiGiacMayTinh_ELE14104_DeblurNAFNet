#!/usr/bin/env python3
"""
STEP 1: Tải ảnh thẻ về làm Ground Truth (GT)
=========================================
Tải từ 2 nguồn:
  1. Ygoprodeck API  → ~300 ảnh thẻ Yugioh (miễn phí, không cần key)
  2. DeckOfCardsAPI  → 52 ảnh bộ bài tây

Chạy: python scripts/step1_download_cards.py
Output: data/raw_gt/ (~350 ảnh PNG/JPG sắc nét)
"""

import os
import sys
import time
import requests
from pathlib import Path

# ───────────────────────────────────────────────
# CONFIG — chỉnh ở đây nếu cần
# ───────────────────────────────────────────────
RAW_GT_DIR   = "data/raw_gt"
YUGIOH_COUNT = 300      # Số thẻ Yugioh muốn tải
BATCH_SIZE   = 100      # Số thẻ mỗi lần gọi API
SLEEP_SEC    = 0.08     # Thời gian chờ giữa các request (tránh bị block)
RETRY_TIMES  = 3        # Số lần retry nếu request lỗi
# ───────────────────────────────────────────────


def create_directories():
    """Tạo toàn bộ thư mục cần thiết cho dự án"""
    dirs = [
        "data/raw_gt",
        "data/train/sharp",
        "data/train/blur",
        "data/test/sharp",
        "data/test/blur",
        "pretrained",
        "results/demo",
        "checkpoints",
        "configs",
        "scripts",
        "notebooks",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Đã tạo cấu trúc thư mục dự án")


def safe_get(url, timeout=15):
    """Request có retry tự động"""
    for attempt in range(RETRY_TIMES):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.exceptions.RequestException as e:
            if attempt < RETRY_TIMES - 1:
                time.sleep(1.0)
    return None


def download_yugioh_cards(output_dir=RAW_GT_DIR, max_cards=YUGIOH_COUNT):
    """
    Tải ảnh thẻ Yugioh via API miễn phí của ygoprodeck.com
    Mỗi ảnh ~421x614 px, chất lượng tốt, nhiều text + hoa văn
    """
    print(f"\n📥 [1/2] Đang tải thẻ Yugioh (target: {max_cards} thẻ)...")

    out = Path(output_dir)
    downloaded = 0
    skipped    = 0
    offset     = 0

    while downloaded < max_cards:
        api_url = (
            f"https://db.ygoprodeck.com/api/v7/cardinfo.php"
            f"?num={BATCH_SIZE}&offset={offset}"
        )
        r = safe_get(api_url)
        if r is None:
            print(f"  ⚠️  API không phản hồi ở offset={offset}, thử offset tiếp theo")
            offset += BATCH_SIZE
            if offset > 12000:   # Yugioh có ~12000+ thẻ
                break
            continue

        data = r.json()
        if "data" not in data or not data["data"]:
            break

        for card in data["data"]:
            if downloaded >= max_cards:
                break

            images = card.get("card_images", [])
            if not images:
                continue

            card_id  = images[0]["id"]
            img_url  = images[0]["image_url"]
            savepath = out / f"yugioh_{card_id}.jpg"

            # Bỏ qua nếu đã tải rồi
            if savepath.exists() and savepath.stat().st_size > 5000:
                skipped += 1
                downloaded += 1
                continue

            img_r = safe_get(img_url)
            if img_r is not None:
                savepath.write_bytes(img_r.content)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"  Đã tải: {downloaded}/{max_cards}...")
                time.sleep(SLEEP_SEC)
            else:
                print(f"  Bỏ qua thẻ {card_id} (lỗi tải)")

        offset += BATCH_SIZE

    print(f"  ✅ Yugioh: {downloaded} thẻ ({skipped} đã có sẵn)")
    return downloaded


def download_playing_cards(output_dir=RAW_GT_DIR):
    """
    Tải 52 lá bài tây từ deckofcardsapi.com
    Ảnh chuẩn, nền trắng, rõ ràng
    """
    print(f"\n📥 [2/2] Đang tải bộ bài tây 52 lá...")

    out  = Path(output_dir)
    BASE = "https://deckofcardsapi.com/static/img"

    # Map giá trị → code trong URL
    values = {
        "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
        "7": "7", "8": "8", "9": "9", "10": "0",
        "J": "J", "Q": "Q", "K": "K", "A": "A"
    }
    suits = {"clubs": "C", "diamonds": "D", "hearts": "H", "spades": "S"}

    downloaded = 0
    failed     = 0

    for val_name, val_code in values.items():
        for suit_name, suit_code in suits.items():
            filename = f"{val_code}{suit_code}"
            url      = f"{BASE}/{filename}.png"
            savepath = out / f"playing_{val_name}_{suit_name}.png"

            if savepath.exists() and savepath.stat().st_size > 3000:
                downloaded += 1
                continue

            r = safe_get(url)
            if r is not None:
                savepath.write_bytes(r.content)
                downloaded += 1
                time.sleep(SLEEP_SEC)
            else:
                failed += 1

    print(f"  ✅ Bài tây: {downloaded}/52 lá ({failed} lỗi)")
    return downloaded


def verify_and_clean(raw_gt_dir=RAW_GT_DIR, min_size_kb=5, min_resolution=100):
    """
    Kiểm tra toàn bộ ảnh đã tải:
    - Xóa file nhỏ hơn min_size_kb (bị hỏng)
    - Xóa ảnh có chiều nhỏ hơn min_resolution px
    """
    import cv2

    path  = Path(raw_gt_dir)
    files = list(path.glob("*.jpg")) + list(path.glob("*.png"))

    valid   = 0
    removed = 0

    for f in files:
        # Kiểm tra kích thước file
        if f.stat().st_size < min_size_kb * 1024:
            f.unlink()
            removed += 1
            continue

        # Kiểm tra ảnh đọc được không
        img = cv2.imread(str(f))
        if img is None or img.shape[0] < min_resolution or img.shape[1] < min_resolution:
            f.unlink()
            removed += 1
            continue

        valid += 1

    print(f"\n📊 Kiểm tra dataset: {valid} ảnh hợp lệ, đã xóa {removed} ảnh lỗi")
    return valid


def print_summary(total):
    """In hướng dẫn bước tiếp theo"""
    print("\n" + "=" * 55)
    print("  BƯỚC 1 HOÀN THÀNH!")
    print("=" * 55)
    print(f"  Tổng ảnh GT trong data/raw_gt/: {total}")
    print(f"  Ước tính cặp train (×15):       {total * 15 * 85 // 100}")
    print(f"  Ước tính cặp test  (×15):       {total * 15 * 15 // 100}")
    print()
    print("  ➡️  Bước tiếp theo:")
    print("     python scripts/step2_gen_dataset.py")
    print("=" * 55)


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  STEP 1: Tải Ảnh Thẻ Ground Truth")
    print("=" * 55)

    # 1. Tạo thư mục
    create_directories()

    # 2. Tải Yugioh
    download_yugioh_cards()

    # 3. Tải bài tây
    download_playing_cards()

    # 4. Kiểm tra và dọn dẹp
    total = verify_and_clean()

    # 5. In hướng dẫn tiếp
    print_summary(total)
