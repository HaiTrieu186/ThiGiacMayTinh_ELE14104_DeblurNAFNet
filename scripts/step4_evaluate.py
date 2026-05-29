#!/usr/bin/env python3
"""
STEP 4: Đánh Giá Kết Quả Model
================================
Tính PSNR, SSIM và OCR accuracy trước/sau deblur.
In ra bảng kết quả để dùng trong báo cáo.

Chạy: python scripts/step4_evaluate.py
      --blur_dir  data/test/blur
      --sharp_dir data/test/sharp
      --result_dir results/nafnet_output
"""

import cv2
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path


# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
BLUR_DIR    = "data/test/blur"
SHARP_DIR   = "data/test/sharp"
RESULT_DIR  = "results/nafnet_output"   # Thư mục chứa output của NAFNet
OUT_JSON    = "results/evaluation.json"
MAX_EVAL    = 200   # Đánh giá tối đa bao nhiêu ảnh (để nhanh)
# ───────────────────────────────────────────────


# ══════════════════════════════════════════════
#  IMAGE QUALITY METRICS
# ══════════════════════════════════════════════

def calc_psnr(img1, img2):
    """
    Peak Signal-to-Noise Ratio.
    Cao hơn = tốt hơn (thường 20–40 dB).
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calc_ssim(img1, img2):
    """
    Structural Similarity Index (0–1, cao hơn = tốt hơn).
    Đánh giá cấu trúc, độ tương phản và độ sáng.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, data_range=255, channel_axis=2)
    except ImportError:
        # Fallback: tính SSIM đơn giản trên grayscale
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        mu1, mu2 = g1.mean(), g2.mean()
        sig1 = g1.std()
        sig2 = g2.std()
        sig12 = np.mean((g1 - mu1) * (g2 - mu2))
        num = (2*mu1*mu2 + C1) * (2*sig12 + C2)
        den = (mu1**2 + mu2**2 + C1) * (sig1**2 + sig2**2 + C2)
        return num / den


# ══════════════════════════════════════════════
#  OCR METRICS
# ══════════════════════════════════════════════

def check_tesseract():
    """Kiểm tra tesseract có cài không"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_read(img):
    """Đọc text từ ảnh bằng Tesseract"""
    import pytesseract
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Tăng contrast trước khi OCR
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary, config='--psm 11 --oem 3')
    return text.strip()


def levenshtein(s1, s2):
    """Edit distance giữa 2 chuỗi"""
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
        prev = curr
    return prev[-1]


def cer(ref, hyp):
    """Character Error Rate: edit_distance / len(ref)"""
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(levenshtein(ref, hyp) / len(ref), 1.0)


# ══════════════════════════════════════════════
#  INFERENCE (nếu chưa có result)
# ══════════════════════════════════════════════

def run_nafnet_inference(blur_dir, result_dir, checkpoint_path):
    """
    Chạy NAFNet inference trên toàn bộ test blur set.
    Lưu kết quả vào result_dir với tên file giống hệt blur_dir.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent / "NAFNet"))

    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError:
        print("❌ NAFNet chưa được cài. Chạy: cd NAFNet && python setup.py develop")
        return False

    print(f"Loading checkpoint: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = NAFNet(img_channel=3, width=32, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Các checkpoint của BasicSR lưu dưới key 'params'
    state_dict = ckpt.get('params', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    Path(result_dir).mkdir(parents=True, exist_ok=True)

    blur_files = sorted(
        list(Path(blur_dir).glob("*.png")) +
        list(Path(blur_dir).glob("*.jpg"))
    )[:MAX_EVAL]

    print(f"Inference trên {len(blur_files)} ảnh...")

    try:
        from tqdm import tqdm
        iterator = tqdm(blur_files, ncols=70)
    except ImportError:
        iterator = blur_files

    for bf in iterator:
        save_path = Path(result_dir) / bf.name
        if save_path.exists():
            continue

        img = cv2.imread(str(bf))
        if img is None:
            continue

        img_t = (torch.from_numpy(img / 255.)
                 .permute(2, 0, 1)
                 .float()
                 .unsqueeze(0)
                 .to(device))

        with torch.no_grad():
            out = model(img_t)

        out_np = (out.squeeze()
                  .permute(1, 2, 0)
                  .cpu()
                  .numpy() * 255).clip(0, 255).astype(np.uint8)

        cv2.imwrite(str(save_path), out_np)

    print(f"✅ Inference xong. Kết quả tại: {result_dir}")
    return True


# ══════════════════════════════════════════════
#  ĐÁNH GIÁ CHÍNH
# ══════════════════════════════════════════════

def evaluate(blur_dir, sharp_dir, result_dir, out_json=OUT_JSON):
    """So sánh 3 chiều: blur | result | sharp"""

    blur_path   = Path(blur_dir)
    sharp_path  = Path(sharp_dir)
    result_path = Path(result_dir)

    if not result_path.exists():
        print(f"❌ Chưa có result. Chạy inference trước!")
        return None

    # Lấy file list
    blur_files = sorted(
        list(blur_path.glob("*.png")) +
        list(blur_path.glob("*.jpg"))
    )[:MAX_EVAL]

    has_ocr = check_tesseract()
    if not has_ocr:
        print("⚠️  Tesseract chưa cài → bỏ qua OCR metrics")
        print("   Cài: pip install pytesseract  +  sudo apt install tesseract-ocr")

    psnr_before, ssim_before, cer_before = [], [], []
    psnr_after,  ssim_after,  cer_after  = [], [], []

    print(f"\nĐang đánh giá {len(blur_files)} ảnh...")

    try:
        from tqdm import tqdm
        files_iter = tqdm(blur_files, ncols=70)
    except ImportError:
        files_iter = blur_files

    for bf in files_iter:
        sf = sharp_path  / bf.name
        rf = result_path / bf.name

        if not sf.exists() or not rf.exists():
            continue

        sharp  = cv2.imread(str(sf))
        blur   = cv2.imread(str(bf))
        result = cv2.imread(str(rf))

        if any(x is None for x in [sharp, blur, result]):
            continue

        # ── Image quality metrics ────────────────────────────
        psnr_before.append(calc_psnr(sharp, blur))
        ssim_before.append(calc_ssim(sharp, blur))

        psnr_after.append(calc_psnr(sharp, result))
        ssim_after.append(calc_ssim(sharp, result))

        # ── OCR metrics (nếu có) ─────────────────────────────
        if has_ocr:
            try:
                ref_text  = ocr_read(sharp)
                blur_text = ocr_read(blur)
                res_text  = ocr_read(result)
                if ref_text:
                    cer_before.append(cer(ref_text, blur_text))
                    cer_after.append(cer(ref_text, res_text))
            except Exception:
                pass

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    # ── Tổng hợp kết quả ──────────────────────────────────────
    results = {
        "num_evaluated": len(psnr_after),
        "before_deblur": {
            "psnr": avg(psnr_before),
            "ssim": avg(ssim_before),
            "cer":  avg(cer_before),
        },
        "after_deblur": {
            "psnr": avg(psnr_after),
            "ssim": avg(ssim_after),
            "cer":  avg(cer_after),
        },
        "improvement": {
            "psnr_gain_db":    (avg(psnr_after) or 0) - (avg(psnr_before) or 0),
            "ssim_gain":       (avg(ssim_after) or 0) - (avg(ssim_before) or 0),
            "cer_reduction":   (avg(cer_before) or 0) - (avg(cer_after) or 0),
        }
    }

    # ── In bảng kết quả ───────────────────────────────────────
    print("\n" + "═" * 52)
    print("  KẾT QUẢ ĐÁNH GIÁ")
    print("═" * 52)
    print(f"  Số ảnh đánh giá: {results['num_evaluated']}")
    print()

    b = results["before_deblur"]
    a = results["after_deblur"]
    g = results["improvement"]

    print(f"  {'Metric':<12} {'Trước Deblur':>14} {'Sau Deblur':>12} {'Cải thiện':>12}")
    print(f"  {'-'*52}")
    if b["psnr"]:
        print(f"  {'PSNR (dB)':<12} {b['psnr']:>14.2f} {a['psnr']:>12.2f} {g['psnr_gain_db']:>+11.2f}")
    if b["ssim"]:
        print(f"  {'SSIM':<12} {b['ssim']:>14.4f} {a['ssim']:>12.4f} {g['ssim_gain']:>+11.4f}")
    if b["cer"] is not None:
        print(f"  {'CER (↓)':<12} {b['cer']:>14.4f} {a['cer']:>12.4f} {-g['cer_reduction']:>+11.4f}")

    print("═" * 52)

    if g["psnr_gain_db"] > 0:
        print(f"\n  ✅ PSNR tăng {g['psnr_gain_db']:.2f} dB sau deblur")
    if g["ssim_gain"] > 0:
        print(f"  ✅ SSIM tăng {g['ssim_gain']:.4f} sau deblur")
    if b["cer"] is not None and g["cer_reduction"] > 0:
        pct = g["cer_reduction"] / b["cer"] * 100 if b["cer"] else 0
        print(f"  ✅ OCR lỗi giảm {pct:.1f}% sau deblur")

    # ── Lưu JSON ──────────────────────────────────────────────
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  💾 Đã lưu chi tiết: {out_json}")

    return results


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá NAFNet deblur")
    parser.add_argument("--blur_dir",    default=BLUR_DIR)
    parser.add_argument("--sharp_dir",   default=SHARP_DIR)
    parser.add_argument("--result_dir",  default=RESULT_DIR)
    parser.add_argument("--checkpoint",  default=None,
                        help="Path checkpoint để tự inference (nếu result_dir trống)")
    args = parser.parse_args()

    print("=" * 52)
    print("  STEP 4: Đánh Giá Kết Quả")
    print("=" * 52)

    # Nếu chưa có result_dir và cung cấp checkpoint → tự inference
    result_path = Path(args.result_dir)
    existing = list(result_path.glob("*.png")) if result_path.exists() else []

    if not existing and args.checkpoint:
        print(f"\nChưa có result → chạy inference từ {args.checkpoint}")
        success = run_nafnet_inference(args.blur_dir, args.result_dir, args.checkpoint)
        if not success:
            sys.exit(1)
    elif not existing:
        print(f"\n⚠️  Thư mục {args.result_dir} trống hoặc không tồn tại.")
        print("    Option 1: Truyền --checkpoint path/to/model.pth")
        print("    Option 2: Chạy inference bằng BasicSR test.py trước")
        print("\n    Ví dụ:")
        print("    python scripts/step4_evaluate.py --checkpoint pretrained/nafnet_cards_best.pth")
        sys.exit(1)

    evaluate(args.blur_dir, args.sharp_dir, args.result_dir)
