#!/usr/bin/env python3
"""
STEP 5: Demo — Perspective Correction + NAFNet Deblur + Unsharp Masking
========================================================================
Pipeline xử lý ảnh thẻ thực tế từ điện thoại:
  Stage 0: Phát hiện 4 góc thẻ → sửa phối cảnh (perspective)
  Stage 1: NAFNet → giải mờ
  Stage 2: Unsharp masking → tăng nét chữ và viền
  Output:  ảnh so sánh 4 cột: [Input | Rectified | Deblurred | Sharpened]

Chạy:
  # 1 ảnh:
  python scripts/step5_demo.py --input my_photo.jpg --checkpoint pretrained/nafnet_cards_best.pth

  # Thư mục:
  python scripts/step5_demo.py --input data/test/blur/ --checkpoint pretrained/nafnet_cards_best.pth

  # Tắt perspective correction:
  python scripts/step5_demo.py --input my_photo.jpg --no-perspective --checkpoint ...

  # Điều chỉnh cường độ sharpen (0.0 = tắt, 1.0 = mạnh):
  python scripts/step5_demo.py --input my_photo.jpg --sharpen 0.7 --checkpoint ...
"""

import cv2
import numpy as np
import torch
import sys
import os
import argparse
from pathlib import Path


# ───────────────────────────────────────────────
# STAGE 0: PERSPECTIVE CORRECTION
# ───────────────────────────────────────────────

def detect_card_contour(img):
    """
    Tìm đường viền tứ giác lớn nhất trong ảnh (= thẻ / card).
    Trả về array 4×2 float32 hoặc None nếu không tìm thấy.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in [(30, 100), (20, 80), (50, 150)]:
        edges  = cv2.Canny(blurred, lo, hi)
        kernel = np.ones((3, 3), np.uint8)
        edges  = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        img_area = img.shape[0] * img.shape[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:8]:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.08:
                break
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def order_corners(pts):
    """Sắp xếp 4 góc: top-left, top-right, bottom-right, bottom-left."""
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl   = pts[np.argmin(s)]
    br   = pts[np.argmax(s)]
    tr   = pts[np.argmin(diff)]
    bl   = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def perspective_correct(img, target_wh=(512, 512)):
    """
    Phát hiện thẻ và căn chỉnh phối cảnh.
    Trả về (corrected_img, success_flag).
    """
    tw, th  = target_wh
    corners = detect_card_contour(img)
    if corners is None:
        return cv2.resize(img, (tw, th)), False
    corners = order_corners(corners)
    dst = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(corners, dst)
    corrected = cv2.warpPerspective(img, M, (tw, th),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_REPLICATE)
    return corrected, True


# ───────────────────────────────────────────────
# STAGE 1: NAFNet INFERENCE
# ───────────────────────────────────────────────

def load_model(checkpoint_path, device):
    """Load NAFNet từ checkpoint file."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "NAFNet"))
    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError:
        print("❌ NAFNet chưa cài!")
        print("   cd NAFNet && pip install -r requirements.txt && python setup.py develop")
        sys.exit(1)

    model = NAFNet(
        img_channel=3,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2],
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"✅ Model loaded on {device}")
    return model


def nafnet_infer(model, img, device):
    """Chạy NAFNet inference trên 1 ảnh BGR uint8."""
    img_t = (torch.from_numpy(img / 255.)
             .permute(2, 0, 1).float().unsqueeze(0).to(device))
    with torch.no_grad():
        out = model(img_t)
    result = (out.squeeze().permute(1, 2, 0)
              .cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return result


# ───────────────────────────────────────────────
# STAGE 2: UNSHARP MASKING  (mới — tăng nét chữ)
# ───────────────────────────────────────────────

def unsharp_mask(img, amount=0.6, radius=1.0):
    """
    Tăng độ sắc nét bằng unsharp masking.
    Công thức: output = img + amount × (img - gaussian_blur(img))

    amount : cường độ sharpen (0.0 = tắt, 0.5 = nhẹ, 1.0 = mạnh)
    radius : bán kính gaussian blur làm base (nhỏ hơn = sắc nét ở pixel nhỏ hơn)

    Lý do thêm stage này:
      NAFNet giải mờ về mặt pixel tốt (PSNR cao) nhưng output vẫn hơi
      soft do training loss (MSE/PSNR) có xu hướng tạo ảnh mịn.
      Unsharp masking lấy lại nét ở cạnh và chữ mà không tạo nhiễu thêm.
    """
    if amount <= 0:
        return img
    sigma  = radius
    ksize  = int(6 * sigma + 1) | 1   # force odd
    blur   = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    sharp  = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def clahe_enhance(img, clip_limit=2.0):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) trên kênh L.
    Tăng độ tương phản cục bộ, giúp chữ nổi rõ hơn trên nền.
    Chỉ apply nếu ảnh nhìn hơi tối hoặc thiếu contrast.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ───────────────────────────────────────────────
# PIPELINE CHÍNH
# ───────────────────────────────────────────────

LABEL_H = 45

def make_comparison(original, rectified, deblurred, sharpened, rect_success):
    """Tạo ảnh so sánh 4 cột."""
    h, w = deblurred.shape[:2]
    o = cv2.resize(original,  (w, h))
    r = cv2.resize(rectified, (w, h))
    d = deblurred.copy()
    s = sharpened.copy()

    row    = np.hstack([o, r, d, s])
    canvas = np.zeros((h + LABEL_H, row.shape[1], 3), dtype=np.uint8)
    canvas[:h] = row
    canvas[h:] = (30, 30, 30)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    y     = h + 30

    status = "(corrected)" if rect_success else "(no card found)"
    cv2.putText(canvas, "INPUT",               (10,        y), font, scale, (180,180,180), thick)
    cv2.putText(canvas, f"RECTIFIED {status}", (w + 5,     y), font, scale, (255,200,  0), thick)
    cv2.putText(canvas, "DEBLURRED",           (2*w + 5,   y), font, scale, (100,255,100), thick)
    cv2.putText(canvas, "SHARPENED",           (3*w + 5,   y), font, scale, (100,200,255), thick)

    return canvas


def process_one(input_path, output_dir, model, device,
                use_perspective=True, target_size=512,
                sharpen_amount=0.6, use_clahe=False):
    """Xử lý 1 ảnh: perspective → deblur → sharpen → save."""

    img = cv2.imread(str(input_path))
    if img is None:
        print(f"  ❌ Không đọc được: {input_path}")
        return

    original = img.copy()
    tw = th  = target_size

    # ── Stage 0: Perspective ────────────────────────────────
    if use_perspective:
        rectified, found = perspective_correct(img, target_wh=(tw, th))
        status = "✅ tìm thấy viền thẻ" if found else "⚠️ không thấy viền, resize toàn ảnh"
    else:
        rectified = cv2.resize(img, (tw, th))
        found     = False
        status    = "(bỏ qua perspective)"
    print(f"  Stage 0: {status}")

    # ── Stage 1: Deblur ──────────────────────────────────────
    deblurred = nafnet_infer(model, rectified, device)
    print(f"  Stage 1: NAFNet deblur OK")

    # ── Stage 2: Unsharp masking ─────────────────────────────
    sharpened = unsharp_mask(deblurred, amount=sharpen_amount, radius=1.0)
    if use_clahe:
        sharpened = clahe_enhance(sharpened)
    print(f"  Stage 2: Unsharp mask (amount={sharpen_amount:.1f}) OK")

    # ── Lưu kết quả ──────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem

    # Ảnh final (sharpened)
    final_path = out_dir / f"{stem}_final.png"
    cv2.imwrite(str(final_path), sharpened)

    # Ảnh deblur (trước sharpen, để so sánh)
    deblur_path = out_dir / f"{stem}_deblurred.png"
    cv2.imwrite(str(deblur_path), deblurred)

    # Ảnh so sánh 4 cột
    comparison   = make_comparison(original, rectified, deblurred, sharpened, found)
    compare_path = out_dir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(compare_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"  💾 Final:      {final_path}")
    print(f"  💾 Deblurred:  {deblur_path}")
    print(f"  📊 So sánh:    {compare_path}")


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo deblur thẻ bài / ID card")
    parser.add_argument("--input",        required=True,
                        help="Ảnh input hoặc thư mục")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path file .pth checkpoint")
    parser.add_argument("--output",       default="results/demo",
                        help="Thư mục lưu kết quả (default: results/demo)")
    parser.add_argument("--no-perspective", action="store_true",
                        help="Bỏ qua bước perspective correction")
    parser.add_argument("--size",         type=int, default=512,
                        help="Kích thước output (default: 512)")
    parser.add_argument("--sharpen",      type=float, default=0.6,
                        help="Cường độ unsharp masking: 0.0=tắt, 0.5=nhẹ, 1.0=mạnh (default: 0.6)")
    parser.add_argument("--clahe",        action="store_true",
                        help="Thêm CLAHE contrast enhancement sau sharpen")
    args = parser.parse_args()

    print("=" * 55)
    print("  STEP 5: Demo Deblur (FIXED)")
    print("=" * 55)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint không tồn tại: {ckpt_path}")
        sys.exit(1)

    model     = load_model(ckpt_path, device)
    use_persp = not args.no_perspective

    input_path = Path(args.input)

    if input_path.is_file():
        print(f"\nXử lý 1 ảnh: {input_path.name}")
        process_one(input_path, args.output, model, device,
                    use_perspective=use_persp, target_size=args.size,
                    sharpen_amount=args.sharpen, use_clahe=args.clahe)

    elif input_path.is_dir():
        files = (sorted(input_path.glob("*.jpg")) +
                 sorted(input_path.glob("*.png")) +
                 sorted(input_path.glob("*.jpeg")))
        print(f"\nXử lý {len(files)} ảnh trong {input_path}")
        for i, f in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {f.name}")
            process_one(f, args.output, model, device,
                        use_perspective=use_persp, target_size=args.size,
                        sharpen_amount=args.sharpen, use_clahe=args.clahe)
    else:
        print(f"❌ Input không tồn tại: {args.input}")
        sys.exit(1)

    print(f"\n✅ Xong! Xem kết quả tại: {args.output}/")
    print("   *_comparison.jpg: input | rectified | deblurred | sharpened")
    print("   *_final.png     : kết quả cuối để dùng")