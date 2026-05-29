#!/usr/bin/env python3
"""
STEP 6: Giao Diện Web Demo (Gradio) — Fixed Version
=====================================================
Thay đổi so với version cũ:
  ✅ Thêm slider điều chỉnh cường độ sharpen
  ✅ Hiển thị 4 ảnh so sánh (thêm cột Sharpened)
  ✅ Thêm CLAHE checkbox
  ✅ Thêm nút Download kết quả

Cài Gradio (1 lần):
    pip install gradio

Chạy:
    python scripts/step6_app.py --checkpoint pretrained/nafnet_cards_best.pth

Mở trình duyệt: http://localhost:7860
"""

import cv2
import numpy as np
import torch
import sys
import argparse
import tempfile
import os
from pathlib import Path
from PIL import Image


# ── Load model ──────────────────────────────────────────────
def load_model(checkpoint_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "NAFNet"))
    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError:
        raise RuntimeError("NAFNet chưa cài! Chạy: cd NAFNet && pip install -e .")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = NAFNet(img_channel=3, width=32, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])

    ckpt = torch.load(checkpoint_path, map_location=device)
    if   'params'     in ckpt: state_dict = ckpt['params']
    elif 'state_dict' in ckpt: state_dict = ckpt['state_dict']
    else:                      state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"✅ Model loaded: {checkpoint_path}")
    return model, device


# ── Perspective correction ───────────────────────────────────
def detect_card_contour(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in [(30, 100), (20, 80), (50, 150)]:
        edges = cv2.Canny(blurred, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        img_area = img.shape[0] * img.shape[1]
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if cv2.contourArea(cnt) < img_area * 0.08:
                break
            approx = cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def order_corners(pts):
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def perspective_correct(img, size=512):
    corners = detect_card_contour(img)
    if corners is None:
        return cv2.resize(img, (size, size)), False
    corners = order_corners(corners)
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    M   = cv2.getPerspectiveTransform(corners, dst)
    out = cv2.warpPerspective(img, M, (size, size),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, True


# ── NAFNet inference ─────────────────────────────────────────
def nafnet_infer(model, device, img_bgr):
    img_t = (torch.from_numpy(img_bgr / 255.)
             .permute(2, 0, 1).float().unsqueeze(0).to(device))
    with torch.no_grad():
        out = model(img_t)
    return (out.squeeze().permute(1, 2, 0)
            .cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


# ── Unsharp masking ──────────────────────────────────────────
def unsharp_mask(img, amount=0.6, radius=1.0):
    """Tăng độ nét bằng unsharp masking."""
    if amount <= 0:
        return img
    ksize = int(6 * radius + 1) | 1
    blur  = cv2.GaussianBlur(img, (ksize, ksize), radius)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def clahe_enhance(img):
    """CLAHE contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── Hàm xử lý chính ─────────────────────────────────────────
def process(pil_image, use_perspective, sharpen_amount, use_clahe):
    """
    Input:  PIL Image + settings từ Gradio
    Output: 4 ảnh PIL + status text + path file kết quả
    """
    if pil_image is None:
        return None, None, None, None, "⚠️ Chưa upload ảnh!", None

    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Stage 0: Perspective
    if use_perspective:
        rectified, found = perspective_correct(img_bgr)
        persp_msg = "✅ Tìm thấy viền thẻ, đã sửa góc" if found else "⚠️ Không thấy viền, resize toàn ảnh"
    else:
        rectified = cv2.resize(img_bgr, (512, 512))
        found     = False
        persp_msg = "— Bỏ qua sửa góc"

    # Stage 1: Deblur
    deblurred = nafnet_infer(MODEL, DEVICE, rectified)

    # Stage 2: Sharpen
    sharpened = unsharp_mask(deblurred, amount=float(sharpen_amount), radius=1.0)
    if use_clahe:
        sharpened = clahe_enhance(sharpened)

    # BGR → RGB
    rect_rgb     = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
    deblur_rgb   = cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB)
    sharpen_rgb  = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    # Lưu file tạm để download
    tmp_path = os.path.join(tempfile.gettempdir(), "deblur_result.png")
    cv2.imwrite(tmp_path, sharpened)

    status = (
        f"{persp_msg}\n"
        f"✅ NAFNet deblur hoàn tất\n"
        f"✅ Unsharp mask (amount={sharpen_amount:.1f})"
        + ("\n✅ CLAHE contrast enhanced" if use_clahe else "")
    )

    return (
        Image.fromarray(np.array(pil_image)),
        Image.fromarray(rect_rgb),
        Image.fromarray(deblur_rgb),
        Image.fromarray(sharpen_rgb),
        status,
        tmp_path
    )


# ── Gradio UI ────────────────────────────────────────────────
def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio chưa cài. Chạy: pip install gradio")
        sys.exit(1)

    with gr.Blocks(title="Card Deblur Demo", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🃏 Card Deblur — NAFNet Demo (Fixed)
        Upload ảnh thẻ bị mờ → model tự sửa góc → khôi phục nét → tăng sắc.

        **Hoạt động tốt với:** Bài tây, Yugioh card, CCCD, thẻ ngân hàng, thẻ tên...
        """)

        with gr.Row():
            # ── Cột trái: Controls ─────────────────────────
            with gr.Column(scale=1):
                inp_image   = gr.Image(type="pil", label="📤 Upload ảnh blur")

                gr.Markdown("### ⚙️ Settings")
                use_persp   = gr.Checkbox(value=True,
                                          label="🔲 Tự động sửa góc (Perspective Correction)")
                sharpen_sl  = gr.Slider(minimum=0.0, maximum=1.5, value=0.6, step=0.1,
                                        label="✨ Cường độ Sharpen (0=tắt, 0.6=mặc định, 1.5=mạnh)")
                use_clahe   = gr.Checkbox(value=False,
                                          label="🎨 CLAHE Contrast Enhancement (dùng nếu ảnh tối)")

                run_btn     = gr.Button("🚀 Chạy Deblur", variant="primary", size="lg")
                status_box  = gr.Textbox(label="📋 Trạng thái", lines=4, interactive=False)
                dl_file     = gr.File(label="💾 Download kết quả (.png)")

            # ── Cột phải: Kết quả ─────────────────────────
            with gr.Column(scale=3):
                with gr.Row():
                    out_input  = gr.Image(label="📷 Ảnh gốc (input)")
                    out_rect   = gr.Image(label="📐 Sau sửa góc")
                with gr.Row():
                    out_deblur = gr.Image(label="🔍 Sau Deblur (NAFNet)")
                    out_sharp  = gr.Image(label="✨ Sau Sharpen (Final)")

        gr.Markdown("""
        ---
        ### 📖 Hướng dẫn
        1. Upload ảnh thẻ bị mờ (jpg/png)
        2. Điều chỉnh settings nếu cần:
           - **Sửa góc**: bật nếu ảnh chụp xiên, tắt nếu ảnh đã thẳng
           - **Sharpen**: tăng nếu chữ vẫn còn hơi mờ, **giảm** nếu ảnh bị quá sắc/nhiễu
           - **CLAHE**: bật nếu ảnh tối, thiếu contrast
        3. Bấm **Chạy Deblur**
        4. Cột **Sau Sharpen** là kết quả cuối cùng để dùng

        ### 💡 Tips
        - Sharpen 0.4–0.6: phù hợp cho hầu hết trường hợp
        - Sharpen 0.8–1.0: dùng khi chữ vẫn mờ sau deblur
        - Sharpen > 1.2: có thể tạo nhiễu quanh cạnh chữ
        """)

        run_btn.click(
            fn=process,
            inputs=[inp_image, use_persp, sharpen_sl, use_clahe],
            outputs=[out_input, out_rect, out_deblur, out_sharp, status_box, dl_file]
        )

    return demo


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Tạo public link để demo online")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌ Không tìm thấy checkpoint: {ckpt}")
        sys.exit(1)

    print("Đang load model...")
    MODEL, DEVICE = load_model(ckpt)

    print("Khởi động giao diện web...")
    ui = build_ui()
    ui.launch(server_port=args.port, share=args.share, inbrowser=True)