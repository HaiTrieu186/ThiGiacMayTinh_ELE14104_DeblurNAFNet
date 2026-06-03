#!/usr/bin/env python3
"""
STEP 6: Giao Diện Web Demo (Gradio) — v2.2
==========================================
  ✅ Nút "Tạo Blur Thử" — blur ảnh sắc nét để test tại chỗ
  ✅ Slider chọn loại blur (motion / defocus / gaussian)
  ✅ Slider chọn mức độ blur (nhẹ / vừa / nặng)
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
        raise RuntimeError("NAFNet chưa cài!")

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
    if amount <= 0:
        return img
    ksize = int(6 * radius + 1) | 1
    blur  = cv2.GaussianBlur(img, (ksize, ksize), radius)
    sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── TẠO BLUR THỬ ────────────────────────────────────────────
def make_blur(pil_image, blur_type, blur_level):
    """
    Tạo blur nhân tạo lên ảnh sắc nét để test.
    blur_type : "Motion", "Defocus", "Gaussian"
    blur_level: 1 (nhẹ) → 3 (nặng)
    """
    if pil_image is None:
        return None, "⚠️ Chưa upload ảnh!"

    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Map level → kernel size
    level_map = {
        1: {"motion": 7,  "defocus": 2, "gaussian": 1.2},
        2: {"motion": 13, "defocus": 4, "gaussian": 2.2},
        3: {"motion": 21, "defocus": 6, "gaussian": 3.5},
    }
    level = int(blur_level)
    params = level_map.get(level, level_map[2])

    if blur_type == "Motion":
        size  = params["motion"]
        angle = 45  # góc cố định để dễ thấy
        k = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        import math
        rad = math.radians(angle)
        for i in range(size):
            offset = i - center
            x = int(round(center + offset * math.cos(rad)))
            y = int(round(center + offset * math.sin(rad)))
            if 0 <= x < size and 0 <= y < size:
                k[y, x] = 1.0
        total = k.sum()
        k = k / total if total > 0 else k
        blurred = cv2.filter2D(img, -1, k)
        desc = f"Motion blur (kernel {size}px, angle 45°)"

    elif blur_type == "Defocus":
        r    = params["defocus"]
        size = 2 * r + 1
        k    = np.zeros((size, size), dtype=np.float32)
        cv2.circle(k, (r, r), r, 1.0, -1)
        k    = k / k.sum()
        blurred = cv2.filter2D(img, -1, k)
        desc = f"Defocus blur (radius {r}px)"

    else:  # Gaussian
        sigma = params["gaussian"]
        ksize = int(6 * sigma + 1) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        desc = f"Gaussian blur (sigma {sigma})"

    # Thêm noise nhẹ cho realistic
    noise   = np.random.normal(0, 2.0, img.shape).astype(np.float32)
    blurred = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    result_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    return result_pil, f"✅ Đã tạo: {desc}\nBây giờ bấm 🚀 Chạy Deblur để xem kết quả!"


# ── Hàm xử lý deblur chính ──────────────────────────────────
def process(pil_image, use_perspective, sharpen_amount, use_clahe):
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

    rect_rgb    = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
    deblur_rgb  = cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB)
    sharpen_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

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
        # 🃏 Card Deblur — NAFNet Demo
        Upload ảnh thẻ bị mờ → model tự sửa góc → khôi phục nét → tăng sắc.

        **Hoạt động tốt với:** Bài tây, Yugioh, CCCD, thẻ ngân hàng...
        """)

        with gr.Row():
            # ── Cột trái: Controls ─────────────────────────
            with gr.Column(scale=1):
                inp_image = gr.Image(type="pil", label="📤 Upload ảnh")

                # ── Blur thử (optional) ──────────────────
                gr.Markdown("### 🧪 Tạo Blur Thử (nếu ảnh chưa blur)")
                gr.Markdown("*Upload ảnh sắc nét → chọn loại blur → bấm nút → ảnh blur tự điền vào ô trên*")
                blur_type  = gr.Radio(
                    choices=["Motion", "Defocus", "Gaussian"],
                    value="Motion",
                    label="Loại blur"
                )
                blur_level = gr.Slider(
                    minimum=1, maximum=3, value=2, step=1,
                    label="Mức độ blur (1=nhẹ, 2=vừa, 3=nặng)"
                )
                blur_btn   = gr.Button("🌀 Tạo Blur Thử", variant="secondary")
                blur_msg   = gr.Textbox(label="", lines=2, interactive=False)

                gr.Markdown("---")

                # ── Deblur settings ──────────────────────
                gr.Markdown("### ⚙️ Settings Deblur")
                use_persp  = gr.Checkbox(value=True,
                                         label="🔲 Tự động sửa góc (Perspective)")
                sharpen_sl = gr.Slider(minimum=0.0, maximum=1.5, value=0.6, step=0.1,
                                       label="✨ Sharpen (0=tắt, 0.6=mặc định, 1.5=mạnh)")
                use_clahe  = gr.Checkbox(value=False,
                                         label="🎨 CLAHE (bật nếu ảnh tối)")

                run_btn    = gr.Button("🚀 Chạy Deblur", variant="primary", size="lg")
                status_box = gr.Textbox(label="📋 Trạng thái", lines=4, interactive=False)
                dl_file    = gr.File(label="💾 Download kết quả")

            # ── Cột phải: Kết quả ─────────────────────────
            with gr.Column(scale=3):
                with gr.Row():
                    out_input  = gr.Image(label="📷 Input")
                    out_rect   = gr.Image(label="📐 Sau sửa góc")
                with gr.Row():
                    out_deblur = gr.Image(label="🔍 Sau Deblur")
                    out_sharp  = gr.Image(label="✨ Final (Deblur + Sharpen)")

        gr.Markdown("""
        ---
        ### 💡 Cách dùng nhanh
        **Có sẵn ảnh blur:** Upload → Chạy Deblur luôn

        **Ảnh sắc nét muốn test:** Upload → Chọn loại blur + mức độ → Bấm Tạo Blur Thử → Bấm Chạy Deblur

        **Sharpen:** 0.4–0.6 cho hầu hết trường hợp | 0.8–1.0 nếu chữ vẫn mờ | >1.2 có thể tạo nhiễu
        """)

        # ── Events ──────────────────────────────────────
        blur_btn.click(
            fn=make_blur,
            inputs=[inp_image, blur_type, blur_level],
            outputs=[inp_image, blur_msg]
        )

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
    parser.add_argument("--share", action="store_true")
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