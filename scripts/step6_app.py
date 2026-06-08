#!/usr/bin/env python3
"""
STEP 6: Giao Diện Desktop Demo (CustomTkinter)
===============================================
Chạy: python scripts/step6_app.py --checkpoint pretrained/nafnet_cards_best_v2.pth
"""

import cv2
import numpy as np
import torch
import sys
import argparse
import math
import threading
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except ImportError:
    print("❌ Chưa cài customtkinter. Chạy: pip install customtkinter")
    sys.exit(1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ── Load model ──────────────────────────────────────────────
def load_model(checkpoint_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "NAFNet"))
    try:
        from basicsr.models.archs.NAFNet_arch import NAFNet
    except ImportError:
        raise RuntimeError("NAFNet chưa cài!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = NAFNet(img_channel=3, width=32, middle_blk_num=12,
                    enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(checkpoint_path, map_location=device)
    if   'params'     in ckpt: sd = ckpt['params']
    elif 'state_dict' in ckpt: sd = ckpt['state_dict']
    else:                      sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, device


# ── Image processing ─────────────────────────────────────────
def detect_card_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in [(30, 100), (20, 80), (50, 150)]:
        edges = cv2.Canny(blurred, lo, hi)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        img_area = img.shape[0] * img.shape[1]
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if cv2.contourArea(cnt) < img_area * 0.08: break
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
    dst = np.float32([[0,0],[size,0],[size,size],[0,size]])
    M   = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, M, (size, size),
                               flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_REPLICATE), True


def nafnet_infer(model, device, img_bgr):
    t = torch.from_numpy(img_bgr/255.).permute(2,0,1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    return (out.squeeze().permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)


def unsharp_mask(img, amount=0.6):
    if amount <= 0: return img
    ksize = 7
    blur  = cv2.GaussianBlur(img, (ksize, ksize), 1.0)
    return np.clip(cv2.addWeighted(img, 1+amount, blur, -amount, 0), 0, 255).astype(np.uint8)


def apply_blur_fn(img_pil, blur_type, level):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    lmap = {1:(7,2,1.2), 2:(13,4,2.2), 3:(21,6,3.5)}
    motion_k, defocus_r, gauss_s = lmap[level]

    if blur_type == "Motion":
        k = np.zeros((motion_k, motion_k), np.float32)
        c = motion_k // 2
        r = math.radians(45)
        for i in range(motion_k):
            o = i - c
            x, y = int(round(c + o*math.cos(r))), int(round(c + o*math.sin(r)))
            if 0 <= x < motion_k and 0 <= y < motion_k:
                k[y, x] = 1.0
        k /= (k.sum() or 1)
        img = cv2.filter2D(img, -1, k)
    elif blur_type == "Defocus":
        s = 2*defocus_r+1
        k = np.zeros((s, s), np.float32)
        cv2.circle(k, (defocus_r, defocus_r), defocus_r, 1.0, -1)
        img = cv2.filter2D(img, -1, k/k.sum())
    else:
        ks = int(6*gauss_s+1)|1
        img = cv2.GaussianBlur(img, (ks, ks), gauss_s)

    noise = np.random.normal(0, 2.0, img.shape).astype(np.float32)
    img   = np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_ctk(pil_img, size=(300, 300)):
    """Chuyển PIL Image sang CTkImage để hiển thị."""
    pil_img = pil_img.copy()
    pil_img.thumbnail(size, Image.LANCZOS)
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)


# ── App chính ────────────────────────────────────────────────
class DeblurApp(ctk.CTk):
    def __init__(self, model, device):
        super().__init__()
        self.model  = model
        self.device = device

        self.img_original  = None   # ảnh gốc lúc upload
        self.img_working   = None   # ảnh đang làm việc (sau crop/blur)
        self.result_pil    = None   # kết quả deblur để save
        self._before_blur  = None   # lưu ảnh trước khi áp blur

        # Crop state
        self.crop_start  = None
        self.crop_rect   = None
        self.canvas_img  = None

        self.title("🃏 Card Deblur — NAFNet")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self._build_ui()

    # ── Build UI ─────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── SIDEBAR ──────────────────────────────────────
        sidebar = ctk.CTkScrollableFrame(self, width=300, corner_radius=0,
                                          fg_color="#1e293b")
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        # Logo
        ctk.CTkLabel(sidebar, text="🃏  Card Deblur",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#f1f5f9").grid(row=0, column=0, pady=(20,2), padx=16, sticky="w")
        ctk.CTkLabel(sidebar, text="NAFNet · PSNR 35dB · PTIT ELE14104",
                     font=ctk.CTkFont(size=11), text_color="#64748b").grid(row=1, column=0, padx=16, sticky="w")

        self._sep(sidebar, 2)

        # ── Upload ───────────────────────────────────────
        self._section(sidebar, 3, "① UPLOAD ẢNH")
        self.open_btn = ctk.CTkButton(sidebar, text="📂  Chọn ảnh từ tập test...",
                      command=self._open_file, height=38,
                      font=ctk.CTkFont(size=13, weight="bold"))
        self.open_btn.grid(row=4, column=0, padx=16, pady=(4,2), sticky="ew")
        ctk.CTkLabel(sidebar, text="Hoặc kéo thả vào vùng xem bên phải",
                     font=ctk.CTkFont(size=11), text_color="#64748b").grid(
            row=5, column=0, padx=16, sticky="w")

        self._sep(sidebar, 6)

        # ── Crop ─────────────────────────────────────────
        self._section(sidebar, 7, "② CROP THỦ CÔNG")
        ctk.CTkLabel(sidebar, text="Kéo chuột trên ảnh để chọn vùng crop",
                     font=ctk.CTkFont(size=11), text_color="#94a3b8",
                     wraplength=260).grid(row=8, column=0, padx=16, sticky="w")

        crop_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        crop_row.grid(row=9, column=0, padx=16, pady=(6,2), sticky="ew")
        crop_row.grid_columnconfigure((0,1), weight=1)

        ctk.CTkButton(crop_row, text="✂️  Crop vùng chọn",
                      command=self._do_crop, height=34,
                      font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0,4), sticky="ew")
        ctk.CTkButton(crop_row, text="↺  Reset crop",
                      command=self._reset_crop, height=34,
                      fg_color="#334155", hover_color="#475569",
                      font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="ew")

        self._sep(sidebar, 10)

        # ── Blur thử ─────────────────────────────────────
        self._section(sidebar, 11, "③ TẠO BLUR THỬ  (tùy chọn)")
        ctk.CTkLabel(sidebar, text="Dùng nếu ảnh chưa blur sẵn",
                     font=ctk.CTkFont(size=11), text_color="#64748b").grid(
            row=12, column=0, padx=16, sticky="w")

        ctk.CTkLabel(sidebar, text="Loại blur:", font=ctk.CTkFont(size=12),
                     text_color="#cbd5e1").grid(row=13, column=0, padx=16, pady=(8,2), sticky="w")
        self.blur_type = ctk.CTkSegmentedButton(
            sidebar, values=["Motion", "Defocus", "Gaussian"])
        self.blur_type.set("Motion")
        self.blur_type.grid(row=14, column=0, padx=16, sticky="ew")

        ctk.CTkLabel(sidebar, text="Mức độ (1=nhẹ, 3=nặng):",
                     font=ctk.CTkFont(size=12), text_color="#cbd5e1").grid(
            row=15, column=0, padx=16, pady=(8,2), sticky="w")
        self.blur_level = ctk.CTkSlider(sidebar, from_=1, to=3, number_of_steps=2)
        self.blur_level.set(2)
        self.blur_level.grid(row=16, column=0, padx=16, sticky="ew")

        blur_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        blur_row.grid(row=17, column=0, padx=16, pady=(8,2), sticky="ew")
        blur_row.grid_columnconfigure((0,1), weight=1)

        ctk.CTkButton(blur_row, text="🌀  Áp Blur",
                      command=self._apply_blur, height=34,
                      font=ctk.CTkFont(size=12)).grid(row=0, column=0, padx=(0,4), sticky="ew")
        ctk.CTkButton(blur_row, text="↺  Undo blur",
                      command=self._undo_blur, height=34,
                      fg_color="#334155", hover_color="#475569",
                      font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="ew")

        self._sep(sidebar, 18)

        # ── Deblur settings ───────────────────────────────
        self._section(sidebar, 19, "④ CÀI ĐẶT & DEBLUR")

        # Mode chọn loại ảnh input
        ctk.CTkLabel(sidebar, text="Loại ảnh input:",
                     font=ctk.CTkFont(size=12), text_color="#cbd5e1").grid(
            row=19, column=0, padx=16, pady=(8,2), sticky="w")
        self.input_mode = ctk.CTkSegmentedButton(
            sidebar,
            values=["📂 Dataset", "📱 Ảnh thực"],
            command=self._on_mode_change)
        self.input_mode.set("📂 Dataset")
        self.input_mode.grid(row=20, column=0, padx=16, pady=(0,6), sticky="ew")

        self.use_persp = ctk.CTkCheckBox(sidebar, text="Tự động sửa góc (chỉ dùng cho ảnh thực)",
                                          text_color="#64748b", font=ctk.CTkFont(size=12),
                                          state="disabled")
        self.use_persp.grid(row=21, column=0, padx=16, pady=(2,2), sticky="w")

        self.use_clahe = ctk.CTkCheckBox(sidebar, text="CLAHE contrast (nếu ảnh tối)",
                                          text_color="#cbd5e1", font=ctk.CTkFont(size=12))
        self.use_clahe.grid(row=22, column=0, padx=16, pady=(2,4), sticky="w")

        ctk.CTkLabel(sidebar, text="Sharpen:", font=ctk.CTkFont(size=12),
                     text_color="#cbd5e1").grid(row=23, column=0, padx=16, pady=(4,0), sticky="w")
        self.sharpen_val = ctk.CTkSlider(sidebar, from_=0, to=1.5, number_of_steps=15,
                                          command=self._update_sharpen_label)
        self.sharpen_val.set(0.6)
        self.sharpen_val.grid(row=24, column=0, padx=16, sticky="ew")
        self.sharpen_label = ctk.CTkLabel(sidebar, text="0.6",
                                           font=ctk.CTkFont(size=11), text_color="#64748b")
        self.sharpen_label.grid(row=25, column=0, padx=16, sticky="w")

        ctk.CTkButton(sidebar, text="🚀  Chạy Deblur",
                      command=self._run_deblur, height=44,
                      font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=26, column=0, padx=16, pady=(12,4), sticky="ew")

        ctk.CTkButton(sidebar, text="💾  Lưu kết quả...",
                      command=self._save_result, height=36,
                      fg_color="#334155", hover_color="#475569",
                      font=ctk.CTkFont(size=12)).grid(
            row=28, column=0, padx=16, pady=(0,4), sticky="ew")

        self._sep(sidebar, 27)
        self.status_label = ctk.CTkLabel(
            sidebar, text="Chờ upload ảnh...",
            font=ctk.CTkFont(size=11), text_color="#64748b",
            wraplength=260, justify="left")
        self.status_label.grid(row=29, column=0, padx=16, pady=(4,20), sticky="w")

        # ── MAIN CONTENT ─────────────────────────────────
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f172a")
        main.grid(row=0, column=1, sticky="nsew")
        # cấu hình grid xử lý bên dưới khi tạo layout

        # Layout: cột trái to (Input nhỏ + Final TO), cột phải nhỏ (2 ảnh phụ)
        main.grid_columnconfigure(0, weight=3)  # cột trái chiếm 3/4
        main.grid_columnconfigure(1, weight=1)  # cột phải chiếm 1/4
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        # ── Cột trái: Input (nhỏ) + Final (TO) ──────────
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(12,6), pady=12)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)   # Input
        left.grid_rowconfigure(3, weight=2)   # Final to gấp đôi Input

        ctk.CTkLabel(left, text="📷  Input",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#64748b").grid(row=0, column=0, sticky="w", pady=(0,2))

        self.canvas = tk.Canvas(left, bg="#1e293b", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=(0,4))
        self.canvas.bind("<ButtonPress-1>",  self._crop_start)
        self.canvas.bind("<B1-Motion>",       self._crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self._crop_end)
        self.canvas.bind("<Enter>", lambda e: self.canvas.configure(bg="#243447"))
        self.canvas.bind("<Leave>", lambda e: self.canvas.configure(bg="#1e293b"))

        ctk.CTkLabel(left, text="✨  Final — Kết quả cuối (Deblur + Sharpen)",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#6366f1").grid(row=2, column=0, sticky="w", pady=(8,2))

        self.lbl_sharp = ctk.CTkLabel(left, text="", fg_color="#0d1a2d", corner_radius=10)
        self.lbl_sharp.grid(row=3, column=0, sticky="nsew")

        # ── Cột phải: 2 ảnh phụ nhỏ ─────────────────────
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6,12), pady=12)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        right.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(right, text="📐  Sau sửa góc",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#64748b").grid(row=0, column=0, sticky="w", pady=(0,2))
        self.lbl_rect = ctk.CTkLabel(right, text="", fg_color="#1e293b", corner_radius=8)
        self.lbl_rect.grid(row=1, column=0, sticky="nsew", pady=(0,4))

        ctk.CTkLabel(right, text="🔍  Sau Deblur",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#64748b").grid(row=2, column=0, sticky="w", pady=(8,2))
        self.lbl_deblur = ctk.CTkLabel(right, text="", fg_color="#1e293b", corner_radius=8)
        self.lbl_deblur.grid(row=3, column=0, sticky="nsew")

        # Placeholders
        self.canvas.create_text(150, 100,
            text="Chọn ảnh để bắt đầu\nhoặc dùng nút Chọn ảnh",
            fill="#334155", font=("Helvetica", 12), justify="center", tags="placeholder")
        for w in [self.lbl_rect, self.lbl_deblur, self.lbl_sharp]:
            w.configure(text="—", text_color="#334155", font=ctk.CTkFont(size=28))

    # ── Helpers ──────────────────────────────────────────────
    def _section(self, parent, row, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#475569").grid(row=row, column=0, padx=16,
                                                pady=(16,4), sticky="w")

    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color="#334155").grid(
            row=row, column=0, padx=16, pady=4, sticky="ew")

    def _img_panel(self, parent, row, col, padx=None):
        lbl = ctk.CTkLabel(parent, text="", fg_color="#1e293b",
                            corner_radius=8)
        if padx is None:
            padx = (8 if col else 16, 16 if col else 8)
        lbl.grid(row=row, column=col, padx=padx, pady=(0, 8), sticky="nsew")
        return lbl

    def _set_status(self, msg, color="#94a3b8"):
        self.status_label.configure(text=msg, text_color=color)

    def _update_sharpen_label(self, val):
        self.sharpen_label.configure(text=f"{float(val):.1f}")

    # ── Upload ───────────────────────────────────────────────
    def _open_file(self):
        mode = self.input_mode.get()
        if mode == "📂 Dataset":
            # Tự mở thẳng vào data/test/blur/
            init_dir = Path(__file__).parent.parent / "data" / "test" / "blur"
            if not init_dir.exists():
                init_dir = Path.home()
        else:
            init_dir = Path.home()

        path = filedialog.askopenfilename(
            initialdir=str(init_dir),
            title="Chọn ảnh" + (" từ tập test" if mode == "📂 Dataset" else " thực tế"),
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self._load_image(path)

    def _on_mode_change(self, mode):
        """Khi đổi mode → đổi label nút + bật/tắt perspective."""
        if mode == "📱 Ảnh thực":
            self.open_btn.configure(text="🖼️  Chọn ảnh thực tế...")
            self.use_persp.configure(state="normal", text_color="#cbd5e1")
            self.use_persp.select()
        else:
            self.open_btn.configure(text="📂  Chọn ảnh từ tập test...")
            self.use_persp.configure(state="disabled", text_color="#64748b")
            self.use_persp.deselect()

    def _load_image(self, path):
        pil = Image.open(path).convert("RGB")
        self.img_original = pil
        self.img_working  = pil.copy()
        self._show_input(pil)
        self._clear_results()
        self._set_status(f"✅ Đã tải: {Path(path).name}")

    def _show_input(self, pil_img):
        """Hiển thị ảnh lên canvas, giữ aspect ratio."""
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(),  200)
        ch = max(self.canvas.winfo_height(), 200)
        self.canvas.delete("all")

        img = pil_img.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._tk_input = ImageTk.PhotoImage(img)

        # Căn giữa
        x = (cw - img.width)  // 2
        y = (ch - img.height) // 2
        self._canvas_offset = (x, y)
        self._canvas_img_size = (img.width, img.height)
        self._current_display_pil = pil_img  # để crop tính tỉ lệ

        self.canvas.create_image(x, y, anchor="nw", image=self._tk_input, tags="img")

    def _clear_results(self):
        for lbl in [self.lbl_rect, self.lbl_deblur, self.lbl_sharp]:
            lbl.configure(image=None, text="—", text_color="#334155",
                          font=ctk.CTkFont(size=30))
        self.result_pil = None

    # ── Crop bằng chuột ──────────────────────────────────────
    def _crop_start(self, e):
        if self.img_working is None: return
        self.crop_start = (e.x, e.y)
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)

    def _crop_drag(self, e):
        if not self.crop_start: return
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
        self.crop_rect = self.canvas.create_rectangle(
            self.crop_start[0], self.crop_start[1], e.x, e.y,
            outline="#6366f1", width=2, dash=(4, 2))

    def _crop_end(self, e):
        if not self.crop_start: return
        self.crop_end_pos = (e.x, e.y)

    def _do_crop(self):
        if self.img_working is None:
            return self._set_status("⚠️ Chưa có ảnh!", "#f59e0b")
        if not hasattr(self, 'crop_end_pos') or not self.crop_start:
            return self._set_status("⚠️ Hãy kéo chuột trên ảnh để chọn vùng crop", "#f59e0b")

        ox, oy   = self._canvas_offset
        cw, ch   = self._canvas_img_size
        pw, ph   = self._current_display_pil.size

        # Canvas coords → image coords
        x1c, y1c = self.crop_start
        x2c, y2c = self.crop_end_pos

        # Clamp vào vùng ảnh
        x1c = max(ox, min(x1c, ox + cw))
        y1c = max(oy, min(y1c, oy + ch))
        x2c = max(ox, min(x2c, ox + cw))
        y2c = max(oy, min(y2c, oy + ch))

        # Tỉ lệ canvas → ảnh gốc
        rx, ry = pw / cw, ph / ch
        x1 = int((x1c - ox) * rx)
        y1 = int((y1c - oy) * ry)
        x2 = int((x2c - ox) * rx)
        y2 = int((y2c - oy) * ry)

        if abs(x2-x1) < 10 or abs(y2-y1) < 10:
            return self._set_status("⚠️ Vùng crop quá nhỏ!", "#f59e0b")

        x1, x2 = min(x1,x2), max(x1,x2)
        y1, y2 = min(y1,y2), max(y1,y2)

        self.img_working = self._current_display_pil.crop((x1, y1, x2, y2))
        self._show_input(self.img_working)
        self.crop_start   = None
        self.crop_end_pos = None
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
            self.crop_rect = None
        self._set_status(f"✅ Đã crop: {x2-x1}×{y2-y1}px", "#22c55e")

    def _reset_crop(self):
        if self.img_original is None:
            return self._set_status("⚠️ Chưa có ảnh gốc!", "#f59e0b")
        self.img_working = self.img_original.copy()
        self._show_input(self.img_working)
        self.crop_start   = None
        self.crop_end_pos = None
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
            self.crop_rect = None
        self._set_status("✅ Đã reset về ảnh gốc", "#22c55e")

    # ── Blur thử ─────────────────────────────────────────────
    def _apply_blur(self):
        if self.img_working is None:
            return self._set_status("⚠️ Chưa có ảnh!", "#f59e0b")
        self._before_blur = self.img_working.copy()
        level = int(round(self.blur_level.get()))
        self.img_working = apply_blur_fn(self.img_working, self.blur_type.get(), level)
        self._show_input(self.img_working)
        self._set_status(f"✅ Đã áp {self.blur_type.get()} blur mức {level}", "#22c55e")

    def _undo_blur(self):
        if self._before_blur is None:
            return self._set_status("⚠️ Chưa áp blur nào!", "#f59e0b")
        self.img_working = self._before_blur
        self._before_blur = None
        self._show_input(self.img_working)
        self._set_status("✅ Đã undo blur", "#22c55e")

    # ── Deblur ───────────────────────────────────────────────
    def _run_deblur(self):
        if self.img_working is None:
            return self._set_status("⚠️ Chưa có ảnh!", "#f59e0b")
        self._set_status("⏳ Đang xử lý...", "#6366f1")
        threading.Thread(target=self._deblur_thread, daemon=True).start()

    def _deblur_thread(self):
        try:
            img = self.img_working.convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            mode = self.input_mode.get()

            if mode == "📂 Dataset":
                # Feed thẳng như step4 — không resize, không warp
                # Chỉ resize nếu ảnh không phải 512x512
                h, w = img_bgr.shape[:2]
                if h != 512 or w != 512:
                    img_bgr = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                rectified = img_bgr
                persp     = "📂 Dataset mode — feed trực tiếp"
            elif self.use_persp.get():
                rectified, found = perspective_correct(img_bgr)
                persp = "✅ Sửa góc" if found else "⚠️ Không thấy viền"
            else:
                rectified = cv2.resize(img_bgr, (512, 512))
                persp     = "— Bỏ qua sửa góc"

            # Deblur
            deblurred = nafnet_infer(self.model, self.device, rectified)

            # Sharpen
            sharpened = unsharp_mask(deblurred, amount=self.sharpen_val.get())
            if self.use_clahe.get():
                lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(2.0, (8,8))
                sharpened = cv2.cvtColor(cv2.merge([clahe.apply(l),a,b]), cv2.COLOR_LAB2BGR)

            # Convert to PIL
            rect_pil   = Image.fromarray(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
            deblur_pil = Image.fromarray(cv2.cvtColor(deblurred, cv2.COLOR_BGR2RGB))
            sharp_pil  = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            self.result_pil = sharp_pil

            # Update UI (phải chạy trên main thread)
            self.after(0, self._update_results, rect_pil, deblur_pil, sharp_pil, persp)

        except Exception as ex:
            self.after(0, self._set_status, f"❌ Lỗi: {ex}", "#ef4444")

    def _update_results(self, rect, deblur, sharp, persp_msg):
        def show(lbl, pil):
            lbl.update_idletasks()
            w = max(lbl.winfo_width(),  200)
            h = max(lbl.winfo_height(), 200)
            cimg = pil.copy()
            cimg.thumbnail((w-16, h-16), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=cimg, dark_image=cimg, size=cimg.size)
            lbl.configure(image=ctk_img, text="")
            lbl._ctk_img = ctk_img  # giữ reference

        show(self.lbl_rect,   rect)
        show(self.lbl_deblur, deblur)
        show(self.lbl_sharp,  sharp)
        self._set_status(f"{persp_msg}  ·  ✅ Deblur xong  ·  Sharpen {self.sharpen_val.get():.1f}",
                         "#22c55e")

    # ── Save ─────────────────────────────────────────────────
    def _save_result(self):
        if self.result_pil is None:
            return self._set_status("⚠️ Chưa có kết quả để lưu!", "#f59e0b")
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="deblur_result.png")
        if path:
            self.result_pil.save(path)
            self._set_status(f"💾 Đã lưu: {Path(path).name}", "#22c55e")


# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"❌ Không tìm thấy checkpoint: {ckpt}")
        sys.exit(1)

    print("Đang load model...")
    model, device = load_model(ckpt)
    print(f"✅ Model loaded on {device}")

    print("Khởi động giao diện...")
    app = DeblurApp(model, device)
    app.mainloop()