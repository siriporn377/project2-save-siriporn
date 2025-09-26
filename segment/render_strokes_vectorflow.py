import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

# === Path ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'output')
img_path = os.path.join(output_dir, 'bg_removed9.png')
frame_dir = os.path.join(output_dir, 'vectorflow_frames')
os.makedirs(frame_dir, exist_ok=True)

# === Load image ===
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ ไม่พบภาพ: {img_path}")

img_rgb = np.array(Image.open(img_path).convert("RGB"))
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
H, W = gray.shape

# === Compute Gradient & Normalize Vector Field ===
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
mag = np.sqrt(gx**2 + gy**2) + 1e-6

# ETF: Tangent vector orthogonal to gradient
tx = -gy / mag
ty = gx / mag

# === Smooth Vector Field (like ETF) ===
tx = cv2.GaussianBlur(tx, (5, 5), 0)
ty = cv2.GaussianBlur(ty, (5, 5), 0)

# === Setup canvas ===
canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
frames = []

# === Mask: avoid background (white/light areas)
mask = gray < 240
y_idx, x_idx = np.where(mask)

# === Draw strokes ===
for _ in range(400):  # จำนวน stroke ทั้งหมด
    idx = np.random.randint(len(x_idx))
    x, y = x_idx[idx], y_idx[idx]

    path = [(x, y)]
    for _ in range(15):  # ความยาว stroke
        dx = int(np.round(np.cos(np.arctan2(ty[y, x], tx[y, x])) * 2))
        dy = int(np.round(np.sin(np.arctan2(ty[y, x], tx[y, x])) * 2))
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H) or not mask[ny, nx]:
            break
        path.append((nx, ny))
        x, y = nx, ny

    # เอาสีจากภาพต้นฉบับ
    color = tuple(img_rgb[path[0][1], path[0][0]])
    draw.line(path, fill=color, width=2)
    frames.append(canvas.copy())

# === Save frames ===
for i, f in enumerate(frames):
    f.save(os.path.join(frame_dir, f"{i:03d}.png"))

print(f"✅ วาดด้วย Vector Field เสร็จแล้ว: {len(frames)} เฟรม → {frame_dir}")
