import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

# === Path Setup ===
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'output')

# === Input files ===
img_path = os.path.join(output_dir, 'bg_removed10.png')
label_path = os.path.join(output_dir, 'labels10.npy')
edge_path = os.path.join(output_dir, 'edge10.png')

# === Output folder ===
frame_dir = os.path.join(output_dir, 'layered_frames10')
os.makedirs(frame_dir, exist_ok=True)

# === Load required data ===
img_rgb = np.array(Image.open(img_path).convert("RGB"))
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
segments = np.load(label_path)
edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)

H, W = gray.shape
canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
frames = []

# ======== LAYER ①: Draw edges ========
edge_points = np.argwhere(edges > 0)
for i, (y, x) in enumerate(edge_points[::8]):
    draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 0, 0))
    if i % 20 == 0:
        frames.append(canvas.copy())

# ======== LAYER ②: Fill base color from SLIC segments ========
segment_ids = np.unique(segments)
for seg_id in segment_ids:
    mask = (segments == seg_id)
    if np.sum(mask) < 50:
        continue

    mean_color = img_rgb[mask].mean(axis=0).astype(np.uint8)
    y_idx, x_idx = np.where(mask)

    for _ in range(10):
        i = np.random.randint(len(x_idx))
        x, y = x_idx[i], y_idx[i]
        draw.ellipse((x-2, y-2, x+2, y+2), fill=tuple(mean_color))

    frames.append(canvas.copy())

# ======== LAYER ③: Stroke-by-stroke with Vector Flow ========
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
mag = np.sqrt(gx**2 + gy**2) + 1e-6
tx, ty = -gy / mag, gx / mag
tx = cv2.GaussianBlur(tx, (5, 5), 0)
ty = cv2.GaussianBlur(ty, (5, 5), 0)

mask = gray < 245
y_idx, x_idx = np.where(mask)

for _ in range(800):
    idx = np.random.randint(len(x_idx))
    x, y = x_idx[idx], y_idx[idx]
    path = [(x, y)]

    for _ in range(20):
        angle = np.arctan2(ty[y, x], tx[y, x])
        dx = int(np.round(np.cos(angle) * 2))
        dy = int(np.round(np.sin(angle) * 2))
        nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H) or not mask[ny, nx]:
            break
        path.append((nx, ny))
        x, y = nx, ny

    color = tuple(img_rgb[path[0][1], path[0][0]])
    draw.line(path, fill=color, width=2)
    frames.append(canvas.copy())

# === Save frames ===
for i, f in enumerate(frames):
    f.save(os.path.join(frame_dir, f"{i:03d}.png"))

print(f"✅ เสร็จแล้ว: วาดภาพ 3 ชั้นลงไปทีละเลเยอร์ใน {frame_dir}")
