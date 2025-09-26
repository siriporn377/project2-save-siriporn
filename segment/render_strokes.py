import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass

#  Path Setup 
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'output')

img_path = os.path.join(output_dir, 'input25_output+kmean.jpg')  # **********************
label_path = os.path.join(output_dir, 'labels25.npy')

img_pil = Image.open(img_path).convert("RGBA")
img_np = np.array(img_pil)
img_rgb = img_np[:, :, :3]
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

segments = np.load(label_path)
H, W = segments.shape

canvas = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
frames = []

# ETF
def compute_etf(gray_img, iterations=5):
    gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=5)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    tx = -gy / mag
    ty = gx / mag
    for _ in range(iterations):
        tx = cv2.GaussianBlur(tx, (11, 11), 0)
        ty = cv2.GaussianBlur(ty, (11, 11), 0)
        norm = np.sqrt(tx**2 + ty**2) + 1e-6
        tx /= norm
        ty /= norm
    return tx, ty

tx, ty = compute_etf(gray)
segment_ids = np.unique(segments)
valid_segment_count = 0

# Stroke Rendering 
for seg_id in segment_ids:
    mask = (segments == seg_id)
    if np.sum(mask) < 100:
        continue

    brightness = np.mean(img_rgb[mask])
    if brightness > 240:
        continue

    valid_segment_count += 1
    mean_color = img_rgb[mask].mean(axis=0).astype(np.uint8)
    color = tuple(mean_color)

    cy, cx = center_of_mass(mask.astype(np.uint8))

    for _ in range(30):  # จำนวนเส้นต่อ segment
        x = cx + np.random.uniform(-10, 10)
        y = cy + np.random.uniform(-10, 10)
        path = [(x, y)]

        # เพิ่มความยาว stroke และให้โค้งขึ้น
        length = np.random.randint(40, 70)
        pressure_base = np.random.randint(1, 3)

        for i in range(length):
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                break
            if not mask[yi, xi]:
                break

            angle = np.arctan2(ty[yi, xi], tx[yi, xi])
            dx = np.cos(angle) * 2
            dy = np.sin(angle) * 2
            x += dx + np.random.normal(scale=0.2)
            y += dy + np.random.normal(scale=0.2)
            path.append((x, y))

        # น้ำหนักเส้นไม่เท่ากันทุกเส้น
        draw.line(path, fill=color, width=pressure_base + np.random.randint(0, 2))

    frames.append(canvas.copy())

print(f" วาดทั้งหมด: {valid_segment_count} segments")
print(f" สร้าง frame ได้: {len(frames)}")

# Save frames 
frame_dir = os.path.join(output_dir, 'frames_input25') #********************
os.makedirs(frame_dir, exist_ok=True)

for i, f in enumerate(frames):
    f.save(os.path.join(frame_dir, f"{i:03d}.png"))

print(f"บันทึกแล้ว {len(frames)} เฟรมใน: {frame_dir}")
