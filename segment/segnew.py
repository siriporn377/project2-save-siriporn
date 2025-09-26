# --- Better background removal + safe pipeline for segmentation/K-means (robust) ---

import os
import sys
import glob
import cv2
import numpy as np
from rembg import remove, new_session
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float
from sklearn.cluster import KMeans

# ===== Path Setup =====
base_dir = os.path.dirname(os.path.abspath(__file__))

# รองรับทั้งส่งชื่อไฟล์มาทาง argv หรือใช้ดีฟอลต์ 'photo/input4.jpg'
candidate_paths = []
if len(sys.argv) > 1:
    # ถ้าผู้ใช้ส่งพาธมาเอง
    p = sys.argv[1]
    candidate_paths.append(p if os.path.isabs(p) else os.path.join(base_dir, p))
else:
    # ลองหาใน photo/
    candidate_paths.extend([
        os.path.join(base_dir, 'photo', 'input22.jpg'),
        os.path.join(base_dir, 'photo', 'input22.png'),
    ])
    # ถ้าไม่เจอ ลองควานไฟล์แรกใน photo/*
    candidate_paths += sorted(glob.glob(os.path.join(base_dir, 'photo', '*.*')))

input_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
if input_path is None:
    raise FileNotFoundError(
        "ไม่พบไฟล์รูปในโฟลเดอร์ 'photo'. โปรดตรวจสอบว่ามีไฟล์ เช่น photo/input4.jpg หรือส่งพาธไฟล์มาที่ argv"
    )

output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)
print("📥 Loading from:", input_path)
print("📤 Output dir   :", output_dir)

# ===== Load image safely =====
img = cv2.imread(input_path, cv2.IMREAD_COLOR)
if img is None:
    raise IOError(f"cv2.imread อ่านไฟล์ไม่ได้: {input_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H0, W0 = img_rgb.shape[:2]

# ===== 1) Remove BG (isnet-general + alpha-matting) =====
# มี fallback ถ้าหาโมเดลนี้ไม่เจอ
try:
    session = new_session('isnet-general')
except Exception:
    print("⚠️ ใช้ 'isnet-general' ไม่ได้ → fallback เป็น 'u2net'")
    session = new_session('u2net')

# ขยายด้านยาวเพื่อให้ขอบเนียนขึ้น (ถ้าเครื่องไหว)
long_side = max(H0, W0)
target = 1536
scale = float(target) / long_side if long_side > target else 1.0
img_big = cv2.resize(img_rgb, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_CUBIC)

# เรียก rembg + รองรับเคสที่คืนค่าเป็น bytes
out_big = remove(
    img_big,
    session=session,
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=5,
    only_mask=False
)
if isinstance(out_big, (bytes, bytearray)):
    arr = np.frombuffer(out_big, dtype=np.uint8)
    img_rgba_big = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
else:
    img_rgba_big = out_big

if img_rgba_big is None:
    raise RuntimeError("rembg คืนค่า None/อ่านผลลัพธ์ไม่ได้")

# บางเคส rembg อาจคืนมาเป็น 3 แชนแนล → เติม alpha 255
if img_rgba_big.ndim == 3 and img_rgba_big.shape[2] == 3:
    a = np.full(img_rgba_big.shape[:2], 255, dtype=np.uint8)
    img_rgba_big = np.dstack([img_rgba_big, a])

# ย่อกลับขนาดเดิม
img_rgba = cv2.resize(img_rgba_big, (W0, H0), interpolation=cv2.INTER_AREA)

# แปลง BGRa→RGBa ถ้าจำเป็น (cv2 ใช้ BGR)
if img_rgba.shape[2] == 4:
    # ตอนนี้ img_rgba น่าจะอยู่ใน BGRa จาก imdecode; ให้แปลงเป็น RGBA เพื่อทำต่อในสเปกเดิม
    img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2RGBA)

alpha = img_rgba[:, :, 3]

# post-process alpha เล็กน้อย (ปิดรูรั่ว + feather ขอบนุ่ม)
kernel = np.ones((3, 3), np.uint8)
alpha_pp = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
alpha_pp = cv2.GaussianBlur(alpha_pp, (0, 0), sigmaX=1.2, sigmaY=1.2)
img_rgba[:, :, 3] = alpha_pp

# เซฟไฟล์โปร่งใส (ไว้พรีเซนต์/เก็บต่อ)
out_png = os.path.join(output_dir, 'input4_removed_bg.png')
cv2.imwrite(out_png, cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA))
print("✅ Better background saved:", out_png)

# ===== 2) เตรียมภาพสำหรับ segmentation แบบไม่ให้พื้นหลังมาป่วน =====
mask = (alpha_pp > 10).astype(np.uint8)
num_obj = int(mask.sum())
if num_obj == 0:
    raise RuntimeError("มาสก์วัตถุว่างเปล่า อาจลบพื้นหลังแรงไป ลองลด erode size หรือตัวแปร threshold")

img_rgb_only = img_rgba[:, :, :3].copy()
bg_color = (0, 0, 0)
img_rgb_only[mask == 0] = bg_color

# เบลอเฉพาะในวัตถุ (ช่วย Felzenszwalb ให้ด้านในเนียนขึ้น)
blurred = cv2.bilateralFilter(img_rgb_only, d=9, sigmaColor=75, sigmaSpace=75)
blurred[mask == 0] = bg_color

image_float = img_as_float(blurred)

# ===== 3) Felzenszwalb segmentation =====
segments = felzenszwalb(image_float, scale=100, sigma=0.5, min_size=50)

# กันนอกวัตถุ
segments_obj = segments.copy()
segments_obj[mask == 0] = -1

# Visualize (เฉพาะโชว์)
segmented_image = label2rgb(segments, image_float, kind='avg')
segmented_image[mask == 0] = (0, 0, 0)
segmented_img_uint8 = (segmented_image * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, 'input22_output_felzen.jpg'),
            cv2.cvtColor(segmented_img_uint8, cv2.COLOR_RGB2BGR))

np.save(os.path.join(output_dir, 'labels4.npy'), segments_obj)

# ===== 4) K-Means เฉพาะพิกเซลในวัตถุ =====
H, W, _ = segmented_img_uint8.shape
flat_rgb = segmented_img_uint8.reshape((-1, 3))
flat_mask = mask.reshape(-1)

obj_pixels = flat_rgb[flat_mask == 1]
if obj_pixels.size == 0:
    raise RuntimeError("ไม่มีพิกเซลของวัตถุสำหรับทำ K-means")

k = 10
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(obj_pixels)
centers_k = kmeans.cluster_centers_.astype(np.uint8)

# map label กลับเข้าเฟรมภาพเต็ม
labels_full = np.full((H * W,), fill_value=-1, dtype=np.int32)
labels_full[flat_mask == 1] = kmeans.labels_
labels_full = labels_full.reshape(H, W)

# ===== 5) Save K-means layers (พื้นหลังขาว ภาพไม่หาย) =====
kmeans_layer_dir = os.path.join(output_dir, 'kmeans_layers22')
os.makedirs(kmeans_layer_dir, exist_ok=True)

for i in range(k):
    layer = np.ones((H, W, 3), dtype=np.uint8) * 255  # BG ขาว
    mask_i = (labels_full == i)
    if np.any(mask_i):
        layer[mask_i] = centers_k[i]
    cv2.imwrite(os.path.join(kmeans_layer_dir, f'layer_kmeans_{i}.jpg'),
                cv2.cvtColor(layer, cv2.COLOR_RGB2BGR))

print("✔️ All layers saved in:", kmeans_layer_dir)
