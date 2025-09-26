from rembg import remove
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import os

# ========= จัดการ path หลัก =========
base_dir = os.path.dirname(os.path.abspath(__file__))             # path ของไฟล์ seg.py
input_path = os.path.join(base_dir, 'photo', 'input29.jpg')       # รูปต้นฉบับ
output_dir = os.path.join(base_dir, 'output')                    # output folder
os.makedirs(output_dir, exist_ok=True)                           # สร้างโฟลเดอร์ output ถ้ายังไม่มี

# ========= โหลดภาพ =========
img = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ========= ลบพื้นหลัง =========
img_no_bg = remove(img_rgb)                          # ได้ RGBA
img_no_bg_rgb = img_no_bg[:, :, :3]                  # เอาเฉพาะ RGB

# ========= ทำ SLIC segmentation =========
image_float = img_as_float(img_no_bg_rgb)
segments = slic(image_float, n_segments=100, compactness=10, sigma=1)
segmented_image = label2rgb(segments, image_float, kind='avg')   # ได้ RGB

# ========= แสดงภาพ =========
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_no_bg)
ax[1].set_title('Background Removed (with alpha)')
ax[1].axis('off')

ax[2].imshow(segmented_image)
ax[2].set_title('SLIC Segmentation')
ax[2].axis('off')

plt.tight_layout()
plt.show()

# ========= บันทึกภาพ =========

# 1. บันทึกภาพที่ลบพื้นหลัง (RGBA)
bg_removed_path = os.path.join(output_dir, 'bg_removed29.png')
cv2.imwrite(bg_removed_path, cv2.cvtColor(img_no_bg, cv2.COLOR_RGBA2BGRA))

# 2. บันทึกภาพ segmentation (RGB → BGR ก่อน)
segmented_bgr = cv2.cvtColor((segmented_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
segmented_path = os.path.join(output_dir, 'segmented-silc29.jpg')
cv2.imwrite(segmented_path, segmented_bgr)

# ========= log ตรวจสอบ =========
print(f"✅ Saved: {bg_removed_path}")
print(f"✅ Saved: {segmented_path}")