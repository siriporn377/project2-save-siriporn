# kmeans_colors.py
import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# ===== Path setup =====
base_dir   = os.path.dirname(os.path.abspath(__file__))
photo_dir  = os.path.join(base_dir, 'photo')                 # รูปต้นฉบับอยู่ที่นี่
image_name = sys.argv[1] if len(sys.argv) > 1 else 'input26.jpg'
input_path = os.path.join(photo_dir, image_name)

# โฟลเดอร์ผลลัพธ์แยกตามชื่อรูป
name_stem  = os.path.splitext(os.path.basename(image_name))[0]
out_dir    = os.path.join(base_dir, 'output', 'kmeans', name_stem)
os.makedirs(out_dir, exist_ok=True)

print(f'>> Input path : {input_path}')
print(f'>> Output dir : {out_dir}')

# ===== Load image =====
img0 = cv.imread(input_path)
if img0 is None:
    raise FileNotFoundError(f'ไม่พบไฟล์ภาพ: {input_path}')

img1 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
print('img size:', img0.shape[0], img0.shape[1], img0.shape[0] * img0.shape[1])

# ===== Prepare data for KMeans =====
Z = img1.reshape((-1, 3)).astype(np.float32)
print('Z.shape:', Z.shape)
print('number of pixels:', len(Z))

# ===== KMeans =====
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 9
compactness, label, center = cv.kmeans(
    Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
)

print('np.shape(label):', label.shape)

# ===== Build cluster images (isolate each cluster, others white) =====
center_u8 = np.uint8(center)  # สีตัวแทนของแต่ละคลัสเตอร์
cluster_img = []
colors = []

# สร้างแมพสีไว้ล่วงหน้าเพื่อลดงานซ้ำ
label_flat = label.flatten()

for i in range(K):
    # temp คือพาเลตต์สี K ชุด โดยเซ็ตคลัสเตอร์อื่นเป็นขาว
    temp = center_u8.copy()
    for j in range(K):
        if i != j:
            temp[j] = np.array([255, 255, 255], dtype=np.uint8)  # white

    colors.append(center_u8[i].tolist())
    res = temp[label_flat]
    res = res.reshape(img1.shape)

    cluster_img.append(res)
    out_path = os.path.join(out_dir, f"{name_stem}_K{i+1}.jpg")
    cv.imwrite(out_path, cv.cvtColor(res, cv.COLOR_RGB2BGR))
    print(f"saved: {out_path}")

# ===== Save preview grid (แทนการ plt.show() ให้บันทึกเป็นไฟล์) =====
figure_size = (9, 7)
plt.figure(figsize=figure_size)

# ช่องที่ 1: รูปต้นฉบับ
plt.subplot(3, 3, 1)
plt.imshow(img1)
plt.title('Input')
plt.xticks([]); plt.yticks([])

# ช่องที่ 2-9: แต่ละคลัสเตอร์
for i in range(2, 10):
    idx = i - 2
    if idx < len(cluster_img):
        plt.subplot(3, 3, i)
        plt.imshow(cluster_img[idx])
        plt.title(f'K{idx+1} {colors[idx]}')
        plt.xticks([]); plt.yticks([])

grid_path = os.path.join(out_dir, f"{name_stem}_grid.png")
plt.tight_layout()
plt.savefig(grid_path, dpi=150)
plt.close()
print(f"saved grid preview: {grid_path}")

print("✅ Done.")
