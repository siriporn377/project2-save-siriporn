import os
import cv2
import numpy as np
from rembg import remove
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float

# === กำหนด path ไปยังโฟลเดอร์ภาพในเครื่อง ===
input_folder = r'C:\Users\User\segment100'
output_folder = input_folder  # เก็บผลลัพธ์ในที่เดียวกัน

# === นามสกุลภาพที่รองรับ ===
supported_exts = ('.jpg', '.jpeg', '.png', '.webp')

# === วนลูปรูปทั้งหมดในโฟลเดอร์ ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(supported_exts):
        input_path = os.path.join(input_folder, filename)
        name, _ = os.path.splitext(filename)

        # === โหลดและลบพื้นหลัง ===
        img = cv2.imread(input_path)
        if img is None:
            print(f"❌ โหลดไม่สำเร็จ: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_no_bg = remove(img_rgb)
        img_no_bg_rgb = img_no_bg[:, :, :3]

        # === ทำ Blur เพื่อลด noise และช่วย segmentation
        blurred = cv2.bilateralFilter(img_no_bg_rgb, d=9, sigmaColor=75, sigmaSpace=75)
        image_float = img_as_float(blurred)

        # === Segment ด้วย Felzenszwalb ===
        segments = felzenszwalb(image_float, scale=100, sigma=0.5, min_size=50)

        # === ทำภาพจาก segment
        segmented_image = label2rgb(segments, image_float, kind='avg')
        segmented_bgr = cv2.cvtColor((segmented_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # === บันทึกผลลัพธ์
        seg_img_path = os.path.join(output_folder, f'{name}_seg.jpg')
        label_path = os.path.join(output_folder, f'{name}_labels.npy')

        cv2.imwrite(seg_img_path, segmented_bgr)
        np.save(label_path, segments)

        print(f"✅ เสร็จ: {filename} → {name}_seg.jpg + {name}_labels.npy")
