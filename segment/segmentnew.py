# segment/segmentnew.py
import os, sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float

# ===================== Config =====================
ROOT = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(ROOT, "photo")
IMG_NAME  = sys.argv[1] if len(sys.argv) > 1 else "input49.jpg" ##############################################
IMG_PATH  = os.path.join(PHOTO_DIR, IMG_NAME)

NAME      = os.path.splitext(os.path.basename(IMG_NAME))[0]
OUT_DIR   = os.path.join(ROOT, "output", "kmeans", NAME)
os.makedirs(OUT_DIR, exist_ok=True)

# Felzenszwalb params
FELZ_SCALE   = 120 #ขยายขนาดสเกลของเซกเมนต์
FELZ_SIGMA   = 0.6 #เบลอภาพ
FELZ_MINSIZE = 60 #ขนาดเล็กสุดของเซกเมนต์

# K-means params
K = 9
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5)
ATTEMPTS = 10 #จำนวนครั้งที่สุ่มเริ่มรัน K-means แล้วเลือกผลที่ดีสุด

# Post-process
MIN_AREA_FRAC = 0.001    # < 0.1% ของภาพ สัดส่วนพื้นที่ขั้นต่ำของชิ้นเล็ก ๆ ที่จะเก็บไว้

# ===================== Load =====================
img_bgr = cv.imread(IMG_PATH) #อ่านไฟล์รูปจากพาธ IMG_PATH ด้วย OpenCV
if img_bgr is None:
    raise FileNotFoundError(f"ไม่พบไฟล์: {IMG_PATH}")
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]
print(f"[INFO] image: {W}x{H}")

# ================= Preprocess ====================
blur = cv.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75) #เบลอแบบรักษาขอบ (ลดนอยส์แต่ไม่ทำให้ขอบแตก)
float_img = img_as_float(blur)

# ============= Felzenszwalb segmentation ========= หั่นภาพออกเปน seg     
segments = felzenszwalb(float_img, scale=FELZ_SCALE, sigma=FELZ_SIGMA, min_size=FELZ_MINSIZE) #ขนาดความหยาบ,Gaussian blur ก่อนหาเซกเมนต์,ขนาดชิ้นเล็กสุด
n_segs = int(segments.max()) + 1 #จำนวนเซกเมนต์ทั้งหมด
print(f"[INFO] segments: {n_segs}")

# ภาพสีเฉลี่ยตาม segment (อ้างอิง)
seg_vis = label2rgb(segments, float_img, kind='avg') #แปลงภาพที่ได้จาก seg เปลี่ยนเป็นภาพตามสีจริง
seg_vis_u8 = (seg_vis * 255).astype(np.uint8) # เป็น 8-bit [0,255] เพื่อให้ OpenCV เขียนไฟล์ได้ถูกต้อง
cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_seg.png"), cv.cvtColor(seg_vis_u8, cv.COLOR_RGB2BGR))
np.save(os.path.join(OUT_DIR, f"{NAME}_labels.npy"), segments)

# ============ Build per-segment features =========
lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB) #ระยะทางสีใน Lab ใกล้เคียงการรับรู้ของมนุษย์ (perceptually uniform) กว่า RGB จึงเหมาะสำหรับ “คำนวณ/จับกลุ่มสี”
seg_lab_mean = np.zeros((n_segs, 3), dtype=np.float32)
seg_rgb_mean = np.zeros((n_segs, 3), dtype=np.float32)

for sid in range(n_segs):
    mask = (segments == sid)
    if not np.any(mask): 
        continue
    seg_lab_mean[sid] = lab[mask].mean(axis=0)
    seg_rgb_mean[sid] = img_rgb[mask].mean(axis=0)

# ================ K-means on segments ============ จับกลุ่มสีของเซกเมนต์ใน Lab เป็น K กลุ่ม
data = seg_lab_mean.astype(np.float32)
_, labels_seg, centers = cv.kmeans(
    data, K, None, CRITERIA, ATTEMPTS, cv.KMEANS_PP_CENTERS
)
labels_seg = labels_seg.flatten()          # cluster id ต่อ segment

# representative RGB ต่อคลัสเตอร์ (เฉลี่ยจากเซ็กเมนต์ที่อยู่ในคลัสเตอร์นั้น)
# สร้าง “สีตัวแทน” (RGB) ให้กับ แต่ละคลัสเตอร์ ของ K-means
cluster_rgb = np.zeros((K, 3), dtype=np.float32)
for k in range(K):
    seg_ids = np.where(labels_seg == k)[0]
    if len(seg_ids) == 0:
        cluster_rgb[k] = np.array([255, 255, 255], dtype=np.float32)#กรณีคลัสเตอร์นั้น “ว่าง” (ไม่มีเซกเมนต์ถูกจัดเข้ามา) → ตั้งสีสำรองเป็น ขาว เพื่อกันค่าไม่กำหนด
    else:
        cluster_rgb[k] = seg_rgb_mean[seg_ids].mean(axis=0)
cluster_rgb = cluster_rgb.clip(0, 255).astype(np.uint8)

# แมปกลับเต็มภาพ
cluster_map = labels_seg[segments]         # HxW ของ id (0..K-1)

# --- SAVE FOR STROKE PIPELINE (สำคัญ) ---
np.save(os.path.join(OUT_DIR, f"{NAME}_cluster_map.npy"), cluster_map.astype(np.int16))
np.save(os.path.join(OUT_DIR, f"{NAME}_cluster_rgb.npy"), cluster_rgb.astype(np.uint8))
# (ออปชัน) รูปโชว์สีตามคลัสเตอร์
cluster_color_img = cluster_rgb[cluster_map]
cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_cluster_color.png"),
           cv.cvtColor(cluster_color_img, cv.COLOR_RGB2BGR))

# ============== Post-process per cluster =========
#หลังจัดคลัสเตอร์/ทำ mask แล้ว มักจะมี เศษจุดเล็ก หรือ เกาะจิ๋ว กระจัดกระจาย ฟังก์ชันนี้ช่วย “ทำความสะอาด” ให้เหลือเฉพาะบริเวณหลัก ๆ ที่มี นัยสำคัญ
def clean_small_regions(mask, min_area):
    num, lbl, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):  # 0 = background
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 1
    return out.astype(bool)

min_area = int(H * W * MIN_AREA_FRAC)

# ============ Export isolated layers & grid =======
#สร้าง “ภาพแยกคลัสเตอร์” ทีละ k
#เตรียมลิสต์เก็บภาพของแต่ละคลัสเตอร์ (layers) และเก็บชื่อ/ค่าของสีไว้แสดงบนกริด
layers = []
color_titles = []

for k in range(K):
    mask = (cluster_map == k)
    if mask.any(): #ถ้าคลัสเตอร์นี้มีพิกเซลอยู่จริง (mask.any() = มีอย่างน้อย 1 จุด)
        mask = clean_small_regions(mask, min_area) #ล้าง “เกาะเล็ก ๆ” ออกด้วย clean_small_regions โดยเก็บเฉพาะชิ้นที่มีพื้นที่ ≥ min_area
    layer = np.full((H, W, 3), 255, dtype=np.uint8) #สร้างภาพพื้นขาวขนาดเท่ารูป
    layer[mask] = cluster_rgb[k]#ทาสี cluster_rgb[k] (RGB ของคลัสเตอร์ k) เฉพาะบริเวณที่ mask เป็นจริง ผล: เห็น “พื้นที่ของคลัสเตอร์ k” ชัด ๆ บนพื้นขาว
    layers.append(layer) #เก็บภาพนี้ลงลิสต์ layers
    color_titles.append(cluster_rgb[k].tolist())#เก็บค่าสีของคลัสเตอร์ (แปลงเป็น list) ไปไว้ทำชื่อหัวข้อบนกริด

    out_path = os.path.join(OUT_DIR, f"{NAME}_K{k+1}.jpg") #เซฟภาพของคลัสเตอร์ k เป็น .../NAME_K{k+1}.jpg
    cv.imwrite(out_path, cv.cvtColor(layer, cv.COLOR_RGB2BGR))#แปลง RGB → BGR ก่อนบันทึก
    print(f"saved: {out_path}")
    
#สร้างรูปสำหรับกริดขนาด 3 คอลัมน์ × 3 แถว
plt.figure(figsize=(9, 7))
plt.subplot(3, 3, 1)
plt.imshow(img_rgb); plt.title("Input"); plt.axis("off")
for i in range(2, 10):
    idx = i - 2
    if idx < len(layers):
        plt.subplot(3, 3, i)
        plt.imshow(layers[idx]); plt.title(f"K{idx+1} {color_titles[idx]}")
        plt.axis("off")
grid_path = os.path.join(OUT_DIR, f"{NAME}_grid.png")
plt.tight_layout(); plt.savefig(grid_path, dpi=150); plt.close()
print(f"saved grid preview: {grid_path}")
print(f"[DONE] Results in: {OUT_DIR}")