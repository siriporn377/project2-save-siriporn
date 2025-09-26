# segment/segment2.py
import os, sys
import argparse
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float

import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation

# -------- Args --------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("IMG_PATH", nargs="?", default="segment/photo/input44.jpg", ######################################################
                   help="พาธรูปอินพุต")
    p.add_argument("--mode", default="bird_branch", choices=["bird_branch","bird_only"],
                   help="เลือกโหมด: bird_branch หรือ bird_only")
    p.add_argument("--max-side", type=int, default=0,
                   help="0 = ไม่ย่อ, >0 = จำกัดด้านยาวสุด (เช่น 500)")
    return p.parse_args()

args = parse_args()
IMG_PATH = args.IMG_PATH
MODE     = args.mode.lower()
MAX_SIDE = max(0, int(args.max_side))

# ชื่อไฟล์/โฟลเดอร์ผลลัพธ์ (มี suffix ตาม max-side เพื่อไม่ทับกัน)
NAME     = os.path.splitext(os.path.basename(IMG_PATH))[0]
NAME_OUT = f"{NAME}_s{MAX_SIDE}" if MAX_SIDE > 0 else NAME
OUT_DIR  = os.path.join("segment", "output", "kmeans_1seg", NAME_OUT)
os.makedirs(OUT_DIR, exist_ok=True)

# -------- Params --------
MODEL_ID   = "CIDAS/clipseg-rd64-refined"
TH_BIRD    = 0.45
TH_RAIL_HI = 0.40
TH_RAIL_LO = 0.30
RAIL_WORDS = [
    "perch","branch","tree branch","log","mossy log","bark","trunk",
    "rail","railing","beam","bar"
]

GRABCUT_ITERS  = 5
PAD_BG         = 15
BRIDGE_DILATE  = 41   # ขยายนกให้ไปแตะราวมากขึ้น

# แถบค้นหาใกล้เท้านก + ช่วง X รอบนก
BAND_TOP_PAD   = 80
BAND_BOT_PAD   = 70
X_MARGIN_LEFT  = 160
X_MARGIN_RIGHT = 260

# Felzenszwalb & K-means
FELZ_SCALE, FELZ_SIGMA, FELZ_MINSIZE = 120, 0.6, 60
K = 9
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5)
ATTEMPTS = 10
MIN_AREA_FRAC = 0.001

# -------- Resize helper --------
def resize_keep_aspect(img, max_side=0):
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

# -------- Load image --------
bgr = cv.imread(IMG_PATH)
if bgr is None:
    raise FileNotFoundError(f"ไม่พบไฟล์รูป: {IMG_PATH}")

# ย่อก่อนประมวลผล (ถ้าเลือก)
bgr = resize_keep_aspect(bgr, MAX_SIDE)

rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
H, W = rgb.shape[:2]
pil_img = Image.fromarray(rgb)

# -------- CLIPSeg (รองรับหลายพรอมป์) --------
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()

def clipseg_mask(prompts, th):
    if isinstance(prompts, str):
        prompts = [prompts]
    images = [pil_img] * len(prompts)
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits  # (n,352,352)
        probs  = torch.sigmoid(logits).cpu().numpy()
    m = probs.max(axis=0)  # union ของหลายพรอมป์
    m = cv.resize(m, (W, H), interpolation=cv.INTER_CUBIC)
    return (m >= th).astype(np.uint8) * 255

def feet_contact_band_and_bbox(bird_mask):
    ys, xs = np.where(bird_mask > 0)
    if len(ys) == 0:
        y0 = int(H*0.65)
        band = np.zeros((H,W), np.uint8); band[max(0,y0-60):min(H,y0+60),:] = 255
        return band, (0, W-1), y0
    y0 = int(np.percentile(ys, 92))  # ใกล้ปลายล่างของนก
    x1, x2 = int(xs.min()), int(xs.max())
    band = np.zeros((H,W), np.uint8)
    y1 = max(0, y0 - BAND_TOP_PAD)
    y2 = min(H-1, y0 + BAND_BOT_PAD)
    band[y1:y2+1, :] = 255
    return band, (x1, x2), y0

def rail_from_texture_fallback(bgr, band, bird_bbox):
    """หาไม้จาก texture (Lap-of-Gaussian) ในแถบ band + จำกัด X รอบตัวนก"""
    x1, x2 = bird_bbox
    rx1 = max(0, x1 - X_MARGIN_LEFT)
    rx2 = min(W-1, x2 + X_MARGIN_RIGHT)

    roi = np.zeros((H,W), np.uint8)
    roi[:, rx1:rx2+1] = 255
    roi = cv.bitwise_and(roi, band)

    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # texture: Laplacian + blur
    lap = cv.Laplacian(gray, cv.CV_32F, ksize=3)
    lap = np.abs(lap)
    lap = cv.GaussianBlur(lap, (5,5), 0)
    lap_u8 = cv.normalize(lap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    vals = lap_u8[roi > 0]
    if vals.size == 0:
        return np.zeros((H,W), np.uint8)
    thr_val, _ = cv.threshold(vals, 0, 255, cv.THRESH_OTSU)
    tex = np.zeros((H,W), np.uint8)
    tex[(lap_u8 >= thr_val) & (roi > 0)] = 255

    # ตัดโทนที่ "เป็นนกจัดๆ" (สีสดมาก) ออก
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    birdish = (hsv[:,:,1] > 140) & (((hsv[:,:,0] < 25) | (hsv[:,:,0] > 150)))  # แดง/ม่วงจัด
    tex[birdish] = 0

    # ทำให้ต่อเนื่องและหนาขึ้น
    tex = cv.morphologyEx(tex, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(17,17)), iterations=1)
    tex = cv.morphologyEx(tex, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), iterations=1)
    tex = cv.dilate(tex, cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,21)), iterations=1)
    return tex

# ---- 1) Bird mask ----
bird = clipseg_mask("bird", TH_BIRD)

if MODE == "bird_only":
    seed = bird.copy()
else:
    # ---- 2) Rail/branch via CLIPSeg ในแถบเท้านก ----
    band, bbox, y0 = feet_contact_band_and_bbox(bird)
    rail = clipseg_mask(RAIL_WORDS, TH_RAIL_HI)
    rail_band = cv.bitwise_and(rail, band)

    # Fallback: ลดเกณฑ์ + ถ้ายังไม่พอ ใช้ texture fallback
    if cv.countNonZero(rail_band) < 500:
        rail = clipseg_mask(RAIL_WORDS, TH_RAIL_LO)
        rail_band = cv.bitwise_and(rail, band)
    if cv.countNonZero(rail_band) < 500:
        rail_band = rail_from_texture_fallback(bgr, band, bbox)

    # ---- 3) เก็บเฉพาะส่วนที่แตะนก ----
    bird_touch = cv.dilate(bird, cv.getStructuringElement(cv.MORPH_ELLIPSE, (BRIDGE_DILATE, BRIDGE_DILATE)))
    union = cv.bitwise_or(rail_band, bird_touch)
    num, lbl, stats, _ = cv.connectedComponentsWithStats(union, connectivity=8)
    keep = np.zeros((H, W), np.uint8)
    if num > 1:
        overlaps = [np.count_nonzero((lbl == i) & (bird_touch > 0)) for i in range(1, num)]
        if len(overlaps):
            idx = 1 + int(np.argmax(overlaps))
            keep[lbl == idx] = 255
    seed = cv.bitwise_or(bird, keep)

cv.imwrite(os.path.join(OUT_DIR, f"{NAME_OUT}_seed.png"), seed)

# ---- 4) GrabCut refine ----
gc = np.full((H, W), cv.GC_PR_BGD, np.uint8)
gc[seed > 0] = cv.GC_PR_FGD
sure_fg = cv.erode(bird, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)))
gc[sure_fg > 0] = cv.GC_FGD
gc[:PAD_BG,:]=0; gc[-PAD_BG:,:]=0; gc[:,:PAD_BG]=0; gc[:,-PAD_BG:]=0
bgdModel = np.zeros((1,65), np.float64); fgdModel = np.zeros((1,65), np.float64)
cv.grabCut(bgr, gc, None, bgdModel, fgdModel, GRABCUT_ITERS, cv.GC_INIT_WITH_MASK)
mask = np.where((gc==cv.GC_FGD)|(gc==cv.GC_PR_FGD), 255, 0).astype(np.uint8)
cv.imwrite(os.path.join(OUT_DIR, f"{NAME_OUT}_mask.png"), mask)

mask_bool  = mask > 0
masked_rgb = rgb.copy(); masked_rgb[~mask_bool] = 255
cv.imwrite(os.path.join(OUT_DIR, f"{NAME_OUT}_masked_input.png"), cv.cvtColor(masked_rgb, cv.COLOR_RGB2BGR))

# ---- 5) Felzenszwalb ----
blur = cv.bilateralFilter(masked_rgb, d=9, sigmaColor=75, sigmaSpace=75)
float_img = img_as_float(blur)
segments = felzenszwalb(float_img, scale=FELZ_SCALE, sigma=FELZ_SIGMA, min_size=FELZ_MINSIZE)
n_segs = int(segments.max()) + 1
seg_vis = label2rgb(segments, float_img, kind='avg')
cv.imwrite(os.path.join(OUT_DIR, f"{NAME_OUT}_seg.png"),
           cv.cvtColor((seg_vis*255).astype(np.uint8), cv.COLOR_RGB2BGR))
np.save(os.path.join(OUT_DIR, f"{NAME_OUT}_labels.npy"), segments)

# ---- 6) K-means เฉพาะใน mask ----
lab = cv.cvtColor(masked_rgb, cv.COLOR_RGB2LAB)
feat_lab, mean_rgb, fg_sids = [], [], []
for sid in range(n_segs):
    m = (segments==sid) & mask_bool
    if not np.any(m): continue
    feat_lab.append(lab[m].mean(axis=0).astype(np.float32))
    mean_rgb.append(masked_rgb[m].mean(axis=0).astype(np.float32))
    fg_sids.append(sid)

if len(fg_sids)==0:
    raise RuntimeError("mask ว่างเกินไป ไม่มีพิกเซลทำ K-means")

feat_lab = np.array(feat_lab, dtype=np.float32)
mean_rgb = np.array(mean_rgb, dtype=np.float32)
_, labels_fg, _ = cv.kmeans(feat_lab, K, None, CRITERIA, ATTEMPTS, cv.KMEANS_PP_CENTERS)
labels_fg = labels_fg.flatten()

cluster_rgb = np.zeros((K,3), np.float32)
for k in range(K):
    idx = np.where(labels_fg==k)[0]
    cluster_rgb[k] = mean_rgb[idx].mean(axis=0) if len(idx) else np.array([255,255,255])
cluster_rgb = cluster_rgb.clip(0,255).astype(np.uint8)

labels_by_sid = np.full(n_segs, -1, dtype=np.int16)
labels_by_sid[np.array(fg_sids, dtype=np.int32)] = labels_fg.astype(np.int16)
cluster_map = labels_by_sid[segments]

np.save(os.path.join(OUT_DIR, f"{NAME_OUT}_cluster_map.npy"), cluster_map.astype(np.int16))
np.save(os.path.join(OUT_DIR, f"{NAME_OUT}_cluster_rgb.npy"), cluster_rgb.astype(np.uint8))

# สีรวม
cluster_color_img = np.full((H,W,3), 255, dtype=np.uint8)
valid = cluster_map >= 0
cluster_color_img[valid] = cluster_rgb[cluster_map[valid]]
cv.imwrite(os.path.join(OUT_DIR, f"{NAME_OUT}_cluster_color.png"),
           cv.cvtColor(cluster_color_img, cv.COLOR_RGB2BGR))

# ---- 7) Export layers & grid ----
def clean_small_regions(mask, min_area):
    num, lbl, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            out[lbl==i] = 1
    return out.astype(bool)

min_area = max(1, int(mask_bool.sum() * MIN_AREA_FRAC))
layers, titles = [], []
for k in range(K):
    mk = (cluster_map==k) & mask_bool
    if mk.any():
        mk = clean_small_regions(mk, min_area)
    layer = np.full((H,W,3), 255, dtype=np.uint8); layer[mk] = cluster_rgb[k]
    layers.append(layer); titles.append(cluster_rgb[k].tolist())
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME_OUT}_K{k+1}.jpg"), cv.cvtColor(layer, cv.COLOR_RGB2BGR))

plt.figure(figsize=(9,7))
plt.subplot(3,3,1); plt.imshow(masked_rgb); plt.title("Input (masked)"); plt.axis("off")
for i in range(2,10):
    idx=i-2
    if idx < len(layers):
        plt.subplot(3,3,i); plt.imshow(layers[idx]); plt.title(f"K{idx+1} {titles[idx]}"); plt.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{NAME_OUT}_grid.png"), dpi=150); plt.close()

print(f"[DONE] Results in: {OUT_DIR}")
