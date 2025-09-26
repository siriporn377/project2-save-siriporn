# segment/segmentnew3.py
# CLIPSeg (bird+branch) -> GrabCut -> de-halo + inpaint + meanShift ->
# K-means (Lab+XY) + optional split-big-cluster -> save outputs
# Default: run ONLY segment/photo/input34.jpg

import os, sys
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation

# ================== CONFIG ==================
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
BRIDGE_DILATE  = 41

BAND_TOP_PAD   = 80
BAND_BOT_PAD   = 70
X_MARGIN_LEFT  = 160
X_MARGIN_RIGHT = 260

K = 12
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5)
ATTEMPTS = 10
MIN_AREA_FRAC = 0.002
SPATIAL_W = 18.0

MS_SP, MS_SR = 6, 8

ENABLE_SPLIT_BIG = True
SPLIT_FRAC = 0.60
K_SUB = 5

# ================ UTILS & MODELS ================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

_model_cache = {"proc": None, "model": None}
def get_clipseg():
    if _model_cache["proc"] is None:
        _model_cache["proc"]  = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        _model_cache["model"] = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()
    return _model_cache["proc"], _model_cache["model"]

def clipseg_mask(pil_img, H, W, prompts, th):
    # ✅ แยก tokenizer กับ image_processor เพื่อใส่ padding/truncation ได้ถูกต้อง
    proc, model = get_clipseg()
    if isinstance(prompts, str):
        prompts = [prompts]
    images = [pil_img] * len(prompts)

    tok = proc.tokenizer(text=prompts, padding=True, truncation=True, return_tensors="pt")
    img = proc.image_processor(images=images, return_tensors="pt")
    inputs = {**tok, **img}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.sigmoid(logits).cpu().numpy()

    m = probs.max(axis=0)
    m = cv.resize(m, (W, H), interpolation=cv.INTER_CUBIC)
    return (m >= th).astype(np.uint8) * 255

def feet_band_and_bbox(bird_mask, H, W, top=BAND_TOP_PAD, bot=BAND_BOT_PAD):
    ys, xs = np.where(bird_mask > 0)
    if len(ys) == 0:
        y0 = int(H*0.65)
        band = np.zeros((H,W), np.uint8); band[max(0,y0-60):min(H,y0+60),:] = 255
        return band, (0, W-1), y0
    y0 = int(np.percentile(ys, 92))
    x1, x2 = int(xs.min()), int(xs.max())
    band = np.zeros((H,W), np.uint8)
    y1 = max(0, y0 - top); y2 = min(H-1, y0 + bot)
    band[y1:y2+1, :] = 255
    return band, (x1, x2), y0

def rail_from_texture_fallback(bgr, band, bird_bbox, H, W):
    x1, x2 = bird_bbox
    rx1 = max(0, x1 - X_MARGIN_LEFT); rx2 = min(W-1, x2 + X_MARGIN_RIGHT)
    roi = np.zeros((H,W), np.uint8); roi[:, rx1:rx2+1] = 255; roi = cv.bitwise_and(roi, band)

    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    lap = cv.Laplacian(gray, cv.CV_32F, ksize=3); lap = np.abs(lap)
    lap = cv.GaussianBlur(lap, (5,5), 0)
    lap_u8 = cv.normalize(lap, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    vals = lap_u8[roi > 0]
    if vals.size == 0: return np.zeros((H,W), np.uint8)
    thr_val, _ = cv.threshold(vals, 0, 255, cv.THRESH_OTSU)
    tex = np.zeros((H,W), np.uint8); tex[(lap_u8 >= thr_val) & (roi > 0)] = 255

    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    birdish = (hsv[:,:,1] > 140) & (((hsv[:,:,0] < 25) | (hsv[:,:,0] > 150)))
    tex[birdish] = 0

    tex = cv.morphologyEx(tex, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(17,17)), 1)
    tex = cv.morphologyEx(tex, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)), 1)
    tex = cv.dilate(tex, cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,21)), 1)
    return tex

def clean_small_regions(mask, min_area):
    num, lbl, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            keep[lbl == i] = 1
    return keep.astype(bool)

# ================ CORE ================
def process_one(img_path, mode="bird_branch"):
    NAME = os.path.splitext(os.path.basename(img_path))[0]
    OUT_DIR = os.path.join("segment", "output", "kmeans_1seg", NAME); ensure_dir(OUT_DIR)

    bgr = cv.imread(img_path)
    if bgr is None: return {"name": NAME, "ok": False, "err": "read_fail"}

    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB); H, W = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)

    # 1) Bird
    bird = clipseg_mask(pil_img, H, W, "bird", TH_BIRD)
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_bird.png"), bird)

    # 2) Branch/Rail near feet
    rail_keep = np.zeros((H,W), np.uint8)
    if mode != "bird_only":
        band, bbox, _ = feet_band_and_bbox(bird, H, W)
        rail = clipseg_mask(pil_img, H, W, RAIL_WORDS, TH_RAIL_HI)
        rail_band = cv.bitwise_and(rail, band)
        if cv.countNonZero(rail_band) < 500:
            rail = clipseg_mask(pil_img, H, W, RAIL_WORDS, TH_RAIL_LO)
            rail_band = cv.bitwise_and(rail, band)
        if cv.countNonZero(rail_band) < 500:
            rail_band = rail_from_texture_fallback(bgr, band, bbox, H, W)
        cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_rail_band.png"), rail_band)

        bird_touch = cv.dilate(bird, cv.getStructuringElement(cv.MORPH_ELLIPSE,(BRIDGE_DILATE, BRIDGE_DILATE)))
        union = cv.bitwise_or(rail_band, bird_touch)
        num, lbl, stats, _ = cv.connectedComponentsWithStats(union, 8)
        if num > 1:
            overlaps = [np.count_nonzero((lbl==i) & (bird_touch>0)) for i in range(1, num)]
            if overlaps:
                rail_keep[lbl == (1+int(np.argmax(overlaps)))] = 255
        cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_rail_keep.png"), rail_keep)

    seed = cv.bitwise_or(bird, rail_keep) if mode!="bird_only" else bird
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_seed.png"), seed)

    # 3) GrabCut
    gc = np.full((H, W), cv.GC_PR_BGD, np.uint8)
    gc[seed>0] = cv.GC_PR_FGD
    sure_fg = cv.erode(bird, cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9)))
    gc[sure_fg>0] = cv.GC_FGD
    gc[:PAD_BG,:]=0; gc[-PAD_BG:,:]=0; gc[:,:PAD_BG]=0; gc[:,-PAD_BG:]=0
    bgdModel = np.zeros((1,65), np.float64); fgdModel = np.zeros((1,65), np.float64)
    cv.grabCut(bgr, gc, None, bgdModel, fgdModel, GRABCUT_ITERS, cv.GC_INIT_WITH_MASK)
    mask = np.where((gc==cv.GC_FGD)|(gc==cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_mask.png"), mask)

    # 4) De-halo + inpaint + mean-shift
    mask_bool = mask>0
    ERODE_PX = 2
    kmeans_mask = cv.erode(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE,(2*ERODE_PX+1, 2*ERODE_PX+1)))
    hole = (kmeans_mask==0).astype(np.uint8)*255
    inpaint_bgr = cv.inpaint(cv.cvtColor(rgb, cv.COLOR_RGB2BGR), hole, 3, cv.INPAINT_TELEA)
    ms_bgr = cv.pyrMeanShiftFiltering(inpaint_bgr, sp=MS_SP, sr=MS_SR)
    ms_rgb = cv.cvtColor(ms_bgr, cv.COLOR_BGR2RGB)

    masked_rgb_show = rgb.copy(); masked_rgb_show[~mask_bool] = 255
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_masked_input.png"), cv.cvtColor(masked_rgb_show, cv.COLOR_RGB2BGR))

    # 5) Direct K-means (Lab+XY)
    lab = cv.cvtColor(ms_rgb, cv.COLOR_RGB2LAB).astype(np.float32)
    ys, xs = np.where(kmeans_mask>0)
    if ys.size == 0: return {"name": NAME, "ok": False, "err": "mask_empty", "out_dir": OUT_DIR}

    feat_col = lab[ys, xs]
    feat_xy  = np.stack([xs/float(W), ys/float(H)], axis=1).astype(np.float32)
    data = np.hstack([feat_col, SPATIAL_W*feat_xy]).astype(np.float32)

    # ใช้สำเนาของ K ภายในฟังก์ชัน
    k_clusters = int(K)

    # รอบแรก
    _, labels_fg, _ = cv.kmeans(data, k_clusters, None, CRITERIA, ATTEMPTS, cv.KMEANS_PP_CENTERS)
    labels_fg = labels_fg.flatten()
    cluster_map = np.full((H, W), -1, dtype=np.int16); cluster_map[ys, xs] = labels_fg.astype(np.int16)

    # 5.1 split biggest cluster (color-only) ถ้าใหญ่เกินไป
    if ENABLE_SPLIT_BIG and labels_fg.size>0:
        areas = np.bincount(labels_fg, minlength=k_clusters)
        big = int(np.argmax(areas))
        frac = areas[big] / float(labels_fg.size)
        if frac >= SPLIT_FRAC:
            idx_big = (labels_fg == big)
            ys_big, xs_big = ys[idx_big], xs[idx_big]
            col_big = lab[ys_big, xs_big].astype(np.float32)
            _K_SUB = min(K_SUB, max(2, int(areas[big] // 5000)))
            _, sub_labels, _ = cv.kmeans(col_big, _K_SUB, None, CRITERIA, ATTEMPTS, cv.KMEANS_PP_CENTERS)
            sub_labels = sub_labels.flatten()
            next_id = int(cluster_map.max()) + 1
            for s in range(_K_SUB):
                m = (sub_labels == s)
                cluster_map[ys_big[m], xs_big[m]] = (next_id + s)
            k_clusters = next_id + _K_SUB

    # 5.2 clean small regions
    min_area = max(1, int(np.count_nonzero(kmeans_mask) * MIN_AREA_FRAC))
    max_label = int(cluster_map.max())
    for k_id in range(max_label + 1):
        mk = (cluster_map == k_id)
        if mk.any():
            keep = clean_small_regions(mk, min_area)
            cluster_map[(mk) & (~keep)] = -1

    # 5.3 representative colors
    max_label = int(cluster_map.max())
    cluster_rgb = np.zeros((max(1, max_label+1), 3), np.uint8)
    for k_id in range(max_label + 1):
        pts = np.where(cluster_map == k_id)
        if len(pts[0]) > 0:
            cluster_rgb[k_id] = np.mean(rgb[pts], axis=0).clip(0,255).astype(np.uint8)
        else:
            cluster_rgb[k_id] = np.array([255,255,255], np.uint8)

    # 6) Save
    np.save(os.path.join(OUT_DIR, f"{NAME}_cluster_map.npy"), cluster_map.astype(np.int16))
    np.save(os.path.join(OUT_DIR, f"{NAME}_cluster_rgb.npy"), cluster_rgb.astype(np.uint8))

    cluster_color_img = np.full((H,W,3), 255, dtype=np.uint8)
    valid = (cluster_map >= 0)
    cluster_color_img[valid] = cluster_rgb[cluster_map[valid]]
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_cluster_color.png"), cv.cvtColor(cluster_color_img, cv.COLOR_RGB2BGR))

    layers, titles = [], []
    for k_id in range(max_label + 1):
        mk = (cluster_map == k_id)
        layer = np.full((H, W, 3), 255, dtype=np.uint8); layer[mk] = cluster_rgb[k_id]
        layers.append(layer); titles.append(cluster_rgb[k_id].tolist())
        cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_K{k_id+1}.jpg"), cv.cvtColor(layer, cv.COLOR_RGB2BGR))

    plt.figure(figsize=(9,7))
    show_inp = rgb.copy(); show_inp[kmeans_mask==0] = 255
    plt.subplot(3,3,1); plt.imshow(show_inp); plt.title("Input (masked)"); plt.axis("off")
    for i in range(2,10):
        idx=i-2
        if idx < len(layers):
            plt.subplot(3,3,i); plt.imshow(layers[idx]); plt.title(f"K{idx+1} {titles[idx]}"); plt.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{NAME}_grid.png"), dpi=150); plt.close()

    bird_area = int((bird>0).sum())
    rail_keep_area = int((rail_keep>0).sum())
    thr = max(300, int(0.02*bird_area))
    branch_ok = (rail_keep_area >= thr) if (mode!="bird_only") else True
    return {"name": NAME, "branch_ok": branch_ok, "out_dir": OUT_DIR}

# ================== MAIN ==================
if __name__ == "__main__":
    in_arg = "segment/photo/input15.png"
    MODE   = "bird_branch"
    if not os.path.isfile(in_arg):
        raise SystemExit(f"ไม่พบไฟล์: {in_arg}")
    info = process_one(in_arg, MODE)
    print(f"[DONE] -> {info}")
