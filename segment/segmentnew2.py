# segment/segment2.py  (single/batch runner + CSV summary)
import os, sys, glob, csv, time
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float

import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation

# ----------------- Configs -----------------
MODEL_ID   = "CIDAS/clipseg-rd64-refined"
TH_BIRD    = 0.45
TH_RAIL_HI = 0.40
TH_RAIL_LO = 0.30
RAIL_WORDS = ["perch","branch","tree branch","log","mossy log","bark","trunk",
              "rail","railing","beam","bar"]

GRABCUT_ITERS  = 5
PAD_BG         = 15
BRIDGE_DILATE  = 41

BAND_TOP_PAD   = 80
BAND_BOT_PAD   = 70
X_MARGIN_LEFT  = 160
X_MARGIN_RIGHT = 260

FELZ_SCALE, FELZ_SIGMA, FELZ_MINSIZE = 120, 0.6, 60
K = 9
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.5)
ATTEMPTS = 10
MIN_AREA_FRAC = 0.001

# ----------------- Helpers -----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def clipseg_mask(processor, model, pil_img, H, W, prompts, th):
    if isinstance(prompts, str): prompts = [prompts]
    images = [pil_img] * len(prompts)
    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
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
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv.CC_STAT_AREA] >= min_area: out[lbl==i] = 1
    return out.astype(bool)

# ----------------- Core process -----------------
def process_one(img_path, mode="bird_branch"):
    IMG_PATH = img_path
    NAME = os.path.splitext(os.path.basename(IMG_PATH))[0]
    OUT_DIR = os.path.join("segment", "output", "kmeans_1seg", NAME)
    ensure_dir(OUT_DIR)

    bgr = cv.imread(IMG_PATH)
    if bgr is None: return {"name": NAME, "ok": False, "err": "read_fail"}
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)

    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_ID).eval()

    # 1) bird
    bird = clipseg_mask(processor, model, pil_img, H, W, "bird", TH_BIRD)
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_bird.png"), bird)

    # 2) rail
    rail_keep = np.zeros((H,W), np.uint8)
    if mode != "bird_only":
        band, bbox, _ = feet_band_and_bbox(bird, H, W)
        rail = clipseg_mask(processor, model, pil_img, H, W, RAIL_WORDS, TH_RAIL_HI)
        rail_band = cv.bitwise_and(rail, band)
        if cv.countNonZero(rail_band) < 500:
            rail = clipseg_mask(processor, model, pil_img, H, W, RAIL_WORDS, TH_RAIL_LO)
            rail_band = cv.bitwise_and(rail, band)
        if cv.countNonZero(rail_band) < 500:
            rail_band = rail_from_texture_fallback(bgr, band, bbox, H, W)
        cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_rail_band.png"), rail_band)

        bird_touch = cv.dilate(bird, cv.getStructuringElement(cv.MORPH_ELLIPSE, (BRIDGE_DILATE, BRIDGE_DILATE)))
        union = cv.bitwise_or(rail_band, bird_touch)
        num, lbl, stats, _ = cv.connectedComponentsWithStats(union, connectivity=8)
        if num > 1:
            overlaps = [np.count_nonzero((lbl == i) & (bird_touch > 0)) for i in range(1, num)]
            if len(overlaps):
                idx = 1 + int(np.argmax(overlaps))
                rail_keep[lbl==idx] = 255
        cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_rail_keep.png"), rail_keep)

    seed = cv.bitwise_or(bird, rail_keep) if mode != "bird_only" else bird
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_seed.png"), seed)

    # 3) grabcut
    gc = np.full((H, W), cv.GC_PR_BGD, np.uint8)
    gc[seed > 0] = cv.GC_PR_FGD
    sure_fg = cv.erode(bird, cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9)))
    gc[sure_fg > 0] = cv.GC_FGD
    gc[:PAD_BG,:]=0; gc[-PAD_BG:,:]=0; gc[:,:PAD_BG]=0; gc[:,-PAD_BG:]=0
    bgdModel = np.zeros((1,65), np.float64); fgdModel = np.zeros((1,65), np.float64)
    cv.grabCut(bgr, gc, None, bgdModel, fgdModel, GRABCUT_ITERS, cv.GC_INIT_WITH_MASK)
    mask = np.where((gc==cv.GC_FGD)|(gc==cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_mask.png"), mask)

    mask_bool  = mask > 0
    masked_rgb = rgb.copy(); masked_rgb[~mask_bool] = 255
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_masked_input.png"), cv.cvtColor(masked_rgb, cv.COLOR_RGB2BGR))

    # 4) felzenszwalb
    blur = cv.bilateralFilter(masked_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    float_img = img_as_float(blur)
    segments = felzenszwalb(float_img, scale=FELZ_SCALE, sigma=FELZ_SIGMA, min_size=FELZ_MINSIZE)
    n_segs = int(segments.max()) + 1
    seg_vis = label2rgb(segments, float_img, kind='avg')
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_seg.png"),
               cv.cvtColor((seg_vis*255).astype(np.uint8), cv.COLOR_RGB2BGR))
    np.save(os.path.join(OUT_DIR, f"{NAME}_labels.npy"), segments)

    # 5) kmeans (mask only)
    lab = cv.cvtColor(masked_rgb, cv.COLOR_RGB2LAB)
    feat_lab, mean_rgb, fg_sids = [], [], []
    for sid in range(n_segs):
        m = (segments==sid) & mask_bool
        if not np.any(m): continue
        feat_lab.append(lab[m].mean(axis=0).astype(np.float32))
        mean_rgb.append(masked_rgb[m].mean(axis=0).astype(np.float32))
        fg_sids.append(sid)

    if len(fg_sids)==0:
        return {"name": NAME, "ok": False, "err": "no_fg_after_mask", "out_dir": OUT_DIR}

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

    np.save(os.path.join(OUT_DIR, f"{NAME}_cluster_map.npy"), cluster_map.astype(np.int16))
    np.save(os.path.join(OUT_DIR, f"{NAME}_cluster_rgb.npy"), cluster_rgb.astype(np.uint8))

    cluster_color_img = np.full((H,W,3), 255, dtype=np.uint8)
    valid = cluster_map >= 0
    cluster_color_img[valid] = cluster_rgb[cluster_map[valid]]
    cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_cluster_color.png"),
               cv.cvtColor(cluster_color_img, cv.COLOR_RGB2BGR))

    # 6) layers & grid
    min_area = max(1, int(mask_bool.sum() * MIN_AREA_FRAC))
    layers, titles = [], []
    for k in range(K):
        mk = (cluster_map==k) & mask_bool
        if mk.any(): mk = clean_small_regions(mk, min_area)
        layer = np.full((H,W,3), 255, dtype=np.uint8); layer[mk] = cluster_rgb[k]
        layers.append(layer); titles.append(cluster_rgb[k].tolist())
        cv.imwrite(os.path.join(OUT_DIR, f"{NAME}_K{k+1}.jpg"), cv.cvtColor(layer, cv.COLOR_RGB2BGR))

    plt.figure(figsize=(9,7))
    plt.subplot(3,3,1); plt.imshow(masked_rgb); plt.title("Input (masked)"); plt.axis("off")
    for i in range(2,10):
        idx=i-2
        if idx < len(layers):
            plt.subplot(3,3,i); plt.imshow(layers[idx]); plt.title(f"K{idx+1} {titles[idx]}"); plt.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"{NAME}_grid.png"), dpi=150); plt.close()

    # ---------- simple success metric ----------
    bird_area = int((bird>0).sum())
    rail_keep_area = int((rail_keep>0).sum())
    rail_keep_frac = rail_keep_area / float(H*W)
    # ถือว่าจับกิ่งได้ถ้ามีพื้นที่มากพอ หรือ >= 2% ของพื้นที่นก
    thr = max(300, int(0.02 * bird_area))
    branch_ok = (rail_keep_area >= thr) if (mode!="bird_only") else True

    return {
        "name": NAME, "h": H, "w": W,
        "bird_area": bird_area, "rail_keep_area": rail_keep_area,
        "rail_keep_frac": round(rail_keep_frac,5),
        "branch_ok": branch_ok, "out_dir": OUT_DIR
    }

# ----------------- Main (single or folder) -----------------
if __name__ == "__main__":
    in_arg = sys.argv[1] if len(sys.argv) > 1 else "segment/photo"
    MODE   = (sys.argv[2] if len(sys.argv) > 2 else "bird_branch").lower()

    # โหมดโฟลเดอร์
    if os.path.isdir(in_arg):
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG","*.JPEG"):
            imgs += glob.glob(os.path.join(in_arg, ext))
        imgs = sorted(imgs)
        if not imgs:
            print(f"[ERR] ไม่พบไฟล์ภาพในโฟลเดอร์: {in_arg}"); sys.exit(1)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        sum_dir = os.path.join("segment","output","kmeans_1seg")
        ensure_dir(sum_dir)
        csv_path = os.path.join(sum_dir, f"_batch_summary_{stamp}.csv")
        ok_cnt, fail_cnt = 0, 0

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["name","h","w","bird_area","rail_keep_area","rail_keep_frac","branch_ok","out_dir","err"])
            for p in imgs:
                try:
                    info = process_one(p, MODE)
                    if info.get("branch_ok", False): ok_cnt += 1
                    else: fail_cnt += 1
                    wr.writerow([info.get("name"), info.get("h"), info.get("w"),
                                 info.get("bird_area"), info.get("rail_keep_area"),
                                 info.get("rail_keep_frac"), info.get("branch_ok"),
                                 info.get("out_dir"), info.get("err","")])
                    print(f"[OK] {info.get('name')} -> branch_ok={info.get('branch_ok')}")
                except Exception as e:
                    fail_cnt += 1
                    nm = os.path.splitext(os.path.basename(p))[0]
                    wr.writerow([nm,"","","","","",False,"",f"EXC:{e}"])
                    print(f"[ERR] {nm}: {e}")

        total = ok_cnt + fail_cnt
        print(f"\n[BATCH DONE] total={total}  ok={ok_cnt}  fail={fail_cnt}")
        print(f"CSV => {csv_path}")

    else:
        # โหมดภาพเดียว
        info = process_one(in_arg, MODE)
        print(f"[DONE] one image -> {info}")
