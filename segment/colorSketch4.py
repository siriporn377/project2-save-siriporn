# ====== colorSketch_kmeans1seg_repo_antibleed_noXY.py ======
# --- make segment/ and segment/sketch_repo importable ---
from pathlib import Path
import sys, os, subprocess
from datetime import datetime

SEG_DIR  = Path(__file__).resolve().parent
REPO_DIR = SEG_DIR / "sketch_repo"
for p in (SEG_DIR, REPO_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

def abspath(p):
    q = Path(p)
    return str(q if q.is_absolute() else (SEG_DIR / q))

print("REPO_DIR =", REPO_DIR)
print("Has LDR.py:", (REPO_DIR / "LDR.py").exists())
print("Has ETF:", (REPO_DIR / "ETF" / "edge_tangent_flow.py").exists())

import cv2
import numpy as np
import torch, random
from PIL import Image

# ----- imports from repo (ตรรกะเส้นเดิมทั้งหมด) -----
from LDR import *
from tone import *
from genStroke_origin import *
from drawpatch import rotate
from tools import *
from ETF.edge_tangent_flow import *
from deblue import deblue
from quicksort import *

# ====== PARAMETERS (สไตล์ input26 สำหรับ "เส้น") ======
np.random.seed(1)
n               = 6
period          = 4
direction       = 10
Freq            = 100
deepen          = 1
transTone       = False
kernel_radius   = 3
iter_time       = 15
background_dir  = 45
CLAHE           = True
edge_CLAHE      = True
draw_new        = True
random_order    = True
ETF_order       = True
process_visible = False

# ====== Anti-bleed (ไม่แตะ x,y กลางภาพ) ======
CENTER_ONLY            = False    # ปิดคุมกลางภาพตามคำขอ
CLIP_TO_CLUSTER        = True     # ลงสีเฉพาะในคลัสเตอร์
SAFE_INSET_PX          = 6        # ดึงขอบเข้าด้านใน (ด้วย distanceTransform) 0=ปิด
MASK_SHRINK_PX         = 0        # สำรอง: erode แบบเดิม (ถ้าอยากใช้แทน DT ให้ตั้ง >0)
OPENING_KERNEL_PX      = 3        # เปิดฝุ่น/ปิดรูเล็ก ๆ ในมาสก์ 0=ปิด
THRESH_STRICT_NOW      = 245      # ลงสีเฉพาะพิกเซลเส้นที่มืดจริง ๆ (ลดการติดคราบฟุ้ง)
DROP_BG_BY_HEURISTIC   = True     # ตัดคลัสเตอร์พื้นหลัง (ใหญ่/ติดขอบ/ขาวจ้า)
BG_TOP_N_BY_AREA       = 2

# ----- บันทึกผลแบบมี timestamp -----
MAX_STROKES     = 1000           # หรือ None ถ้าอยากวาดครบทุกเส้น
SAVE_FRAMES     = True
SAVE_EVERY      = 5
VIDEO_FPS       = 30
MAKE_MP4        = True
MAKE_GIF        = True
GIF_FPS         = 12
RUN_PROCESS_DIR = None

# ---------- helpers ----------
def frames_to_mp4(process_dir: Path, out_mp4: Path, fps: int = 30):
    frames = sorted(process_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"[warn] ไม่พบเฟรมใน {process_dir}")
        return
    first = cv2.imread(str(frames[0])); h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is None: continue
        if img.shape[:2] != (h, w): img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release(); print("MP4 saved ->", out_mp4)

def frames_to_gif(process_dir: Path, out_gif: Path, fps: int = 12):
    frames = sorted(process_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"[warn] ไม่พบเฟรมใน {process_dir}")
        return
    imgs = []
    first = Image.open(frames[0]).convert("RGB"); W, H = first.size
    for f in frames:
        im = Image.open(f).convert("RGB")
        if im.size != (W, H): im = im.resize((W, H))
        imgs.append(im)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    first.save(out_gif, save_all=True, append_images=imgs[1:], duration=max(1,int(1000/fps)), loop=0, optimize=True)
    print("GIF saved ->", out_gif)

def build_center_mask(h, w, fraction=0.6, shape='ellipse'):
    # (ไม่ได้ใช้ในเวอร์ชันนี้ แต่คงไว้เผื่อ)
    fraction = max(0.05, min(0.98, float(fraction)))
    if shape == 'rect':
        cy, cx = h//2, w//2; hh=int(h*fraction/2.0); hw=int(w*fraction/2.0)
        m = np.zeros((h,w), bool)
        m[max(0,cy-hh):min(h,cy+hh), max(0,cx-hw):min(w,cx+hw)] = True
        return m
    yy, xx = np.ogrid[:h,:w]; cy, cx=(h-1)/2.0,(w-1)/2.0; ry=(h*fraction)/2.0; rx=(w*fraction)/2.0
    return ((yy-cy)**2)/(ry**2+1e-6) + ((xx-cx)**2)/(rx**2+1e-6) <= 1.0

def pick_background_ids(cluster_map, src_rgb, top_n_by_area=2,
                        min_area_frac=0.15, min_border_touch=0.25,
                        max_sat_keep=25, min_v_nearwhite=220):
    cm = np.array(cluster_map, dtype=np.int32); valid = cm >= 0
    if not np.any(valid): return []
    H, W = cm.shape; total = int(valid.sum())
    border = np.zeros_like(cm, bool); border[0,:]=border[-1,:]=True; border[:,0]=True; border[:,-1]=True
    border &= valid; border_count = int(border.sum()) or 1
    hsv = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2HSV)
    labels_valid = cm[valid]; K = int(labels_valid.max()) + 1
    counts = np.bincount(labels_valid.reshape(-1), minlength=K); order = np.argsort(counts)[::-1]
    bg = set(order[:max(1, top_n_by_area)])
    for k in range(K):
        mask = (cm == k) & valid
        if not np.any(mask): continue
        area_frac   = float(mask.sum()) / float(total)
        border_frac = float((mask & border).sum()) / float(border_count)
        sat_mean    = float(hsv[...,1][mask].mean()); val_mean = float(hsv[...,2][mask].mean())
        if (area_frac >= min_area_frac and border_frac >= min_border_touch) or \
           (sat_mean < max_sat_keep and val_mean > min_v_nearwhite and border_frac > 0.10):
            bg.add(k)
    return sorted(bg)

# ---------- core drawing (ตรรกะเส้นจาก repo 100%) ----------
def colorSketch(img=None, color=None, result_bgr=None, paint_mask=None):
    """
    img: path รูป grayscale เฉพาะคลัสเตอร์หนึ่งใบ
    color: [R,G,B]  (จะถูกเขียนลง result_bgr เฉพาะจุดที่มีเส้น)
    result_bgr: แคนวาสสะสมสี (ถ้า None จะเริ่มพื้นขาว)
    paint_mask:  bool (H,W) บังคับลงสีเฉพาะในคลัสเตอร์ (แบบขอบถอยใน = anti-bleed)
    """
    img_path = abspath(img)
    file_name = Path(img_path).stem
    output_root = SEG_DIR / "output"
    output_path = output_root / file_name
    (output_path / "mask").mkdir(parents=True, exist_ok=True)
    (output_path / "process").mkdir(parents=True, exist_ok=True)

    print(file_name)

    # ======= ETF (เรียกจาก repo เดิม) =======
    ETF(input_path=img_path, output_path=str(output_path / "mask"),
        dir_num=direction, kernel_radius=kernel_radius,
        iter_time=iter_time, background_dir=background_dir).forward()

    # อ่านรูปเทา
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise FileNotFoundError(f"อ่านภาพไม่ได้: {img_path}")
    (h0, w0) = input_img.shape
    cv2.imwrite(str(output_path / "input_gray.jpg"), input_img)

    if transTone:
        input_img = transferTone(input_img)

    if draw_new:
        stroke_sequence = []
        stroke_temp = {'angle':None, 'grayscale':None, 'row':None, 'begin':None, 'end':None}

        for dirs in range(direction):
            angle = -90 + dirs * 180 / direction
            stroke_temp['angle'] = angle
            img_rot, _ = rotate(input_img, -angle)
            img_use = HistogramEqualization(img_rot) if CLAHE else img_rot

            # gradient (แบบเดิม)
            pad = cv2.copyMakeBorder(img_use, 2*period, 2*period, 2*period, 2*period, cv2.BORDER_REPLICATE)
            norm = cv2.normalize(pad.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)
            x_der = torch.from_numpy(cv2.Sobel(norm, cv2.CV_32FC1, 1, 0, ksize=5)) + 1e-12
            y_der = torch.from_numpy(cv2.Sobel(norm, cv2.CV_32FC1, 0, 1, ksize=5)) + 1e-12
            grad = torch.sqrt(x_der**2.0 + y_der**2.0)
            grad_norm = grad / (grad.max() + 1e-12)

            # LDR + cumulate (ตาม repo)
            ldr = LDR(img_use, n)
            LDR_single_add(ldr, n, str(output_path))
            (h, w) = ldr.shape

            for j in range(n):
                stroke_temp['grayscale'] = j*255/n
                mp  = output_path / 'mask' / f'mask{j}.png'
                dmp = output_path / 'mask' / f'dir_mask{dirs}.png'
                mask = cv2.imread(str(mp),  cv2.IMREAD_GRAYSCALE) / 255
                dirm = cv2.imread(str(dmp), cv2.IMREAD_GRAYSCALE)
                dirm, _ = rotate(dirm, -angle, pad_color=0)
                dirm[dirm < 128] = 0; dirm[dirm > 127] = 1

                dist = Gassian((1, int(h/period)+4), mean=period, var=1)
                dist = np.uint8(np.round(np.clip(dist, period*0.8, period*1.25)))
                raw = -int(period/2)

                for step in np.squeeze(dist).tolist():
                    if raw < h:
                        y = raw + 2*period; raw += step
                        for interval in get_start_end(mask[y-2*period] * dirm[y-2*period]):
                            begin = interval[0] - 2*period
                            end   = interval[1] + 2*period
                            stroke_temp['begin'] = begin
                            stroke_temp['end']   = end
                            stroke_temp['row']   = y-int(period/2)
                            stroke_temp['importance'] = (
                                (255-stroke_temp['grayscale']) *
                                torch.sum(grad_norm[y:y+period, interval[0]+2*period:interval[1]+2*period]).numpy()
                            )
                            stroke_sequence.append(stroke_temp.copy())

        if random_order:
            random.shuffle(stroke_sequence)

        # แคนวาสผลสะสมเส้น + สี
        result = Gassian((h0, w0), mean=255, var=0)
        if result_bgr is None:
            result_bgr = cv2.merge([result, result, result])

        canvases = []
        for dirs in range(direction):
            angle = -90 + dirs * 180 / direction
            canvas, _ = rotate(result, -angle)
            canvas = np.pad(canvas, pad_width=2*period, mode='constant', constant_values=(255, 255))
            canvases.append(canvas)

        # โฟลเดอร์เซฟเฟรม
        proc_dir = Path(RUN_PROCESS_DIR) if RUN_PROCESS_DIR else (output_path / "process")
        proc_dir.mkdir(parents=True, exist_ok=True)
        frame_id = len(list(proc_dir.glob("frame_*.jpg")))
        step_id = 0

        seq = stroke_sequence if MAX_STROKES is None else stroke_sequence[:MAX_STROKES]
        for st in seq:
            angle = st['angle']; dirs = int((angle + 90) * direction / 180)
            distribution = ChooseDistribution(period=period, Grayscale=st['grayscale'])
            row, begin, end = st['row'], st['begin'], st['end']
            if end - begin > 100:
                continue

            newline = Getline(distribution=distribution, length=end-begin)
            canvas = canvases[dirs]
            temp = canvas[row:row+2*period, 2*period+begin:2*period+end]
            temp = np.minimum(temp, newline[:, :temp.shape[1]])
            canvas[row:row+2*period, 2*period+begin:2*period+end] = temp

            now, _ = rotate(canvas[2*period:-2*period, 2*period:-2*period], angle)
            (H, W) = now.shape
            now    = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]

            # ===== สะสมเส้น (ตรรกะเดิมจาก repo) =====
            result = np.minimum(now, result)

            # ===== ลงสีเฉพาะพิกเซลเส้น + กันเปื้อนด้วย paint_mask =====
            # ใช้ threshold เข้มขึ้นเพื่อตัดเส้นจางๆ ที่ฟุ้ง
            row_hit = now < THRESH_STRICT_NOW
            for y in range(now.shape[0]):
                allow = row_hit[y]
                if paint_mask is not None:
                    allow = allow & paint_mask[y]
                idx = np.where(allow)[0]
                if idx.size:
                    # BGR from RGB
                    result_bgr[y, idx, 0] = color[2]
                    result_bgr[y, idx, 1] = color[1]
                    result_bgr[y, idx, 2] = color[0]

            if process_visible:
                try: cv2.imshow('step', result_bgr); cv2.waitKey(1)
                except Exception: pass

            if SAVE_FRAMES and (step_id % SAVE_EVERY == 0):
                cv2.imwrite(str(proc_dir / f"frame_{frame_id:06d}.jpg"), result_bgr); frame_id += 1
            step_id += 1

        if SAVE_FRAMES:
            cv2.imwrite(str(proc_dir / f"frame_{frame_id:06d}.jpg"), result_bgr)

    return result_bgr

# ================== MAIN (อ่านจาก kmeans_1seg) ==================
if __name__ == '__main__':
    # ชื่ออินพุต (ไม่ส่งอาร์กิวเมนต์จะใช้ "input47")
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("input47")
    if raw.suffix.lower() in (".jpg", ".jpeg", ".png"):
        NAME = raw.stem
        SRC  = raw if raw.is_absolute() else (SEG_DIR / raw)
    else:
        NAME = raw.name
        SRC  = None

    BASE = SEG_DIR / "output" / "kmeans_1seg" / NAME
    rgb_npy  = BASE / f"{NAME}_cluster_rgb.npy"
    cmap_npy = BASE / f"{NAME}_cluster_map.npy"

    # ถ้ายังไม่มี .npy → พยายามรัน segmentation อัตโนมัติ (ต้องมีภาพต้นฉบับ)
    if not (rgb_npy.exists() and cmap_npy.exists()):
        if SRC is None or not Path(SRC).exists():
            for c in (SEG_DIR / "photo" / f"{NAME}.jpg", SEG_DIR / "photo" / f"{NAME}.png",
                      BASE / f"{NAME}.jpg", BASE / f"{NAME}.png"):
                if c.exists(): SRC = c; break
        if SRC is None or not Path(SRC).exists():
            raise SystemExit(f"ไม่พบ .npy และไม่พบภาพต้นฉบับ {NAME} (วาง {NAME}.jpg/.png ใน segment/photo/)")
        seg_script = SEG_DIR / "segmentnew2.py"
        cmd = [sys.executable, str(seg_script), str(SRC)]
        print("[auto] run segmentation:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # โหลดผล kmeans_1seg
    cluster_rgb = np.load(str(rgb_npy)).astype(np.uint8)   # (K,3) RGB
    cluster_map = np.load(str(cmap_npy)).astype(np.int32)  # (H,W)

    # โหลดภาพต้นฉบับเพื่อดึงโทนเทา
    if SRC is None or not Path(SRC).exists():
        for c in (BASE / f"{NAME}.jpg", BASE / f"{NAME}.png",
                  SEG_DIR / "photo" / f"{NAME}.jpg", SEG_DIR / "photo" / f"{NAME}.png"):
            if c.exists(): SRC = c; break
    if SRC is None or not Path(SRC).exists():
        raise SystemExit(f"ไม่พบภาพต้นฉบับของ {NAME}")

    orig_bgr  = cv2.imread(str(SRC))
    if orig_bgr is None:
        raise SystemExit(f"อ่านรูปไม่ได้: {SRC}")
    orig_rgb  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

    # ให้ cluster_map มีขนาดเท่ารูปจริง
    H0, W0 = orig_gray.shape[:2]
    if cluster_map.shape != (H0, W0):
        cluster_map = cv2.resize(cluster_map, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # ทำโฟลเดอร์ process แบบ timestamp (เก็บทุกครั้ง)
    STAMP = datetime.now().strftime('%Y%m%d-%H%M%S')
    RUN_PROCESS_DIR = str(SEG_DIR / "output" / NAME / f"process_{STAMP}")
    Path(RUN_PROCESS_DIR).mkdir(parents=True, exist_ok=True)

    # (ตัวเลือก) ตัดคลัสเตอร์พื้นหลังทิ้งก่อนลงสี
    drop_ids = []
    if DROP_BG_BY_HEURISTIC:
        drop_ids = pick_background_ids(cluster_map, orig_rgb, top_n_by_area=BG_TOP_N_BY_AREA)
        if drop_ids:
            print(f"[BG] drop clusters (heuristic): {[i+1 for i in drop_ids]}")

    # โฟลเดอร์ temp สำหรับภาพเทาของแต่ละคลัสเตอร์
    tmp_dir = SEG_DIR / "_tmp_kmeans_inputs" / NAME / STAMP
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # วาดทีละคลัสเตอร์จาก cluster_map / cluster_rgb
    res_bgr = None
    K = int(cluster_rgb.shape[0])
    for k in range(K):
        if k in drop_ids:
            continue

        mask = (cluster_map == k)
        if not np.any(mask):
            continue

        # ---- ทำ paint_mask แบบ anti-bleed (เฉพาะ "ลงสี")
        pm_uint8 = (mask.astype(np.uint8) * 255)
        # ตัดฝุ่น/รูเล็ก ๆ
        if OPENING_KERNEL_PX > 0:
            r = int(OPENING_KERNEL_PX)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            pm_uint8 = cv2.morphologyEx(pm_uint8, cv2.MORPH_OPEN, ker, iterations=1)
        # ถอยขอบเข้าด้านในด้วย distance transform (เนียนและกันล้นดี)
        if SAFE_INSET_PX > 0:
            dist = cv2.distanceTransform(pm_uint8, cv2.DIST_L2, 5)  # float32
            safe = (dist >= float(SAFE_INSET_PX))
            pm_bool = safe.astype(bool)
        else:
            pm_bool = pm_uint8 > 0
            # หรือใช้ erode แบบเดิมแทน
            if MASK_SHRINK_PX > 0:
                s = int(MASK_SHRINK_PX)
                ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*s+1, 2*s+1))
                pm_uint8_er = cv2.erode(pm_uint8, ker, iterations=1)
                pm_bool = pm_uint8_er > 0

        # ---- ทำรูปเทาเฉพาะพื้นที่ของคลัสเตอร์ k (สำหรับตรรกะเส้นเดิม)
        cluster_gray = np.full_like(orig_gray, 255)
        cluster_gray[mask] = orig_gray[mask]
        tmp = tmp_dir / f"{NAME}_K{k+1}_gray.png"
        cv2.imwrite(str(tmp), cluster_gray)

        rgb = cluster_rgb[k].tolist()  # [R,G,B]
        print(f"[DRAW] K{k+1} RGB={rgb}")
        res_bgr = colorSketch(
            img=str(tmp),
            color=rgb,
            result_bgr=res_bgr,
            paint_mask=pm_bool    # <<< กันเปื้อนตรงนี้ (เฉพาะตอน "ลงสี")
        )

    # เซฟผลรวม + รวมวิดีโอ/GIF (มี timestamp)
    out_dir = SEG_DIR / "output" / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / f"result_from_kmeans1_{STAMP}.jpg"
    if res_bgr is not None:
        cv2.imwrite(str(out_img), res_bgr); print("Saved ->", out_img)
    else:
        print("[!] ไม่มีผลลัพธ์ให้เซฟ (ตรวจ .npy / ภาพต้นฉบับ)")

    if MAKE_MP4:
        frames_to_mp4(Path(RUN_PROCESS_DIR), out_dir / f"{NAME}_process_{STAMP}.mp4", fps=VIDEO_FPS)
    if MAKE_GIF:
        frames_to_gif(Path(RUN_PROCESS_DIR), out_dir / f"{NAME}_process_{STAMP}.gif", fps=GIF_FPS)

    if process_visible:
        cv2.waitKey(0)
