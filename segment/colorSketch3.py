# ====== colorSketch_like_input26.py (match input26 style) ======
# --- make segment/ and segment/sketch_repo importable ---
from pathlib import Path
import sys, os, re, glob
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

# ====== PARAMETERS (ตาม input26) ======
np.random.seed(1)
n               = 6
period          = 6
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

# — สไตล์ input26: วาดเต็มภาพ (ถ้าจะลองเฉพาะกลางภาพค่อยเปลี่ยนเป็น True) —
CENTER_ONLY     = False         # ให้เหมือน input26 → False
CENTER_SHAPE    = 'ellipse'
CENTER_FRACTION = 0.60

# เซฟเฟรม/วิดีโอแบบ input26
MAX_STROKES     = 1000
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
    fraction = max(0.05, min(0.98, float(fraction)))
    if shape == 'rect':
        cy, cx = h//2, w//2; hh=int(h*fraction/2.0); hw=int(w*fraction/2.0)
        m = np.zeros((h,w), bool)
        m[max(0,cy-hh):min(h,cy+hh), max(0,cx-hw):min(w,cx+hw)] = True
        return m
    yy, xx = np.ogrid[:h,:w]; cy, cx=(h-1)/2.0,(w-1)/2.0; ry=(h*fraction)/2.0; rx=(w*fraction)/2.0
    return ((yy-cy)**2)/(ry**2+1e-6) + ((xx-cx)**2)/(rx**2+1e-6) <= 1.0

# ---------- core drawing (ตรรกะเส้นเดิมจาก repo) ----------
def colorSketch(img=None, color=None, result_bgr=None, center_mask=None):
    """
    img: path รูป grayscale เฉพาะคลัสเตอร์หนึ่งใบ
    color: [R,G,B]
    result_bgr: แคนวาสสะสม (ถ้า None จะสร้างพื้นขาวให้)
    center_mask: bool (H,W) วาดเฉพาะกลางภาพ (ถ้าเปิด CENTER_ONLY)
    """
    img_path = abspath(img)
    file_name = Path(img_path).stem
    output_root = SEG_DIR / "output"
    output_path = output_root / file_name
    (output_path / "mask").mkdir(parents=True, exist_ok=True)
    (output_path / "process").mkdir(parents=True, exist_ok=True)

    print(file_name)

    # ======= ETF (ตาม repo) =======
    ETF_filter = ETF(
        input_path=img_path,
        output_path=str(output_path / "mask"),
        dir_num=direction,
        kernel_radius=kernel_radius,
        iter_time=iter_time,
        background_dir=background_dir
    )
    ETF_filter.forward()

    # อ่านรูปเทา
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise FileNotFoundError(f"อ่านภาพไม่ได้: {img_path}")
    (h0, w0) = input_img.shape
    cv2.imwrite(str(output_path / "input_gray.jpg"), input_img)

    if transTone:
        input_img = transferTone(input_img)

    # mask กลางภาพเพื่อ “ลงสี” (ไม่แตะตรรกะเส้น)
    if center_mask is None and CENTER_ONLY:
        center_mask = build_center_mask(h0, w0, CENTER_FRACTION, CENTER_SHAPE)

    if draw_new:
        stroke_sequence = []
        stroke_temp = {'angle':None, 'grayscale':None, 'row':None, 'begin':None, 'end':None}

        for dirs in range(direction):
            angle = -90 + dirs * 180 / direction
            stroke_temp['angle'] = angle
            img_rot, _ = rotate(input_img, -angle)
            img_use = HistogramEqualization(img_rot) if CLAHE else img_rot

            # gradient (เหมือนเดิม)
            pad = cv2.copyMakeBorder(img_use, 2*period, 2*period, 2*period, 2*period, cv2.BORDER_REPLICATE)
            norm = cv2.normalize(pad.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)
            x_der = cv2.Sobel(norm, cv2.CV_32FC1, 1, 0, ksize=5)
            y_der = cv2.Sobel(norm, cv2.CV_32FC1, 0, 1, ksize=5)
            x_der = torch.from_numpy(x_der) + 1e-12
            y_der = torch.from_numpy(y_der) + 1e-12
            gradient_magnitude = torch.sqrt(x_der**2.0 + y_der**2.0)
            gradient_norm = gradient_magnitude / gradient_magnitude.max()

            # LDR + cumulate ตาม repo
            ldr = LDR(img_use, n)
            LDR_single_add(ldr, n, str(output_path))
            (h, w) = ldr.shape

            for j in range(n):
                stroke_temp['grayscale'] = j*255/n
                mask_path    = output_path / 'mask' / f'mask{j}.png'
                dir_mask_png = output_path / 'mask' / f'dir_mask{dirs}.png'
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) / 255
                dir_mask = cv2.imread(str(dir_mask_png), cv2.IMREAD_GRAYSCALE)
                dir_mask, _ = rotate(dir_mask, -angle, pad_color=0)
                dir_mask[dir_mask < 128] = 0; dir_mask[dir_mask > 127] = 1

                dist = Gassian((1, int(h/period)+4), mean=period, var=1)
                dist = np.uint8(np.round(np.clip(dist, period*0.8, period*1.25)))
                raw = -int(period/2)

                for step in np.squeeze(dist).tolist():
                    if raw < h:
                        y = raw + 2*period; raw += step
                        for interval in get_start_end(mask[y-2*period] * dir_mask[y-2*period]):
                            begin = interval[0] - 2*period
                            end   = interval[1] + 2*period
                            stroke_temp['begin'] = begin
                            stroke_temp['end']   = end
                            stroke_temp['row']   = y-int(period/2)
                            stroke_temp['importance'] = (
                                (255-stroke_temp['grayscale']) *
                                torch.sum(gradient_norm[y:y+period, interval[0]+2*period:interval[1]+2*period]).numpy()
                            )
                            stroke_sequence.append(stroke_temp.copy())

        if random_order:
            random.shuffle(stroke_sequence)

        # แคนวาส
        result = Gassian((h0, w0), mean=255, var=0)
        if result_bgr is None:
            result_bgr = cv2.merge([result, result, result])

        canvases = []
        for dirs in range(direction):
            angle = -90 + dirs * 180 / direction
            canvas, _ = rotate(result, -angle)
            canvas = np.pad(canvas, pad_width=2*period, mode='constant', constant_values=(255, 255))
            canvases.append(canvas)

        # โฟลเดอร์เซฟเฟรม (รวมทุก K ไว้ที่เดียว)
        proc_dir = Path(RUN_PROCESS_DIR) if RUN_PROCESS_DIR else (output_path / "process")
        proc_dir.mkdir(parents=True, exist_ok=True)
        frame_id = len(list(proc_dir.glob("frame_*.jpg")))
        step = 0

        seq = stroke_sequence if MAX_STROKES is None else stroke_sequence[:MAX_STROKES]
        for s in seq:
            angle        = s['angle']
            dirs         = int((angle + 90) * direction / 180)
            grayscale    = s['grayscale']
            distribution = ChooseDistribution(period=period, Grayscale=grayscale)
            row          = s['row']
            begin        = s['begin']
            end          = s['end']
            length       = end - begin
            if length > 100:
                continue

            newline = Getline(distribution=distribution, length=length)
            canvas = canvases[dirs]
            temp = canvas[row:row+2*period, 2*period+begin:2*period+end]
            m    = np.minimum(temp, newline[:, :temp.shape[1]])
            canvas[row:row+2*period, 2*period+begin:2*period+end] = m

            now, _ = rotate(canvas[2*period:-2*period, 2*period:-2*period], angle)
            (H, W) = now.shape
            now    = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]

            # สะสมเส้นตาม repo
            result = np.minimum(now, result)

            # ลงสี “แข็ง” ตาม input26 (ทับเฉพาะพิกเซลที่มีเส้น)
            # + กรองด้วย center_mask เมื่อเปิด CENTER_ONLY
            row_hit = now < 247
            for y in range(now.shape[0]):
                allow = row_hit[y]
                if CENTER_ONLY and (center_mask is not None):
                    allow = allow & center_mask[y]
                idx = np.where(allow)[0]
                if idx.size:
                    result_bgr[y, idx, 0] = color[2]  # B
                    result_bgr[y, idx, 1] = color[1]  # G
                    result_bgr[y, idx, 2] = color[0]  # R

            if process_visible:
                try:
                    cv2.imshow('step', result_bgr); cv2.waitKey(1)
                except Exception:
                    pass

            if SAVE_FRAMES and (step % SAVE_EVERY == 0):
                cv2.imwrite(str(proc_dir / f"frame_{frame_id:06d}.jpg"), result_bgr)
                frame_id += 1
            step += 1

        if SAVE_FRAMES:
            cv2.imwrite(str(proc_dir / f"frame_{frame_id:06d}.jpg"), result_bgr)

    return result_bgr

# ================== MAIN: แบบ input26 ==================
if __name__ == '__main__':
    BASE = SEG_DIR / "output" / "kmeans" / "input26"   # โฟลเดอร์ของ input26

    # หา “ภาพต้นฉบับ” เพื่อดึงโทนเทาจริงภายในคลัสเตอร์
    cand = [
        BASE / "input26.jpg", BASE / "input26.png",
        BASE / "input26_seg.png", BASE / "input26_cluster_color.png",
        SEG_DIR / "photo" / "input26.jpg"
    ]
    SRC = next((p for p in cand if p.exists()), None)
    if SRC is None:
        raise SystemExit("ไม่พบภาพต้นฉบับของ input26 (ลองวาง input26.jpg ไว้ที่ segment/photo/ หรือ output/kmeans/input26/)")

    orig_bgr  = cv2.imread(str(SRC))
    if orig_bgr is None:
        raise SystemExit(f"อ่านรูปไม่ได้: {SRC}")
    orig_rgb  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

    # รวมเฟรมทุก K ไว้ที่เดียว (ทับไฟล์เก่าเพื่อได้ลำดับเรียบ)
    NAME = "input26"
    RUN_PROCESS_DIR = str(SEG_DIR / "output" / NAME / "process")
    pdir = Path(RUN_PROCESS_DIR)
    pdir.mkdir(parents=True, exist_ok=True)
    for old in list(pdir.glob("frame_*.jpg")):
        try: os.remove(str(old))
        except Exception: pass

    # เตรียม center mask (ถ้าเปิด)
    if CENTER_ONLY:
        center_mask = build_center_mask(*orig_gray.shape, fraction=CENTER_FRACTION, shape=CENTER_SHAPE)
    else:
        center_mask = None

    # หาไฟล์ K ทั้งหมดของ input26 (เรียงตามเลข)
    k_files = sorted(BASE.glob("input26_K*.jpg"), key=lambda p: int(re.search(r"K(\d+)", p.stem).group(1)))
    if not k_files:
        raise SystemExit(f"ไม่พบไฟล์ K ของ input26 ใน {BASE} (*_K#.jpg)")

    # สีของแต่ละคลัสเตอร์: ใช้ไฟล์ .npy ถ้ามี, ไม่มีก็ “เฉลี่ยสีจริง” จากภาพต้นฉบับใต้ mask
    rgb_npy = BASE / "input26_cluster_rgb.npy"
    cluster_rgbs = None
    if rgb_npy.exists():
        cluster_rgbs = np.load(str(rgb_npy)).astype(np.uint8)  # (K,3) RGB
    else:
        cluster_rgbs = []

    def run(k_idx, k_path, res=None):
        """สร้างภาพเทาเฉพาะคลัสเตอร์ (ไม่ใช่สีทึบ) แล้วส่งเข้า colorSketch()"""
        kimg = cv2.imread(str(k_path), cv2.IMREAD_GRAYSCALE)  # พื้นหลังขาว คลัสเตอร์ดำ
        if kimg is None:
            print(f"[warn] ข้าม (อ่านไม่ได้): {k_path}")
            return res
        _, mask = cv2.threshold(kimg, 250, 255, cv2.THRESH_BINARY_INV)  # ในคลัสเตอร์=255

        # เตรียมสีของคลัสเตอร์ k_idx
        if isinstance(cluster_rgbs, np.ndarray):
            rgb = cluster_rgbs[k_idx].tolist()
        else:
            # เฉลี่ยสีจริงจากภาพต้นฉบับตรงบริเวณ mask
            m = mask == 255
            if m.sum() == 0:
                rgb = [0, 0, 0]
            else:
                r = int(orig_rgb[...,0][m].mean())
                g = int(orig_rgb[...,1][m].mean())
                b = int(orig_rgb[...,2][m].mean())
                rgb = [r, g, b]

        # ทำภาพเทาเฉพาะพื้นที่ของคลัสเตอร์
        cluster_gray = np.full_like(orig_gray, 255)
        cluster_gray[mask == 255] = orig_gray[mask == 255]

        tmp_dir = SEG_DIR / "_tmp_kmeans_inputs" / NAME
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{k_path.stem}_gray.jpg"
        cv2.imwrite(str(tmp), cluster_gray)

        return colorSketch(img=str(tmp), color=rgb, result_bgr=res, center_mask=center_mask)

    # วาดตามลำดับไฟล์ K (สไตล์ input26)
    res_bgr = None
    for idx, kpath in enumerate(k_files):
        print(f"[DRAW] {kpath.name}")
        res_bgr = run(idx, kpath, res_bgr)

    # เซฟภาพรวม + รวมวิดีโอ/GIF
    out_dir = SEG_DIR / "output" / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "result_from_kmeans1.jpg"
    if res_bgr is not None:
        cv2.imwrite(str(out_img), res_bgr)
        print("Saved ->", out_img)
    else:
        print("[!] ไม่มีผลลัพธ์ให้เซฟ (ตรวจรายชื่อไฟล์ K ที่ส่งเข้า)")

    if MAKE_MP4:
        frames_to_mp4(Path(RUN_PROCESS_DIR), out_dir / f"{NAME}_process.mp4", fps=VIDEO_FPS)
    if MAKE_GIF:
        frames_to_gif(Path(RUN_PROCESS_DIR), out_dir / f"{NAME}_process.gif", fps=GIF_FPS)

    if process_visible:
        cv2.waitKey(0)
