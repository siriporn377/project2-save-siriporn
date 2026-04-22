from pathlib import Path
import sys, os, re, glob, time
from datetime import datetime
import subprocess  

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
import torch
from PIL import Image

# ----- imports from repo -----
from LDR import *
from tone import *
from genStroke_origin import *
from drawpatch import rotate
from tools import *
from ETF.edge_tangent_flow import *
from deblue import deblue
from quicksort import *

# ====== PARAMETERS ======
np.random.seed(1)
n               = 6
period          = 8
direction       = 8
Freq            = 100
deepen          = 1
transTone       = False
kernel_radius   = 3
iter_time       = 15
background_dir  = 45
CLAHE           = True
edge_CLAHE      = True
draw_new        = True
random_order    = False
ETF_order       = False
process_visible = False

CENTER_ONLY     = False
CENTER_SHAPE    = 'rect'
CENTER_FRACTION = 0.70

# ====== บันทึกเฟรม/วิดีโอ ======
MAX_STROKES     = None
SAVE_FRAMES     = True
SAVE_EVERY      = 5        #
VIDEO_FPS       = 30
MAKE_MP4        = True
MAKE_GIF        = False

# ====== สีคลัสเตอร์ ======
USE_RGB_NPY     = False
RGB_NPY_IS_RGB  = True

# ---- ENV override สำหรับ backend pipeline ----
SAVE_EVERY = int(os.getenv("SKETCH_SAVE_EVERY", SAVE_EVERY))
VIDEO_FPS  = int(os.getenv("SKETCH_FPS", VIDEO_FPS))

# ---------- helpers ----------
def frames_to_mp4(process_dir: Path, out_mp4: Path, fps: int = 30):
    """
    รวมเฟรม JPG → MP4 ที่เล่นบนเว็บได้แน่นอน:
      - libx264
      - yuv420p
      - +faststart (ย้าย moov ไปหัวไฟล์)
      - บังคับมิติภาพเป็นเลขคู่ (ffmpeg pad)
    """
    frames = sorted(process_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"[warn] ไม่พบเฟรมใน {process_dir}")
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"")
        return

    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # --- 1) ใช้ ffmpeg ของ imageio-ffmpeg ---
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
        input_pattern = str(process_dir / "frame_%06d.jpg")
        vf = "pad=iw+mod(iw\\,2):ih+mod(ih\\,2),format=yuv420p"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True)
        print("MP4 saved (ffmpeg/libx264 yuv420p + faststart) ->", out_mp4)
        return
    except Exception as e:
        print("[warn] ffmpeg encode failed, fallback imageio/libx264 >", e)
    try:
        import imageio.v3 as iio, cv2
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        if w % 2: w -= 1
        if h % 2: h -= 1
        writer = iio.get_writer(str(out_mp4), fps=fps, codec="libx264", pix_fmt="yuv420p")
        for f in frames:
            img_bgr = cv2.imread(str(f))
            if img_bgr is None:
                continue
            if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
                img_bgr = cv2.resize(img_bgr, (w, h))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            writer.append_data(img_rgb)
        writer.close()
        print("MP4 saved (imageio/libx264 yuv420p) ->", out_mp4)
        return
    except Exception as e:
        print("[warn] imageio/libx264 fallback failed >", e)

 
    try:
        import cv2
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        if w % 2: w -= 1
        if h % 2: h -= 1
        vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            print("[error] เปิด VideoWriter mp4v ไม่ได้"); return
        for f in frames:
            img = cv2.imread(str(f))
            if img is None: continue
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            vw.write(img)
        vw.release()
        print("MP4 saved (mp4v fallback) ->", out_mp4)
    except Exception as e:
        print("[error] mp4v fallback failed >", e)


# ---------- core ----------
def colorSketch(img=None, color=None, result_bgr=None, center_mask=None, run_process_dir: Path=None):
    img_path = abspath(img)
    file_name = Path(img_path).stem
    output_root = SEG_DIR / "output"
    output_path = output_root / file_name
    (output_path / "mask").mkdir(parents=True, exist_ok=True)
    (output_path / "process").mkdir(parents=True, exist_ok=True)

    print(file_name)

    # ======= ETF =======
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

    # มาสก์คลัสเตอร์เดิม (กันสีเปื้อน)
    cluster_mask_orig = (input_img < 255)

    if transTone:
        input_img = transferTone(input_img)

    if center_mask is None and CENTER_ONLY:
        center_mask = build_center_mask(h0, w0, CENTER_FRACTION, CENTER_SHAPE)

    if draw_new:
        stroke_sequence = []
        stroke_temp = {'angle':None, 'grayscale':None, 'row':None, 'begin':None, 'end':None} 

        for dirs in range(direction):
            angle = -90 + dirs * 180 / direction #เส้นเอียง
            stroke_temp['angle'] = angle
            img_rot, _ = rotate(input_img, -angle)
            img_use = HistogramEqualization(img_rot) if CLAHE else img_rot

            # gradient หาความชั้นจุดที่สำคัญ วสย
            pad = cv2.copyMakeBorder(img_use, 2*period, 2*period, 2*period, 2*period, cv2.BORDER_REPLICATE)
            norm = cv2.normalize(pad.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)
            x_der = cv2.Sobel(norm, cv2.CV_32FC1, 1, 0, ksize=5)
            y_der = cv2.Sobel(norm, cv2.CV_32FC1, 0, 1, ksize=5)
            x_der = torch.from_numpy(x_der) + 1e-12
            y_der = torch.from_numpy(y_der) + 1e-12
            gradient_magnitude = torch.sqrt(x_der**2.0 + y_der**2.0)
            gradient_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-12)

            # LDR แบ่งรูปเทา
            ldr = LDR(img_use, n)
            LDR_single_add(ldr, n, str(output_path))
            (h, w) = ldr.shape

            # มาสก์คลัสเตอร์ (หมุนด้วยพื้นขาว)
            cluster_mask = (input_img < 255).astype(np.uint8) * 255
            cluster_mask, _ = rotate(cluster_mask, -angle, pad_color=255)
            cluster_mask[cluster_mask < 128] = 0
            cluster_mask[cluster_mask >= 128] = 1
            #เตรียม stroke สำหรับโทนความสว่าง
            for j in range(n):
                stroke_temp['grayscale'] = j*255/n
                mask_path    = output_path / 'mask' / f'mask{j}.png' 
                dir_mask_png = output_path / 'mask' / f'dir_mask{dirs}.png'
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) / 255
                #มากสำหรับทิศทางของ etf (dir_mask)
                dir_mask = cv2.imread(str(dir_mask_png), cv2.IMREAD_GRAYSCALE)
                dir_mask, _ = rotate(dir_mask, -angle, pad_color=255)
                dir_mask[dir_mask < 128] = 0; dir_mask[dir_mask > 127] = 1
                
                for y in range(2*period, h, period):
                    row_mask = mask[y-2*period] * dir_mask[y-2*period] * cluster_mask[y-2*period]
                    for interval in get_start_end(row_mask):
                        begin = interval[0] - 2*period
                        end   = interval[1] + 2*period
                        stroke_temp['begin'] = begin
                        stroke_temp['end']   = end
                        stroke_temp['row']   = y - int(period/2)
                        stroke_temp['importance'] = (
                            (255-stroke_temp['grayscale']) *
                            torch.sum(gradient_norm[y:y+period, interval[0]+2*period:interval[1]+2*period]).numpy()
                        )
                        stroke_sequence.append(stroke_temp.copy())

        
        stroke_sequence.sort(key=lambda s: (s['row'], s['begin'])) #*****

        # แคนวาสเริ่ม
        result = Gassian((h0, w0), mean=255, var=0)
        if result_bgr is None:
            result_bgr = cv2.merge([result, result, result])
        
        canvases = [] #เก็บกระดาษแต่ละทิศ 
        for dirs in range(direction):
            angle = -90 + dirs * 180 / direction
            canvas, _ = rotate(result, -angle)
            canvas = np.pad(canvas, pad_width=2*period, mode='constant', constant_values=(255, 255))
            canvases.append(canvas)

        # โฟลเดอร์เฟรมของรันนี้
        if run_process_dir is None:
            run_process_dir = Path(SEG_DIR / "output" / "process_fallback")
        run_process_dir.mkdir(parents=True, exist_ok=True)
        frame_id = len(list(run_process_dir.glob("frame_*.jpg")))
        step = 0

        # เตรียมก่อนวาด
        seq = stroke_sequence if MAX_STROKES is None else stroke_sequence[:MAX_STROKES]
        for s in seq:
            angle        = s['angle']
            dirs         = int((angle + 90) * direction / 180)
            grayscale    = s['grayscale']
            distribution = ChooseDistribution(period=period, Grayscale=grayscale) #หน้าตาของเส้น
            row          = s['row']
            begin        = s['begin']
            end          = s['end']
            full_len     = end - begin
            if full_len <= 0:
                step += 1
                continue
        # วาดทีละเส้น
            newline_full = Getline(distribution=distribution, length=full_len) #วาดจริง
            MAX_SEG = 256 #ท่อนจะวาดได้แค่ 256 px 
            offset = 0
            s0 = begin
            while s0 < end:
                seg_len = min(MAX_SEG, end - s0)
                s1 = s0 + seg_len
                seg = newline_full[:, offset:offset + seg_len]

                canvas  = canvases[dirs]
                temp    = canvas[row:row+2*period, 2*period+s0:2*period+s1]
                m       = np.minimum(temp, seg[:, :temp.shape[1]]) 
                canvas[row:row+2*period, 2*period+s0:2*period+s1] = m

                offset += seg_len
                s0 = s1

            # หมุนกลับมา + อัปเดตผลรวม
            now, _ = rotate(canvas[2*period:-2*period, 2*period:-2*period], angle)
            (H, W) = now.shape
            now    = now[int((H-h0)/2):int((H-h0)/2)+h0, int((W-w0)/2):int((W-w0)/2)+w0]

            # เติมสีเฉพาะจุดที่เพิ่งวาดภายใต้มาสก์เดิม
            PAINT_EPS = 3
            prev = result
            just_painted = (prev - now) > PAINT_EPS
            row_hit = (now < 247) & just_painted
            allow = row_hit & cluster_mask_orig

            kernel = np.ones((3,3), np.uint8)
            allow_u8 = (allow.astype(np.uint8) * 255)
            allow_u8 = cv2.morphologyEx(allow_u8, cv2.MORPH_OPEN, kernel)
            allow = allow_u8.astype(bool)

            ys, xs = np.where(allow)
            if ys.size: #เติมสีจริง
                result_bgr[ys, xs, 0] = color[2]  # B
                result_bgr[ys, xs, 1] = color[1]  # G
                result_bgr[ys, xs, 2] = color[0]  # R

            result = np.minimum(now, prev) #รวมเส้นลงภาพเทา

            if process_visible:
                try:
                    cv2.imshow('step', result_bgr); cv2.waitKey(1)
                except Exception:
                    pass

            if SAVE_FRAMES and (step % SAVE_EVERY == 0):
                cv2.imwrite(str(run_process_dir / f"frame_{frame_id:06d}.jpg"), result_bgr)
                frame_id += 1
            step += 1

        if SAVE_FRAMES:
            cv2.imwrite(str(run_process_dir / f"frame_{frame_id:06d}.jpg"), result_bgr)

    return result_bgr #ภาพสี

# ===================== main =====================
if __name__ == '__main__':
    # อ่านชื่อชุดงานจาก ENV ที่ backend ใส่มา (เช่น job_1712345678_s960)
    NAME = os.getenv("SKETCH_NAME", "input45")
    BASE = SEG_DIR / "output" / "kmeans_1seg" / NAME

    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
    RUN_DIR = SEG_DIR / "output" / NAME / f"run_{RUN_ID}"
    RUN_PROCESS_DIR = RUN_DIR / "process"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    # หา “ภาพต้นฉบับ”
    cand = [
        BASE / f"{NAME}.jpg", BASE / f"{NAME}.png",
        BASE / f"{NAME}_seg.png", BASE / f"{NAME}_cluster_color.png",
        SEG_DIR / "photo" / f"{NAME}.jpg"
    ]
    SRC = next((p for p in cand if p.exists()), None)
    if SRC is None:
        raise SystemExit(f"ไม่พบภาพต้นฉบับของ {NAME}")

    orig_bgr  = cv2.imread(str(SRC))
    if orig_bgr is None:
        raise SystemExit(f"อ่านรูปไม่ได้: {SRC}")
    orig_rgb  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)

    center_mask = None  # ไม่บังคับ

    # รายการไฟล์ K ตามลำดับ
    k_files = sorted(
        BASE.glob(f"{NAME}_K*.jpg"),
        key=lambda p: int(re.search(r"K(\d+)", p.stem).group(1))
    )
    if not k_files:
        raise SystemExit(f"ไม่พบไฟล์ K ของ {NAME} ใน {BASE} (*_K#.jpg)")

    def run(k_idx, k_path, res=None):
        kimg = cv2.imread(str(k_path), cv2.IMREAD_GRAYSCALE)
        if kimg is None:
            print(f"[warn] ข้าม (อ่านไม่ได้): {k_path}")
            return res
        _, mask = cv2.threshold(kimg, 250, 255, cv2.THRESH_BINARY_INV)

        # สีเฉลี่ยจากภาพจริงใต้คลัสเตอร์
        m = (mask == 255)
        if m.sum() == 0:
            rgb = [0, 0, 0]
        else:
            r = int(orig_rgb[...,0][m].mean())
            g = int(orig_rgb[...,1][m].mean())
            b = int(orig_rgb[...,2][m].mean())
            rgb = [r, g, b]

        # ทำภาพเทาเฉพาะคลัสเตอร์
        cluster_gray = np.full_like(orig_gray, 255)
        cluster_gray[mask == 255] = orig_gray[mask == 255]

        tmp_dir = SEG_DIR / "_tmp_kmeans_inputs" / NAME
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{k_path.stem}_gray.jpg"
        cv2.imwrite(str(tmp), cluster_gray)

        return colorSketch(
            img=str(tmp),
            color=rgb,
            result_bgr=res,
            center_mask=center_mask,
            run_process_dir=RUN_PROCESS_DIR
        )

    # วาดครบทุก K
    res_bgr = None
    for idx, kpath in enumerate(k_files):
        print(f"[DRAW] {kpath.name}")
        res_bgr = run(idx, kpath, res_bgr)

    # เซฟภาพรวม + ทำ MP4 จากเฟรม
    out_img = RUN_DIR / f"result_{NAME}_{RUN_ID}.jpg"
    if res_bgr is not None:
        cv2.imwrite(str(out_img), res_bgr)
        print("Saved ->", out_img)

    if MAKE_MP4:
        out_mp4 = RUN_DIR / f"{NAME}_process_{RUN_ID}.mp4"
        frames_to_mp4(RUN_PROCESS_DIR, out_mp4, fps=VIDEO_FPS)

    if process_visible:
        cv2.waitKey(0)
