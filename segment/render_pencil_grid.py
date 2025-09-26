# ===== segment/render_pencil_grid.py =====
# วาดลายเส้นดินสอทีละคลัสเตอร์ (KMeans) โดยไม่พึ่งโมเดล
# จุดเด่น:
#  - ใช้ Structure Tensor หา orientation (แนวเส้นสัมผัส)
#  - "วางเส้นแบบกริด" ภายใน mask ของคลัสเตอร์ → ได้เส้นเยอะ ไม่โล่ง
#  - สีเส้นเทาเข้ม (ไม่ผูกกับสีคลัสเตอร์) เห็นชัดบนพื้นขาว
#  - วาดคลัสเตอร์เรียงจากบนลงล่าง (เริ่มหัวก่อน)
#
# ต้องมีไฟล์:
#   segment/output/kmeans/<NAME>/<NAME>_cluster_map.npy
#   segment/output/kmeans/<NAME>/<NAME>_cluster_rgb.npy   (มีไว้เช็ค K อย่างเดียว)
#
# ใช้:
#   py segment\render_pencil_grid.py input40

import os, sys, cv2, numpy as np
from pathlib import Path
from datetime import datetime

SEG_DIR  = Path(__file__).resolve().parent
PROJ_DIR = SEG_DIR.parent
EXPORTS  = PROJ_DIR / "exports"
EXPORTS.mkdir(parents=True, exist_ok=True)

# ---------- CONFIG ----------
NAME_DEFAULT      = "input41"
ORDER             = "top"            # "top" | "area" | "id"
FPS               = 24
HOLD_FRAMES       = 3
BG_COLOR          = (255,255,255)    # พื้นหลังขาวสนิท เพื่อคอนทราสต์สูง

# หัวดินสอ (แกนสีเทาเข้ม)
PENCIL_RADIUS     = 14               # ครึ่งความหนา
PENCIL_ASPECT     = 3.4              # ยาว/หนา (มากขึ้น = ยาวขึ้น)
PENCIL_SOFTEN     = 5
PENCIL_DARK       = 45               # 0 ดำมาก → ลดเลข = เข้มขึ้น
PENCIL_COLOR_BGR  = (40,40,40)       # ใช้สีเทาเข้มคงที่ (ไม่ผูกสีคลัสเตอร์)

# การวางจุดแบบกริด (ยิ่ง step เล็ก → เส้นเยอะ)
GRID_STEP         = 10               # px ระยะกริดพื้นฐาน
GRID_JITTER       = 0.35             # สุ่มเลื่อนตำแหน่งภายในเซลล์ 0..1
MAG_THRESH_PCT    = 55               # เกณฑ์คัดตาม gradient magnitude เป็นเปอร์เซ็นไทล์ของแต่ละคลัสเตอร์

# จำกัดภาระงาน
MAX_POINTS_PER_CLUSTER = 8000        # ลิมิตจำนวนจุดสูงสุด/คลัสเตอร์ (กันหนักเครื่อง)

ANGLE_JITTER_DEG  = 6.0              # สุ่มแกว่งมุม ± องศา
LEN_JITTER_PCT    = 0.25             # สุ่มยาว/สั้นของแสตมป์เล็กน้อย

# ---------- helpers ----------
def find_image_path(name: str) -> str:
    for ext in (".jpg",".png",".jpeg"):
        p = SEG_DIR/"photo"/f"{name}{ext}"
        if p.exists(): return str(p)
    return ""

def clear_dir(p: Path):
    if p.exists():
        for f in p.glob("*"):
            try: f.unlink()
            except: pass
    else:
        p.mkdir(parents=True, exist_ok=True)

def make_pencil_stamp(radius=10, aspect=3.0, soften=5, darkness=60):
    h = int(2*radius)
    w = max(2, int(2*radius*aspect))
    stamp = np.full((h, w), 255, np.uint8)
    cv2.ellipse(stamp, (w//2, h//2), (w//2-1, h//2-1), 0, 0, 360, int(darkness), -1)
    if soften>0 and soften%2==1:
        stamp = cv2.GaussianBlur(stamp, (soften, soften), 0)
    return stamp

def rotate_and_blend_stamp(canvas, center_xy, angle_deg, stamp_gray, color_bgr):
    H,W = canvas.shape[:2]
    sh, sw = stamp_gray.shape
    M = cv2.getRotationMatrix2D((sw/2, sh/2), angle_deg, 1.0)
    rot = cv2.warpAffine(stamp_gray, M, (sw,sh), flags=cv2.INTER_LINEAR, borderValue=255)
    alpha = (255.0 - rot).astype(np.float32) / 255.0  # 0..1
    if alpha.max() <= 0: return
    cy, cx = center_xy
    top = int(cy - sh/2); left = int(cx - sw/2)
    bot = top + sh;       right = left + sw
    if bot <= 0 or right <= 0 or top >= H or left >= W: return
    t0, l0 = max(0, top), max(0, left)
    b0, r0 = min(H, bot), min(W, right)
    a = alpha[(t0-top):(t0-top)+(b0-t0), (l0-left):(l0-left)+(r0-l0)]
    if a.size == 0: return
    roi = canvas[t0:b0, l0:r0].astype(np.float32)
    col = np.array(color_bgr, np.float32)
    canvas[t0:b0, l0:r0] = (roi*(1.0-a[...,None]) + col*a[...,None]).astype(np.uint8)

def build_video_from_frames(frames_dir: Path, out_mp4: Path, out_gif: Path, fps: int):
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames: 
        print("[WARN] no frames:", frames_dir); 
        return False
    first = cv2.imread(str(frames[0])); h,w = first.shape[:2]
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None: continue
        if img.shape[:2] != (h,w): img = cv2.resize(img, (w,h))
        vw.write(img)
    vw.release()
    try:
        import imageio.v2 as imageio
        imgs = [imageio.imread(str(fp)) for fp in frames]
        imageio.mimsave(str(out_gif), imgs, duration=1.0/fps)
    except Exception as e:
        print("[GIF] skip:", e)
    return True

# ---------- orientation (Structure Tensor) ----------
def structure_tensor_orientation_and_mag(gray, blur_sigma=1.5, ksize=3, smooth_sigma=3.0):
    g = cv2.GaussianBlur(gray, (0,0), blur_sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)  # gradient magnitude
    Jxx = cv2.GaussianBlur(gx*gx, (0,0), smooth_sigma)
    Jyy = cv2.GaussianBlur(gy*gy, (0,0), smooth_sigma)
    Jxy = cv2.GaussianBlur(gx*gy, (0,0), smooth_sigma)
    theta = 0.5 * np.arctan2(2*Jxy, (Jxx - Jyy))
    tangent_deg = (np.degrees(theta) + 360.0) % 360.0
    return tangent_deg, mag

# ---------- MAIN ----------
def main():
    NAME = sys.argv[1] if len(sys.argv)>1 else NAME_DEFAULT
    KMEANS_DIR = SEG_DIR/"output"/"kmeans"/NAME
    map_path   = KMEANS_DIR/f"{NAME}_cluster_map.npy"
    rgb_path   = KMEANS_DIR/f"{NAME}_cluster_rgb.npy"
    assert map_path.exists() and rgb_path.exists(), \
        f"ไม่พบ {map_path.name}/{rgb_path.name} ใน {KMEANS_DIR} (รัน: py segment\\segmentnew.py {NAME}.jpg)"

    # โหลดข้อมูล KMeans
    cluster_map = np.load(str(map_path))     # HxW int
    cluster_rgb = np.load(str(rgb_path))     # Kx3 (ไม่ใช้สี แต่นับ K)
    H,W = cluster_map.shape

    # รีแมป label ให้เป็น 0..K-1
    uniq = np.unique(cluster_map)
    remap = {int(old):i for i,old in enumerate(uniq.tolist())}
    cluster_map = np.vectorize(lambda v: remap[int(v)])(cluster_map)
    K = len(uniq)
    if cluster_rgb.shape[0] < K:
        raise RuntimeError(f"cluster_rgb มี {cluster_rgb.shape[0]} สี แต่มีคลัสเตอร์ {K}")

    # ภาพเทาอ้างอิง
    base_gray = np.full((H,W), 200, np.uint8)
    ip = find_image_path(NAME)
    if ip:
        g = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if g is not None and g.shape[:2] != (H,W):
            g = cv2.resize(g, (W,H), interpolation=cv2.INTER_AREA)
        if g is not None:
            base_gray = g

    # คำนวณ orientation + magnitude
    tangent_deg, grad_mag = structure_tensor_orientation_and_mag(base_gray, 1.5, 3, 3.0)

    # เตรียม output เฟรม
    FRAMES_DIR = KMEANS_DIR/"_frames_final"
    clear_dir(FRAMES_DIR)
    frame_id = 0
    canvas = np.zeros((H,W,3), np.uint8); canvas[:] = BG_COLOR
    cv2.imwrite(str(FRAMES_DIR/f"frame_{frame_id:06d}.png"), canvas); frame_id += 1

    base_stamp = make_pencil_stamp(PENCIL_RADIUS, PENCIL_ASPECT, PENCIL_SOFTEN, PENCIL_DARK)
    rng = np.random.default_rng(2025)

    # ลำดับคลัสเตอร์ (เริ่มจากด้านบน)
    ids = list(range(K))
    if ORDER=="area":
        ids = sorted(ids, key=lambda k: -int((cluster_map==k).sum()))
    elif ORDER=="top":
        tops=[]
        for k in range(K):
            ys,_ = np.where(cluster_map==k)
            tops.append((k, ys.mean() if len(ys)>0 else 1e9))
        ids = [k for k,_ in sorted(tops, key=lambda t:t[1])]

    print(f"[info] clusters={len(ids)}")

    # วาดทีละคลัสเตอร์
    for idx,k in enumerate(ids,1):
        mask = (cluster_map==k)
        if not mask.any(): 
            continue

        # เกณฑ์ magnitude เฉพาะภายในคลัสเตอร์
        gm = grad_mag.copy()
        gm[~mask] = 0
        vals = gm[mask]
        if vals.size == 0:
            continue
        thr = float(np.percentile(vals, MAG_THRESH_PCT))

        # สร้างกริดของพิกเซล
        ys = np.arange(0, H, GRID_STEP)
        xs = np.arange(0, W, GRID_STEP)
        count = 0

        for y in ys:
            # หมายเหตุ: ใช้ jitter ให้ดูเป็นธรรมชาติ
            for x in xs:
                yy = int(np.clip(y + (rng.random()*2-1)*GRID_STEP*GRID_JITTER, 0, H-1))
                xx = int(np.clip(x + (rng.random()*2-1)*GRID_STEP*GRID_JITTER, 0, W-1))
                if not mask[yy, xx]:
                    continue
                if gm[yy, xx] < thr:
                    continue

                ang = float(tangent_deg[yy, xx])
                jitter = (rng.random()*2 - 1) * ANGLE_JITTER_DEG

                # random ความยาวเล็กน้อย
                stamp = base_stamp
                if LEN_JITTER_PCT > 0:
                    jl = 1.0 + (rng.random()*2 - 1) * float(LEN_JITTER_PCT)
                    jl = max(0.7, min(1.3, jl))
                    h, w = base_stamp.shape
                    stamp = cv2.resize(base_stamp, (int(w*jl), h), interpolation=cv2.INTER_LINEAR)

                rotate_and_blend_stamp(canvas, (yy, xx), ang + jitter, stamp, PENCIL_COLOR_BGR)

                count += 1
                if count >= MAX_POINTS_PER_CLUSTER:
                    break
            if count >= MAX_POINTS_PER_CLUSTER:
                break

        # บันทึกเฟรมพักให้เห็นความคืบหน้า
        for _ in range(HOLD_FRAMES):
            cv2.imwrite(str(FRAMES_DIR/f"frame_{frame_id:06d}.png"), canvas); frame_id += 1

    # export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_mp4 = KMEANS_DIR/f"{NAME}_strokes_{ts}.mp4"
    out_gif = KMEANS_DIR/f"{NAME}_strokes_{ts}.gif"
    ok = build_video_from_frames(FRAMES_DIR, out_mp4, out_gif, FPS)
    if ok:
        final_png = KMEANS_DIR/f"{NAME}_strokes_{ts}.png"
        cv2.imwrite(str(final_png), canvas, [int(cv2.IMWRITE_JPEG_QUALITY),95])
        try:
            import shutil
            for p in (out_mp4,out_gif,final_png):
                shutil.copy2(str(p), str(EXPORTS/p.name))
        except: pass
        print("video  :", out_mp4)
        print("gif    :", out_gif)
        print("image  :", final_png)
        print("exports:", EXPORTS)
    else:
        print("[ERR] no frames:", FRAMES_DIR)

if __name__ == "__main__":
    main()
