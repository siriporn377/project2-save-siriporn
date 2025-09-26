# render_from_k_layers.py
import os, sys, re
import cv2
import numpy as np
import colorSketch3 as sk  # ใช้ไฟล์โมเดลของคุณ

def list_k_layers(kdir):
    files = []
    for fn in os.listdir(kdir):
        m = re.search(r'_K(\d+)\.(jpg|png)$', fn, flags=re.IGNORECASE)
        if m:
            files.append((int(m.group(1)), os.path.join(kdir, fn)))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]

def mean_color_rgb(path, white_thresh=250):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = np.any(rgb < white_thresh, axis=2)  # ไม่เอาพื้นขาว
    if not np.any(mask):
        return [0, 0, 0]
    return rgb[mask].mean(axis=0).astype(np.uint8).tolist()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def write_video_mp4(frames_dir, out_mp4, fps=20):
    frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith('.png')]
    )
    if not frames:
        print("[WARN] no frames to write.")
        return
    first = cv2.imread(frames[0]); H, W = first.shape[:2]
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for fp in frames:
        img = cv2.imread(fp)
        if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H))
        vw.write(img)
    vw.release()
    print(f"[OK] video saved: {out_mp4}")

def run(name_stem, kmeans_root='output/kmeans', fps=20):
    kdir = os.path.join(kmeans_root, name_stem)
    assert os.path.isdir(kdir), f"not found: {kdir}"

    # ปรับพารามิเตอร์ของโมเดลจากภายนอกได้ที่นี่
    sk.Freq = 50               # บันทึกเฟรมบ่อยขึ้นภายในเลเยอร์ (ถ้าเปิดบันทึกในโมเดล)
    sk.process_visible = False # ปิดหน้าต่าง imshow
    sk.random_order = True     # สุ่มลำดับสโตรก
    # sk.ETF_order = True      # ถ้าจะเรียงตาม importance (เปิด/แก้ในไฟล์โมเดลตามที่ต้องการ)

    out_root   = ensure_dir(os.path.join('output', 'strokes', name_stem))
    frames_dir = ensure_dir(os.path.join(out_root, 'frames'))

    layers = list_k_layers(kdir)
    print(f"[INFO] {len(layers)} layers: {[os.path.basename(p) for p in layers]}")

    result_bgr = None
    frame_idx = 0

    for li, layer_path in enumerate(layers, start=1):
        rgb = mean_color_rgb(layer_path)  # [R,G,B]
        print(f"[RUN] Layer {li}/{len(layers)}: {os.path.basename(layer_path)}  color={rgb}")
        result_bgr = sk.colorSketch(img=layer_path, color=rgb, result_bgr=result_bgr)

        snap = os.path.join(frames_dir, f'frame_{frame_idx:06d}.png')
        cv2.imwrite(snap, result_bgr); frame_idx += 1

    final_png = os.path.join(out_root, f'{name_stem}_final.png')
    cv2.imwrite(final_png, result_bgr)
    print(f"[OK] final image: {final_png}")

    mp4_path = os.path.join(out_root, f'{name_stem}.mp4')
    write_video_mp4(frames_dir, mp4_path, fps=fps)
    print(f"[DONE] outputs in: {out_root}")

if __name__ == '__main__':
    # ใช้: python render_from_k_layers.py input26
    name = sys.argv[1] if len(sys.argv) > 1 else 'input26'
    run(name)
