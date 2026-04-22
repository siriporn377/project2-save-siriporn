from pathlib import Path
import shutil, subprocess, sys, os, time, glob, json
import numpy as np  # ⬅️ ใช้โหลด cluster_rgb.npy

# ตำแหน่งโฟลเดอร์
ROOT         = Path(__file__).resolve().parent      # .../backend
PROJECT_ROOT = ROOT.parent                          # .../
SEG_DIR      = PROJECT_ROOT / "segment"             # .../segment
PYTHON       = sys.executable

def _latest(pattern: str):
    files = [Path(p) for p in glob.glob(pattern)]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def _pick_segment_script() -> Path:
    cand = [SEG_DIR / "segmentnew4.py", SEG_DIR / "segment2.py", SEG_DIR / "segmentnew.py"]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError("ไม่พบ segmentnew4.py / segment2.py / segmentnew.py ในโฟลเดอร์ segment/")

def _hex(rgb):
    r,g,b = rgb
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

# ---------- พาเล็ต: Fallback เดิมจากภาพสุดท้าย ----------
def _extract_palette_jpg(img_path: Path, k: int, out_json: Path):
    """
    สกัดพาเล็ตจากภาพสุดท้ายด้วย PIL (adaptive palette)
    เก็บเป็น { "colors": ["#RRGGBB", ...] } เรียงจากสีที่พบมาก → น้อย
    ตัด near-white/near-black ออกเล็กน้อย
    """
    try:
        from PIL import Image
    except Exception:
        return
    try:
        im = Image.open(img_path).convert("RGB")
        pal_img = im.convert("P", palette=Image.ADAPTIVE, colors=max(1, min(9, k)))
        pal = pal_img.getpalette()  # flat list
        counts = pal_img.getcolors() or []

        def idx_rgb(i):
            base = i*3
            return (pal[base], pal[base+1], pal[base+2])

        pairs = sorted(counts, key=lambda t: t[0], reverse=True)
        colors = []
        for cnt, idx in pairs:
            rgb = idx_rgb(idx)
            if max(rgb) < 18:   # almost black
                continue
            if min(rgb) > 238:  # almost white
                continue
            hx = _hex(rgb)
            if hx not in colors:
                colors.append(hx)

        if not colors:
            colors = [_hex(im.resize((1,1)).getpixel((0,0)))]

        out_json.write_text(json.dumps({"colors": colors}, ensure_ascii=False))
    except Exception:
        # ถ้าพัง ก็ปล่อยให้ไม่มีพาเล็ต (frontend จะไม่แสดงวงกลม)
        pass

# ---------- พาเล็ต: ใหม่ จากผลการ segment (cluster_rgb.npy) ----------
def _write_palette_from_cluster(name_out: str, k: int, out_json: Path) -> bool:
    """
    อ่านสีคลัสเตอร์จาก segment/output/kmeans_1seg/<name_out>/<name_out>_cluster_rgb.npy
    แล้วเขียน palette.json ให้ได้จำนวนเท่ากับ K ที่ร้องขอจริง ๆ
    """
    try:
        base = SEG_DIR / "output" / "kmeans_1seg" / name_out
        npy  = base / f"{name_out}_cluster_rgb.npy"
        if not npy.exists():
            print(f"[warn] palette: missing {npy}")
            return False

        arr = np.load(npy)  # shape: (K,3) uint8
        if arr.ndim != 2 or arr.shape[1] != 3:
            print(f"[warn] palette: bad shape in {npy} -> {arr.shape}")
            return False

        want = int(k)
        cols = []
        upto = min(want, len(arr))
        for i in range(upto):
            r, g, b = (int(arr[i,0]), int(arr[i,1]), int(arr[i,2]))
            cols.append(f"#{r:02X}{g:02X}{b:02X}")

        out_json.write_text(json.dumps({"colors": cols}, ensure_ascii=False))
        print(f"[ok] palette from cluster ({upto} colors) -> {out_json}")
        return True
    except Exception as e:
        print("[warn] palette from cluster failed >", e)
        return False


def process_image(original_path: str, job_dir: Path, k: int = 6) -> Path:
    """
    1) copy รูปไป segment/photo
    2) รันสคริปต์ segment
    3) รัน color5.py สร้าง MP4
    4) copy MP4 + JPG สุดท้าย → backend/static/<job_id>/
    5) สร้าง palette.json จาก cluster_rgb.npy (ถ้าไม่ได้ ค่อย fallback จาก final.jpg)
    """
    original_path = Path(original_path)
    job_dir = Path(job_dir)

    name = f"job_{int(time.time())}"
    max_side = int(os.getenv("SKETCH_MAX_SIDE", "300"))
    k = max(1, min(9, int(os.getenv("SKETCH_K", str(k)))))

    # 1) รูปทำงาน
    seg_photo = SEG_DIR / "photo"
    seg_photo.mkdir(parents=True, exist_ok=True)
    work_img = seg_photo / f"{name}{original_path.suffix.lower() or '.png'}"
    shutil.copyfile(original_path, work_img)

    # 2) segment
    seg_script = _pick_segment_script()
    cmd1 = [
        PYTHON, str(seg_script), str(work_img),
        "--mode", "bird_branch",
        "--max-side", str(max_side),
        "--k", str(k),
    ]
    subprocess.run(cmd1, check=True, cwd=str(PROJECT_ROOT))

    name_out = f"{name}_s{max_side}_k{k}"

    env = os.environ.copy()
    env["SKETCH_NAME"]       = name_out
    env["SKETCH_SAVE_EVERY"] = env.get("SKETCH_SAVE_EVERY", "2")
    env["SKETCH_FPS"]        = env.get("SKETCH_FPS", "30")

    cmd2 = [PYTHON, str(SEG_DIR / "color5.py")]
    subprocess.run(cmd2, check=True, cwd=str(SEG_DIR), env=env)

    # 4) คัดลอกผลลัพธ์ล่าสุด
    run_base = SEG_DIR / "output" / name_out

    mp4 = _latest((run_base / "run_*" / f"{name_out}_process_*.mp4").as_posix())
    out_mp4 = job_dir / "result.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if mp4 and mp4.exists():
        shutil.copyfile(mp4, out_mp4)
    else:
        out_mp4.write_bytes(b"")

    jpg = _latest((run_base / "run_*" / f"result_{name_out}_*.jpg").as_posix())
    out_jpg = job_dir / "final.jpg"
    if jpg and jpg.exists():
        shutil.copyfile(jpg, out_jpg)

    # 5) ทำ palette.json จาก "cluster_rgb.npy" ให้จำนวนเท่ากับ K ที่ร้องขอจริง ๆ
    #    ถ้าอ่านไม่ได้ ค่อย fallback เป็นการควอนไทซ์จาก final.jpg
    palette_json = job_dir / "palette.json"
    ok = _write_palette_from_cluster(name_out, k, palette_json)
    if not ok and out_jpg.exists():
        _extract_palette_jpg(out_jpg, k, palette_json)

    return out_mp4
