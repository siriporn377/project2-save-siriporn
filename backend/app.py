from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pathlib import Path
import uuid, shutil, os, io, json

from .processing import process_image  # <- ของเดิมคุณ

ROOT         = Path(__file__).resolve().parent
STATIC_DIR   = ROOT / "static"
UPLOAD_DIR   = ROOT / "uploads"
FRONTEND_DIR = ROOT.parent / "frontend"

for p in (STATIC_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SketchColor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _unlink(p: Path):
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass

def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def _exists_nonempty(p: Path):
    return p.exists() and p.stat().st_size > 0

def _cleanup_if_empty(job_id: str):
    d = STATIC_DIR / job_id
    try:
        if d.exists() and d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    except Exception:
        pass

def _promote_cached(job_dir: Path, k: int) -> bool:
    """
    ถ้ามีไฟล์ที่แคชไว้ตาม K ให้ 'โปรโมต' ขึ้นมาเป็นชื่อปัจจุบัน (result.mp4 / final.jpg / result.gif / palette.json)
    แล้วรีเทิร์น True ถ้าทำได้ครบอย่างน้อยไฟล์วิดีโอ
    """
    src_mp4 = job_dir / f"result_k{k}.mp4"
    src_jpg = job_dir / f"final_k{k}.jpg"
    src_gif = job_dir / f"result_k{k}.gif"
    src_pal = job_dir / f"palette_k{k}.json"

    ok = False
    if _exists_nonempty(src_mp4):
        _copy(src_mp4, job_dir / "result.mp4")
        ok = True
    if _exists_nonempty(src_jpg):
        _copy(src_jpg, job_dir / "final.jpg")
    if _exists_nonempty(src_gif):
        _copy(src_gif, job_dir / "result.gif")
    if _exists_nonempty(src_pal):
        _copy(src_pal, job_dir / "palette.json")
    return ok

def _snapshot_current_as_k(job_dir: Path, k: int) -> bool:
    """
    ถ้ามีไฟล์ปัจจุบัน (generic) ให้สำเนาเก็บเป็น result_k{k}.*
    """
    cur_mp4 = job_dir / "result.mp4"
    cur_jpg = job_dir / "final.jpg"
    cur_gif = job_dir / "result.gif"
    cur_pal = job_dir / "palette.json"

    done = False
    if _exists_nonempty(cur_mp4):
        _copy(cur_mp4, job_dir / f"result_k{k}.mp4")
        done = True
    if _exists_nonempty(cur_jpg):
        _copy(cur_jpg, job_dir / f"final_k{k}.jpg")
    if _exists_nonempty(cur_gif):
        _copy(cur_gif, job_dir / f"result_k{k}.gif")
    if _exists_nonempty(cur_pal):
        _copy(cur_pal, job_dir / f"palette_k{k}.json")
    return done

# --- shrink upload to 500px / ~500KB (configurable via ENV) ---
def _shrink_image_to_jpg(in_file, out_path: Path,
                         max_side: int = None,
                         target_kb: int = None,
                         bg_color=(255,255,255)):
    from PIL import Image
    max_side  = int(os.getenv("SKETCH_UPLOAD_MAX_SIDE", str(max_side or 500)))
    target_kb = int(os.getenv("SKETCH_UPLOAD_TARGET_KB", str(target_kb or 500)))

    im = Image.open(in_file)
    if im.mode in ("RGBA","LA"):
        bg = Image.new("RGB", im.size, bg_color)
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    im.thumbnail((max_side, max_side), Image.LANCZOS)

    lo, hi = 30, 95
    best = None
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        sz_kb = len(buf.getvalue()) / 1024
        if sz_kb <= target_kb:
            best = buf.getvalue()
            lo = q + 1
        else:
            hi = q - 1
    if best is None:
        im.save(out_path, format="JPEG", quality=35, optimize=True, progressive=True)
    else:
        out_path.write_bytes(best)

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/upload")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    k: int = Form(6)
):
    job_id = uuid.uuid4().hex
    job_dir = STATIC_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    original_path = job_dir / "input.jpg"
    try:
        file.file.seek(0)
        _shrink_image_to_jpg(file.file, original_path, max_side=500, target_kb=500)
    except Exception:
        file.file.seek(0)
        with original_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

    k = max(1, min(9, int(k)))
    # ประมวลผลครั้งแรก
    background_tasks.add_task(process_image, str(original_path), job_dir, k=k)

    return JSONResponse({
        "job_id": job_id,
        "original_url": f"/static/{job_id}/{original_path.name}",
        "result_url":   f"/static/{job_id}/result.mp4",
        "image_url":    f"/static/{job_id}/final.jpg",
        "gif_url":      f"/static/{job_id}/result.gif",
        "k": k,
        "status": "processing",
    })

@app.post("/api/reprocess/{job_id}")
async def reprocess_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    k: int = Form(...),
    force: int = Form(0)
):
    """ใช้รูปเดิมของ job นี้ แล้วประมวลผลใหม่ด้วย K ใหม่
       - ถ้ามี cache ของ K นี้แล้ว และ force=0 -> โปรโมต cache ขึ้นมาและตอบกลับทันที (ไม่รันใหม่)
       - ถ้าไม่มี cache -> ล้างผลเดิมแล้วคิวรันใหม่
    """
    job_dir = STATIC_DIR / job_id
    original_path = job_dir / "input.jpg"
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Original image for this job not found")

    k = max(1, min(9, int(k)))
    if force == 0:
        # ถ้ามี cache อยู่แล้ว ให้โปรโมตขึ้นมาเป็นผลปัจจุบันเลย (เร็ว)
        if _promote_cached(job_dir, k):
            return JSONResponse({"job_id": job_id, "k": k, "status": "cached"})

    # ล้างผลลัพธ์เก่า (เพื่อให้หน้า result เห็นสถานะกำลังประมวลผลใหม่)
    for name in ("result.mp4", "final.jpg", "result.gif", "palette.json"):
        _unlink(job_dir / name)

    background_tasks.add_task(process_image, str(original_path), job_dir, k=k)
    return JSONResponse({"job_id": job_id, "k": k, "status": "reprocessing"})

@app.get("/api/cache/{job_id}")
def cache_current_as_k(job_id: str, k: int = Query(..., ge=1, le=9)):
    """
    เรียกหลังจากผลล่าสุดพร้อมแล้ว เพื่อ snapshot เก็บเป็น result_k{k}.*
    ใช้ได้ทั้งตอนรอบแรก และหลัง reprocess แต่ละครั้ง
    """
    job_dir = STATIC_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    ok = _snapshot_current_as_k(job_dir, int(k))
    return {"ok": ok, "k": int(k)}

@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    d = STATIC_DIR / job_id
    mp4 = d / "result.mp4"
    jpg = d / "final.jpg"
    gif = d / "result.gif"

    ready_video = _exists_nonempty(mp4)
    ready_image = _exists_nonempty(jpg)
    ready_gif   = _exists_nonempty(gif)

    out = {
        "ready": ready_video,
        "result_url": f"/static/{job_id}/result.mp4" if ready_video else None,
        "image_ready": ready_image,
        "image_url": f"/static/{job_id}/final.jpg" if ready_image else None,
        "gif_ready": ready_gif,
        "gif_url": f"/static/{job_id}/result.gif" if ready_gif else None,
    }
    return out

@app.get("/api/colors/{job_id}")
def get_colors(job_id: str):
    p = STATIC_DIR / job_id / "palette.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Palette not ready")
    try:
        data = json.loads(p.read_text())
        return data
    except Exception:
        raise HTTPException(status_code=500, detail="Palette read error")

def _file_response_keep_or_delete(p: Path, media_type: str, filename: str, keep: int, job_id: str):
    if not (_exists_nonempty(p)):
        raise HTTPException(status_code=404, detail="Result not ready")
    if keep == 1:
        return FileResponse(p, media_type=media_type, filename=filename)
    return FileResponse(
        p,
        media_type=media_type,
        filename=filename,
        background=BackgroundTask(lambda: (_unlink(p), _cleanup_if_empty(job_id)))
    )

@app.get("/api/video/{job_id}")
def stream_video(
    job_id: str,
    keep: int = Query(1, ge=0, le=1),
    download: int = Query(0, ge=0, le=1),
):
    mp4 = STATIC_DIR / job_id / "result.mp4"
    if not _exists_nonempty(mp4):
        raise HTTPException(status_code=404, detail="Result not ready")

    if download == 1:
        return _file_response_keep_or_delete(mp4, "video/mp4", f"sketch_{job_id}.mp4", keep, job_id)

    if keep == 1:
        return FileResponse(mp4, media_type="video/mp4")
    else:
        return FileResponse(
            mp4,
            media_type="video/mp4",
            background=BackgroundTask(lambda: (_unlink(mp4), _cleanup_if_empty(job_id)))
        )

@app.get("/api/image/{job_id}")
def stream_image(job_id: str, keep: int = Query(1, ge=0, le=1)):
    jpg = STATIC_DIR / job_id / "final.jpg"
    return _file_response_keep_or_delete(jpg, "image/jpeg", f"sketch_{job_id}.jpg", keep, job_id)

@app.get("/api/gif/{job_id}")
def stream_gif(job_id: str, keep: int = Query(1, ge=0, le=1)):
    gif = STATIC_DIR / job_id / "result.gif"
    return _file_response_keep_or_delete(gif, "image/gif", f"sketch_{job_id}.gif", keep, job_id)

# ======== เพิ่ม: ดึงไฟล์ที่ cache ตาม K โดยตรง ========
@app.get("/api/video_k/{job_id}") #k ปัจจุบัน
def stream_video_k(
    job_id: str,
    k: int = Query(..., ge=1, le=9),
    keep: int = Query(1, ge=0, le=1),
    download: int = Query(0, ge=0, le=1),
):
    p = STATIC_DIR / job_id / f"result_k{k}.mp4"
    if not _exists_nonempty(p):
        raise HTTPException(status_code=404, detail="Cached video for this K not found")
    if download == 1:
        return _file_response_keep_or_delete(p, "video/mp4", f"sketch_{job_id}_K{k}.mp4", keep, job_id)
    return FileResponse(p, media_type="video/mp4")

@app.get("/api/colors_k/{job_id}") # ส่งพาเจตสี
def get_colors_k(job_id: str, k: int = Query(..., ge=1, le=9)):
    p = STATIC_DIR / job_id / f"palette_k{k}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Cached palette for this K not found")
    try:
        return json.loads(p.read_text())
    except Exception:
        raise HTTPException(status_code=500, detail="Palette read error")

# Static ก่อนหน้า frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
