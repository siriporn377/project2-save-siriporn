# segment/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
from segment.render_strokes import process_image  # ใช้ฟังก์ชันวาดภาพ

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    upload_path = "segment/uploaded.jpg"
    result_path = "segment/output/result.jpg"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    process_image(upload_path, result_path)  # ← คุณมี logic อยู่แล้วใน render_strokes.py

    return FileResponse(result_path, media_type="image/jpeg")
