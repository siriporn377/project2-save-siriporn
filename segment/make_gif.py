# make_gif.py
from PIL import Image, UnidentifiedImageError
import os
import glob


base_dir = os.path.dirname(os.path.abspath(__file__))
frame_dir = os.path.join(base_dir, 'output', 'frames_input25')  #ชื่อเฟรม
output_path = os.path.join(base_dir, 'output', 'stroke_render_input25.gif')  # ชื่อไฟล์ GIF 

#  Load frames 
frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
print(f"🔍 เจอ {len(frame_files)} เฟรมที่: {frame_dir}")

if not frame_files:
    raise FileNotFoundError(f" ไม่พบภาพใน {frame_dir}")

frames = []

for f in frame_files:
    print("กำลังโหลด:", f)
    try:
        img = Image.open(f).convert("RGB")
        frames.append(img)
    except UnidentifiedImageError:
        print(f"ข้ามภาพที่โหลดไม่ได้: {f}")
    except Exception as e:
        print(f"Error ไม่คาดคิด: {f} → {e}")

if not frames:
    raise ValueError("ไม่สามารถสร้าง GIF ได้")

#  Save GIF
try:
    print(f" กำลังสร้าง GIF ที่: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ระยะเวลาเฟรม (ms)
        loop=0
    )
    print(f" GIF สร้างแล้วสำเร็จที่: {output_path}")
except Exception as e:
    print(f" เกิดข้อผิดพลาดตอนบันทึก GIF: {e}")
