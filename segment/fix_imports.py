import os
import re

# โฟลเดอร์ sketch_repo ที่อยู่ในโปรเจกต์
ROOT = os.path.join(os.path.dirname(__file__), "sketch_repo")

# รายชื่อโมดูลที่ต้องการให้แก้เป็น relative import
targets = {
    "simulate", "drawpatch", "tools", "tone", "LDR",
    "genStroke_origin", "quicksort", "deblue"
}

# สร้าง regex จับบรรทัด import ที่ต้องแก้
pattern = re.compile(r"^(\s*)from\s+(" + "|".join(targets) + r")\s+import\s+", re.M)

changed = 0

# ไล่ทุกไฟล์ .py ใน sketch_repo
for fn in os.listdir(ROOT):
    path = os.path.join(ROOT, fn)
    if not (os.path.isfile(path) and fn.endswith(".py")):
        continue

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # แก้ไขให้เป็น from .xxx import ...
    new_src = pattern.sub(lambda m: f"{m.group(1)}from .{m.group(2)} import ", src)

    if new_src != src:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_src)
        print(f"patched: {fn}")
        changed += 1

# ตรวจว่ามี __init__.py หรือยัง ถ้าไม่มีก็สร้าง
init_path = os.path.join(ROOT, "__init__.py")
if not os.path.exists(init_path):
    open(init_path, "w", encoding="utf-8").close()
    print("created __init__.py")

print(f"done. files changed: {changed}")
