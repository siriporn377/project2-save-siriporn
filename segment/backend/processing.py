from pathlib import Path
import shutil, subprocess, sys, os, time, glob

ROOT = Path(__file__).resolve().parent          # ...\segment\backend
PROJECT_ROOT = ROOT.parent.parent               # ...\pro-2-backend   (ขึ้นไป 2 ชั้น)
SEG_DIR = PROJECT_ROOT / "segment"              # ...\pro-2-backend\segment
PYTHON = sys.executable
