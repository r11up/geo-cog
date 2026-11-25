"""Kaggle GPU runner for the deep models (geo-cog revision, Study 2). MODELS is set per kernel."""
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

MODELS = ["dualtower"]
START = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "braindecode", "pyriemann"], check=True)

def find(*markers):
    """Root of the mounted dataset holding one of `markers` (relative paths). Kaggle mounts
    at /kaggle/input/<slug> or deeper (/kaggle/input/datasets/<owner>/<slug>)."""
    for marker in markers:
        for depth in range(1, 5):
            hits = glob.glob("/kaggle/input/" + "*/" * depth + marker)
            if hits:
                return hits[0][:-len(marker)].rstrip("/")
    subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"])
    raise SystemExit(f"none of {markers} found under /kaggle/input")


SRC = find("scene_features.csv")
CODE_SRC = find("code/scripts/run_deep.py", "scripts/run_deep.py", "code.zip")
print("inputs:", SRC, CODE_SRC, flush=True)
W, D = "/kaggle/working", "/kaggle/working/derived"
for name, src_root in (("eeg", SRC), ("features", SRC), ("code", CODE_SRC)):
    dst = f"{W}/code" if name == "code" else f"{D}/{name}"
    os.makedirs(dst, exist_ok=True)
    if os.path.isdir(f"{src_root}/{name}"):
        shutil.copytree(f"{src_root}/{name}", dst, dirs_exist_ok=True)
    elif os.path.exists(f"{src_root}/{name}.zip"):
        zipfile.ZipFile(f"{src_root}/{name}.zip").extractall(dst)
    elif name == "code" and os.path.isdir(f"{src_root}/scripts"):   # folder contents mounted at the root
        shutil.copytree(src_root, dst, dirs_exist_ok=True)
    nested = f"{dst}/{name}"                       # zips may contain the folder itself
    if os.path.isdir(nested):
        for f in os.listdir(nested):
            shutil.move(f"{nested}/{f}", dst)
        os.rmdir(nested)
print("files:", {n: len(os.listdir(f"{D}/{n}")) for n in ("eeg", "features")}, flush=True)

os.environ.update(URBEEG_DERIVED=D, URBEEG_RESULTS=f"{W}/results",
                  URBEEG_SCENE_CSV=f"{SRC}/scene_features.csv",
                  URBEEG_DEADLINE=str(START + 11 * 3600))       # Kaggle sessions end at 12 h
subprocess.run([sys.executable, f"{W}/code/scripts/run_deep.py", "--models", *MODELS], check=True)
shutil.make_archive(f"{W}/deep_results", "zip", f"{W}/results")
print("done in", round((time.time() - START) / 60), "min")
