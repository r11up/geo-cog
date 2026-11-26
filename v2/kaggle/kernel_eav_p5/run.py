"""EAV, EEG only, protocol P5 (Malallah & Iqbal 2025; declared 26 Nov 2026): runs
scripts/eav_newmethods.py with the preprocessed EAV epochs of a private dataset. EEG-derived working files
are deleted at the end, so the output holds only result tables and the log."""
import glob
import os
import shutil
import subprocess
import sys

PROTOCOL, JOBS = "p5", "4"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "braindecode", "pyriemann", "huggingface_hub"], check=False)
import numpy as np  # noqa: E402


def find(*markers):
    for depth in range(1, 6):
        for marker in markers:
            hits = glob.glob("/kaggle/input/" + "*/" * depth + marker)
            if hits:
                return hits[0][:-len(marker)].rstrip("/")
    raise SystemExit(f"{markers} not found")


DATA = find("eav_raw.npy")
CODE = find("urbeeg/config.py", "code/urbeeg/config.py")
CODE = CODE + "/code" if os.path.isdir(CODE + "/code/urbeeg") else CODE
WORK = "/kaggle/working"
code_dir, derived = f"{WORK}/code", f"{WORK}/derived"
shutil.copytree(CODE, code_dir, dirs_exist_ok=True)
os.makedirs(f"{derived}/eav", exist_ok=True)
os.symlink(f"{DATA}/eav_raw.npy", f"{derived}/eav/eav_raw.npy")
z = np.load(f"{DATA}/eav_labels.npz")
np.savez(f"{derived}/eav/eav_features_v2.npz", y=z["y"], subject=z["subject"], trial=z["trial"])
env = dict(os.environ, URBEEG_DERIVED=derived, URBEEG_RESULTS=f"{WORK}/results", OMP_NUM_THREADS="4")
win = 200 if PROTOCOL == "p3" else 500
subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = ['.', 'scripts']; import eav_newmethods as nm; "
                f"nm.load_windows({win}, 4)"], cwd=code_dir, env=env, check=True)
log_path = f"{WORK}/{PROTOCOL}.log"
with open(log_path, "w") as log:
    args = [sys.executable, "scripts/eav_newmethods.py", "--protocol", PROTOCOL, "--jobs", JOBS]
    rc = subprocess.run(args, cwd=code_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print(open(log_path).read()[-8000:], flush=True)
shutil.rmtree(derived, ignore_errors=True)                  # no EEG-derived files in the output
shutil.rmtree(code_dir, ignore_errors=True)
print("exit code", rc, "; outputs:", sorted(glob.glob(f"{WORK}/results/*")), flush=True)
