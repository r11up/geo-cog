"""Study 1, round 6 (declared 26 Nov 2026, 21:57): fusion-operator controls and baselines on the
subject-level cross-validation folds of the final run, with the EAV features of a private dataset.
Runs scripts/run_study1_fusion.py; EEG-derived working files are deleted at the end, so the output
holds only result tables and the log."""
import glob
import os
import shutil
import subprocess
import sys


def find(*markers):
    for depth in range(1, 6):
        for marker in markers:
            hits = glob.glob("/kaggle/input/" + "*/" * depth + marker)
            if hits:
                return hits[0][:-len(marker)].rstrip("/")
    raise SystemExit(f"{markers} not found")


DATA = find("eav_features_v2.npz")
CODE = find("urbeeg/config.py", "code/urbeeg/config.py")
CODE = CODE + "/code" if os.path.isdir(CODE + "/code/urbeeg") else CODE
WORK = "/kaggle/working"
code_dir, derived, results = f"{WORK}/code", f"{WORK}/derived", f"{WORK}/results"
shutil.copytree(CODE, code_dir, dirs_exist_ok=True)
os.makedirs(f"{derived}/eav", exist_ok=True)
os.symlink(f"{DATA}/eav_features_v2.npz", f"{derived}/eav/eav_features_v2.npz")
for f in ("lockbox_subjects.txt", "astana_grid.csv"):
    shutil.copy(f"{DATA}/{f}", f"{derived}/eav/{f}")
ref = glob.glob(f"{DATA}/ref/study1v2_cv_*")[0]
os.makedirs(results, exist_ok=True)
shutil.copytree(ref, f"{results}/{os.path.basename(ref)}", dirs_exist_ok=True)
subprocess.run(["nvidia-smi", "-L"], check=False)
env = dict(os.environ, URBEEG_DERIVED=derived, URBEEG_RESULTS=results, OMP_NUM_THREADS="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           PYTHONWARNINGS="ignore")
log_path = f"{WORK}/study1fusion.log"
with open(log_path, "w") as log:
    rc = subprocess.run([sys.executable, "-I", "scripts/run_study1_fusion.py", "--jobs", "4"], cwd=code_dir,
                        env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print(open(log_path).read()[-6000:], flush=True)
shutil.rmtree(derived, ignore_errors=True)                  # no EEG-derived inputs in the output
shutil.rmtree(code_dir, ignore_errors=True)
shutil.rmtree(f"{results}/{os.path.basename(ref)}", ignore_errors=True)
for f in glob.glob(f"{results}/tracker.*"):                 # keep the local tracker the only tracker
    os.rename(f, f.replace("tracker.", "tracker_round6."))
print("exit code", rc, "; outputs:", sorted(glob.glob(f"{results}/*")), flush=True)
