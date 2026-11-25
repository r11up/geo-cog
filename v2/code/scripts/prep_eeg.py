"""Preprocess every fully downloaded participant that has not been processed yet.

Safe to run repeatedly while the download is in progress:
    ~/.venvs/urbeeg/bin/python -I scripts/prep_eeg.py --jobs 3
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from joblib import Parallel, delayed  # noqa: E402

from urbeeg.config import DERIVED  # noqa: E402
from urbeeg.download_check import complete_participants  # noqa: E402
from urbeeg.eeg import preprocess_participant  # noqa: E402

OUT = DERIVED / "eeg"


def run_one(sub: str) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    t = time.time()
    try:
        preprocess_participant(sub, OUT)
        return {"participant": sub, "ok": True, "seconds": round(time.time() - t)}
    except Exception as e:  # keep going; record the failure for inspection
        return {"participant": sub, "ok": False, "error": repr(e),
                "trace": traceback.format_exc()[-2000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="process just these participants")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done, pending = complete_participants()
    todo = [s for s in (args.only or done) if not (OUT / f"{s}_epochs.npz").exists()]
    print(f"downloaded: {len(done)}  still downloading: {len(pending)}  to process now: {len(todo)}",
          flush=True)
    results = Parallel(n_jobs=args.jobs, verbose=0)(delayed(run_one)(s) for s in todo)
    log = OUT / "prep_log.jsonl"
    with log.open("a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
            print(r if not r["ok"] else f"{r['participant']} ok in {r['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
