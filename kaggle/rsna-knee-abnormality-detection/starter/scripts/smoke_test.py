#!/usr/bin/env python3
"""End-to-end wiring check on the 3 sample test studies (CPU is fine, ~1 min):
cache -> fake labels/folds -> 1 training epoch -> inference -> submission.csv.

  python scripts/smoke_test.py --data-dir /kaggle/input/competitions/rsna-knee-abnormality-detection
It only proves the plumbing works; the numbers it prints mean nothing.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from knee.config import LABELS  # noqa: E402


def run(*args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--backbone", default="resnet18")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="knee_smoke_")
    cache = os.path.join(tmp, "cache")
    run(os.path.join(HERE, "01_build_cache.py"), "--data-dir", a.data_dir, "--split", "test",
        "--out", cache, "--workers", "2", "--img-size", "128")

    uids = pd.read_csv(os.path.join(a.data_dir, "test.csv"))["StudyInstanceUID"].astype(str).tolist()
    rng = np.random.default_rng(0)
    fake = pd.DataFrame(rng.uniform(0, 1, (len(uids), len(LABELS))), columns=LABELS)
    fake.insert(0, "StudyInstanceUID", uids)
    fake.insert(1, "Report", "")
    fake[LABELS] = np.nan  # no gold rows
    fake.to_csv(os.path.join(tmp, "train.csv"), index=False)
    lab = pd.DataFrame(rng.uniform(0, 1, (len(uids), len(LABELS))), index=pd.Index(uids, name="StudyInstanceUID"),
                       columns=LABELS)
    lab.to_csv(os.path.join(tmp, "labels.csv"))
    pd.DataFrame({"StudyInstanceUID": uids, "fold": [0] + [1] * (len(uids) - 1)}).to_csv(
        os.path.join(tmp, "folds.csv"), index=False)

    run(os.path.join(HERE, "04_train.py"), "--cache", cache, "--labels", os.path.join(tmp, "labels.csv"),
        "--folds", os.path.join(tmp, "folds.csv"), "--train-csv", os.path.join(tmp, "train.csv"),
        "--fold", "0", "--out", os.path.join(tmp, "run"), "--backbone", a.backbone, "--pretrained", "0",
        "--img", "96", "--k", "4", "--bs", "2", "--epochs", "1", "--workers", "0")
    out = os.path.join(tmp, "submission.csv")
    run(os.path.join(HERE, "06_infer_submit.py"), "--ckpt", os.path.join(tmp, "run", "best.pt"),
        "--data-dir", a.data_dir, "--out", out, "--workers", "2")
    print(pd.read_csv(out).round(3).to_string())
    print("\nsmoke test OK, artefacts in", tmp)


if __name__ == "__main__":
    main()
